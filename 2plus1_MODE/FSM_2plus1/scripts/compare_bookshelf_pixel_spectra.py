#!/usr/bin/env python3
"""Per-pixel frequency comparison of four-step and 2+1, plus a valid-pixel table.

Each selected camera pixel is plotted the same way as FSM_decode_example.visualize_spectrum:
raw intensities, quadrature signals, complex magnitude, linear spectrum, log spectrum,
and a zoom around the main peak. Four-step is solid; 2+1 is dashed.

The summary table uses every pixel whose four-step normalized peak is at least 1.5.
This capture is vertical fringes, so the coordinate is the projector column.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(
    r"E:\images\before_2026_5_23\before_2025_12_27\images_FSM"
    r"\place1\03_bookshelf_white_cycle1\FSM_vertical_f0_f150\images_cache.npy"
)
OUT = ROOT / "2plus1_逐像素频谱"

N_FREQ = 151
FFT_N = 1920
STEP_HZ = 1.0
PEAK_VALID = 1.5
REFINE_MIN = 0.1
BATCH = 8
POINTS = (
    (1200, 1000),
    (1409, 1176),
    (900, 400),
    (1100, 1100),
    (1500, 700),
    (700, 900),
)


def setup_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


def decode_peaks(signal: np.ndarray):
    """signal (F, B, W) complex64 -> idx, coord, value (B, W) and magnitude (FFT_N, B, W)."""
    spectrum = np.fft.fft(signal, n=FFT_N, axis=0)
    mag = np.abs(spectrum)
    idx = np.argmax(mag, axis=0)
    batch, width = idx.shape
    rows = np.arange(batch)[:, None]
    cols = np.arange(width)[None, :]
    center = mag[idx, rows, cols]
    left = mag[(idx - 1) % FFT_N, rows, cols]
    right = mag[(idx + 1) % FFT_N, rows, cols]
    value = center / FFT_N
    coord = idx.astype(np.float64)
    denom = left + center + right
    ok = (value > REFINE_MIN) & (denom > 0)
    coord[ok] += 0.5 * (right[ok] - left[ok]) / denom[ok]
    return idx.astype(np.int32), coord.astype(np.float32), value.astype(np.float32), mag / FFT_N


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_pixel(sample: dict, folder: Path) -> None:
    freq = np.arange(N_FREQ) * STEP_HZ
    bins = np.arange(FFT_N)
    title = f"像素({sample['x']}, {sample['y']})  VER"
    valid = "有效" if sample["value4"] >= PEAK_VALID else "无效"
    spec4 = sample["spec4"]
    spec21 = sample["spec21"]
    peak4 = sample["idx4"]
    peak21 = sample["idx21"]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(freq, sample["I0"], "o-", label="I0(0°)", lw=1.2, ms=3)
    ax.plot(freq, sample["I1"], "s-", label="I1(90°)", lw=1.2, ms=3)
    ax.plot(freq, sample["I2"], "^-", label="I2(180°)", lw=1.2, ms=3)
    ax.plot(freq, sample["I3"], "d-", label="I3(270°)", lw=1.2, ms=3)
    ax.plot(freq, sample["A4"], "k--", label="A4 四帧平均", lw=1.4)
    ax.set(xlabel="频率 (Hz)", ylabel="强度", title=f"{title}\n1. 原始强度序列（M={N_FREQ}，步长={STEP_HZ:g} Hz）")
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    save_fig(folder / "1_原始强度.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(freq, sample["Dc4"], "-", label="四步 Dc=I0-I2", lw=1.4)
    ax.plot(freq, sample["Ds4"], "-", label="四步 Ds=I3-I1", lw=1.4)
    ax.plot(freq, sample["Dc21"], "--", label="2+1 Dc=2(I0-A4)", lw=1.4)
    ax.plot(freq, sample["Ds21"], "--", label="2+1 Ds=2(A4-I1)", lw=1.4)
    ax.set(xlabel="频率 (Hz)", ylabel="差分值", title=f"{title}\n2. 差分信号")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(folder / "2_差分信号.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(freq, sample["mag4"], "-", color="purple", label="四步 |C|", lw=1.5)
    ax.plot(freq, sample["mag21"], "--", color="C1", label="2+1 |C|", lw=1.5)
    ax.set(xlabel="频率 (Hz)", ylabel="|C|", title=f"{title}\n3. 复数信号幅值")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(folder / "3_复数幅值.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(bins, spec4, "-", lw=1.1, label="四步")
    ax.plot(bins, spec21, "--", lw=1.1, label="2+1")
    ax.axvline(peak4, color="C0", ls=":", lw=1.2, label=f"四步峰={peak4}")
    ax.axvline(peak21, color="C1", ls=":", lw=1.2, label=f"2+1峰={peak21}")
    ax.set(xlabel="投影仪列坐标", ylabel="归一化幅值",
           title=f"{title}  {valid}，峰值={sample['value4']:.3f}\n4. 有效 DFT 频谱（线性）")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(folder / "4_频谱线性.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(bins, np.log10(spec4 + 1e-10), "-", color="green", lw=1.1, label="四步")
    ax.plot(bins, np.log10(spec21 + 1e-10), "--", color="C1", lw=1.1, label="2+1")
    ax.axvline(peak4, color="C0", ls=":", lw=1.2, label=f"四步峰={peak4}")
    ax.axvline(peak21, color="C1", ls=":", lw=1.2, label=f"2+1峰={peak21}")
    ax.set(xlabel="投影仪列坐标", ylabel="log10(幅值)", title=f"{title}\n5. 有效 DFT 频谱（对数）")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(folder / "5_频谱对数.png")

    zoom_s = max(0, int(peak4) - 100)
    zoom_e = min(FFT_N, int(peak4) + 100)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    sl = slice(zoom_s, zoom_e)
    ax.plot(bins[sl], spec4[sl], "o-", lw=1.5, ms=3, label="四步")
    ax.plot(bins[sl], spec21[sl], "s--", lw=1.5, ms=3, label="2+1")
    ax.axvline(peak4, color="C0", ls="--", lw=1.2, label=f"四步峰={peak4}")
    ax.axvline(peak21, color="C1", ls="--", lw=1.2, label=f"2+1峰={peak21}")
    ax.set(xlabel="投影仪列坐标", ylabel="归一化幅值",
           title=f"{title}\n6. 峰值局部放大 [{zoom_s}, {zoom_e}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(folder / "6_峰值放大.png")

    lines = [
        f"pixel ({sample['x']}, {sample['y']})",
        f"four-step peak index {sample['idx4']}  value {sample['value4']:.6f}  coord {sample['coord4']:.6f}",
        f"2+1 peak index {sample['idx21']}  value {sample['value21']:.6f}  coord {sample['coord21']:.6f}",
        f"abs column error {abs(sample['coord21'] - sample['coord4']):.6f}",
        f"valid peak>={PEAK_VALID}: {sample['value4'] >= PEAK_VALID}",
    ]
    (folder / "峰值.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(stats: dict, samples: list[dict], out: Path) -> None:
    n = stats["n"]
    total = stats["total"]
    err = stats["err"]
    bins = stats["bins"]

    def pct(count: int) -> str:
        return f"{100.0 * count / n:.4f}%" if n else "nan"

    limits = (0.5, 1.0, 2.0, 5.0)
    within = {lim: int(np.count_nonzero(err <= lim)) for lim in limits}
    qs = np.percentile(err, (50, 90, 95, 99)) if n else [float("nan")] * 4
    same = int(np.count_nonzero(bins == 0))
    off1 = int(np.count_nonzero(bins == 1))
    offmore = int(np.count_nonzero(bins > 1))
    gt1 = int(np.count_nonzero(err > 1))
    gt5 = int(np.count_nonzero(err > 5))

    rows = [
        "| 项目 | 数值 |",
        "| --- | ---: |",
        f"| 全图像素 | {total} |",
        f"| 有效像素（四步峰值 ≥ {PEAK_VALID:g}） | {n} |",
        f"| 有效像素占全图 | {100.0 * n / total:.4f}% |",
        f"| 整数列号相同 | {same}（{pct(same)}） |",
        f"| 整数列号差 1 | {off1}（{pct(off1)}） |",
        f"| 整数列号差超过 1 | {offmore}（{pct(offmore)}） |",
    ]
    for lim in limits:
        rows.append(f"| 亚像素绝对差 ≤ {lim:g} 列 | {within[lim]}（{pct(within[lim])}） |")
    rows += [
        f"| 绝对差中位数 | {qs[0]:.6f} |",
        f"| 绝对差 90% 分位 | {qs[1]:.6f} |",
        f"| 绝对差 95% 分位 | {qs[2]:.6f} |",
        f"| 绝对差 99% 分位 | {qs[3]:.6f} |",
        f"| 绝对差 > 1 列 | {gt1} |",
        f"| 绝对差 > 5 列 | {gt5} |",
    ]
    sample_lines = [
        "",
        "抽样像素（峰值低于 1.5 的点不用于支持替代结论）：",
        "",
        "| 像素 (x, y) | 四步峰值 | 四步列 | 2+1 列 | 绝对差 | 是否有效 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for sample in samples:
        flag = "有效" if sample["value4"] >= PEAK_VALID else "无效"
        sample_lines.append(
            f"| ({sample['x']}, {sample['y']}) | {sample['value4']:.4f} | "
            f"{sample['coord4']:.4f} | {sample['coord21']:.4f} | "
            f"{abs(sample['coord21'] - sample['coord4']):.4f} | {flag} |"
        )
    text = "\n".join([
        "# 有效像素的投影仪列坐标误差",
        "",
        "数据为垂直条纹 `FSM_vertical_f0_f150`，频率 0 到 150 Hz，共 151 个。",
        "有效像素指四步归一化峰值不低于 1.5。误差是 2+1 与四步的亚像素列坐标绝对差。",
        "",
        *rows,
        *sample_lines,
        "",
    ])
    (out / "有效像素列坐标误差表.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"加载缓存 {CACHE}")
    images = np.load(CACHE)
    height, width = images.shape[1], images.shape[2]
    wanted = {(x, y): None for x, y in POINTS}
    err_parts = []
    bin_parts = []
    valid_n = 0
    total = height * width

    for y0 in range(0, height, BATCH):
        t0 = time.perf_counter()
        y1 = min(height, y0 + BATCH)
        block = images[:, y0:y1, :].astype(np.float32)
        i0, i1, i2, i3 = block[0::4], block[1::4], block[2::4], block[3::4]
        a4 = (i0 + i1 + i2 + i3) / 4.0
        signal4 = ((i0 - i2) + 1j * (i3 - i1)).astype(np.complex64)
        signal21 = (2.0 * ((i0 - a4) + 1j * (a4 - i1))).astype(np.complex64)
        idx4, coord4, value4, mag4 = decode_peaks(signal4)
        idx21, coord21, value21, mag21 = decode_peaks(signal21)
        valid = value4 >= PEAK_VALID
        valid_n += int(valid.sum())
        if np.any(valid):
            err_parts.append(np.abs(coord21 - coord4)[valid].astype(np.float32))
            bin_parts.append(np.abs(idx21.astype(np.int32) - idx4)[valid])
        for x, y in wanted:
            if y0 <= y < y1:
                ly = y - y0
                wanted[(x, y)] = {
                    "x": x,
                    "y": y,
                    "I0": i0[:, ly, x].copy(),
                    "I1": i1[:, ly, x].copy(),
                    "I2": i2[:, ly, x].copy(),
                    "I3": i3[:, ly, x].copy(),
                    "A4": a4[:, ly, x].copy(),
                    "Dc4": (i0[:, ly, x] - i2[:, ly, x]).copy(),
                    "Ds4": (i3[:, ly, x] - i1[:, ly, x]).copy(),
                    "Dc21": (2.0 * (i0[:, ly, x] - a4[:, ly, x])).copy(),
                    "Ds21": (2.0 * (a4[:, ly, x] - i1[:, ly, x])).copy(),
                    "mag4": np.abs(signal4[:, ly, x]).copy(),
                    "mag21": np.abs(signal21[:, ly, x]).copy(),
                    "spec4": mag4[:, ly, x].copy(),
                    "spec21": mag21[:, ly, x].copy(),
                    "idx4": int(idx4[ly, x]),
                    "idx21": int(idx21[ly, x]),
                    "coord4": float(coord4[ly, x]),
                    "coord21": float(coord21[ly, x]),
                    "value4": float(value4[ly, x]),
                    "value21": float(value21[ly, x]),
                }
        print(f"\r峰值行 {y1}/{height}  {time.perf_counter() - t0:.2f}s", end="", flush=True)
    print()

    samples = []
    for x, y in POINTS:
        sample = wanted[(x, y)]
        if sample is None:
            raise SystemExit(f"missing pixel ({x}, {y})")
        folder = OUT / f"pixel_{x}_{y}"
        plot_pixel(sample, folder)
        samples.append(sample)
        flag = "有效" if sample["value4"] >= PEAK_VALID else "无效"
        print(
            f"({x},{y}) {flag}  peak4={sample['value4']:.4f}  "
            f"col {sample['coord4']:.4f} vs {sample['coord21']:.4f}"
        )

    err = np.concatenate(err_parts) if err_parts else np.zeros(0, np.float32)
    bins = np.concatenate(bin_parts) if bin_parts else np.zeros(0, np.int32)
    write_table({"n": valid_n, "total": total, "err": err, "bins": bins}, samples, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
