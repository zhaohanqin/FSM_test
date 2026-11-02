#!/usr/bin/env python3
"""Compare four-step and 2+1 on the bookshelf vertical capture.

The 2+1 background is the four-frame average of the same frequency:

    A4 = (I0 + I1 + I2 + I3) / 4
    C4 = (I0 - I2) + 1j * (I3 - I1)
    C21 = 2 * ((I0 - A4) + 1j * (A4 - I1))

Peak detection matches FSM_decode_core low-memory mode: float32,
fft length 1920, centroid refinement when the normalized peak exceeds 0.1.
"""
from __future__ import annotations

import csv
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
OUT = ROOT / "results" / "2plus1_bookshelf_vertical"

N_FREQ = 151
FFT_N = 1920
ROI_MAG = 8.0
PEAK_MIN = 0.1
ABS_MAX = 64.0
ABS_BINS = 6400
PHASE_MAX = 180.0
PHASE_BINS = 3600
DISP_MAX = 20.0
DISP_BINS = 20000
STEP = 4
BATCH = 8
SPECTRUM_POINTS = ((1200, 1000),)


def setup_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def empty_hist(nbins: int) -> np.ndarray:
    return np.zeros(nbins, dtype=np.int64)


def add_hist(hist: np.ndarray, values: np.ndarray, vmax: float, overflow: int) -> int:
    if values.size == 0:
        return overflow
    counts, _ = np.histogram(values, bins=hist.size, range=(0.0, vmax))
    hist += counts
    overflow += int(np.count_nonzero(values >= vmax))
    return overflow


def hist_percentile(hist: np.ndarray, vmax: float, overflow: int, q: float) -> float:
    total = int(hist.sum()) + int(overflow)
    if total <= 0:
        return float("nan")
    target = (q / 100.0) * total
    csum = np.cumsum(hist)
    if target > csum[-1]:
        return float(vmax)
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = min(idx, hist.size - 1)
    prev = 0.0 if idx == 0 else float(csum[idx - 1])
    width = vmax / hist.size
    frac = 0.0 if hist[idx] == 0 else (target - prev) / float(hist[idx])
    return float((idx + frac) * width)


class Running:
    def __init__(self, vmax: float, nbins: int):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq_diff = 0.0
        self.sum_sq_ref = 0.0
        self.max_abs = 0.0
        self.hist = empty_hist(nbins)
        self.overflow = 0
        self.vmax = vmax

    def update(self, diff: np.ndarray, ref: np.ndarray) -> None:
        if diff.size == 0:
            return
        mag = np.abs(diff)
        self.n += int(mag.size)
        self.sum_abs += float(mag.sum())
        self.sum_sq_diff += float(np.square(mag).sum())
        self.sum_sq_ref += float(np.square(np.abs(ref)).sum())
        self.max_abs = max(self.max_abs, float(mag.max()))
        self.overflow = add_hist(self.hist, mag, self.vmax, self.overflow)

    def summary(self) -> dict:
        denom = max(self.sum_sq_ref, np.finfo(float).tiny)
        return {
            "n": self.n,
            "max": self.max_abs if self.n else float("nan"),
            "mean": self.sum_abs / self.n if self.n else float("nan"),
            "p95": hist_percentile(self.hist, self.vmax, self.overflow, 95),
            "p99": hist_percentile(self.hist, self.vmax, self.overflow, 99),
            "rel": float(np.sqrt(self.sum_sq_diff / denom)) if self.n else float("nan"),
        }


def decode_peaks(signal: np.ndarray):
    """signal (F, B, W) complex64 -> idx, coord, value, each (B, W)."""
    spectrum = np.fft.fft(signal, n=FFT_N, axis=0)
    mag = np.abs(spectrum)
    idx = np.argmax(mag, axis=0)
    batch, width = idx.shape
    rows, cols = np.indices((batch, width))
    center = mag[idx, rows, cols]
    left = mag[(idx - 1) % FFT_N, rows, cols]
    right = mag[(idx + 1) % FFT_N, rows, cols]
    value = center / FFT_N
    coord = idx.astype(np.float64)
    denom = left + center + right
    ok = (value > PEAK_MIN) & (denom > 0)
    coord[ok] += 0.5 * (right[ok] - left[ok]) / denom[ok]
    return idx.astype(np.int32), coord.astype(np.float32), value.astype(np.float32), spectrum


def save_gray(path: Path, image: np.ndarray, cmap: str, vmin: float, vmax: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x / 4")
    ax.set_ylabel("y / 4")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def compare_complex(images: np.ndarray, out: Path) -> dict:
    height, width = images.shape[1], images.shape[2]
    full_main, roi_main = Running(ABS_MAX, ABS_BINS), Running(ABS_MAX, ABS_BINS)
    full_f0, roi_f0 = Running(ABS_MAX, ABS_BINS), Running(ABS_MAX, ABS_BINS)
    phase_hist = empty_hist(PHASE_BINS)
    phase_overflow = 0
    phase_n = 0
    mag_abs_sum = 0.0
    mag_rel_sum = 0.0
    rows = []
    error_map = np.zeros((height // STEP, width // STEP), dtype=np.float64)
    best = None
    worst = None
    identity_max = 0.0

    for freq in range(N_FREQ):
        t0 = time.perf_counter()
        block = images[4 * freq: 4 * freq + 4].astype(np.float64)
        i0, i1, i2, i3 = block
        a4 = (i0 + i1 + i2 + i3) / 4.0
        c4 = (i0 - i2) + 1j * (i3 - i1)
        c21 = 2.0 * ((i0 - a4) + 1j * (a4 - i1))
        pair_gap = ((i0 + i2) - (i1 + i3)) / 2.0
        diff = c21 - c4
        identity = float(np.max(np.abs(diff - pair_gap * (1 + 1j))))
        identity_max = max(identity_max, identity)
        abs_diff = np.abs(diff)
        magnitude = np.abs(c4)
        roi = magnitude >= ROI_MAG

        target_full = full_f0 if freq == 0 else full_main
        target_roi = roi_f0 if freq == 0 else roi_main
        target_full.update(diff, c4)
        target_roi.update(diff[roi], c4[roi])

        roi_diff = abs_diff[roi]
        if freq != 0 and roi_diff.size:
            phase = np.abs(np.angle(c21[roi] * np.conj(c4[roi]))) * (180.0 / np.pi)
            phase_n += int(phase.size)
            phase_overflow = add_hist(phase_hist, phase, PHASE_MAX, phase_overflow)
            mag_delta = np.abs(np.abs(c21[roi]) - magnitude[roi])
            mag_abs_sum += float(mag_delta.sum())
            mag_rel_sum += float((mag_delta / np.maximum(magnitude[roi], 1e-12)).sum())
            error_map += abs_diff[::STEP, ::STEP]

        p95, p99 = (float("nan"), float("nan"))
        rp95, rp99 = (float("nan"), float("nan"))
        if abs_diff.size:
            p95, p99 = [float(v) for v in np.percentile(abs_diff, (95, 99))]
        if roi_diff.size:
            rp95, rp99 = [float(v) for v in np.percentile(roi_diff, (95, 99))]
        rel_full = float(np.linalg.norm(diff) / max(np.linalg.norm(c4), np.finfo(float).tiny))
        rel_roi = float("nan")
        if roi_diff.size:
            rel_roi = float(np.linalg.norm(diff[roi]) / max(np.linalg.norm(c4[roi]), np.finfo(float).tiny))
        rows.append({
            "f": freq,
            "max": float(abs_diff.max()),
            "mean": float(abs_diff.mean()),
            "p95": p95,
            "p99": p99,
            "rel": rel_full,
            "roi_n": int(roi.sum()),
            "roi_max": float(roi_diff.max()) if roi_diff.size else float("nan"),
            "roi_mean": float(roi_diff.mean()) if roi_diff.size else float("nan"),
            "roi_p95": rp95,
            "roi_p99": rp99,
            "roi_rel": rel_roi,
            "median_mag": float(np.median(magnitude)),
            "roi_median_mag": float(np.median(magnitude[roi])) if roi.any() else float("nan"),
            "pair_gap_mean": float(np.mean(np.abs(pair_gap))),
            "identity": identity,
        })

        if freq != 0:
            y = int(np.argmax(magnitude.mean(axis=1)))
            score = float(np.median(magnitude[roi])) if roi.any() else 0.0
            if best is None or score > best["score"]:
                candidates = np.argsort(magnitude[y])[-8:]
                samples = []
                for x in sorted(int(v) for v in candidates):
                    samples.append((x, y, i0[y, x], i1[y, x], i2[y, x], i3[y, x],
                                    a4[y, x], complex(c4[y, x]), complex(c21[y, x])))
                best = {
                    "score": score,
                    "f": freq,
                    "y": y,
                    "real4": np.real(c4[y]).copy(),
                    "imag4": np.imag(c4[y]).copy(),
                    "real21": np.real(c21[y]).copy(),
                    "imag21": np.imag(c21[y]).copy(),
                    "abs_diff": abs_diff[y].copy(),
                    "samples": samples,
                }
            flat = int(np.argmax(abs_diff))
            wy, wx = np.unravel_index(flat, abs_diff.shape)
            peak_err = float(abs_diff[wy, wx])
            if worst is None or peak_err > worst["abs"]:
                worst = {
                    "abs": peak_err,
                    "f": freq,
                    "x": int(wx),
                    "y": int(wy),
                    "I": (float(i0[wy, wx]), float(i1[wy, wx]), float(i2[wy, wx]), float(i3[wy, wx])),
                    "A4": float(a4[wy, wx]),
                    "C4": complex(c4[wy, wx]),
                    "C21": complex(c21[wy, wx]),
                    "mag": float(magnitude[wy, wx]),
                }
        print(f"\r复数 f={freq:3d}/{N_FREQ - 1}  {time.perf_counter() - t0:.2f}s", end="", flush=True)

    print()
    if identity_max > 1e-6:
        raise SystemExit(f"2+1 residual does not match the pair-average identity: {identity_max}")
    error_map /= max(N_FREQ - 1, 1)

    with (out / "per_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.save(out / "mean_abs_error_f1_f150.npy", error_map.astype(np.float32))
    vmax = float(np.percentile(error_map, 99.5))
    save_gray(
        out / "01_mean_abs_error.png", error_map, "magma", 0.0, max(vmax, 1e-6),
        "f=1…150 平均 |C21-C4|（每 4 像素）",
    )

    y = best["y"]
    x = np.arange(width)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(x, best["real4"], lw=1.0, label="四步")
    axes[0].plot(x, best["real21"], lw=1.0, ls="--", label="2+1")
    axes[0].set_ylabel("实部")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, best["imag4"], lw=1.0, label="四步")
    axes[1].plot(x, best["imag21"], lw=1.0, ls="--", label="2+1")
    axes[1].set_ylabel("虚部")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    axes[2].plot(x, best["abs_diff"], color="C3", lw=1.0)
    axes[2].set_ylabel("|C21-C4|")
    axes[2].set_xlabel(f"x（f={best['f']}，y={y}，该频率条纹区中位幅值最大）")
    axes[2].grid(alpha=0.3)
    fig.suptitle("四帧平均背景下的复数剖线", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "02_profile.png", dpi=140)
    plt.close(fig)

    lines = [
        "A4=(I0+I1+I2+I3)/4, float64, not rounded back to uint8.",
        "C4=(I0-I2)+1j*(I3-I1)",
        "C21=2*((I0-A4)+1j*(A4-I1))",
        "Identity: C21-C4 = ((I0+I2)-(I1+I3))/2 * (1+1j)",
        f"identity max abs residual = {identity_max:.3e}",
        "",
        f"strongest modulated frequency f={best['f']}, row y={best['y']}",
        f"{'x':>6} {'y':>6} {'I0':>8} {'I1':>8} {'I2':>8} {'I3':>8} {'A4':>10} {'C4':>24} {'C21':>24} {'|diff|':>10}",
    ]
    for item in best["samples"]:
        x_i, y_i, a, b, c, d, a4_i, c4_i, c21_i = item
        lines.append(
            f"{x_i:6d} {y_i:6d} {a:8.1f} {b:8.1f} {c:8.1f} {d:8.1f} {a4_i:10.4f} "
            f"{c4_i.real:+10.4f}{c4_i.imag:+10.4f}j "
            f"{c21_i.real:+10.4f}{c21_i.imag:+10.4f}j {abs(c21_i - c4_i):10.4f}"
        )
    lines += [
        "",
        f"max-error pixel f={worst['f']} (x,y)=({worst['x']},{worst['y']}) "
        f"|C4|={worst['mag']:.4f} |diff|={worst['abs']:.4f}",
        f"I={tuple(round(v, 1) for v in worst['I'])} A4={worst['A4']:.4f}",
        f"C4={worst['C4'].real:+.4f}{worst['C4'].imag:+.4f}j  "
        f"C21={worst['C21'].real:+.4f}{worst['C21'].imag:+.4f}j",
    ]
    (out / "pixel_values.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    phase_p95 = hist_percentile(phase_hist, PHASE_MAX, phase_overflow, 95)
    phase_p99 = hist_percentile(phase_hist, PHASE_MAX, phase_overflow, 99)
    return {
        "full_main": full_main.summary(),
        "roi_main": roi_main.summary(),
        "full_f0": full_f0.summary(),
        "roi_f0": roi_f0.summary(),
        "phase_p95_deg": phase_p95,
        "phase_p99_deg": phase_p99,
        "phase_n": phase_n,
        "mag_abs_mean": mag_abs_sum / phase_n if phase_n else float("nan"),
        "mag_rel_mean": mag_rel_sum / phase_n if phase_n else float("nan"),
        "identity_max": identity_max,
        "best_f": best["f"],
        "best_y": best["y"],
        "worst": worst,
        "per_frequency": rows,
    }


def compare_peaks(images: np.ndarray, out: Path) -> dict:
    height, width = images.shape[1], images.shape[2]
    same = off1 = offmore = 0
    conf_n = conf_same = conf_off1 = conf_offmore = 0
    conf_weak21 = 0
    disp_hist = empty_hist(DISP_BINS)
    disp_overflow = 0
    delta_map = np.zeros((height // STEP, width // STEP), dtype=np.float32)
    spectra = {}
    best_peak = None

    for y0 in range(0, height, BATCH):
        t0 = time.perf_counter()
        y1 = min(height, y0 + BATCH)
        block = images[:, y0:y1, :].astype(np.float32)
        i0, i1, i2, i3 = block[0::4], block[1::4], block[2::4], block[3::4]
        signal4 = ((i0 - i2) + 1j * (i3 - i1)).astype(np.complex64)
        a4 = (i0 + i1 + i2 + i3) / 4.0
        signal21 = (2.0 * ((i0 - a4) + 1j * (a4 - i1))).astype(np.complex64)
        idx4, coord4, value4, spec4 = decode_peaks(signal4)
        idx21, coord21, value21, spec21 = decode_peaks(signal21)
        delta_idx = np.abs(idx21.astype(np.int32) - idx4.astype(np.int32))
        same += int(np.count_nonzero(delta_idx == 0))
        off1 += int(np.count_nonzero(delta_idx == 1))
        offmore += int(np.count_nonzero(delta_idx > 1))
        conf = value4 > PEAK_MIN
        conf_n += int(conf.sum())
        conf_same += int(np.count_nonzero(conf & (delta_idx == 0)))
        conf_off1 += int(np.count_nonzero(conf & (delta_idx == 1)))
        conf_offmore += int(np.count_nonzero(conf & (delta_idx > 1)))
        conf_weak21 += int(np.count_nonzero(conf & (value21 <= PEAK_MIN)))
        du = np.abs(coord21 - coord4)
        disp_overflow = add_hist(disp_hist, du[conf], DISP_MAX, disp_overflow)
        ys = np.arange(y0, y1)
        take = (ys % STEP) == 0
        if np.any(take):
            delta_map[ys[take] // STEP] = du[take][:, ::STEP]

        local = int(np.argmax(value4))
        ly, lx = np.unravel_index(local, value4.shape)
        peak_here = float(value4[ly, lx])
        point = (int(y0 + ly), int(lx), peak_here)
        if best_peak is None or peak_here > best_peak[2]:
            best_peak = point
            spectra["max_peak"] = {
                "point": (int(lx), int(y0 + ly)),
                "spec4": (np.abs(spec4[:, ly, lx]) / FFT_N).astype(np.float32),
                "spec21": (np.abs(spec21[:, ly, lx]) / FFT_N).astype(np.float32),
                "idx4": int(idx4[ly, lx]),
                "idx21": int(idx21[ly, lx]),
                "value4": peak_here,
                "value21": float(value21[ly, lx]),
            }
        for x, y in SPECTRUM_POINTS:
            if y0 <= y < y1 and 0 <= x < width:
                ly = y - y0
                spectra[f"{x}_{y}"] = {
                    "point": (x, y),
                    "spec4": (np.abs(spec4[:, ly, x]) / FFT_N).astype(np.float32),
                    "spec21": (np.abs(spec21[:, ly, x]) / FFT_N).astype(np.float32),
                    "idx4": int(idx4[ly, x]),
                    "idx21": int(idx21[ly, x]),
                    "value4": float(value4[ly, x]),
                    "value21": float(value21[ly, x]),
                }
        print(f"\r峰值行 {y1}/{height}  {time.perf_counter() - t0:.2f}s", end="", flush=True)

    print()
    np.save(out / "peak_abs_delta.npy", delta_map)
    vmax = float(np.percentile(delta_map, 99.5))
    save_gray(
        out / "03_peak_abs_delta.png", delta_map, "magma", 0.0, max(vmax, 1e-3),
        "四步与 2+1 的投影仪列坐标绝对差（每 4 像素）",
    )

    bins = np.arange(FFT_N)
    fig, axes = plt.subplots(len(spectra), 1, figsize=(11, 3.4 * len(spectra)), squeeze=False)
    for ax, (name, item) in zip(axes[:, 0], spectra.items()):
        x, y = item["point"]
        ax.plot(bins, item["spec4"], lw=1.1, label="四步")
        ax.plot(bins, item["spec21"], lw=1.1, ls="--", label="2+1")
        ax.set_xlim(0, FFT_N - 1)
        ax.set_ylabel("归一化幅值")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        ax.set_title(
            f"{name} 像素({x},{y})  四步峰={item['idx4']} ({item['value4']:.4f})  "
            f"2+1峰={item['idx21']} ({item['value21']:.4f})"
        )
    axes[-1, 0].set_xlabel("FFT bin（步长 1.0 时等于投影仪列）")
    fig.tight_layout()
    fig.savefig(out / "04_spectra.png", dpi=140)
    plt.close(fig)

    total = height * width
    conf_du = du_percentiles(disp_hist, disp_overflow)
    return {
        "total": total,
        "same": same,
        "off1": off1,
        "offmore": offmore,
        "conf_n": conf_n,
        "conf_same": conf_same,
        "conf_off1": conf_off1,
        "conf_offmore": conf_offmore,
        "conf_weak21": conf_weak21,
        "conf_du": conf_du,
        "spectra": {k: {kk: vv for kk, vv in item.items() if kk != "spec4" and kk != "spec21"}
                    for k, item in spectra.items()},
    }


def du_percentiles(hist: np.ndarray, overflow: int) -> dict:
    return {
        "median": hist_percentile(hist, DISP_MAX, overflow, 50),
        "p95": hist_percentile(hist, DISP_MAX, overflow, 95),
        "p99": hist_percentile(hist, DISP_MAX, overflow, 99),
        "overflow": overflow,
    }


def fmt(stats: dict) -> str:
    return (
        f"n={stats['n']} max={stats['max']:.6f} mean={stats['mean']:.6f} "
        f"p95={stats['p95']:.6f} p99={stats['p99']:.6f} relL2={stats['rel']:.6e}"
    )


def write_summary(complex_stats: dict, peak_stats: dict, out: Path) -> None:
    conf_n = max(peak_stats["conf_n"], 1)
    total = peak_stats["total"]
    lines = [
        "four-frame average background, vertical bookshelf, f0..f150",
        f"ROI |C4| >= {ROI_MAG}",
        f"identity max residual {complex_stats['identity_max']:.3e}",
        "",
        "complex f=1..150 full  " + fmt(complex_stats["full_main"]),
        "complex f=1..150 roi   " + fmt(complex_stats["roi_main"]),
        "complex f=0 full       " + fmt(complex_stats["full_f0"]),
        "complex f=0 roi        " + fmt(complex_stats["roi_f0"]),
        f"roi phase abs deg p95={complex_stats['phase_p95_deg']:.4f} "
        f"p99={complex_stats['phase_p99_deg']:.4f} n={complex_stats['phase_n']}",
        f"roi magnitude abs mean={complex_stats['mag_abs_mean']:.6f} "
        f"relative mean={complex_stats['mag_rel_mean']:.6e}",
        f"profile f={complex_stats['best_f']} y={complex_stats['best_y']}",
        "",
        f"pixels {total}",
        f"bin same {peak_stats['same']} ({peak_stats['same'] / total:.6f})",
        f"bin off1 {peak_stats['off1']} ({peak_stats['off1'] / total:.6f})",
        f"bin off>1 {peak_stats['offmore']} ({peak_stats['offmore'] / total:.6f})",
        f"confident n {peak_stats['conf_n']} ({peak_stats['conf_n'] / total:.6f})",
        f"confident bin same {peak_stats['conf_same']} ({peak_stats['conf_same'] / conf_n:.6f})",
        f"confident bin off1 {peak_stats['conf_off1']} ({peak_stats['conf_off1'] / conf_n:.6f})",
        f"confident bin off>1 {peak_stats['conf_offmore']} ({peak_stats['conf_offmore'] / conf_n:.6f})",
        f"confident but 2+1 peak<=0.1 {peak_stats['conf_weak21']}",
        "confident |du| " + " ".join(f"{k}={v}" for k, v in peak_stats["conf_du"].items()),
        "spectra " + str(peak_stats["spectra"]),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"加载缓存 {CACHE}")
    images = np.load(CACHE)
    if images.shape != (N_FREQ * 4, 2048, 2448) or images.dtype != np.uint8:
        raise SystemExit(f"unexpected cache: {images.shape} {images.dtype}")
    print(f"已加载 {images.shape}")
    complex_stats = compare_complex(images, OUT)
    peak_stats = compare_peaks(images, OUT)
    write_summary(complex_stats, peak_stats, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
