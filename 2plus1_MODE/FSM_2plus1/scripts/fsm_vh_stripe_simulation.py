#!/usr/bin/env python3
"""Vertical and horizontal fringe check of four-step vs 2+1.

The earlier 256x256 demo used a tilted phase, so the saved camera images
look like diagonal stripes. This script rebuilds the same camera-side A(x,y)
and B(x,y), but with phase that depends on only one axis:

    vertical stripes:   phi = 2*pi*16*x/W     (intensity varies along x)
    horizontal stripes: phi = 2*pi*16*y/H     (intensity varies along y)

Phase convention matches scripts/fsm_2plus1_integration.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsm_2plus1_integration import four_step_complex, two_plus_one_with_gray  # noqa: E402

H = W = 256
N_PERIODS = 16
SAMPLE_INDEX = [0, 2, 4, 8, 12, 64, 80, 128, 192]


def setup_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def camera_baseline(xx: np.ndarray, yy: np.ndarray):
    """Spatially varying camera response. Same A, B as the earlier demo."""
    ambient = 128.0 + 18.0 * np.sin(2 * np.pi * yy / H) + 10.0 * np.cos(2 * np.pi * xx / W)
    modulation = 72.0 + 14.0 * np.exp(-((xx - 80.0) ** 2 + (yy - 180.0) ** 2) / (2 * 50.0**2))
    return ambient, modulation


def synthesize(phi: np.ndarray):
    yy, xx = np.mgrid[0:H, 0:W]
    ambient, modulation = camera_baseline(xx, yy)
    i0 = ambient + modulation * np.cos(phi)
    i1 = ambient - modulation * np.sin(phi)
    i2 = ambient - modulation * np.cos(phi)
    i3 = ambient + modulation * np.sin(phi)
    c4 = four_step_complex(i0, i1, i2, i3)
    c21 = two_plus_one_with_gray(i0, i1, ambient)
    expected = 2.0 * modulation * np.exp(1j * phi)
    return {
        "I0": i0,
        "I1": i1,
        "I2": i2,
        "I3": i3,
        "A": ambient,
        "B": modulation,
        "phi": phi,
        "C4": c4,
        "C21": c21,
        "expected": expected,
    }


def quantize_8bit(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.float64)


def stats(pack: dict) -> dict:
    diff = pack["C21"] - pack["C4"]
    q0, q1, q2, q3, qa = (quantize_8bit(pack[k]) for k in ("I0", "I1", "I2", "I3", "A"))
    c4q = four_step_complex(q0, q1, q2, q3)
    c21q = two_plus_one_with_gray(q0, q1, qa)
    diffq = c21q - c4q
    denom = max(float(np.linalg.norm(c4q)), np.finfo(float).eps)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "max_vs_model": float(np.max(np.abs(pack["C4"] - pack["expected"]))),
        "rel_l2": float(np.linalg.norm(diff) / max(float(np.linalg.norm(pack["C4"])), np.finfo(float).eps)),
        "q_max_abs": float(np.max(np.abs(diffq))),
        "q_rel_l2": float(np.linalg.norm(diffq) / denom),
    }


def fmt_c(z: complex) -> str:
    return f"{z.real:+.6f}{z.imag:+.6f}j"


def write_pixel_table(path: Path, pack: dict, axis: str) -> str:
    """axis='x' samples a row (vertical fringes). axis='y' samples a column."""
    lines = [
        "C4=(I0-I2)+1j*(I3-I1)",
        "C21=2*((I0-A)+1j*(A-I1))",
        "A,B are camera-side quantities, not the projector gray pattern.",
        "phi_deg is wrapped into [0, 360).",
        "",
        f"{'x':>4} {'y':>4} {'phi_deg':>10} {'A':>10} {'B':>10} {'I0':>10} {'I1':>10} {'I2':>10} {'I3':>10} {'C4':>28} {'C21':>28} {'|diff|':>12}",
    ]
    md = [
        "| 像素 (x,y) | φ (°) | A | B | I0 | I1 | I2 | I3 | 四步法 C4 | 2+1 法 C21 | 绝对误差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    if axis == "x":
        points = [(x, 128) for x in SAMPLE_INDEX]
    else:
        points = [(128, y) for y in SAMPLE_INDEX]
    for x, y in points:
        phi = float(np.mod(pack["phi"][y, x], 2 * np.pi))
        a = float(pack["A"][y, x])
        b = float(pack["B"][y, x])
        i0 = float(pack["I0"][y, x])
        i1 = float(pack["I1"][y, x])
        i2 = float(pack["I2"][y, x])
        i3 = float(pack["I3"][y, x])
        c4 = complex(pack["C4"][y, x])
        c21 = complex(pack["C21"][y, x])
        err = abs(c21 - c4)
        lines.append(
            f"{x:4d} {y:4d} {np.degrees(phi):10.3f} {a:10.4f} {b:10.4f} "
            f"{i0:10.4f} {i1:10.4f} {i2:10.4f} {i3:10.4f} {fmt_c(c4):>28} {fmt_c(c21):>28} {err:12.3e}"
        )
        md.append(
            f"| ({x},{y}) | {np.degrees(phi):.3f} | {a:.4f} | {b:.4f} | {i0:.4f} | {i1:.4f} | "
            f"{i2:.4f} | {i3:.4f} | `{fmt_c(c4)}` | `{fmt_c(c21)}` | {err:.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(md)


def save_gray(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, np.clip(image, 0, 255), cmap="gray", vmin=0, vmax=255)


def save_signed(path: Path, image: np.ndarray, limit: float, cmap: str) -> None:
    plt.imsave(path, image, cmap=cmap, vmin=-limit, vmax=limit)


def save_positive(path: Path, image: np.ndarray, limit: float, cmap: str) -> None:
    plt.imsave(path, image, cmap=cmap, vmin=0, vmax=limit)


def plot_intermediate(path: Path, pack: dict, title: str) -> None:
    names = [("I0  0°", "I0"), ("I1  90°", "I1"), ("I2  180°", "I2"), ("I3  270°", "I3"), ("相机背景 A", "A")]
    fig, axes = plt.subplots(1, 5, figsize=(16.6, 3.9))
    image = None
    for ax, (label, key) in zip(axes, names):
        image = ax.imshow(pack[key], cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.90, top=0.78, bottom=0.08, wspace=0.15)
    colorbar_ax = fig.add_axes((0.915, 0.14, 0.012, 0.58))
    fig.colorbar(image, cax=colorbar_ax)
    fig.suptitle(title, fontsize=14, y=0.96)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_complex(path: Path, pack: dict, title: str, info: dict) -> None:
    real_lim = max(np.max(np.abs(np.real(pack["C4"]))), np.max(np.abs(np.real(pack["C21"]))))
    imag_lim = max(np.max(np.abs(np.imag(pack["C4"]))), np.max(np.abs(np.imag(pack["C21"]))))
    signed_lim = max(real_lim, imag_lim)
    mag_lim = max(np.max(np.abs(pack["C4"])), np.max(np.abs(pack["C21"])))
    panels = [
        ("四步 实部", np.real(pack["C4"]), "coolwarm", -signed_lim, signed_lim),
        ("四步 虚部", np.imag(pack["C4"]), "coolwarm", -signed_lim, signed_lim),
        ("四步 幅值", np.abs(pack["C4"]), "viridis", 0, mag_lim),
        ("四步 相位", np.angle(pack["C4"]), "twilight", -np.pi, np.pi),
        ("2+1 实部", np.real(pack["C21"]), "coolwarm", -signed_lim, signed_lim),
        ("2+1 虚部", np.imag(pack["C21"]), "coolwarm", -signed_lim, signed_lim),
        ("2+1 幅值", np.abs(pack["C21"]), "viridis", 0, mag_lim),
        ("2+1 相位", np.angle(pack["C21"]), "twilight", -np.pi, np.pi),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.6))
    for ax, (label, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{title}\n浮点最大 |C21−C4| = {info['max_abs']:.3e}，相对 L2 = {info['rel_l2']:.3e}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_difference(path: Path, pack: dict, title: str, info: dict) -> None:
    diff = pack["C21"] - pack["C4"]
    panels = [
        ("实部差", np.real(diff)),
        ("虚部差", np.imag(diff)),
        ("复数绝对误差", np.abs(diff)),
    ]
    peak = max(info["max_abs"], np.finfo(float).eps)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    for ax, (label, image) in zip(axes, panels):
        if label.startswith("复数"):
            im = ax.imshow(image, cmap="magma", vmin=0, vmax=peak)
        else:
            im = ax.imshow(image, cmap="coolwarm", vmin=-peak, vmax=peak)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{title}：色标按真实最大误差 {info['max_abs']:.3e} 固定，没有再自动拉伸",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profile(path: Path, pack: dict, along: str) -> None:
    if along == "x":
        sl = pack["C4"][128, :]
        sl21 = pack["C21"][128, :]
        coord = np.arange(W)
        xlabel = "x（y = 128）"
    else:
        sl = pack["C4"][:, 128]
        sl21 = pack["C21"][:, 128]
        coord = np.arange(H)
        xlabel = "y（x = 128）"
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True)
    for ax, part, title in (
        (axes[0], np.real, "实部"),
        (axes[1], np.imag, "虚部"),
    ):
        ax.plot(coord, part(sl), color="C0", lw=1.8, label="四步 C4")
        ax.plot(coord, part(sl21), color="C1", lw=1.2, ls="--", label="2+1 C21")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    err = np.abs(sl21 - sl)
    axes[2].plot(coord, err, color="C3", lw=1.2)
    axes[2].set_ylabel("|C21-C4|")
    axes[2].set_xlabel(xlabel)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].grid(alpha=0.3)
    fig.suptitle("中心剖线：实部与虚部重合，差值停留在舍入误差", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_direction(name: str, title: str, phi: np.ndarray, along: str) -> str:
    data_dir = ROOT / "data" / name
    fig_dir = ROOT / "figures" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pack = synthesize(phi)
    info = stats(pack)
    np.savez_compressed(
        data_dir / "simulation.npz",
        I0=pack["I0"],
        I1=pack["I1"],
        I2=pack["I2"],
        I3=pack["I3"],
        A=pack["A"],
        B=pack["B"],
        phi=pack["phi"],
        C4=pack["C4"],
        C21=pack["C21"],
    )
    np.save(data_dir / "C4_complex.npy", pack["C4"])
    np.save(data_dir / "C21_complex.npy", pack["C21"])
    for key in ("I0", "I1", "I2", "I3"):
        save_gray(data_dir / f"{key}.png", pack[key])
    save_gray(data_dir / "A_camera.png", pack["A"])
    signed_lim = max(np.max(np.abs(np.real(pack["C4"]))), np.max(np.abs(np.imag(pack["C4"]))))
    mag_lim = float(np.max(np.abs(pack["C4"])))
    for prefix, field in (("C4", pack["C4"]), ("C21", pack["C21"])):
        save_signed(data_dir / f"{prefix}_real.png", np.real(field), signed_lim, "coolwarm")
        save_signed(data_dir / f"{prefix}_imag.png", np.imag(field), signed_lim, "coolwarm")
        save_positive(data_dir / f"{prefix}_mag.png", np.abs(field), mag_lim, "viridis")
        plt.imsave(data_dir / f"{prefix}_phase.png", np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plot_intermediate(fig_dir / "01_intermediate.png", pack, title + "：中间结果 I0–I3 与相机背景 A")
    plot_complex(fig_dir / "02_complex.png", pack, title + "：最终复数", info)
    plot_difference(fig_dir / "03_difference.png", pack, title, info)
    plot_profile(fig_dir / "04_profile.png", pack, along)
    table = write_pixel_table(data_dir / "pixel_values.txt", pack, along)
    summary = (
        f"## {title}\n\n"
        f"- 浮点最大绝对误差 `|C21-C4|` = `{info['max_abs']:.8e}`\n"
        f"- 相对 L2 = `{info['rel_l2']:.8e}`\n"
        f"- 相对解析式 `2B exp(jφ)` 的最大误差 = `{info['max_vs_model']:.8e}`\n"
        f"- 各帧独立四舍五入到 8-bit 后，最大绝对误差 = `{info['q_max_abs']:.8e}`\n"
        f"- 8-bit 相对 L2 = `{info['q_rel_l2']:.8e}`\n\n"
        f"{table}\n"
    )
    print(title)
    for key, value in info.items():
        print(f"  {key}: {value:.8e}")
    return summary


def main() -> None:
    setup_font()
    yy, xx = np.mgrid[0:H, 0:W]
    vertical_phi = 2 * np.pi * N_PERIODS * xx / W
    horizontal_phi = 2 * np.pi * N_PERIODS * yy / H
    parts = [
        save_direction("vertical", "竖直条纹", vertical_phi, "x"),
        save_direction("horizontal", "水平条纹", horizontal_phi, "y"),
    ]
    print("\n".join(parts))


if __name__ == "__main__":
    main()
