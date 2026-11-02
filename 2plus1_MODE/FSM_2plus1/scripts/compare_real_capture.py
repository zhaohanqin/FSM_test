#!/usr/bin/env python3
"""Compare four-step and 2+1 on a real 8-bit capture.

The background used by 2+1 is synthesized from the same four frames:

    A4 = (I0 + I1 + I2 + I3) / 4

It is a per-pixel camera image, not one constant gray level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsm_2plus1_integration import four_step_complex, two_plus_one_with_gray  # noqa: E402


def setup_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_frames(folder: Path):
    frames = []
    for name in ("v1.bmp", "v2.bmp", "v3.bmp", "v4.bmp"):
        image = np.array(Image.open(folder / name))
        if image.ndim != 2:
            raise SystemExit(f"{name} is not single-channel: {image.shape}")
        frames.append(image.astype(np.float64))
    return frames


def save_gray(path: Path, image: np.ndarray, vmin: float, vmax: float) -> None:
    plt.imsave(path, image, cmap="gray", vmin=vmin, vmax=vmax)


def main() -> None:
    setup_font()
    folder = ROOT / "1"
    out = folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    i0, i1, i2, i3 = load_frames(folder)
    a4 = (i0 + i1 + i2 + i3) / 4.0
    even = (i0 + i2) / 2.0
    odd = (i1 + i3) / 2.0
    pair_gap = even - odd
    c4 = four_step_complex(i0, i1, i2, i3)
    c21 = two_plus_one_with_gray(i0, i1, a4)
    diff = c21 - c4
    # Algebra: C21 - C4 = (E - O) * (1 + j) when A4 is the four-frame average.
    if not np.allclose(diff, pair_gap * (1 + 1j)):
        raise SystemExit("2+1 residual does not match the pair-average identity")

    abs_diff = np.abs(diff)
    magnitude = np.abs(c4)
    # Fringe support: the projected rectangle, not the dark surround.
    roi = magnitude >= 8.0

    def summarize(mask: np.ndarray) -> dict:
        d = abs_diff[mask]
        ref = c4[mask]
        denom = max(float(np.linalg.norm(ref)), np.finfo(float).eps)
        return {
            "n": int(mask.sum()),
            "max": float(d.max()) if d.size else float("nan"),
            "mean": float(d.mean()) if d.size else float("nan"),
            "p95": float(np.percentile(d, 95)) if d.size else float("nan"),
            "p99": float(np.percentile(d, 99)) if d.size else float("nan"),
            "rel": float(np.linalg.norm(diff[mask]) / denom),
            "exact": float(np.mean(d == 0)) if d.size else float("nan"),
            "median_mag": float(np.median(np.abs(ref))) if ref.size else float("nan"),
        }

    full = summarize(np.ones(a4.shape, dtype=bool))
    bright = summarize(roi)
    print("A4 min/mean/max/std", a4.min(), a4.mean(), a4.max(), a4.std())
    print("ROI fraction", roi.mean())
    print("full", full)
    print("roi", bright)
    print("|pair gap| max/mean", np.max(np.abs(pair_gap)), np.mean(np.abs(pair_gap)))

    np.save(out / "A4.npy", a4.astype(np.float32))
    np.save(out / "C4_complex.npy", c4.astype(np.complex64))
    np.save(out / "C21_complex.npy", c21.astype(np.complex64))

    # Viewable copies. Calculation above uses the exact float average.
    save_gray(out / "A4.png", a4, 0, max(float(np.percentile(a4, 99.5)), 1))
    for name, image in (("I0", i0), ("I1", i1), ("I2", i2), ("I3", i3)):
        save_gray(out / f"{name}.png", image, 0, 255)

    step = 4
    view = (slice(None, None, step), slice(None, None, step))

    fig, axes = plt.subplots(1, 5, figsize=(16.6, 4.2))
    panels = [("v1 / I0", i0), ("v2 / I1", i1), ("v3 / I2", i2), ("v4 / I3", i3), ("A4", a4)]
    vmax = float(np.percentile(np.stack([i0, i1, i2, i3]), 99.5))
    for ax, (label, image) in zip(axes, panels):
        im = ax.imshow(image[view], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.90, top=0.78, bottom=0.08, wspace=0.12)
    fig.colorbar(im, cax=fig.add_axes((0.915, 0.16, 0.012, 0.55)))
    fig.suptitle("实拍四步图与合成背景 A4（显示拉伸到 99.5 分位，计算仍用原始灰度）", fontsize=13, y=0.96)
    fig.savefig(out / "01_inputs_and_A4.png", dpi=140)
    plt.close(fig)

    signed = max(np.percentile(np.abs(np.real(c4)), 99.5), np.percentile(np.abs(np.imag(c4)), 99.5))
    mag_lim = float(np.percentile(magnitude, 99.5))
    complex_panels = [
        ("四步 实部", np.real(c4), "coolwarm", -signed, signed),
        ("四步 虚部", np.imag(c4), "coolwarm", -signed, signed),
        ("四步 幅值", magnitude, "viridis", 0, mag_lim),
        ("四步 相位", np.angle(c4), "twilight", -np.pi, np.pi),
        ("2+1 实部", np.real(c21), "coolwarm", -signed, signed),
        ("2+1 虚部", np.imag(c21), "coolwarm", -signed, signed),
        ("2+1 幅值", np.abs(c21), "viridis", 0, mag_lim),
        ("2+1 相位", np.angle(c21), "twilight", -np.pi, np.pi),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 8.0))
    for ax, (label, image, cmap, vmin, vmax_i) in zip(axes.ravel(), complex_panels):
        im = ax.imshow(image[view], cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"实拍复数对比    全图最大 |C21-C4| = {full['max']:.3f}    "
        f"条纹区相对 L2 = {bright['rel']:.3e}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "02_complex.png", dpi=140)
    plt.close(fig)

    peak = max(full["max"], 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for ax, label, image, cmap, vmin, vmax_i in (
        (axes[0], "实部差", np.real(diff), "coolwarm", -peak, peak),
        (axes[1], "虚部差", np.imag(diff), "coolwarm", -peak, peak),
        (axes[2], "绝对误差", abs_diff, "magma", 0, peak),
    ):
        im = ax.imshow(image[view], cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("C21 − C4，色标按全图最大误差固定", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / "03_difference.png", dpi=140)
    plt.close(fig)

    y = int(np.argmax(magnitude.mean(axis=1)))
    x = np.arange(i0.shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(x, np.real(c4[y]), lw=1.0, label="四步")
    axes[0].plot(x, np.real(c21[y]), lw=1.0, ls="--", label="2+1")
    axes[0].set_ylabel("实部")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, np.imag(c4[y]), lw=1.0, label="四步")
    axes[1].plot(x, np.imag(c21[y]), lw=1.0, ls="--", label="2+1")
    axes[1].set_ylabel("虚部")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    axes[2].plot(x, abs_diff[y], color="C3", lw=1.0)
    axes[2].set_ylabel("|C21-C4|")
    axes[2].set_xlabel(f"x（y = {y}，该行平均幅值最大）")
    axes[2].grid(alpha=0.3)
    fig.suptitle("条纹最强一行的复数剖线", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "04_profile.png", dpi=140)
    plt.close(fig)

    # Sample pixels on that row where the fringe is actually modulated.
    candidates = np.argsort(magnitude[y])[-8:]
    lines = [
        "v1,v2,v3,v4 interpreted as I0,I1,I2,I3 (0, 90, 180, 270 deg).",
        "A4=(I0+I1+I2+I3)/4 computed in float64, not rounded back to uint8.",
        "C4=(I0-I2)+1j*(I3-I1)",
        "C21=2*((I0-A4)+1j*(A4-I1))",
        "Identity: C21-C4 = ((I0+I2)-(I1+I3))/2 * (1+1j)",
        "",
        f"strongest row y={y}",
        f"{'x':>6} {'I0':>8} {'I1':>8} {'I2':>8} {'I3':>8} {'A4':>10} {'C4':>24} {'C21':>24} {'|diff|':>10}",
    ]
    for x_i in sorted(int(v) for v in candidates):
        c4_i = complex(c4[y, x_i])
        c21_i = complex(c21[y, x_i])
        lines.append(
            f"{x_i:6d} {i0[y, x_i]:8.1f} {i1[y, x_i]:8.1f} {i2[y, x_i]:8.1f} {i3[y, x_i]:8.1f} "
            f"{a4[y, x_i]:10.4f} {c4_i.real:+10.4f}{c4_i.imag:+10.4f}j "
            f"{c21_i.real:+10.4f}{c21_i.imag:+10.4f}j {abs(c21_i - c4_i):10.4f}"
        )
    worst = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    wy, wx = (int(worst[0]), int(worst[1]))
    lines += [
        "",
        f"max-error pixel (x,y)=({wx},{wy}) |C4|={abs(c4[wy, wx]):.4f} |diff|={abs_diff[wy, wx]:.4f}",
        f"I=({i0[wy, wx]:.1f},{i1[wy, wx]:.1f},{i2[wy, wx]:.1f},{i3[wy, wx]:.1f}) A4={a4[wy, wx]:.4f}",
        "",
        f"full max={full['max']:.6f} mean={full['mean']:.6f} p95={full['p95']:.6f} relL2={full['rel']:.6e} exact={full['exact']:.4f}",
        f"roi  max={bright['max']:.6f} mean={bright['mean']:.6f} p95={bright['p95']:.6f} relL2={bright['rel']:.6e} exact={bright['exact']:.4f} n={bright['n']}",
    ]
    (out / "pixel_values.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
