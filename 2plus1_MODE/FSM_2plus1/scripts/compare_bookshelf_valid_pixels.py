#!/usr/bin/env python3
"""Column-coordinate error of 2+1 versus four-step, only on high-peak pixels.

The mask follows the visualization convention: a pixel is valid when the
four-step normalized peak is at least 0.5. Low-peak background is ignored.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_bookshelf_vertical as base

OUT = base.OUT
THRESHOLDS = (0.5, 1.0)
BATCH = base.BATCH


def main() -> None:
    base.setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"加载缓存 {base.CACHE}")
    images = np.load(base.CACHE)
    height, width = images.shape[1], images.shape[2]
    idx4 = np.empty((height, width), np.int32)
    idx21 = np.empty((height, width), np.int32)
    coord4 = np.empty((height, width), np.float32)
    coord21 = np.empty((height, width), np.float32)
    value4 = np.empty((height, width), np.float32)
    value21 = np.empty((height, width), np.float32)

    for y0 in range(0, height, BATCH):
        t0 = time.perf_counter()
        y1 = min(height, y0 + BATCH)
        block = images[:, y0:y1, :].astype(np.float32)
        i0, i1, i2, i3 = block[0::4], block[1::4], block[2::4], block[3::4]
        signal4 = ((i0 - i2) + 1j * (i3 - i1)).astype(np.complex64)
        a4 = (i0 + i1 + i2 + i3) / 4.0
        signal21 = (2.0 * ((i0 - a4) + 1j * (a4 - i1))).astype(np.complex64)
        a, c4, v4, _ = base.decode_peaks(signal4)
        b, c21, v21, _ = base.decode_peaks(signal21)
        idx4[y0:y1] = a
        idx21[y0:y1] = b
        coord4[y0:y1] = c4
        coord21[y0:y1] = c21
        value4[y0:y1] = v4
        value21[y0:y1] = v21
        print(f"\r峰值行 {y1}/{height}  {time.perf_counter() - t0:.2f}s", end="", flush=True)
    print()

    du = np.abs(coord21 - coord4)
    dbin = np.abs(idx21.astype(np.int32) - idx4.astype(np.int32))
    lines = ["valid-pixel column error, four-step peak threshold"]
    for threshold in THRESHOLDS:
        valid = value4 >= threshold
        n = int(valid.sum())
        err = du[valid]
        bins = dbin[valid]
        lines.append(
            f"threshold {threshold:.1f}  n={n} ({n / valid.size:.4%})  "
            f"bin0={np.mean(bins == 0):.4%}  bin1={np.mean(bins == 1):.4%}  "
            f"bin>1={np.mean(bins > 1):.4%}"
        )
        for limit in (0.5, 1.0, 2.0, 5.0):
            lines.append(f"  |du| <= {limit:.1f}: {np.mean(err <= limit):.4%}")
        qs = np.percentile(err, (50, 90, 95, 99, 99.9))
        lines.append(
            "  percentiles "
            + " ".join(f"p{p}={v:.4f}" for p, v in zip((50, 90, 95, 99, 99.9), qs))
        )
        lines.append(
            f"  |du|>1: {int(np.count_nonzero(err > 1))}  "
            f"|du|>5: {int(np.count_nonzero(err > 5))}  "
            f"2+1 peak<0.5: {int(np.count_nonzero(valid & (value21 < 0.5)))}"
        )

    valid = value4 >= 0.5
    show = np.full((height // base.STEP, width // base.STEP), np.nan, np.float32)
    show_valid = valid[::base.STEP, ::base.STEP]
    show[show_valid] = du[::base.STEP, ::base.STEP][show_valid]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    im = ax.imshow(show, cmap="magma", vmin=0.0, vmax=2.0, interpolation="nearest")
    ax.set_title("峰值 ≥ 0.5 的有效像素上，列坐标绝对差（色标 0–2 像素）")
    ax.set_xlabel("x / 4")
    ax.set_ylabel("y / 4")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="|Δ列|")
    fig.tight_layout()
    fig.savefig(OUT / "05_valid_peak_abs_delta.png", dpi=140)
    plt.close(fig)

    text = "\n".join(lines) + "\n"
    (OUT / "valid_pixel_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
