"""
同一垂直书架数据：四步 vs 2+1(a4) 解码 A/B，比较 disp_ver / peak_ver。

用法（仓库根目录）:
  python 2plus1_MODE/FSM_2plus1/scripts/verify_decode_2plus1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_MODE = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from FSM_decode_core import FSADecoder  # noqa: E402

ROOT = _REPO

IMAGE_DIR = Path(
    r"E:\images\before_2026_5_23\before_2025_12_27\images_FSM\place1"
    r"\03_bookshelf_white_cycle1\FSM_vertical_f0_f150"
)
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
NUM_FREQUENCIES = 151
FREQUENCY_STEP = 1.0
PEAK_MASK = 1.5
OUT_FOUR = ROOT / "results" / "fsa_decode_four"
OUT_21 = ROOT / "results" / "fsa_decode_2plus1"
REPORT = _MODE / "2plus1_逐像素频谱" / "decode_ab_test.md"


def run_decode(out_dir: Path, phase_shift_mode: str) -> tuple[np.ndarray, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder = FSADecoder(
        PROJECTOR_WIDTH,
        PROJECTOR_HEIGHT,
        FREQUENCY_STEP,
        phase_shift_mode=phase_shift_mode,
        background_mode="a4_per_frequency",
    )
    cache = IMAGE_DIR / decoder.demodulator.default_cache_filename()
    if cache.is_file():
        images = np.load(cache)
        print(f"loaded cache {cache} shape={images.shape}")
    else:
        images = decoder.load_images(str(IMAGE_DIR), NUM_FREQUENCIES, use_cache=True)
    disp, peak, _ = decoder.decode_direction(
        images=images,
        direction="ver",
        num_frequencies=NUM_FREQUENCIES,
        block_size=100,
        low_memory=True,
    )
    decoder.save_results(disp, peak, str(out_dir), "ver")
    return disp, peak


def compare(d4: np.ndarray, p4: np.ndarray, d21: np.ndarray, p21: np.ndarray) -> dict:
    mask = p4 >= PEAK_MASK
    n_valid = int(mask.sum())
    total = mask.size
    du = np.abs(d21 - d4)
    du_v = du[mask]
    int_d = np.abs(np.round(d21) - np.round(d4)).astype(np.int32)
    int_v = int_d[mask]
    bins = {
        "int_same": int((int_v == 0).sum()),
        "int_1": int((int_v == 1).sum()),
        "int_gt1": int((int_v > 1).sum()),
    }
    return {
        "n_valid": n_valid,
        "total": total,
        "pct_le_1": float((du_v <= 1.0).mean() * 100) if n_valid else 0.0,
        "median_du": float(np.median(du_v)) if n_valid else 0.0,
        "max_du": float(du_v.max()) if n_valid else 0.0,
        "bins": bins,
    }


def write_report(stats: dict) -> None:
    b = stats["bins"]
    pct_same = 100.0 * b["int_same"] / stats["n_valid"] if stats["n_valid"] else 0
    pct_1 = 100.0 * b["int_1"] / stats["n_valid"] if stats["n_valid"] else 0
    text = f"""# 解码 A/B：四步 vs 2+1 (A4)

数据：`{IMAGE_DIR}`  
有效掩膜：四步 `peak_ver >= {PEAK_MASK}`  

| 指标 | 值 |
| --- | ---: |
| 全图像素 | {stats['total']:,} |
| 有效像素 | {stats['n_valid']:,} |
| 整数列相同 | {b['int_same']:,} ({pct_same:.2f}%) |
| 整数列差 1 | {b['int_1']:,} ({pct_1:.2f}%) |
| 整数列差 >1 | {b['int_gt1']:,} |
| 亚像素 |Δu| 中位数 | {stats['median_du']:.6f} 列 |
| 亚像素 |Δu| 最大 | {stats['max_du']:.4f} 列 |
| |Δu| ≤ 1 列占比 | {stats['pct_le_1']:.4f}% |

输出目录：`results/fsa_decode_four/` 与 `results/fsa_decode_2plus1/`（各含 `disp_ver.npy`, `peak_ver.npy`）。

实用结论：亚像素 ≤1 列 {'通过' if stats['pct_le_1'] >= 99.99 else '未通过'}；整数 100% 一致 {'通过' if b['int_1'] == 0 and b['int_gt1'] == 0 else '未通过'}。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print("wrote", REPORT)


def main() -> None:
    if not IMAGE_DIR.is_dir() and not (IMAGE_DIR / "images_cache.npy").is_file():
        print("skip: image dir not found:", IMAGE_DIR)
        sys.exit(1)
    print("=== four_step ===")
    d4, p4 = run_decode(OUT_FOUR, "four_step")
    print("=== two_plus_one ===")
    d21, p21 = run_decode(OUT_21, "two_plus_one")
    stats = compare(d4, p4, d21, p21)
    print(stats)
    write_report(stats)


if __name__ == "__main__":
    main()
