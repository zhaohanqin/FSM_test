"""
从四步相移序列生成原生 2+1 目录（不修改源文件夹）。

规则（与 FSM_decode_core 的 shared_gray / 2M+1 布局一致）：
  - 0.png：源目录**最后四张** (I0,I1,I2,I3) 的像素平均，作为共享背景 A
  - 每个频率 f：取源目录第 f 组的**前两帧**作为 I0、I1
    源索引 f*4, f*4+1 → 新索引 1+2f, 2+2f

源：4M 张 (0.png … (4M-1).png)
目标：2M+1 张
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def load_gray(path: str) -> np.ndarray:
    try:
        import cv2
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(path).convert("L"))
    if img is None:
        raise ValueError(f"无法读取: {path}")
    return img


def save_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import cv2
        ext = os.path.splitext(path)[1] or ".png"
        ok, enc = cv2.imencode(ext, img)
        if not ok:
            raise ValueError(f"编码失败: {path}")
        enc.tofile(path)
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


def count_source_frames(src: Path, pattern: str = "{:d}.png") -> int:
    n = 0
    while (src / pattern.format(n)).is_file():
        n += 1
    return n


def convert(src_dir: str | Path, dst_dir: str | Path | None = None,
            image_pattern: str = "{:d}.png") -> Path:
    src = Path(src_dir)
    if not src.is_dir():
        raise FileNotFoundError(src)
    n_src = count_source_frames(src, image_pattern)
    if n_src % 4 != 0:
        raise ValueError(f"源目录张数 {n_src} 不是 4 的倍数，无法按四步分组")
    m = n_src // 4
    if dst_dir is None:
        dst = src.parent / f"{src.name}_2plus1"
    else:
        dst = Path(dst_dir)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"目标目录已存在且非空: {dst}")

    dst.mkdir(parents=True, exist_ok=True)
    last_base = n_src - 4
    phases = []
    for k in range(4):
        p = src / image_pattern.format(last_base + k)
        phases.append(load_gray(str(p)).astype(np.float32))
    background = np.mean(phases, axis=0)
    background_u8 = np.clip(np.round(background), 0, 255).astype(np.uint8)
    save_gray(str(dst / image_pattern.format(0)), background_u8)

    out_i = 1
    for f in range(m):
        for phase in (0, 1):
            src_idx = f * 4 + phase
            img = load_gray(str(src / image_pattern.format(src_idx)))
            save_gray(str(dst / image_pattern.format(out_i)), img)
            out_i += 1

    expected = 1 + 2 * m
    meta = (
        f"源目录: {src}\n"
        f"四步张数: {n_src} ({m} 频率 × 4)\n"
        f"背景: 源索引 [{last_base}..{last_base + 3}] 四帧平均 → 0.png\n"
        f"I0/I1: 每频率源索引 f*4, f*4+1\n"
        f"输出张数: {expected} (2M+1)\n"
        f"解码: PHASE_SHIFT_MODE=two_plus_one, BACKGROUND_MODE=shared_gray 或 auto\n"
    )
    (dst / "2plus1_layout.txt").write_text(meta, encoding="utf-8")
    print(f"[OK] {src.name} → {dst}")
    print(f"     输出 {expected} 张，背景来自 {last_base}.png–{last_base + 3}.png")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="四步相移目录 → 2+1 (2M+1) 目录")
    parser.add_argument("src_dirs", nargs="+", help="一个或多个四步图像文件夹")
    parser.add_argument("--dst", default=None, help="单个源时可选指定目标目录")
    args = parser.parse_args()
    if len(args.src_dirs) > 1 and args.dst:
        parser.error("--dst 仅适用于单个源目录")
    for src in args.src_dirs:
        convert(src, args.dst if len(args.src_dirs) == 1 else None)


if __name__ == "__main__":
    main()
