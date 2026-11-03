"""同一 2+1 解码结果，用标定结果.txt / 标定结果2.txt 各重建一次。"""
import os

import cv2
import numpy as np

from FSM_reconstruct_core import FSAReconstructor

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "results", "fsa_decode_2plus1_bookshelf")
CAMERA_IMAGE_SIZE = (2448, 2048)
PROJECTOR_WIDTH, PROJECTOR_HEIGHT = 1920, 1080
FREQUENCY_STEP = 1.0
NUM_FREQUENCIES = 151
PEAK_LIMIT = 5.0   # 2+1 解码 peak 量级较低，过高阈值会过少；5 可抑制重建离群
OVERSAMPLE = 4
WHITE_IMAGE = (
    r"E:\images\before_2026_5_23\before_2025_12_27\images_FSM\place1"
    r"\03_bookshelf_white_cycle1\FSM_vertical_f0_f150_2plus1\0.png"
)


def load_white(path: str, shape) -> np.ndarray:
    if path and os.path.isfile(path):
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    return np.ones(shape, dtype=np.uint8) * 128


def run_one(calib_file: str, out_ply: str) -> None:
    calib_path = os.path.join(ROOT, calib_file)
    disp_h = np.load(os.path.join(RESULT_DIR, "disp_hor.npy"))
    disp_v = np.load(os.path.join(RESULT_DIR, "disp_ver.npy"))
    peak_h = np.load(os.path.join(RESULT_DIR, "peak_hor.npy"))
    peak_v = np.load(os.path.join(RESULT_DIR, "peak_ver.npy"))
    white = load_white(WHITE_IMAGE, disp_h.shape)

    rec = FSAReconstructor(
        calibration_file=calib_path,
        camera_image_size=CAMERA_IMAGE_SIZE,
        projector_width=PROJECTOR_WIDTH,
        projector_height=PROJECTOR_HEIGHT,
        frequency_step=FREQUENCY_STEP,
        num_frequencies=NUM_FREQUENCIES,
    )
    pts, colors = rec.reconstruct_pointcloud(
        disparity_hor=disp_h,
        disparity_ver=disp_v,
        white_pattern=white,
        peak_limit=PEAK_LIMIT,
        peak_hor=peak_h,
        peak_ver=peak_v,
        oversample=OVERSAMPLE,
    )
    out_path = os.path.join(ROOT, out_ply)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rec.save_pointcloud(pts, colors, out_path)
    print(f"[OK] {calib_file}: {len(pts):,} 点 → {out_path}")
    print(
        f"     Z [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}] mm, "
        f"X [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}] mm"
    )


if __name__ == "__main__":
    for c, o in (
        ("标定结果.txt", "results/pointcloud_fsa_2plus1_calib1.ply"),
        ("标定结果2.txt", "results/pointcloud_fsa_2plus1_calib2.ply"),
    ):
        run_one(c, o)
