"""
FSA (Frequency Shift Analysis) 频移法解码核心模块（重构版）

本模块实现了频移法的核心DFT解码算法，用于从采集的条纹图像中
解码出相机像素与投影仪像素之间的对应关系。

主要功能：
1. 读取四步相移条纹图像序列
2. 使用DFT（离散傅里叶变换）解码投影仪坐标
3. 检测频谱峰值并计算子像素精度
4. 输出视差图和置信度图

重构说明（2025-11-09）：
------------------------
为了提高代码的可读性和可维护性，将原有功能细分为以下专门的类：

1. **ImageLoader** - 图像加载器（新增）
   负责加载条纹图像序列，支持缓存和中文路径

2. **CalibrationReader** - 标定文件读取器（新增，移植自FSM_reconstruct_core.py）
   负责读取标定结果文件，提取极线约束校正所需的参数

3. **DFTProcessor** - DFT处理器（新增）
   负责执行DFT傅里叶变换，将时域信号转换到频域

4. **PeakDetector** - 峰值检测器（新增）
   负责在频谱中找到最高响应峰值，并计算子像素精度

5. **MultiPeakDetector** - 多峰检测器（新增，恢复自decode_fsa.py）
   在频谱中找出所有局部极大值，支持近峰加权合并，用于后续极线约束校正

6. **EpipolarCorrector** - 极线约束校正器（新增，恢复自decode_fsa.py）
   利用相机-投影仪极线几何约束，从多个候选峰值中选出正确匹配

7. **FSADecoder** - FSA解码协调器（重构）
   整合上述组件，提供统一的解码接口，保持向后兼容性

修复说明：
---------
- **DFT长度选择**：恢复原始代码decode_fsa.py中的DFT长度策略。
  当frequency_step==1.0时，FFT长度使用投影仪分辨率（而非频率数量M），
  通过零填充实现频谱插值，使argmax直接输出投影仪坐标（1像素精度）。

- **frequency_step=0.5 支持（新增）**：
  采用通用公式 fft_length = int(round(proj_dim / frequency_step))，
  当step=0.5时得到 fft_length = 2*proj_dim，FFT bin编号仍直接对应
  投影仪坐标（coord_scale=1.0），且argmax只在前proj_dim个有效bin中执行，
  排除后半段负频率区域的干扰，保证输出坐标在 [0, proj_dim-1] 范围内。
  此通用公式可自然扩展至 step=0.25（→4*proj_dim）等任意正数步长。

  旧非标准分支（fft_length=M, coord_scale=proj_dim/M）会产生明显量化阶梯
  （如 step=0.5, M=301 时坐标量化间隔约 6.38 像素），已被通用公式替代。

作者：从decode_fsa.py提取并重构
日期：2025-11-01（初版），2025-11-09（重构），2025-xx-xx（DFT修复+后处理恢复+0.5步长支持）
"""

import numpy as np
import os
from typing import Tuple, Optional, List, Union, Dict


class ImageLoader:
    """
    图像加载器

    负责加载条纹图像序列，支持缓存和中文路径。
    """

    @staticmethod
    def load_images(image_dir: str,
                    num_frequencies: int,
                    image_pattern: str = "{:d}.png",
                    use_cache: bool = True,
                    cache_path: Optional[str] = None,
                    verify_sequence: bool = False) -> np.ndarray:
        """
        加载条纹图像序列

        Parameters:
        -----------
        image_dir : str
            图像文件夹路径
        num_frequencies : int
            频率数量（M），总图像数为 M*4
        image_pattern : str
            图像文件名模式，默认为 "{:d}.png"
        use_cache : bool
            是否使用缓存（.npy文件），可加速重复加载
        cache_path : str, optional
            缓存文件路径，默认为 image_dir/images_cache.npy
        verify_sequence : bool
            是否验证图像序列的组织方式，默认为 False

        Returns:
        --------
        images : np.ndarray
            图像数组，形状为 (M*4, height, width)
        """
        if cache_path is None:
            cache_path = os.path.join(image_dir, "images_cache.npy")

        if use_cache and os.path.exists(cache_path):
            print(f"从缓存加载图像: {cache_path}")
            images = np.load(cache_path)
            print(f"已加载 {images.shape[0]} 张图像，尺寸: {images.shape[1]}x{images.shape[2]}")
            return images

        print(f"从文件夹加载图像: {image_dir}")
        total_images = num_frequencies * 4
        images_list = []

        for i in range(total_images):
            img_path = os.path.join(image_dir, image_pattern.format(i))
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")

            try:
                import cv2
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            except ImportError:
                try:
                    from PIL import Image
                    img = np.array(Image.open(img_path).convert('L'))
                except ImportError:
                    raise ImportError("需要安装 opencv-python 或 Pillow 来读取图像")
            except Exception as e:
                try:
                    from PIL import Image
                    img = np.array(Image.open(img_path).convert('L'))
                except Exception:
                    raise ValueError(f"无法读取图像: {img_path}，错误: {e}")

            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")

            images_list.append(img)

            if (i + 1) % 10 == 0 or i == total_images - 1:
                print(f"\r加载进度: {i+1}/{total_images}", end="")

        print()
        images = np.array(images_list)

        if use_cache:
            print(f"保存缓存到: {cache_path}")
            np.save(cache_path, images)

        print(f"已加载 {images.shape[0]} 张图像，尺寸: {images.shape[1]}x{images.shape[2]}")

        if verify_sequence:
            ImageLoader.verify_image_sequence(images, num_frequencies)

        return images

    @staticmethod
    def verify_image_sequence(images: np.ndarray, num_frequencies: int, sample_points: int = 5) -> bool:
        """
        验证图像序列的组织方式是否正确
        """
        print("\n" + "="*60)
        print("验证图像序列组织方式")
        print("="*60)

        if images.shape[0] != num_frequencies * 4:
            print(f"❌ 错误: 图像数量不匹配！")
            print(f"   期望: {num_frequencies * 4} 张 ({num_frequencies} 频率 × 4 相移)")
            print(f"   实际: {images.shape[0]} 张")
            return False

        print(f"[OK] 图像数量正确: {images.shape[0]} 张")

        print(f"\n检查相位图像差异 (采样 {sample_points} 个频率)...")
        sample_freqs = np.linspace(0, num_frequencies-1, min(sample_points, num_frequencies), dtype=int)

        has_variation = True
        for freq_idx in sample_freqs:
            base_idx = freq_idx * 4
            i0 = images[base_idx + 0].astype(np.float32)
            i1 = images[base_idx + 1].astype(np.float32)
            i2 = images[base_idx + 2].astype(np.float32)
            i3 = images[base_idx + 3].astype(np.float32)

            diff_01 = np.mean(np.abs(i0 - i1))
            diff_12 = np.mean(np.abs(i1 - i2))
            diff_23 = np.mean(np.abs(i2 - i3))
            diff_30 = np.mean(np.abs(i3 - i0))

            avg_diff = (diff_01 + diff_12 + diff_23 + diff_30) / 4

            print(f"  频率 {freq_idx:3d}: 相位间平均差异 = {avg_diff:.2f}", end="")

            if avg_diff < 5.0:
                print(" ⚠ (差异较小，可能有问题)")
                has_variation = False
            else:
                print(" [OK]")

        if not has_variation:
            print("\n⚠ 警告: 某些频率的相位图像差异很小，请检查图像顺序！")
            return False

        print("\n[OK] 图像序列验证通过！")
        print("="*60)
        return True


class CalibrationReader:
    """
    标定结果读取器

    移植自 FSM_reconstruct_core.py，用于从标定结果文件中读取
    相机和投影仪的标定参数。

    在解码模块中主要用于极线约束校正（EpipolarCorrector），
    使用户无需手动填写标定参数，只需指定标定文件路径即可。
    """

    @staticmethod
    def read_calibration_file(calib_file_path: str) -> Dict:
        """
        读取标定结果文件

        Parameters:
        -----------
        calib_file_path : str
            标定结果文件路径（.txt格式）

        Returns:
        --------
        calib_data : dict
            包含相机和投影仪标定参数的字典
        """
        print(f"\n读取标定文件: {calib_file_path}")

        with open(calib_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        calib_data = {}

        cam_params = []
        for i in range(4, 12):
            cam_params.append(float(lines[i].strip()))

        calib_data['camera'] = {
            'fx': cam_params[0], 'fy': cam_params[1],
            'cx': cam_params[2], 'cy': cam_params[3],
            'k1': cam_params[4], 'k2': cam_params[5],
            'p1': cam_params[6], 'p2': cam_params[7]
        }

        proj_params = []
        for i in range(17, 25):
            proj_params.append(float(lines[i].strip()))

        calib_data['projector'] = {
            'fx': proj_params[0], 'fy': proj_params[1],
            'cx': proj_params[2], 'cy': proj_params[3],
            'k1': proj_params[4], 'k2': proj_params[5],
            'p1': proj_params[6], 'p2': proj_params[7]
        }

        rotation_matrix = []
        for i in range(27, 30):
            row = [float(x) for x in lines[i].strip().split()]
            rotation_matrix.append(row)
        calib_data['rotation_matrix'] = np.array(rotation_matrix)

        translation = [float(x) for x in lines[32].strip().split()]
        calib_data['translation'] = np.array(translation)

        print("[OK] 标定文件读取成功")
        print(f"  相机参数: fx={calib_data['camera']['fx']:.2f}, fy={calib_data['camera']['fy']:.2f}")
        print(f"  投影仪参数: fx={calib_data['projector']['fx']:.2f}, fy={calib_data['projector']['fy']:.2f}")
        print(f"  平移向量: {calib_data['translation']}")

        return calib_data

    @staticmethod
    def extract_epipolar_params(calib_data: Dict) -> Dict:
        """
        从标定数据中提取极线约束校正所需的参数
        """
        import cv2

        cam = calib_data['camera']
        cam_intrinsic = np.array([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        cam_distortion = np.array([cam['k1'], cam['k2'], cam['p1'], cam['p2'], 0], dtype=np.float64)
        cam_rotation = np.zeros(3, dtype=np.float64)
        cam_translation = np.zeros(3, dtype=np.float64)

        proj = calib_data['projector']
        proj_intrinsic = np.array([
            [proj['fx'], 0, proj['cx']],
            [0, proj['fy'], proj['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        proj_distortion = np.array([proj['k1'], proj['k2'], proj['p1'], proj['p2'], 0], dtype=np.float64)

        R = np.array(calib_data['rotation_matrix'], dtype=np.float64)
        proj_rotation, _ = cv2.Rodrigues(R)
        proj_rotation = proj_rotation.flatten()
        proj_translation = np.array(calib_data['translation'], dtype=np.float64).flatten()

        params = {
            'cam_intrinsic': cam_intrinsic,
            'cam_distortion': cam_distortion,
            'cam_rotation': cam_rotation,
            'cam_translation': cam_translation,
            'proj_intrinsic': proj_intrinsic,
            'proj_distortion': proj_distortion,
            'proj_rotation': proj_rotation,
            'proj_translation': proj_translation,
        }

        print("\n[OK] 已从标定数据中提取极线约束校正参数:")
        print(f"  相机内参: fx={cam['fx']:.2f}, fy={cam['fy']:.2f}, cx={cam['cx']:.2f}, cy={cam['cy']:.2f}")
        print(f"  投影仪内参: fx={proj['fx']:.2f}, fy={proj['fy']:.2f}, cx={proj['cx']:.2f}, cy={proj['cy']:.2f}")
        print(f"  投影仪旋转向量(Rodrigues): {proj_rotation}")
        print(f"  投影仪平移向量: {proj_translation}")

        return params


class DFTProcessor:
    """
    DFT处理器

    负责执行离散傅里叶变换（DFT），将时域信号转换到频域。
    """

    @staticmethod
    def construct_complex_signal(I0, I1, I2, I3):
        """从四步相移图像构造复数信号 D = (I0 - I2) + j*(I3 - I1)"""
        return (I0 - I2) + 1j * (I3 - I1)

    @staticmethod
    def perform_dft(signal, N, axis=0):
        """执行DFT变换"""
        return np.fft.fft(signal, n=N, axis=axis)

    @staticmethod
    def compute_magnitude_spectrum(spectrum):
        """计算幅值谱"""
        return np.abs(spectrum)


class PeakDetector:
    """
    峰值检测器

    负责在频谱中找到最高响应峰值，并计算子像素精度。
    """

    @staticmethod
    def find_peak_index(magnitude_spectrum, axis=0):
        """找到频谱中的最大峰值位置"""
        return np.argmax(magnitude_spectrum, axis=axis)

    @staticmethod
    def compute_peak_value(magnitude_spectrum, peak_indices, normalization_factor=1.0):
        """计算峰值强度"""
        if len(magnitude_spectrum.shape) == 2:
            rows = np.arange(magnitude_spectrum.shape[1])
            return magnitude_spectrum[peak_indices, rows] / normalization_factor
        return np.max(magnitude_spectrum, axis=0) / normalization_factor

    @staticmethod
    def refine_peak_subpixel(magnitude_spectrum, peak_indices, N, threshold=0.1):
        """使用质心法计算子像素精度的峰值位置"""
        refined_peaks = peak_indices.astype(np.float32)
        if len(magnitude_spectrum.shape) == 1:
            if magnitude_spectrum[peak_indices] / N > threshold:
                idx_left  = (peak_indices - 1) % N
                idx_right = (peak_indices + 1) % N
                vl = magnitude_spectrum[idx_left]
                vc = magnitude_spectrum[peak_indices]
                vr = magnitude_spectrum[idx_right]
                if vl + vc + vr > 0:
                    offset = (vr - vl) / (vl + vc + vr)
                    refined_peaks = peak_indices + offset * 0.5
        return refined_peaks


# ===========================================================================
# 内部辅助函数：计算通用 FFT 参数
# ===========================================================================

def _calc_fft_params(proj_dim: int, frequency_step: float):
    """
    计算通用 FFT 参数（被 FSADecoder 和 MultiPeakDetector 共用）。

    通用公式推导
    ------------
    设真实频率序列 f_k = s·k（s = frequency_step），投影仪坐标为 u，
    理想复数信号：D[k] ≈ A · exp(j·2π·s·k·u / proj_dim)。
    FFT 长度 L 时，第 q 个 bin 的匹配条件：q/L = s·u/proj_dim，
    解得 u = q·proj_dim/(L·s)。令 coord_scale = 1，则 L = proj_dim/s。

    结论
    ----
    fft_length   = int(round(proj_dim / frequency_step))
    valid_length = proj_dim        ← argmax 只在前 proj_dim 个 bin 中搜索
    coord_scale  ≈ 1.0            ← FFT bin 编号直接对应投影仪坐标

    典型情况
    --------
    frequency_step = 1.0  → fft_length = proj_dim,    valid_length = proj_dim
    frequency_step = 0.5  → fft_length = 2*proj_dim,  valid_length = proj_dim
    frequency_step = 0.25 → fft_length = 4*proj_dim,  valid_length = proj_dim

    Parameters
    ----------
    proj_dim      : int   投影仪在该方向上的分辨率
    frequency_step : float 频率步长（Hz）

    Returns
    -------
    fft_length   : int
    valid_length : int
    coord_scale  : float  （理论上 = 1.0）
    """
    if frequency_step == 1.0:
        # 标准模式：与原始 decode_fsa.py 完全一致，零填充到投影仪分辨率
        fft_length   = proj_dim
        valid_length = proj_dim
        coord_scale  = 1.0
    else:
        # 通用模式：fft_length = proj_dim / step，保证 bin 编号对应投影仪坐标
        fft_length   = int(round(proj_dim / frequency_step))
        valid_length = proj_dim   # 只取前 proj_dim 个有效 bin
        coord_scale  = proj_dim / (fft_length * frequency_step)   # ≈ 1.0
    return fft_length, valid_length, coord_scale


class MultiPeakDetector:
    """
    多峰检测器

    恢复自原始代码 decode_fsa.py 中的 find_peaks() 和 get_peaks_position() 函数。

    功能说明：
    --------
    在FSA频谱中，由于反射、遮挡或多路径干扰，一个相机像素可能对应频谱中的
    多个峰值。MultiPeakDetector负责找出所有显著峰值，为后续的极线约束校正
    提供候选匹配点。
    """

    @staticmethod
    def find_peaks(spectrum, limit, merge_distance=5):
        """
        在频谱中找出所有局部极大值。

        Parameters:
        -----------
        spectrum : np.ndarray
            归一化频谱幅值（1D数组）
        limit : float
            峰值强度阈值，低于此值的峰值将被忽略
        merge_distance : int
            近峰合并距离，默认为5

        Returns:
        --------
        peaks : np.ndarray
            峰值位置数组（浮点数，加权合并后可为非整数）
        """
        peaks = []  # [(position, amplitude), ...]

        for i in range(len(spectrum)):
            value = float(spectrum[i])
            if value < limit:
                continue

            left_ok  = (i == 0) or (spectrum[i] > spectrum[i - 1])
            right_ok = (i == len(spectrum) - 1) or (spectrum[i] > spectrum[i + 1])

            if left_ok and right_ok:
                if peaks and (i - peaks[-1][0]) < merge_distance:
                    prev_pos, prev_val = peaks[-1]
                    ki = value / (value + prev_val)
                    new_pos = ki * i + (1.0 - ki) * prev_pos
                    peaks[-1] = (new_pos, max(value, prev_val))
                else:
                    peaks.append((float(i), value))

        if not peaks:
            return np.array([])
        return np.array([p[0] for p in peaks])

    @staticmethod
    def get_peaks_for_pixels(images, pixel_coords, direction,
                             proj_height, proj_width,
                             num_frequencies, frequency_step=1.0,
                             peak_limit=0.5):
        """
        批量获取指定像素点的所有频谱峰值位置。

        恢复自 decode_fsa.py 的 get_peaks_position() 函数，并扩展为支持
        任意 frequency_step（含 0.5）的通用实现。

        Parameters:
        -----------
        images : np.ndarray
            原始图像数组，形状为 (M*4, height, width)
        pixel_coords : list of (x, y) tuples
            需要分析的相机像素坐标列表
        direction : str
            条纹方向 'hor' 或 'ver'
        proj_height, proj_width : int
            投影仪分辨率
        num_frequencies : int
            频率数量 M
        frequency_step : float
            频率步长（Hz）
        peak_limit : float
            峰值强度阈值

        Returns:
        --------
        results : list of np.ndarray
            每个像素点对应的峰值位置数组（投影仪坐标）
        """
        proj_dim = proj_height if direction == "hor" else proj_width
        # ★ 使用通用公式（与 FSADecoder._get_fft_params 一致）
        fft_length, valid_length, coord_scale = _calc_fft_params(proj_dim, frequency_step)

        images_f = images.astype(np.float32) if images.dtype != np.float32 else images
        I0 = images_f[0::4];  I1 = images_f[1::4]
        I2 = images_f[2::4];  I3 = images_f[3::4]
        D = (I0 - I2) + 1j * (I3 - I1)

        results = []
        for x, y in pixel_coords:
            # FFT 后只在有效 bin [0, valid_length) 范围内搜索峰值
            C_full = np.abs(np.fft.fft(D[:, y, x], n=fft_length)) / fft_length
            C_valid = C_full[:valid_length]
            peaks = MultiPeakDetector.find_peaks(C_valid, peak_limit)

            # coord_scale 理论上 = 1.0，此处保留以防极端浮点误差
            if abs(coord_scale - 1.0) > 1e-9 and len(peaks) > 0:
                peaks = peaks * coord_scale

            results.append(peaks)

        return results

    @staticmethod
    def scan_image_for_multi_peaks(images, direction, proj_height, proj_width,
                                   num_frequencies, frequency_step=1.0,
                                   peak_limit=0.5, suppress_radius=30,
                                   block_size=100, max_report=200):
        """
        扫描整幅图像，找出所有存在多峰（多个显著频谱峰值）的像素。

        Parameters:
        -----------
        images : np.ndarray
            图像数组，形状为 (M*4, height, width)
        direction : str
            'hor' 或 'ver'
        proj_height, proj_width : int
            投影仪分辨率
        num_frequencies : int
            频率数量 M
        frequency_step : float
            频率步长（Hz）
        peak_limit : float
            次峰判定阈值
        suppress_radius : int
            主峰抑制半径（bin 数）
        block_size : int
            分块行数
        max_report : int
            最多报告的多峰像素数量

        Returns:
        --------
        multi_peak_pixels : list of dict
        """
        proj_dim = proj_height if direction == "hor" else proj_width
        # ★ 使用通用公式
        fft_length, valid_length, coord_scale = _calc_fft_params(proj_dim, frequency_step)

        height, width = images.shape[1], images.shape[2]

        print(f"\n{'='*60}")
        print(f"多峰像素扫描 - {direction.upper()}方向")
        print(f"{'='*60}")
        print(f"  图像尺寸: {width} × {height}")
        print(f"  FFT长度: {fft_length}，有效bin范围: 0~{valid_length-1}")
        print(f"  次峰判定阈值: {peak_limit}，主峰抑制半径: ±{suppress_radius} bins")

        images_f = images.astype(np.float32) if images.dtype != np.float32 else images
        I0 = images_f[0::4];  I1 = images_f[1::4]
        I2 = images_f[2::4];  I3 = images_f[3::4]
        D = (I0 - I2) + 1j * (I3 - I1)
        del I0, I1, I2, I3, images_f

        multi_peak_pixels = []
        # ★ freq_indices 基于 valid_length（有效 bin 范围）
        freq_indices = np.arange(valid_length)[:, None]  # (valid_length, 1)

        for row_start in range(0, height, block_size):
            row_end = min(row_start + block_size, height)
            D_block = D[:, row_start:row_end, :]

            # FFT，归一化后截取有效 bin
            C_full = np.fft.fft(D_block, n=fft_length, axis=0)
            C_mag  = np.abs(C_full)[:valid_length, :, :] / fft_length  # (valid_length, block_h, width)

            block_h = row_end - row_start

            for local_row in range(block_h):
                C_row = C_mag[:, local_row, :]  # (valid_length, width)

                primary_idx = np.argmax(C_row, axis=0)           # (width,)
                col_indices = np.arange(width)
                primary_val = C_row[primary_idx, col_indices]

                # 圆周距离（基于 valid_length）
                dist = np.abs(freq_indices - primary_idx[None, :])
                dist = np.minimum(dist, valid_length - dist)

                # 抑制主峰邻域
                C_masked = C_row.copy()
                C_masked[dist <= suppress_radius] = 0

                secondary_idx = np.argmax(C_masked, axis=0)
                secondary_val = C_masked[secondary_idx, col_indices]

                mask = secondary_val > peak_limit
                abs_row = row_start + local_row
                for col in np.where(mask)[0]:
                    if len(multi_peak_pixels) < max_report * 10:
                        pv = primary_val[col]
                        sv = secondary_val[col]
                        multi_peak_pixels.append({
                            'x': int(col), 'y': abs_row,
                            'primary_pos':   int(primary_idx[col]),
                            'primary_val':   float(pv),
                            'secondary_pos': int(secondary_idx[col]),
                            'secondary_val': float(sv),
                            'ratio': float(sv / pv) if pv > 0 else 0.0,
                        })

            progress = min(row_end, height)
            print(f"\r  扫描进度: {progress}/{height} ({progress/height*100:.1f}%), "
                  f"已发现 {len(multi_peak_pixels)} 个多峰像素", end="")

        print()

        multi_peak_pixels.sort(key=lambda p: p['secondary_val'], reverse=True)
        if len(multi_peak_pixels) > max_report:
            total_found = len(multi_peak_pixels)
            multi_peak_pixels = multi_peak_pixels[:max_report]
            print(f"  共发现 {total_found} 个多峰像素，仅报告次峰最强的前 {max_report} 个")

        print(f"[OK] 多峰扫描完成，共 {len(multi_peak_pixels)} 个多峰像素")
        return multi_peak_pixels

    @staticmethod
    def generate_spectrum_data_for_pixels(images, pixel_coords, direction,
                                          proj_height, proj_width,
                                          num_frequencies, frequency_step=1.0):
        """
        为指定像素生成完整的频谱分析数据（与 visualize_spectrum 兼容的格式）。

        Parameters:
        -----------
        images : np.ndarray
            图像数组，形状为 (M*4, height, width)
        pixel_coords : list of (x, y) tuples
        direction : str
            'hor' 或 'ver'
        proj_height, proj_width : int
        num_frequencies : int
        frequency_step : float

        Returns:
        --------
        spectrum_data_dict : dict
        """
        proj_dim = proj_height if direction == "hor" else proj_width
        # ★ 使用通用公式
        fft_length, valid_length, coord_scale = _calc_fft_params(proj_dim, frequency_step)

        images_f = images.astype(np.float32) if images.dtype != np.float32 else images
        I0 = images_f[0::4];  I1 = images_f[1::4]
        I2 = images_f[2::4];  I3 = images_f[3::4]
        Dc = I0 - I2;  Ds = I3 - I1
        D  = Dc + 1j * Ds

        spectrum_data_dict = {}

        for idx, (x, y) in enumerate(pixel_coords):
            if not (0 <= x < images.shape[2] and 0 <= y < images.shape[1]):
                print(f"  ⚠ 像素({x},{y}) 超出图像范围，跳过")
                continue

            signal = D[:, y, x]
            C = np.fft.fft(signal, n=fft_length)
            C_magnitude = np.abs(C)

            # ★ 峰值搜索限制在有效 bin 范围内
            peak_idx   = int(np.argmax(C_magnitude[:valid_length]))
            peak_value = float(C_magnitude[peak_idx] / fft_length)

            spectrum_data_dict[idx] = {
                'spectrum':         C_magnitude / fft_length,  # 完整频谱（含无效区域），供可视化
                'spectrum_raw':     C_magnitude.copy(),
                'spectrum_complex': C.copy(),
                'frequencies':      np.arange(fft_length),
                'peak_index':       peak_idx,
                'peak_value':       peak_value,
                'point':            (x, y),
                'direction':        direction,
                'N':                fft_length,
                'valid_length':     valid_length,            # ★ 新增：有效 bin 数量
                'num_frequencies':  num_frequencies,
                'frequency_step':   frequency_step,
                'I0': I0[:, y, x].copy(), 'I1': I1[:, y, x].copy(),
                'I2': I2[:, y, x].copy(), 'I3': I3[:, y, x].copy(),
                'Dc': Dc[:, y, x].copy(), 'Ds': Ds[:, y, x].copy(),
            }

        return spectrum_data_dict


class EpipolarCorrector:
    """
    极线约束校正器

    恢复自原始代码 decode_fsa.py 中的相关函数，利用相机-投影仪极线几何约束
    从多个候选峰值中选出正确匹配，修正因频谱多峰导致的误匹配。
    """

    def __init__(self, cam_intrinsic, cam_distortion, cam_rotation, cam_translation,
                 proj_intrinsic, proj_distortion, proj_rotation, proj_translation):
        import cv2

        self.Kc = np.array(cam_intrinsic,  dtype=np.float64)
        self.dc = np.array(cam_distortion, dtype=np.float64)
        self.Kp = np.array(proj_intrinsic,  dtype=np.float64)
        self.dp = np.array(proj_distortion, dtype=np.float64)

        self.Rc, _ = cv2.Rodrigues(np.array(cam_rotation,  dtype=np.float64).reshape(-1))
        self.tc    = np.array(cam_translation, dtype=np.float64).flatten()
        self.Rp, _ = cv2.Rodrigues(np.array(proj_rotation, dtype=np.float64).reshape(-1))
        self.tp    = np.array(proj_translation, dtype=np.float64).flatten()

        Pc = np.empty((3, 4), dtype=np.float64)
        Pc[:, :3] = self.Rc;  Pc[:, -1] = self.tc
        self.Pc = self.Kc @ Pc

        Pp = np.empty((3, 4), dtype=np.float64)
        Pp[:, :3] = self.Rp;  Pp[:, -1] = self.tp
        self.Pp = self.Kp @ Pp

        self.camera_center = -self.Rc.T @ self.tc
        ep = self.Kp @ (self.Rp @ self.camera_center + self.tp)
        self.epipole = ep / ep[-1]

    def get_epipolar_line(self, cx, cy):
        """计算相机像素 (cx, cy) 在投影仪图像中对应的极线 [a, b, c]"""
        import cv2

        xc_undis = cv2.undistortPoints(
            np.array([[cx, cy]], dtype=np.float64),
            self.Kc, self.dc, P=self.Kc
        )[0][0]
        xc = np.array([xc_undis[0], xc_undis[1], 1.0], dtype=np.float64)

        xp = self.Pp @ np.linalg.pinv(self.Pc) @ xc
        xp = xp / xp[-1]

        lp = np.cross(self.epipole, xp)
        lp = lp / lp[-1]
        return lp

    def correct_disparity(self, pixel_coords, peaks_hor, peaks_ver,
                          raw_disp_hor=None, raw_disp_ver=None):
        """利用极线约束从候选峰值中选出最佳匹配，返回校正结果列表"""
        import cv2

        results = []
        for i, (cx, cy) in enumerate(pixel_coords):
            lp     = self.get_epipolar_line(cx, cy)
            pk_hor = peaks_hor[i]
            pk_ver = peaks_ver[i]

            if len(pk_hor) == 0 or len(pk_ver) == 0:
                results.append({
                    'py': raw_disp_hor[cy, cx] if raw_disp_hor is not None else np.nan,
                    'px': raw_disp_ver[cy, cx] if raw_disp_ver is not None else np.nan,
                    'distance': np.inf, 'corrected': False
                })
                continue

            pys = np.repeat(pk_hor, len(pk_ver))
            pxs = np.tile(pk_ver, len(pk_hor))

            proj_pts = np.column_stack([pxs, pys]).astype(np.float64)
            undis    = cv2.undistortPoints(proj_pts, self.Kp, self.dp, P=self.Kp).reshape(-1, 2)

            distances = np.abs(
                lp[0] * undis[:, 0] + lp[1] * undis[:, 1] + lp[2]
            ) / np.sqrt(lp[0]**2 + lp[1]**2)

            best = int(np.argmin(distances))
            result = {
                'py': float(pys[best]), 'px': float(pxs[best]),
                'distance': float(distances[best]), 'corrected': True,
                'all_candidates': [(float(px), float(py), float(d))
                                   for px, py, d in zip(pxs, pys, distances)]
            }
            if raw_disp_hor is not None and raw_disp_ver is not None:
                result['original_py'] = float(raw_disp_hor[cy, cx])
                result['original_px'] = float(raw_disp_ver[cy, cx])
            results.append(result)

        return results

    @staticmethod
    def draw_epipolar(image, lp):
        """在图像上绘制极线"""
        import cv2
        x0 = 0
        y0 = -lp[2] / lp[1]
        x1 = image.shape[1]
        y1 = -(lp[2] + lp[0] * x1) / lp[1]
        cv2.line(image,
                 np.array((x0, y0), dtype=np.int64),
                 np.array((x1, y1), dtype=np.int64),
                 (0, 0, 255), 1)


class FSADecoder:
    """
    频移法解码协调器

    整合 ImageLoader、DFTProcessor、PeakDetector、MultiPeakDetector、EpipolarCorrector，
    提供统一的解码接口。

    ★ FFT 参数策略（通用公式，支持任意 frequency_step）：
    -------------------------------------------------------
    fft_length   = int(round(proj_dim / frequency_step))
    valid_length = proj_dim       ← argmax 只在前 proj_dim 个 bin 中搜索
    coord_scale  ≈ 1.0           ← FFT bin 编号直接对应投影仪坐标

    典型情况：
      frequency_step = 1.0  → fft_length = proj_dim   （原始逻辑，完全向后兼容）
      frequency_step = 0.5  → fft_length = 2*proj_dim （新增支持）
      frequency_step = 0.25 → fft_length = 4*proj_dim （可自然扩展）

    为什么 valid_length < fft_length 时必须截断：
      当 fft_length > proj_dim 时，FFT 输出后半段 [proj_dim, fft_length-1] 属于
      负频率区域（物理上无效）。对全频谱做 argmax 可能因噪声或反光在无效区域
      找到强峰，输出超出投影仪范围的坐标。限制 argmax 范围到 [0, proj_dim-1]
      可强制结果落在合法坐标区间内。
    """

    def __init__(self, projector_width: int, projector_height: int,
                 frequency_step: float = 1.0):
        """
        Parameters:
        -----------
        projector_width  : int   投影仪宽度（像素）
        projector_height : int   投影仪高度（像素）
        frequency_step   : float 频率步长（Hz），默认 1.0；支持 0.5、0.25 等任意正数值
        """
        self.proj_width    = projector_width
        self.proj_height   = projector_height
        self.frequency_step = frequency_step

    def verify_image_sequence(self, images, num_frequencies, sample_points=5):
        return ImageLoader.verify_image_sequence(images, num_frequencies, sample_points)

    def load_images(self, image_dir, num_frequencies,
                    image_pattern="{:d}.png",
                    use_cache=True, cache_path=None,
                    verify_sequence=False):
        return ImageLoader.load_images(
            image_dir=image_dir, num_frequencies=num_frequencies,
            image_pattern=image_pattern, use_cache=use_cache,
            cache_path=cache_path, verify_sequence=verify_sequence
        )

    def _get_fft_params(self, direction: str, num_frequencies: int):
        """
        计算 FFT 参数（通用公式，由模块级 _calc_fft_params 实现）。

        Parameters:
        -----------
        direction       : str  'hor' 或 'ver'
        num_frequencies : int  频率数量 M（保留参数，向后兼容）

        Returns:
        --------
        fft_length   : int
        coord_scale  : float  ≈ 1.0
        proj_dim     : int    投影仪对应维度的分辨率
        valid_length : int    有效 bin 数，argmax 只在 [0, valid_length) 中搜索
        """
        proj_dim = self.proj_height if direction == "hor" else self.proj_width
        fft_length, valid_length, coord_scale = _calc_fft_params(proj_dim, self.frequency_step)
        return fft_length, coord_scale, proj_dim, valid_length

    # ------------------------------------------------------------------
    # 低内存模式（逐行处理）
    # ------------------------------------------------------------------
    def _decode_direction_low_memory(self,
                                     images, direction, num_frequencies,
                                     fft_length, coord_scale, valid_length,
                                     debug_point=None, spectrum_points=None):
        """低内存模式：逐行处理，内存占用极低"""
        height, width = images.shape[1], images.shape[2]

        print("\n使用低内存模式 - 逐行处理")
        print(f"  总行数: {height}")
        print(f"  FFT长度: {fft_length}，有效bin范围: 0~{valid_length-1}")
        print(f"  坐标映射比例: {coord_scale:.6f}")

        if spectrum_points is not None:
            if (isinstance(spectrum_points, tuple) and len(spectrum_points) == 2
                    and not isinstance(spectrum_points[0], tuple)):
                spectrum_points_list = [spectrum_points]
            else:
                spectrum_points_list = list(spectrum_points)
            print(f"\n⚡ 频谱分析点: {len(spectrum_points_list)} 个点")
        else:
            spectrum_points_list = []

        disparity    = np.zeros((height, width), dtype=np.float32)
        peak         = np.zeros((height, width), dtype=np.float32)
        spectrum_data = None

        for row in range(height):
            if (row + 1) % 50 == 0 or row == height - 1:
                print(f"\r  处理进度: {row+1}/{height} 行 ({(row+1)/height*100:.1f}%)", end="")

            I0_row = images[0::4, row, :].astype(np.float32)
            I1_row = images[1::4, row, :].astype(np.float32)
            I2_row = images[2::4, row, :].astype(np.float32)
            I3_row = images[3::4, row, :].astype(np.float32)
            Dc_row = I0_row - I2_row
            Ds_row = I3_row - I1_row
            D_row  = Dc_row + 1j * Ds_row

            # 预存本行中频谱分析点的原始数据
            row_sp_data = {}
            for pt_idx, (x, y) in enumerate(spectrum_points_list):
                if row == y and 0 <= x < width:
                    row_sp_data[pt_idx] = {
                        'I0': I0_row[:, x].copy(), 'I1': I1_row[:, x].copy(),
                        'I2': I2_row[:, x].copy(), 'I3': I3_row[:, x].copy(),
                        'Dc': Dc_row[:, x].copy(), 'Ds': Ds_row[:, x].copy(),
                        'point': (x, row),
                    }

            for col in range(width):
                C = np.fft.fft(D_row[:, col], n=fft_length)
                C_mag   = np.abs(C)
                C_valid = C_mag[:valid_length]

                peak_idx   = int(np.argmax(C_valid))
                peak_value = float(C_valid[peak_idx] / fft_length)

                # 频谱分析点保存
                for pt_idx, pt_data in row_sp_data.items():
                    if col == pt_data['point'][0]:
                        if spectrum_data is None:
                            spectrum_data = {}
                        spectrum_data[pt_idx] = {
                            'spectrum':         C_mag / fft_length,
                            'spectrum_raw':     C_mag.copy(),
                            'spectrum_complex': C.copy(),
                            'frequencies':      np.arange(fft_length),
                            'peak_index':       peak_idx,
                            'peak_value':       peak_value,
                            'point':            (col, row),
                            'direction':        direction,
                            'N':                fft_length,
                            'valid_length':     valid_length,
                            'num_frequencies':  num_frequencies,
                            'frequency_step':   self.frequency_step,
                            'I0': pt_data['I0'], 'I1': pt_data['I1'],
                            'I2': pt_data['I2'], 'I3': pt_data['I3'],
                            'Dc': pt_data['Dc'], 'Ds': pt_data['Ds'],
                        }
                        print(f"\n\n  ⚡ [频谱分析] 点{pt_idx+1} ({col},{row}) 已保存，"
                              f"FFT={fft_length}，有效={valid_length}，"
                              f"峰值位置={peak_idx}，强度={peak_value:.4f}")

                # 子像素精度（质心法）
                if peak_value > 0.1:
                    vl = C_mag[(peak_idx - 1) % fft_length]
                    vc = C_mag[peak_idx]
                    vr = C_mag[(peak_idx + 1) % fft_length]
                    if vl + vc + vr > 0:
                        peak_coord = peak_idx + 0.5 * (vr - vl) / (vl + vc + vr)
                    else:
                        peak_coord = float(peak_idx)
                else:
                    peak_coord = float(peak_idx)

                disparity[row, col] = peak_coord
                peak[row, col]      = peak_value

            if debug_point is not None and row == debug_point[1]:
                x = debug_point[0]
                print(f"\n\n  [调试] 点 ({x},{row}): 坐标={disparity[row,x]:.2f}, "
                      f"峰值强度={peak[row,x]:.4f}")

        print()
        print("\n[OK] 低内存模式解码完成")

        if abs(coord_scale - 1.0) > 1e-9:
            disparity = disparity * coord_scale
            print(f"  已应用坐标映射: × {coord_scale:.6f}")

        return disparity, peak, spectrum_data

    # ------------------------------------------------------------------
    # 主解码入口
    # ------------------------------------------------------------------
    def decode_direction(self,
                         images: np.ndarray,
                         direction: str,
                         num_frequencies: int,
                         block_size: int = 100,
                         debug_point: Optional[Tuple[int, int]] = None,
                         low_memory: bool = False,
                         spectrum_points: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None
                         ) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        对特定方向的条纹图像进行DFT解码。

        Parameters:
        -----------
        images          : np.ndarray  形状 (M*4, height, width)
        direction       : str         'hor' 或 'ver'
        num_frequencies : int         频率数量 M
        block_size      : int         分块大小
        debug_point     : tuple       调试点坐标 (x, y)
        low_memory      : bool        是否逐行处理
        spectrum_points : tuple or list  频谱分析点坐标

        Returns:
        --------
        disparity    : np.ndarray  视差图（投影仪坐标）
        peak         : np.ndarray  峰值强度图（置信度）
        spectrum_data : dict|None  频谱分析数据
        """
        fft_length, coord_scale, proj_dim, valid_length = self._get_fft_params(direction, num_frequencies)

        print(f"\n{'='*60}")
        print(f"开始解码 {direction} 方向 (投影仪{'行' if direction == 'hor' else '列'}坐标)")
        print(f"图像数量: {images.shape[0]} ({num_frequencies} 频率 × 4 相移)")
        print(f"图像尺寸: {images.shape[1]} × {images.shape[2]}")
        print(f"投影仪{'高度' if direction == 'hor' else '宽度'}: {proj_dim}")
        print(f"频率步长: {self.frequency_step}")
        print(f"★ FFT长度: {fft_length}")
        print(f"★ 有效bin范围: 0 ~ {valid_length-1} (共 {valid_length} 个)")
        if self.frequency_step != 1.0:
            print(f"★ 注: fft_length = proj_dim({proj_dim}) / step({self.frequency_step}) = {fft_length}")
        print(f"★ 坐标映射: {'直接输出投影仪坐标 (coord_scale≈1.0)' if abs(coord_scale-1.0)<1e-9 else f'argmax × {coord_scale:.6f}'}")
        if low_memory:
            print(f"模式: 低内存模式 (逐行处理)")
        print(f"{'='*60}")

        if low_memory:
            return self._decode_direction_low_memory(
                images, direction, num_frequencies,
                fft_length, coord_scale, valid_length,
                debug_point, spectrum_points
            )

        # ---------- 标准分块模式 ----------
        print("\n步骤1: 提取四步相移图像序列...")
        I0 = images[0::4].astype(np.float32)
        I1 = images[1::4].astype(np.float32)
        I2 = images[2::4].astype(np.float32)
        I3 = images[3::4].astype(np.float32)
        print(f"  I0~I3 形状: {I0.shape}")
        del images

        print("\n步骤2: 构造复数信号 D = (I0-I2) + j*(I3-I1)...")
        Dc = I0 - I2;  Ds = I3 - I1
        D  = Dc + 1j * Ds
        print(f"  D 形状: {D.shape}, 类型: {D.dtype}")

        if debug_point is not None:
            x, y = debug_point
            print(f"\n  [调试] 点({x},{y}): I0均值={I0[:,y,x].mean():.2f}, Dc均值={Dc[:,y,x].mean():.2f}")

        # 保存频谱分析点原始数据
        spectrum_raw_data = {}
        if spectrum_points is not None:
            if (isinstance(spectrum_points, tuple) and len(spectrum_points) == 2
                    and not isinstance(spectrum_points[0], tuple)):
                spectrum_points_list = [spectrum_points]
            else:
                spectrum_points_list = list(spectrum_points)
            for pt_idx, (x, y) in enumerate(spectrum_points_list):
                if 0 <= x < D.shape[2] and 0 <= y < D.shape[1]:
                    spectrum_raw_data[pt_idx] = {
                        'I0': I0[:,y,x].copy(), 'I1': I1[:,y,x].copy(),
                        'I2': I2[:,y,x].copy(), 'I3': I3[:,y,x].copy(),
                        'Dc': Dc[:,y,x].copy(), 'Ds': Ds[:,y,x].copy(),
                        'point': (x, y),
                    }
                    print(f"  [OK] 已保存频谱分析点{pt_idx+1} ({x},{y}) 的原始数据")
                else:
                    print(f"  ⚠ 频谱分析点{pt_idx+1} ({x},{y}) 超出图像范围")
        else:
            spectrum_points_list = []

        del I0, I1, I2, I3, Dc, Ds

        disparity     = np.empty(D.shape[1:], dtype=np.float32)
        peak          = np.empty(D.shape[1:], dtype=np.float32)
        spectrum_data = None

        total_blocks = (((D.shape[1] + block_size - 1) // block_size) *
                        ((D.shape[2] + block_size - 1) // block_size))
        processed_blocks = 0

        print(f"\n步骤3: DFT变换（FFT长度={fft_length}，有效bin={valid_length}，块大小={block_size}）...")

        for ty in range(0, D.shape[1], block_size):
            for tx in range(0, D.shape[2], block_size):
                ey = min(ty + block_size, D.shape[1])
                ex = min(tx + block_size, D.shape[2])

                D_block = D[:, ty:ey, tx:ex]

                # FFT
                C = np.fft.fft(D_block, n=fft_length, axis=0)
                C_magnitude = np.abs(C)

                # ★ 只在有效 bin [0, valid_length) 内搜索峰值
                C_valid    = C_magnitude[:valid_length, :, :]
                freq_index = np.argmax(C_valid, axis=0)

                # coord_scale ≈ 1.0，直接输出投影仪坐标
                if abs(coord_scale - 1.0) < 1e-9:
                    disparity[ty:ey, tx:ex] = freq_index
                else:
                    disparity[ty:ey, tx:ex] = freq_index * coord_scale

                peak[ty:ey, tx:ex] = np.max(C_valid, axis=0) / fft_length

                processed_blocks += 1
                print(f"\r  进度: {processed_blocks}/{total_blocks} "
                      f"({processed_blocks/total_blocks*100:.1f}%)", end="")

                # 调试点
                if debug_point is not None:
                    x, y = debug_point
                    if ty <= y < ey and tx <= x < ex:
                        lx, ly = x - tx, y - ty
                        C_vp = C_valid[:, ly, lx]
                        print(f"\n\n  [调试] 点({x},{y}): 坐标={disparity[y,x]:.2f}, 强度={peak[y,x]:.4f}")
                        top5 = np.argsort(C_vp)[-5:][::-1]
                        for ii, idx in enumerate(top5):
                            print(f"      #{ii+1}: 位置={idx}, 强度={C_vp[idx]/fft_length:.4f}")
                        print()

                # 频谱分析点
                if spectrum_raw_data:
                    for pt_idx, raw_data in spectrum_raw_data.items():
                        x, y = raw_data['point']
                        if ty <= y < ey and tx <= x < ex:
                            lx, ly = x - tx, y - ty
                            C_pt     = C[:, ly, lx]
                            C_mag_pt = np.abs(C_pt)
                            C_val_pt = C_mag_pt[:valid_length]

                            if spectrum_data is None:
                                spectrum_data = {}
                            spectrum_data[pt_idx] = {
                                'spectrum':         C_mag_pt / fft_length,
                                'spectrum_raw':     C_mag_pt,
                                'spectrum_complex': C_pt,
                                'frequencies':      np.arange(fft_length),
                                'peak_index':       int(disparity[y, x]),
                                'peak_value':       float(peak[y, x]),
                                'point':            (x, y),
                                'direction':        direction,
                                'N':                fft_length,
                                'valid_length':     valid_length,
                                'num_frequencies':  num_frequencies,
                                'frequency_step':   self.frequency_step,
                                'I0': raw_data['I0'], 'I1': raw_data['I1'],
                                'I2': raw_data['I2'], 'I3': raw_data['I3'],
                                'Dc': raw_data['Dc'], 'Ds': raw_data['Ds'],
                            }
                            print(f"\n\n  ⚡ [频谱分析] 点{pt_idx+1} ({x},{y}) 已保存，"
                                  f"峰值位置={int(disparity[y,x])}，强度={peak[y,x]:.4f}")
                            top10 = np.argsort(C_val_pt)[-10:][::-1]
                            print(f"    频谱前10个峰值（有效范围内）:")
                            for ii, idx in enumerate(top10):
                                print(f"      #{ii+1}: 投影仪坐标={idx:4d}, 强度={C_val_pt[idx]/fft_length:.4f}")
                            print()

        print()
        print("\n步骤4: 解码完成，统计信息:")
        print(f"  FFT长度: {fft_length}，有效bin数: {valid_length}")
        print(f"  coord_scale: {coord_scale:.6f}")
        print(f"  视差范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        print(f"  峰值强度范围: [{peak.min():.4f}, {peak.max():.4f}]")
        print(f"  峰值强度均值: {peak.mean():.4f}，中值: {np.median(peak):.4f}")

        return disparity, peak, spectrum_data

    def save_results(self, disparity, peak, output_dir, direction):
        """保存视差图和峰值图"""
        os.makedirs(output_dir, exist_ok=True)
        disp_path = os.path.join(output_dir, f"disp_{direction}.npy")
        peak_path = os.path.join(output_dir, f"peak_{direction}.npy")
        np.save(disp_path, disparity)
        np.save(peak_path, peak)
        print(f"\n结果已保存:")
        print(f"  视差图: {disp_path}")
        print(f"  峰值图: {peak_path}")


def demonstrate_fsa_decoding():
    """演示FSA解码的完整流程"""
    print("="*80)
    print("FSA (频移法) 解码演示程序")
    print("="*80)

    PROJECTOR_WIDTH  = 1920
    PROJECTOR_HEIGHT = 1080
    NUM_FREQUENCIES  = 151
    IMAGE_DIR_HOR    = "./data/hor"
    IMAGE_DIR_VER    = "./data/ver"
    OUTPUT_DIR       = "./results"

    decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

    for direction, image_dir in [('hor', IMAGE_DIR_HOR), ('ver', IMAGE_DIR_VER)]:
        print(f"\n{'='*80}")
        print(f"解码{'水平' if direction=='hor' else '垂直'}条纹")
        print(f"{'='*80}")
        try:
            images = decoder.load_images(image_dir=image_dir,
                                         num_frequencies=NUM_FREQUENCIES,
                                         use_cache=True)
            disp, pk, _ = decoder.decode_direction(
                images=images, direction=direction,
                num_frequencies=NUM_FREQUENCIES,
                block_size=100, debug_point=(500, 500)
            )
            decoder.save_results(disp, pk, OUTPUT_DIR, direction)
        except FileNotFoundError as e:
            print(f"\n错误: {e}")

    print("\n" + "="*80)
    print("解码完成！")
    print("="*80)


if __name__ == "__main__":
    demonstrate_fsa_decoding()
