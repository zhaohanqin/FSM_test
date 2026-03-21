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

2. **DFTProcessor** - DFT处理器（新增）
   负责执行DFT傅里叶变换，将时域信号转换到频域

3. **PeakDetector** - 峰值检测器（新增）
   负责在频谱中找到最高响应峰值，并计算子像素精度

4. **MultiPeakDetector** - 多峰检测器（新增，恢复自decode_fsa.py）
   在频谱中找出所有局部极大值，支持近峰加权合并，用于后续极线约束校正

5. **EpipolarCorrector** - 极线约束校正器（新增，恢复自decode_fsa.py）
   利用相机-投影仪极线几何约束，从多个候选峰值中选出正确匹配

6. **FSADecoder** - FSA解码协调器（重构）
   整合上述组件，提供统一的解码接口，保持向后兼容性

修复说明：
---------
- **DFT长度选择**：恢复原始代码decode_fsa.py中的DFT长度策略。
  当frequency_step==1.0时，FFT长度使用投影仪分辨率（而非频率数量M），
  通过零填充实现频谱插值，使argmax直接输出投影仪坐标（1像素精度）。
  当frequency_step!=1.0时，FFT长度使用频率数量M，argmax后需坐标映射。

作者：从decode_fsa.py提取并重构
日期：2025-11-01（初版），2025-11-09（重构），2025-xx-xx（DFT修复+后处理恢复）
"""

import numpy as np
import os
from typing import Tuple, Optional, List, Union


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
        
        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_path):
            print(f"从缓存加载图像: {cache_path}")
            images = np.load(cache_path)
            print(f"已加载 {images.shape[0]} 张图像，尺寸: {images.shape[1]}x{images.shape[2]}")
            return images
        
        # 从文件加载图像
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


class DFTProcessor:
    """
    DFT处理器
    
    负责执行离散傅里叶变换（DFT），将时域信号转换到频域。
    """
    
    @staticmethod
    def construct_complex_signal(I0: np.ndarray, 
                                 I1: np.ndarray,
                                 I2: np.ndarray,
                                 I3: np.ndarray) -> np.ndarray:
        """
        从四步相移图像构造复数信号
        
        D = (I0 - I2) + j*(I3 - I1)
        """
        Dc = I0 - I2
        Ds = I3 - I1
        D = Dc + 1j * Ds
        return D
    
    @staticmethod
    def perform_dft(signal: np.ndarray, N: int, axis: int = 0) -> np.ndarray:
        """执行DFT变换"""
        return np.fft.fft(signal, n=N, axis=axis)
    
    @staticmethod
    def compute_magnitude_spectrum(spectrum: np.ndarray) -> np.ndarray:
        """计算幅值谱"""
        return np.abs(spectrum)


class PeakDetector:
    """
    峰值检测器
    
    负责在频谱中找到最高响应峰值，并计算子像素精度。
    """
    
    @staticmethod
    def find_peak_index(magnitude_spectrum: np.ndarray, axis: int = 0) -> np.ndarray:
        """找到频谱中的最大峰值位置"""
        return np.argmax(magnitude_spectrum, axis=axis)
    
    @staticmethod
    def compute_peak_value(magnitude_spectrum: np.ndarray, 
                          peak_indices: np.ndarray,
                          normalization_factor: float = 1.0) -> np.ndarray:
        """计算峰值强度"""
        if len(magnitude_spectrum.shape) == 2:
            rows = np.arange(magnitude_spectrum.shape[1])
            peak_values = magnitude_spectrum[peak_indices, rows] / normalization_factor
        else:
            peak_values = np.max(magnitude_spectrum, axis=0) / normalization_factor
        return peak_values
    
    @staticmethod
    def refine_peak_subpixel(magnitude_spectrum: np.ndarray,
                            peak_indices: np.ndarray,
                            N: int,
                            threshold: float = 0.1) -> np.ndarray:
        """使用质心法计算子像素精度的峰值位置"""
        refined_peaks = peak_indices.astype(np.float32)
        
        if len(magnitude_spectrum.shape) == 1:
            if magnitude_spectrum[peak_indices] / N > threshold:
                idx_left = (peak_indices - 1) % N
                idx_right = (peak_indices + 1) % N
                
                val_left = magnitude_spectrum[idx_left]
                val_center = magnitude_spectrum[peak_indices]
                val_right = magnitude_spectrum[idx_right]
                
                if val_left + val_center + val_right > 0:
                    offset = (val_right - val_left) / (val_left + val_center + val_right)
                    refined_peaks = peak_indices + offset * 0.5
        
        return refined_peaks


class MultiPeakDetector:
    """
    多峰检测器
    
    恢复自原始代码 decode_fsa.py 中的 find_peaks() 和 get_peaks_position() 函数。
    
    功能说明：
    --------
    在FSA频谱中，由于反射、遮挡或多路径干扰，一个相机像素可能对应频谱中的
    多个峰值。MultiPeakDetector负责找出所有显著峰值，为后续的极线约束校正
    提供候选匹配点。
    
    原始代码中的逻辑：
    - find_peaks: 遍历频谱，找出所有局部极大值（高于阈值），近距离峰值进行加权合并
    - get_peaks_position: 对指定像素批量执行多峰检测
    """
    
    @staticmethod
    def find_peaks(spectrum, limit, merge_distance=5):
        """
        在频谱中找出所有局部极大值
        
        恢复自 decode_fsa.py 的 find_peaks(C, limit) 函数。
        
        Parameters:
        -----------
        spectrum : np.ndarray
            归一化频谱幅值（1D数组）
        limit : float
            峰值强度阈值，低于此值的峰值将被忽略
        merge_distance : int
            近峰合并距离，默认为5。距离小于此值的相邻峰值会被加权合并
            
        Returns:
        --------
        peaks : np.ndarray
            峰值位置数组（可能包含浮点数，因为加权合并后位置可以是非整数）
            
        原理说明：
        --------
        原始代码中，当两个峰值距离小于merge_distance时，使用幅值加权合并：
            ki = value / (value + prev_value)
            new_pos = ki * i + (1-ki) * prev_pos
        这实现了一种简单的质心定位，提高了峰值位置的精度。
        """
        # 使用 (position, value) 元组存储，避免原始代码中浮点索引访问频谱的潜在问题
        peaks = []  # [(position, amplitude), ...]
        
        for i in range(len(spectrum)):
            value = float(spectrum[i])
            if value < limit:
                continue
            
            left_peak = (i == 0) or (spectrum[i] > spectrum[i - 1])
            right_peak = (i == len(spectrum) - 1) or (spectrum[i] > spectrum[i + 1])
            is_peak = left_peak and right_peak
            
            if is_peak:
                if len(peaks) > 0 and (i - peaks[-1][0]) < merge_distance:
                    # 加权合并近距离峰值（恢复自原始代码逻辑）
                    prev_pos, prev_val = peaks[-1]
                    ki = value / (value + prev_val)
                    new_pos = ki * i + (1.0 - ki) * prev_pos
                    # 保留较大的幅值用于后续可能的合并比较
                    peaks[-1] = (new_pos, max(value, prev_val))
                else:
                    peaks.append((float(i), value))
        
        if len(peaks) == 0:
            return np.array([])
        return np.array([p[0] for p in peaks])
    
    @staticmethod
    def get_peaks_for_pixels(images, pixel_coords, direction,
                             proj_height, proj_width,
                             num_frequencies, frequency_step=1.0,
                             peak_limit=0.5):
        """
        批量获取指定像素点的所有频谱峰值位置
        
        恢复自 decode_fsa.py 的 get_peaks_position(ays, axs, dir, limit) 函数。
        
        Parameters:
        -----------
        images : np.ndarray
            原始图像数组，形状为 (M*4, height, width)
        pixel_coords : list of (x, y) tuples
            需要分析的相机像素坐标列表
        direction : str
            条纹方向 'hor' 或 'ver'
        proj_height : int
            投影仪高度
        proj_width : int
            投影仪宽度
        num_frequencies : int
            频率数量 M
        frequency_step : float
            频率步长
        peak_limit : float
            峰值强度阈值
            
        Returns:
        --------
        results : list of np.ndarray
            每个像素点对应的峰值位置数组（投影仪坐标空间）
            
        说明：
        -----
        此函数恢复了原始代码中 get_peaks_position 的关键逻辑：
        1. 使用与主解码相同的FFT参数（投影仪分辨率作为FFT长度，frequency_step=1.0时）
        2. 对每个像素的频谱执行多峰检测
        3. 返回投影仪坐标空间中的峰值位置
        """
        # ★ 关键：使用与主解码一致的FFT长度策略（恢复自decode_fsa.py）
        proj_dim = proj_height if direction == "hor" else proj_width
        
        if frequency_step == 1.0:
            fft_length = proj_dim   # 投影仪分辨率，与decode_fsa.py中N=H/W一致
            coord_scale = 1.0       # argmax直接是投影仪坐标
        else:
            fft_length = num_frequencies
            coord_scale = proj_dim / num_frequencies
        
        # 构造复数信号
        images_f = images.astype(np.float32) if images.dtype != np.float32 else images
        I0 = images_f[0::4]
        I1 = images_f[1::4]
        I2 = images_f[2::4]
        I3 = images_f[3::4]
        Dc = I0 - I2
        Ds = I3 - I1
        D = Dc + 1j * Ds
        
        results = []
        for x, y in pixel_coords:
            # 与原始代码一致：C = np.abs(np.fft.fft(D[:,y,x], N)) / N
            C = np.abs(np.fft.fft(D[:, y, x], n=fft_length)) / fft_length
            peaks = MultiPeakDetector.find_peaks(C, peak_limit)
            
            # 当需要坐标映射时（frequency_step != 1.0），将频率索引转换为投影仪坐标
            if coord_scale != 1.0 and len(peaks) > 0:
                peaks = peaks * coord_scale
            
            results.append(peaks)
        
        return results


class EpipolarCorrector:
    """
    极线约束校正器
    
    恢复自原始代码 decode_fsa.py 中的以下函数：
    - get_epipolar_line(cx, cy): 计算相机像素在投影仪中的极线
    - fix_fsa_disparity(): 利用极线约束修正误匹配
    - draw_epipolar(image, lp): 在图像上绘制极线
    
    功能说明：
    --------
    在FSA解码中，由于频谱多峰的存在，argmax可能选择了错误的峰值
    （尤其在反射、遮挡或多路径干扰场景下）。极线约束校正的原理是：
    
    对于相机中的一个像素点，其在投影仪中的对应点必须落在极线上（或极线附近）。
    因此，可以枚举所有候选峰值组合（水平方向×垂直方向），计算每个候选点
    到极线的距离，选择距离最小的候选点作为正确匹配。
    
    使用前提：
    --------
    需要已完成相机-投影仪的双目标定，获取内参、外参和畸变参数。
    需要安装 opencv-python (cv2)。
    """
    
    def __init__(self, cam_intrinsic, cam_distortion, cam_rotation, cam_translation,
                 proj_intrinsic, proj_distortion, proj_rotation, proj_translation):
        """
        初始化极线约束校正器
        
        Parameters:
        -----------
        cam_intrinsic : np.ndarray, shape (3,3)
            相机内参矩阵
        cam_distortion : np.ndarray
            相机畸变系数
        cam_rotation : np.ndarray
            相机旋转向量（Rodrigues格式，(3,)或(3,1)）
        cam_translation : np.ndarray
            相机平移向量 (3,) 或 (3,1)
        proj_intrinsic : np.ndarray, shape (3,3)
            投影仪内参矩阵
        proj_distortion : np.ndarray
            投影仪畸变系数
        proj_rotation : np.ndarray
            投影仪旋转向量（Rodrigues格式）
        proj_translation : np.ndarray
            投影仪平移向量
        
        说明：
        -----
        恢复自 decode_fsa.py 中 get_epipolar_line() 的初始化部分。
        原始代码通过 CalibrationData 从文件加载参数，此处改为直接传入参数，
        消除对 pyaoi 库的依赖。
        """
        import cv2
        
        self.Kc = np.array(cam_intrinsic, dtype=np.float64)
        self.dc = np.array(cam_distortion, dtype=np.float64)
        self.Kp = np.array(proj_intrinsic, dtype=np.float64)
        self.dp = np.array(proj_distortion, dtype=np.float64)
        
        # 旋转向量转旋转矩阵（与原始代码一致：Rc,_ = cv2.Rodrigues(cam_calib_data.rotation)）
        self.Rc, _ = cv2.Rodrigues(np.array(cam_rotation, dtype=np.float64).reshape(-1))
        self.tc = np.array(cam_translation, dtype=np.float64).flatten()
        self.Rp, _ = cv2.Rodrigues(np.array(proj_rotation, dtype=np.float64).reshape(-1))
        self.tp = np.array(proj_translation, dtype=np.float64).flatten()
        
        # 相机投影矩阵 Pc = Kc @ [Rc | tc]（恢复自原始代码）
        Pc = np.empty((3, 4), dtype=np.float64)
        Pc[:, :3] = self.Rc
        Pc[:, -1] = self.tc
        self.Pc = self.Kc @ Pc
        
        # 投影仪投影矩阵 Pp = Kp @ [Rp | tp]（恢复自原始代码）
        Pp = np.empty((3, 4), dtype=np.float64)
        Pp[:, :3] = self.Rp
        Pp[:, -1] = self.tp
        self.Pp = self.Kp @ Pp
        
        # 相机光心（世界坐标）: camera_center = -Rc.T @ tc（恢复自原始代码）
        self.camera_center = -self.Rc.T @ self.tc
        
        # 极点（投影仪图像中）: ep = Kp @ (Rp @ camera_center + tp)（恢复自原始代码）
        ep = self.Kp @ (self.Rp @ self.camera_center + self.tp)
        self.epipole = ep / ep[-1]
    
    def get_epipolar_line(self, cx, cy):
        """
        计算相机像素 (cx, cy) 在投影仪图像中对应的极线
        
        恢复自 decode_fsa.py 的 get_epipolar_line(cx, cy) 函数。
        
        Parameters:
        -----------
        cx, cy : float
            相机像素坐标
            
        Returns:
        --------
        lp : np.ndarray, shape (3,)
            极线方程 [a, b, c]，满足 a*x + b*y + c = 0
            已归一化使 c 分量为1
        """
        import cv2
        
        # 去畸变（恢复自原始代码）
        xc_undis = cv2.undistortPoints(
            np.array([[cx, cy]], dtype=np.float64),
            self.Kc, self.dc, P=self.Kc
        )[0][0]
        xc = np.array([xc_undis[0], xc_undis[1], 1.0], dtype=np.float64)
        
        # 将相机点投影到投影仪（恢复自原始代码）
        # xp = Pp @ pinv(Pc) @ xc
        xp = self.Pp @ np.linalg.pinv(self.Pc) @ xc
        xp = xp / xp[-1]
        
        # 极线 = 极点 × 投影点（恢复自原始代码）
        lp = np.cross(self.epipole, xp)
        lp = lp / lp[-1]
        
        return lp
    
    def correct_disparity(self, pixel_coords, peaks_hor, peaks_ver,
                          raw_disp_hor=None, raw_disp_ver=None):
        """
        利用极线约束校正视差
        
        恢复自 decode_fsa.py 的 fix_fsa_disparity() 函数的核心逻辑。
        原始代码中该函数主要用于诊断和可视化，此处将其改造为自动校正函数，
        自动选择离极线最近的候选点。
        
        Parameters:
        -----------
        pixel_coords : list of (x, y) tuples
            需要校正的相机像素坐标列表
        peaks_hor : list of np.ndarray
            每个像素的水平方向候选峰值列表（投影仪行坐标）
        peaks_ver : list of np.ndarray
            每个像素的垂直方向候选峰值列表（投影仪列坐标）
        raw_disp_hor : np.ndarray, optional
            原始水平视差图，用于参考和比较
        raw_disp_ver : np.ndarray, optional
            原始垂直视差图，用于参考和比较
            
        Returns:
        --------
        results : list of dict
            每个像素的校正结果，包含：
            - 'py': 校正后的投影仪行坐标
            - 'px': 校正后的投影仪列坐标
            - 'distance': 最佳候选点到极线的距离
            - 'corrected': 是否成功校正
            - 'all_candidates': 所有候选点及其到极线的距离
            - 'original_py', 'original_px': 原始视差值（如果提供了raw_disp）
        """
        import cv2
        
        results = []
        
        for i, (cx, cy) in enumerate(pixel_coords):
            lp = self.get_epipolar_line(cx, cy)
            pk_hor = peaks_hor[i]
            pk_ver = peaks_ver[i]
            
            if len(pk_hor) == 0 or len(pk_ver) == 0:
                result = {
                    'py': raw_disp_hor[cy, cx] if raw_disp_hor is not None else np.nan,
                    'px': raw_disp_ver[cy, cx] if raw_disp_ver is not None else np.nan,
                    'distance': np.inf,
                    'corrected': False
                }
                results.append(result)
                continue
            
            # 排列组合得到所有候选点（恢复自原始代码）
            # pys = np.repeat(pk_hor, pk_ver.shape[0])
            # pxs = np.repeat(pk_ver.reshape(1,-1), pk_hor.shape[0], axis=0).flatten()
            pys = np.repeat(pk_hor, len(pk_ver))
            pxs = np.tile(pk_ver, len(pk_hor))
            
            # 去畸变投影仪点（恢复自原始代码）
            proj_points = np.column_stack([pxs, pys]).astype(np.float64)
            undis_points = cv2.undistortPoints(
                proj_points, self.Kp, self.dp, P=self.Kp
            )
            undis_points = undis_points.reshape(-1, 2)
            
            # 计算每个候选点到极线的距离（恢复自原始代码）
            # d = |a*x + b*y + c| / sqrt(a^2 + b^2)
            distances = np.abs(
                lp[0] * undis_points[:, 0] + lp[1] * undis_points[:, 1] + lp[2]
            ) / np.sqrt(lp[0]**2 + lp[1]**2)
            
            # 选择距离最小的候选点
            best_idx = np.argmin(distances)
            
            result = {
                'py': float(pys[best_idx]),
                'px': float(pxs[best_idx]),
                'distance': float(distances[best_idx]),
                'corrected': True,
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
        """
        在图像上绘制极线
        
        恢复自 decode_fsa.py 的 draw_epipolar(image, lp) 函数。
        
        Parameters:
        -----------
        image : np.ndarray
            目标图像 (H, W, 3)
        lp : np.ndarray
            极线方程 [a, b, c]
        """
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
    
    ★ DFT长度修复说明：
    ------------------
    原始代码 decode_fsa.py 中的DFT长度选择逻辑：
    
    N = H if dir == "hor" else W  # N = 投影仪分辨率
    if frequency_step == 1.0:
        fft_length = N            # 使用投影仪分辨率作为FFT长度
    else:
        fft_length = M            # 使用频率数量作为FFT长度
    
    当 frequency_step=1.0 且 M<N 时（最常见场景），FFT会将M个样本零填充到N个点。
    零填充等价于频谱的sinc插值，使得频谱有N个bin，每个bin对应1个投影仪像素。
    因此argmax直接输出投影仪坐标，精度为1像素。
    
    之前改进版的错误：
    将FFT长度固定为 num_frequencies（即M），然后乘以 coord_scale 映射。
    这导致输出只能取 coord_scale 间隔的离散值（如每隔~5.65或~15.87个像素），
    在三维重建中表现为严重的量化阶梯效应。
    """
    
    def __init__(self, projector_width: int, projector_height: int, frequency_step: float = 1.0):
        """
        初始化FSA解码器
        
        Parameters:
        -----------
        projector_width : int
            投影仪的宽度（像素数）
        projector_height : int
            投影仪的高度（像素数）
        frequency_step : float
            频率步长，默认为1.0
        """
        self.proj_width = projector_width
        self.proj_height = projector_height
        self.frequency_step = frequency_step
        
    def verify_image_sequence(self, images: np.ndarray, num_frequencies: int, sample_points: int = 5):
        """验证图像序列的组织方式是否正确"""
        return ImageLoader.verify_image_sequence(images, num_frequencies, sample_points)
    
    def load_images(self, 
                    image_dir: str, 
                    num_frequencies: int,
                    image_pattern: str = "{:d}.png",
                    use_cache: bool = True,
                    cache_path: Optional[str] = None,
                    verify_sequence: bool = False) -> np.ndarray:
        """
        加载条纹图像序列
        
        Parameters / Returns：同 ImageLoader.load_images
        """
        return ImageLoader.load_images(
            image_dir=image_dir,
            num_frequencies=num_frequencies,
            image_pattern=image_pattern,
            use_cache=use_cache,
            cache_path=cache_path,
            verify_sequence=verify_sequence
        )
    
    def _get_fft_params(self, direction: str, num_frequencies: int):
        """
        ★ 核心修复：计算FFT参数
        
        恢复原始代码 decode_fsa.py 中的DFT长度选择逻辑。
        
        Parameters:
        -----------
        direction : str
            'hor' 或 'ver'
        num_frequencies : int
            频率数量 M
            
        Returns:
        --------
        fft_length : int
            FFT长度
        coord_scale : float
            坐标映射比例（1.0表示argmax直接是投影仪坐标）
        proj_dim : int
            投影仪对应维度的分辨率
        """
        # 投影仪分辨率（恢复自原始代码：N = H if dir == "hor" else W）
        proj_dim = self.proj_height if direction == "hor" else self.proj_width
        
        # ★ 恢复原始代码的FFT长度选择逻辑
        if self.frequency_step == 1.0:
            # 标准模式：FFT长度 = 投影仪分辨率
            # 零填充效果：M个样本补零到proj_dim个点，实现频谱sinc插值
            # argmax直接输出投影仪坐标，精度1像素
            fft_length = proj_dim
            coord_scale = 1.0
        else:
            # 非标准步长：FFT长度 = 频率数量M
            # argmax输出频率索引，需要映射到投影仪坐标
            fft_length = num_frequencies
            coord_scale = proj_dim / num_frequencies
        
        return fft_length, coord_scale, proj_dim
    
    def _decode_direction_low_memory(self,
                                     images: np.ndarray,
                                     direction: str,
                                     num_frequencies: int,
                                     fft_length: int,
                                     coord_scale: float,
                                     debug_point: Optional[Tuple[int, int]] = None,
                                     spectrum_points: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        低内存模式解码（逐行处理）
        
        Parameters:
        -----------
        fft_length : int
            FFT长度（由_get_fft_params计算）
        coord_scale : float
            坐标映射比例（由_get_fft_params计算）
        spectrum_points : tuple(x, y) or list of tuples, optional
            频谱分析点坐标
        """
        height, width = images.shape[1], images.shape[2]
        
        print("\n使用低内存模式 - 逐行处理")
        print(f"  每次只处理1行，内存占用极低")
        print(f"  总行数: {height}")
        print(f"  FFT长度: {fft_length}")
        print(f"  坐标映射比例: {coord_scale}")
        
        spectrum_data = None
        
        if spectrum_points is not None:
            if isinstance(spectrum_points, tuple) and len(spectrum_points) == 2 and not isinstance(spectrum_points[0], tuple):
                spectrum_points_list = [spectrum_points]
            else:
                spectrum_points_list = list(spectrum_points)
            print(f"\n⚡ 频谱分析点: {len(spectrum_points_list)} 个点")
            for i, point in enumerate(spectrum_points_list):
                print(f"   点{i+1}: ({point[0]}, {point[1]})")
        else:
            spectrum_points_list = []
        
        disparity = np.zeros((height, width), dtype=np.float32)
        peak = np.zeros((height, width), dtype=np.float32)
        
        for row in range(height):
            if (row + 1) % 50 == 0 or row == height - 1:
                print(f"\r  处理进度: {row+1}/{height} 行 ({(row+1)/height*100:.1f}%)", end="")
            
            I0_row = images[0::4, row, :].astype(np.float32)
            I1_row = images[1::4, row, :].astype(np.float32)
            I2_row = images[2::4, row, :].astype(np.float32)
            I3_row = images[3::4, row, :].astype(np.float32)
            
            Dc_row = I0_row - I2_row
            Ds_row = I3_row - I1_row
            D_row = Dc_row + 1j * Ds_row  # (M, width)
            
            # 保存当前行中频谱分析点的原始数据
            current_row_spectrum_data = {}
            for point_idx, point in enumerate(spectrum_points_list):
                x, y = point
                if row == y and 0 <= x < width:
                    current_row_spectrum_data[point_idx] = {
                        'I0': I0_row[:, x].copy(),
                        'I1': I1_row[:, x].copy(),
                        'I2': I2_row[:, x].copy(),
                        'I3': I3_row[:, x].copy(),
                        'Dc': Dc_row[:, x].copy(),
                        'Ds': Ds_row[:, x].copy(),
                        'D': D_row[:, x].copy(),
                        'point': (x, row),
                        'direction': direction,
                        'num_frequencies': num_frequencies,
                        'fft_length': fft_length
                    }
            
            # 逐像素处理
            for col in range(width):
                signal = D_row[:, col]
                
                # ★ 使用正确的fft_length（恢复自原始代码）
                C = np.fft.fft(signal, n=fft_length)
                C_magnitude = np.abs(C)
                
                peak_idx = np.argmax(C_magnitude)
                peak_value = C_magnitude[peak_idx] / fft_length
                
                # 保存频谱分析点的完整频谱
                for point_idx in current_row_spectrum_data:
                    point_data = current_row_spectrum_data[point_idx]
                    if col == point_data['point'][0]:
                        if spectrum_data is None:
                            spectrum_data = {}
                        
                        spectrum_data[point_idx] = {
                            'spectrum': C_magnitude / fft_length,
                            'spectrum_raw': C_magnitude.copy(),
                            'spectrum_complex': C.copy(),
                            'frequencies': np.arange(fft_length),
                            'peak_index': int(peak_idx),
                            'peak_value': float(peak_value),
                            'point': (col, row),
                            'direction': direction,
                            'N': fft_length,
                            'num_frequencies': num_frequencies,
                            'frequency_step': self.frequency_step,
                            'I0': point_data['I0'],
                            'I1': point_data['I1'],
                            'I2': point_data['I2'],
                            'I3': point_data['I3'],
                            'Dc': point_data['Dc'],
                            'Ds': point_data['Ds'],
                        }
                        
                        print(f"\n\n  ⚡ [频谱分析] 点{point_idx+1} ({col}, {row}) 的完整频谱已保存")
                        print(f"    FFT长度: {fft_length}")
                        print(f"    最大峰值位置: {peak_idx}")
                        print(f"    最大峰值强度: {peak_value:.4f}")
                        
                        top10_indices = np.argsort(C_magnitude)[-10:][::-1]
                        print(f"    频谱前10个峰值:")
                        for ii, idx in enumerate(top10_indices):
                            print(f"      #{ii+1}: 投影仪坐标={idx:4d}, 强度={C_magnitude[idx]/fft_length:.4f}")
                        print()
                
                # 子像素精度（质心法）
                if peak_value > 0.1:
                    # ★ 使用fft_length做模运算（修复）
                    idx_left = (peak_idx - 1) % fft_length
                    idx_right = (peak_idx + 1) % fft_length
                    
                    val_left = C_magnitude[idx_left]
                    val_center = C_magnitude[peak_idx]
                    val_right = C_magnitude[idx_right]
                    
                    if val_left + val_center + val_right > 0:
                        offset = (val_right - val_left) / (val_left + val_center + val_right)
                        peak_coord = peak_idx + offset * 0.5
                    else:
                        peak_coord = float(peak_idx)
                else:
                    peak_coord = float(peak_idx)
                
                disparity[row, col] = peak_coord
                peak[row, col] = peak_value
            
            if debug_point is not None and row == debug_point[1]:
                x = debug_point[0]
                print(f"\n\n  [调试] 点 ({x}, {row}):")
                print(f"    解码坐标(FFT域): {disparity[row, x]:.2f}")
                print(f"    峰值强度: {peak[row, x]:.4f}")
        
        print()
        print(f"\n[OK] 低内存模式解码完成")
        
        # ★ 应用坐标映射（当frequency_step != 1.0时需要）
        if coord_scale != 1.0:
            disparity = disparity * coord_scale
            print(f"  已应用坐标映射: × {coord_scale:.4f}")
        
        return disparity, peak, spectrum_data
    
    def decode_direction(self,
                        images: np.ndarray,
                        direction: str,
                        num_frequencies: int,
                        block_size: int = 100,
                        debug_point: Optional[Tuple[int, int]] = None,
                        low_memory: bool = False,
                        spectrum_points: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        对特定方向的条纹图像进行DFT解码
        
        Parameters:
        -----------
        images : np.ndarray
            图像序列，形状为 (M*4, height, width)
        direction : str
            条纹方向，'hor' 或 'ver'
        num_frequencies : int
            采集的频率数量 M
        block_size : int
            分块处理的块大小
        debug_point : tuple(x, y), optional
            调试点坐标
        low_memory : bool
            是否使用低内存模式
        spectrum_points : tuple(x, y) or list of tuples, optional
            频谱分析点坐标
            
        Returns:
        --------
        disparity : np.ndarray
            视差图（投影仪坐标），形状为 (height, width)
        peak : np.ndarray
            峰值强度图（置信度），形状为 (height, width)
        spectrum_data : dict or None
            频谱分析数据
        """
        # ★ 核心修复：使用正确的FFT参数（恢复自decode_fsa.py）
        fft_length, coord_scale, proj_dim = self._get_fft_params(direction, num_frequencies)
        
        print(f"\n{'='*60}")
        print(f"开始解码 {direction} 方向 (投影仪{'行' if direction == 'hor' else '列'}坐标)")
        print(f"图像数量: {images.shape[0]} ({num_frequencies} 频率 × 4 相移)")
        print(f"图像尺寸: {images.shape[1]} × {images.shape[2]}")
        print(f"投影仪{'高度' if direction == 'hor' else '宽度'}: {proj_dim}")
        print(f"频率步长: {self.frequency_step}")
        print(f"★ FFT长度: {fft_length} ({'投影仪分辨率' if self.frequency_step == 1.0 else '频率数量M'})")
        print(f"★ 坐标映射: {'直接输出投影仪坐标' if coord_scale == 1.0 else f'argmax × {coord_scale:.4f}'}")
        if low_memory:
            print(f"模式: 低内存模式 (逐行处理)")
        print(f"{'='*60}")
        
        # 低内存模式
        if low_memory:
            return self._decode_direction_low_memory(
                images, direction, num_frequencies, fft_length, coord_scale,
                debug_point, spectrum_points
            )
        
        # 步骤1: 提取四个相移图像序列
        print("\n步骤1: 提取四步相移图像序列...")
        
        I0 = images[0::4].astype(np.float32)
        I1 = images[1::4].astype(np.float32)
        I2 = images[2::4].astype(np.float32)
        I3 = images[3::4].astype(np.float32)
        
        print(f"  - I0 (0°):   形状 {I0.shape}")
        print(f"  - I1 (90°):  形状 {I1.shape}")
        print(f"  - I2 (180°): 形状 {I2.shape}")
        print(f"  - I3 (270°): 形状 {I3.shape}")
        
        del images
        
        # 步骤2: 构造复数信号
        print("\n步骤2: 构造复数信号 D = (I0-I2) + j*(I3-I1)...")
        Dc = I0 - I2
        Ds = I3 - I1
        D = Dc + 1j * Ds
        print(f"  - D (复数信号): 形状 {D.shape}, 类型 {D.dtype}")
        
        if debug_point is not None:
            x, y = debug_point
            print(f"\n  [调试] 点 ({x}, {y}) 的信号:")
            print(f"    I0: 均值={I0[:, y, x].mean():.2f}, 标准差={I0[:, y, x].std():.2f}")
            print(f"    Dc: 均值={Dc[:, y, x].mean():.2f}, 标准差={Dc[:, y, x].std():.2f}")
        
        # 保存频谱分析点的原始数据
        spectrum_raw_data = {}
        if spectrum_points is not None:
            if isinstance(spectrum_points, tuple) and len(spectrum_points) == 2 and not isinstance(spectrum_points[0], tuple):
                spectrum_points_list = [spectrum_points]
            else:
                spectrum_points_list = list(spectrum_points)
                
            for point_idx, point in enumerate(spectrum_points_list):
                x, y = point
                if 0 <= x < D.shape[2] and 0 <= y < D.shape[1]:
                    spectrum_raw_data[point_idx] = {
                        'I0': I0[:, y, x].copy(),
                        'I1': I1[:, y, x].copy(),
                        'I2': I2[:, y, x].copy(),
                        'I3': I3[:, y, x].copy(),
                        'Dc': Dc[:, y, x].copy(),
                        'Ds': Ds[:, y, x].copy(),
                        'D': D[:, y, x].copy(),
                        'point': (x, y),
                        'direction': direction,
                        'num_frequencies': num_frequencies,
                        'fft_length': fft_length
                    }
                    print(f"\n  [OK] 已保存频谱分析点{point_idx+1} ({x}, {y}) 的原始数据")
                else:
                    print(f"\n  ⚠ 警告: 频谱分析点{point_idx+1} ({x}, {y}) 超出图像范围")
        else:
            spectrum_points_list = []
        
        del I0, I1, I2, I3, Dc, Ds
        
        # 初始化输出数组
        disparity = np.empty(D.shape[1:], dtype=np.float32)
        peak = np.empty(D.shape[1:], dtype=np.float32)
        
        spectrum_data = None
        if spectrum_points_list:
            print(f"\n⚡ 频谱分析点: {len(spectrum_points_list)} 个点")
        
        # 步骤3: 分块DFT处理
        print(f"\n步骤3: 执行DFT变换（分块处理，FFT长度={fft_length}）...")
        print(f"  块大小: {block_size} × {block_size}")
        
        total_blocks = ((D.shape[1] + block_size - 1) // block_size) * \
                       ((D.shape[2] + block_size - 1) // block_size)
        processed_blocks = 0
        
        for ty in range(0, D.shape[1], block_size):
            for tx in range(0, D.shape[2], block_size):
                ey = min(ty + block_size, D.shape[1])
                ex = min(tx + block_size, D.shape[2])
                
                D_block = D[:, ty:ey, tx:ex]
                
                # ★ 核心修复：使用正确的fft_length
                # 当frequency_step=1.0时，fft_length=投影仪分辨率（零填充）
                # 当frequency_step!=1.0时，fft_length=频率数量M
                C = np.fft.fft(D_block, n=fft_length, axis=0)
                C_magnitude = np.abs(C)
                
                # ★ 峰值检测
                freq_index = np.argmax(C_magnitude, axis=0)
                
                # ★ 恢复原始代码的坐标映射逻辑
                if coord_scale == 1.0:
                    # frequency_step=1.0: argmax直接是投影仪坐标
                    disparity[ty:ey, tx:ex] = freq_index
                else:
                    # frequency_step!=1.0: 需要映射
                    disparity[ty:ey, tx:ex] = freq_index * coord_scale
                
                # ★ 峰值归一化（使用fft_length，与原始代码一致）
                peak[ty:ey, tx:ex] = np.max(C_magnitude, axis=0) / fft_length
                
                processed_blocks += 1
                progress = processed_blocks / total_blocks * 100
                print(f"\r  进度: {processed_blocks}/{total_blocks} ({progress:.1f}%)", end="")
                
                # 调试点输出
                if debug_point is not None:
                    x, y = debug_point
                    if ty <= y < ey and tx <= x < ex:
                        local_y = y - ty
                        local_x = x - tx
                        C_point = C_magnitude[:, local_y, local_x]
                        
                        print(f"\n\n  [调试] 点 ({x}, {y}) 的DFT结果:")
                        print(f"    投影仪坐标: {disparity[y, x]:.2f}")
                        print(f"    峰值强度: {peak[y, x]:.4f}")
                        print(f"    频谱前5个峰值:")
                        top5_indices = np.argsort(C_point)[-5:][::-1]
                        for ii, idx in enumerate(top5_indices):
                            print(f"      #{ii+1}: 位置={idx}, 强度={C_point[idx]/fft_length:.4f}")
                        print()
                
                # 保存频谱分析点的完整频谱
                if spectrum_raw_data:
                    for point_idx, raw_data in spectrum_raw_data.items():
                        x, y = raw_data['point']
                        if ty <= y < ey and tx <= x < ex:
                            local_y = y - ty
                            local_x = x - tx
                            C_point = C[:, local_y, local_x]
                            C_magnitude_point = np.abs(C_point)
                            
                            if spectrum_data is None:
                                spectrum_data = {}
                                
                            spectrum_data[point_idx] = {
                                'spectrum': C_magnitude_point / fft_length,
                                'spectrum_raw': C_magnitude_point,
                                'spectrum_complex': C_point,
                                'frequencies': np.arange(fft_length),
                                'peak_index': int(disparity[y, x]),
                                'peak_value': float(peak[y, x]),
                                'point': (x, y),
                                'direction': direction,
                                'N': fft_length,
                                'num_frequencies': num_frequencies,
                                'frequency_step': self.frequency_step,
                                'I0': raw_data['I0'],
                                'I1': raw_data['I1'],
                                'I2': raw_data['I2'],
                                'I3': raw_data['I3'],
                                'Dc': raw_data['Dc'],
                                'Ds': raw_data['Ds'],
                            }
                            
                            print(f"\n\n  ⚡ [频谱分析] 点{point_idx+1} ({x}, {y}) 的完整频谱已保存")
                            print(f"    FFT长度: {fft_length}")
                            print(f"    最大峰值位置: {spectrum_data[point_idx]['peak_index']}")
                            print(f"    最大峰值强度: {spectrum_data[point_idx]['peak_value']:.4f}")
                            
                            top10_indices = np.argsort(C_magnitude_point)[-10:][::-1]
                            print(f"    频谱前10个峰值:")
                            for ii, idx in enumerate(top10_indices):
                                print(f"      #{ii+1}: 投影仪坐标={idx:4d}, 强度={C_magnitude_point[idx]/fft_length:.4f}")
                            print()
        
        print()
        
        # 步骤4: 输出统计信息
        print("\n步骤4: 解码完成，统计信息:")
        print(f"  FFT长度: {fft_length}")
        print(f"  坐标映射比例: {coord_scale}")
        print(f"  视差范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        print(f"  峰值强度范围: [{peak.min():.4f}, {peak.max():.4f}]")
        print(f"  峰值强度均值: {peak.mean():.4f}")
        print(f"  峰值强度中值: {np.median(peak):.4f}")
        
        return disparity, peak, spectrum_data
    
    def save_results(self,
                    disparity: np.ndarray,
                    peak: np.ndarray,
                    output_dir: str,
                    direction: str):
        """
        保存解码结果
        
        Parameters:
        -----------
        disparity : np.ndarray
            视差图
        peak : np.ndarray
            峰值强度图
        output_dir : str
            输出目录
        direction : str
            方向标识 ('hor' 或 'ver')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        disp_path = os.path.join(output_dir, f"disp_{direction}.npy")
        peak_path = os.path.join(output_dir, f"peak_{direction}.npy")
        
        np.save(disp_path, disparity)
        np.save(peak_path, peak)
        
        print(f"\n结果已保存:")
        print(f"  视差图: {disp_path}")
        print(f"  峰值图: {peak_path}")


def demonstrate_fsa_decoding():
    """
    演示FSA解码的完整流程
    """
    print("="*80)
    print("FSA (频移法) 解码演示程序")
    print("="*80)
    
    PROJECTOR_WIDTH = 1920
    PROJECTOR_HEIGHT = 1080
    NUM_FREQUENCIES = 151
    
    IMAGE_DIR_HOR = "./data/hor"
    IMAGE_DIR_VER = "./data/ver"
    OUTPUT_DIR = "./results"
    
    decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    
    print("\n" + "="*80)
    print("第一步: 解码水平条纹（获取投影仪行坐标）")
    print("="*80)
    
    try:
        images_hor = decoder.load_images(
            image_dir=IMAGE_DIR_HOR,
            num_frequencies=NUM_FREQUENCIES,
            use_cache=True
        )
        
        disparity_hor, peak_hor, _ = decoder.decode_direction(
            images=images_hor,
            direction='hor',
            num_frequencies=NUM_FREQUENCIES,
            block_size=100,
            debug_point=(500, 500)
        )
        
        decoder.save_results(disparity_hor, peak_hor, OUTPUT_DIR, 'hor')
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print(f"请确保图像文件夹存在: {IMAGE_DIR_HOR}")
    
    print("\n" + "="*80)
    print("第二步: 解码垂直条纹（获取投影仪列坐标）")
    print("="*80)
    
    try:
        images_ver = decoder.load_images(
            image_dir=IMAGE_DIR_VER,
            num_frequencies=NUM_FREQUENCIES,
            use_cache=True
        )
        
        disparity_ver, peak_ver, _ = decoder.decode_direction(
            images=images_ver,
            direction='ver',
            num_frequencies=NUM_FREQUENCIES,
            block_size=100,
            debug_point=(500, 500)
        )
        
        decoder.save_results(disparity_ver, peak_ver, OUTPUT_DIR, 'ver')
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print(f"请确保图像文件夹存在: {IMAGE_DIR_VER}")
    
    print("\n" + "="*80)
    print("解码完成！")
    print("="*80)


if __name__ == "__main__":
    demonstrate_fsa_decoding()
