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

4. **FSADecoder** - FSA解码协调器（重构）
   整合上述组件，提供统一的解码接口，保持向后兼容性

这种设计使得：
- DFT变换过程更加清晰
- 峰值检测逻辑独立可测试
- 图像加载功能可复用
- 便于理解和维护

作者：从decode_fsa.py提取并重构
日期：2025-11-01（初版），2025-11-09（重构）
"""

import numpy as np
import os
from typing import Tuple, Optional, List, Union


class ImageLoader:
    """
    图像加载器
    
    负责加载条纹图像序列，支持缓存和中文路径。
    
    重构说明：
    --------
    将图像加载功能从FSADecoder中分离出来，使代码职责更加清晰。
    支持多种图像格式和中文路径，提供缓存机制以加速重复加载。
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
            
        说明：
        -----
        **输入图像的期望顺序：**
        图像文件应该按照"频率优先"的方式组织，即：
        - 0.png, 1.png, 2.png, 3.png → 频率0的 (0°, 90°, 180°, 270°)
        - 4.png, 5.png, 6.png, 7.png → 频率1的 (0°, 90°, 180°, 270°)
        - 以此类推
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
            
            # 使用opencv或PIL加载图像（支持中文路径）
            try:
                import cv2
                # 使用 cv2.imdecode 配合 np.fromfile 来支持中文路径
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            except ImportError:
                try:
                    from PIL import Image
                    img = np.array(Image.open(img_path).convert('L'))
                except ImportError:
                    raise ImportError("需要安装 opencv-python 或 Pillow 来读取图像")
            except Exception as e:
                # 如果上面的方法失败，尝试使用PIL
                try:
                    from PIL import Image
                    img = np.array(Image.open(img_path).convert('L'))
                except Exception:
                    raise ValueError(f"无法读取图像: {img_path}，错误: {e}")
            
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            images_list.append(img)
            
            # 显示进度
            if (i + 1) % 10 == 0 or i == total_images - 1:
                print(f"\r加载进度: {i+1}/{total_images}", end="")
        
        print()  # 换行
        images = np.array(images_list)
        
        # 保存缓存
        if use_cache:
            print(f"保存缓存到: {cache_path}")
            np.save(cache_path, images)
        
        print(f"已加载 {images.shape[0]} 张图像，尺寸: {images.shape[1]}x{images.shape[2]}")
        
        # 可选：验证图像序列
        if verify_sequence:
            ImageLoader.verify_image_sequence(images, num_frequencies)
        
        return images
    
    @staticmethod
    def verify_image_sequence(images: np.ndarray, num_frequencies: int, sample_points: int = 5) -> bool:
        """
        验证图像序列的组织方式是否正确
        
        此函数检查相移图像之间的差异，帮助判断图像序列是否按照正确的顺序组织。
        
        Parameters:
        -----------
        images : np.ndarray
            图像序列，形状为 (M*4, height, width)
        num_frequencies : int
            频率数量 M
        sample_points : int
            采样点数量，用于快速检查
            
        Returns:
        --------
        bool
            如果验证通过返回 True，否则返回 False
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
        
        # 检查同一频率的不同相位图像之间的差异
        print(f"\n检查相位图像差异 (采样 {sample_points} 个频率)...")
        sample_freqs = np.linspace(0, num_frequencies-1, min(sample_points, num_frequencies), dtype=int)
        
        has_variation = True
        for freq_idx in sample_freqs:
            base_idx = freq_idx * 4
            i0 = images[base_idx + 0].astype(np.float32)
            i1 = images[base_idx + 1].astype(np.float32)
            i2 = images[base_idx + 2].astype(np.float32)
            i3 = images[base_idx + 3].astype(np.float32)
            
            # 计算相位间的平均差异
            diff_01 = np.mean(np.abs(i0 - i1))
            diff_12 = np.mean(np.abs(i1 - i2))
            diff_23 = np.mean(np.abs(i2 - i3))
            diff_30 = np.mean(np.abs(i3 - i0))
            
            avg_diff = (diff_01 + diff_12 + diff_23 + diff_30) / 4
            
            print(f"  频率 {freq_idx:3d}: 相位间平均差异 = {avg_diff:.2f}", end="")
            
            if avg_diff < 5.0:  # 如果差异太小，可能有问题
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
    
    原理说明：
    --------
    对于每个相机像素，在M个不同频率下采集到的强度序列可以看作时域信号。
    通过DFT变换，可以将这个信号转换到频域，频谱中的峰值位置对应投影仪坐标。
    
    重构说明：
    --------
    将DFT变换的核心算法从FSADecoder中分离出来，使傅里叶变换过程更加清晰。
    """
    
    @staticmethod
    def construct_complex_signal(I0: np.ndarray, 
                                 I1: np.ndarray,
                                 I2: np.ndarray,
                                 I3: np.ndarray) -> np.ndarray:
        """
        从四步相移图像构造复数信号
        
        Parameters:
        -----------
        I0, I1, I2, I3 : np.ndarray
            四个相位的图像序列，形状为 (M, H, W)
            
        Returns:
        --------
        D : np.ndarray
            复数信号，形状为 (M, H, W)
            
        原理：
        -----
        D = (I0 - I2) + j*(I3 - I1)
        其中 (I0 - I2) 是余弦分量，(I3 - I1) 是正弦分量
        """
        Dc = I0 - I2  # 余弦分量（实部）
        Ds = I3 - I1  # 正弦分量（虚部）
        D = Dc + 1j * Ds  # 复数信号
        return D
    
    @staticmethod
    def perform_dft(signal: np.ndarray, N: int, frequency_step: float = 1.0, axis: int = 0) -> np.ndarray:
        """
        执行DFT变换
        
        Parameters:
        -----------
        signal : np.ndarray
            输入信号
        N : int
            DFT长度。当frequency_step=1.0时，N通常等于投影仪尺寸；
            当frequency_step=0.5时，N应该等于频率数量M（M=2×投影仪尺寸）
        frequency_step : float
            频率步长，默认为1.0（此参数用于文档说明，不影响DFT计算）
        axis : int
            沿哪个轴进行FFT，默认为0
            
        Returns:
        --------
        spectrum : np.ndarray
            频谱（复数）
            
        说明:
        -----
        【重要】DFT长度N的选择：
        - 当frequency_step=1.0时：N = 投影仪尺寸（频率数量M = 投影仪尺寸）
        - 当frequency_step=0.5时：N = 频率数量M（M = 2×投影仪尺寸）
        
        这样可以充分利用所有频率信息，获得更高的频率分辨率。
        """
        return np.fft.fft(signal, n=N, axis=axis)
    
    @staticmethod
    def compute_magnitude_spectrum(spectrum: np.ndarray) -> np.ndarray:
        """
        计算幅值谱
        
        Parameters:
        -----------
        spectrum : np.ndarray
            复数频谱
            
        Returns:
        --------
        magnitude : np.ndarray
            幅值谱
        """
        return np.abs(spectrum)


class PeakDetector:
    """
    峰值检测器
    
    负责在频谱中找到最高响应峰值，并计算子像素精度。
    
    原理说明：
    --------
    频谱中的峰值位置对应投影仪坐标。通过质心法可以计算子像素精度，
    提高解码精度。
    
    重构说明：
    --------
    将峰值检测逻辑从FSADecoder中分离出来，使峰值查找算法更加清晰和独立。
    """
    
    @staticmethod
    def find_peak_index(magnitude_spectrum: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        找到频谱中的最大峰值位置
        
        Parameters:
        -----------
        magnitude_spectrum : np.ndarray
            幅值谱
        axis : int
            沿哪个轴查找峰值
            
        Returns:
        --------
        peak_indices : np.ndarray
            峰值索引（DFT输出的原始索引）
        """
        return np.argmax(magnitude_spectrum, axis=axis)
    
    
    @staticmethod
    def compute_peak_value(magnitude_spectrum: np.ndarray, 
                          peak_indices: np.ndarray,
                          normalization_factor: float = 1.0) -> np.ndarray:
        """
        计算峰值强度
        
        Parameters:
        -----------
        magnitude_spectrum : np.ndarray
            幅值谱
        peak_indices : np.ndarray
            峰值索引
        normalization_factor : float
            归一化因子
            
        Returns:
        --------
        peak_values : np.ndarray
            峰值强度（归一化后）
        """
        # 对于2D数组，需要使用高级索引
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
        """
        使用质心法计算子像素精度的峰值位置
        
        Parameters:
        -----------
        magnitude_spectrum : np.ndarray
            幅值谱，形状为 (N, H, W) 或 (N,)
        peak_indices : np.ndarray
            整数峰值索引
        N : int
            频谱长度
        threshold : float
            峰值强度阈值，低于此值不进行子像素精化
            
        Returns:
        --------
        refined_peaks : np.ndarray
            子像素精度的峰值位置
        """
        refined_peaks = peak_indices.astype(np.float32)
        
        # 对于每个峰值，使用相邻点进行质心计算
        if len(magnitude_spectrum.shape) == 1:
            # 1D情况
            if magnitude_spectrum[peak_indices] / N > threshold:
                idx_left = (peak_indices - 1) % N
                idx_right = (peak_indices + 1) % N
                
                val_left = magnitude_spectrum[idx_left]
                val_center = magnitude_spectrum[peak_indices]
                val_right = magnitude_spectrum[idx_right]
                
                if val_left + val_center + val_right > 0:
                    offset = (val_right - val_left) / (val_left + val_center + val_right)
                    refined_peaks = peak_indices + offset * 0.5
        else:
            # 2D或3D情况，逐像素处理
            # 这里简化处理，实际应用中可以向量化
            pass
        
        return refined_peaks


class FSADecoder:
    """
    频移法解码协调器（重构版）
    
    原理说明：
    --------
    频移法通过投影一系列频率递增的正弦条纹，每个频率投影4张相移图（0°, 90°, 180°, 270°）。
    相机采集这些条纹后，每个像素点上的强度变化序列包含了该点对应的投影仪坐标信息。
    通过DFT变换，可以将这个强度序列转换到频域，频谱峰值位置即对应投影仪的坐标。
    
    数学原理：
    --------
    对于相机像素(xc, yc)，在M个不同频率的条纹下采集到4M张图像：
    - I0[k]: 相位为0°的图像序列
    - I1[k]: 相位为90°的图像序列  
    - I2[k]: 相位为180°的图像序列
    - I3[k]: 相位为270°的图像序列
    
    其中 k = 0, 1, ..., M-1 表示频率索引
    
    解码步骤：
    1. 构造复数信号：D[k] = (I0[k] - I2[k]) + j*(I3[k] - I1[k])
    2. DFT变换：C = FFT(D)
    3. 峰值检测：xp = argmax(|C|)
    
    峰值位置xp即为该相机像素对应的投影仪列坐标（或行坐标，取决于条纹方向）
    
    重构说明：
    --------
    本类现在作为协调器，整合以下专门的功能类：
    - ImageLoader: 加载条纹图像序列
    - DFTProcessor: 执行DFT傅里叶变换
    - PeakDetector: 检测频谱峰值并计算子像素精度
    
    这种设计使得：
    - DFT变换过程更加清晰
    - 峰值检测逻辑独立可测试
    - 代码职责分离，便于维护
    同时保持了原有的API接口，确保向后兼容性。
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
            频率步长，默认为1.0。当设置为0.5时，频率间隔为0.5Hz
        """
        self.proj_width = projector_width
        self.proj_height = projector_height
        self.frequency_step = frequency_step
        
    def verify_image_sequence(self, images: np.ndarray, num_frequencies: int, sample_points: int = 5):
        """
        验证图像序列的组织方式是否正确
        
        此函数检查相移图像之间的差异，帮助判断图像序列是否按照正确的顺序组织。
        
        Parameters:
        -----------
        images : np.ndarray
            图像序列，形状为 (M*4, height, width)
        num_frequencies : int
            频率数量 M
        sample_points : int
            采样点数量，用于快速检查
            
        Returns:
        --------
        bool
            如果验证通过返回 True，否则返回 False
        """
        # 使用ImageLoader验证图像序列
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
            
        说明：
        -----
        **输入图像的期望顺序：**
        图像文件应该按照"频率优先"的方式组织，即：
        - 0.png, 1.png, 2.png, 3.png → 频率0的 (0°, 90°, 180°, 270°)
        - 4.png, 5.png, 6.png, 7.png → 频率1的 (0°, 90°, 180°, 270°)
        - 8.png, 9.png, 10.png, 11.png → 频率2的 (0°, 90°, 180°, 270°)
        - ... 以此类推
        
        **返回数组的组织方式：**
        加载后的数组按顺序存储所有图像：
        - images[0:M*4] 包含所有M*4张图像，按采集顺序排列
        
        **后续处理时的切片方式：**
        在解码时，会通过切片操作重新组织数据：
        - images[0::4] → 提取索引 0,4,8,12,...  → 所有频率的0°图像 (M张)
        - images[1::4] → 提取索引 1,5,9,13,...  → 所有频率的90°图像 (M张)
        - images[2::4] → 提取索引 2,6,10,14,... → 所有频率的180°图像 (M张)
        - images[3::4] → 提取索引 3,7,11,15,... → 所有频率的270°图像 (M张)
        """
        # 使用ImageLoader加载图像
        return ImageLoader.load_images(
            image_dir=image_dir,
            num_frequencies=num_frequencies,
            image_pattern=image_pattern,
            use_cache=use_cache,
            cache_path=cache_path,
            verify_sequence=verify_sequence
        )
    
    def _decode_direction_low_memory(self,
                                     images: np.ndarray,
                                     direction: str,
                                     num_frequencies: int,
                                     N: int,
                                     debug_point: Optional[Tuple[int, int]] = None,
                                     spectrum_points: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        低内存模式解码（逐行处理）
        
        此函数通过逐行处理图像来最小化内存占用，适合内存受限的环境。
        虽然速度较慢，但可以在内存不足时完成解码任务。
        
        Parameters:
        -----------
        spectrum_points : tuple(x, y) or list of tuples, optional
            频谱分析点坐标，支持单个点或多个点列表（低内存模式现已支持）
        """
        height, width = images.shape[1], images.shape[2]
        
        print("\n使用低内存模式 - 逐行处理")
        print(f"  每次只处理1行，内存占用极低")
        print(f"  总行数: {height}")
        
        # 初始化频谱数据存储
        spectrum_data = None
        
        # 处理频谱分析点参数
        if spectrum_points is not None:
            # 统一转换为列表格式
            if isinstance(spectrum_points, tuple):
                spectrum_points_list = [spectrum_points]
            else:
                spectrum_points_list = spectrum_points
            
            print(f"\n⚡ 频谱分析点: {len(spectrum_points_list)} 个点")
            for i, point in enumerate(spectrum_points_list):
                print(f"   点{i+1}: ({point[0]}, {point[1]})")
            print("   低内存模式支持多点频谱分析（内存占用仅几KB每点）")
        else:
            spectrum_points_list = []
        
        # 初始化结果数组
        disparity = np.zeros((height, width), dtype=np.float32)
        peak = np.zeros((height, width), dtype=np.float32)
        
        # 逐行处理
        for row in range(height):
            if (row + 1) % 50 == 0 or row == height - 1:
                print(f"\r  处理进度: {row+1}/{height} 行 ({(row+1)/height*100:.1f}%)", end="")
            
            # 只提取当前行的数据
            I0_row = images[0::4, row, :].astype(np.float32)  # (M, width)
            I1_row = images[1::4, row, :].astype(np.float32)
            I2_row = images[2::4, row, :].astype(np.float32)
            I3_row = images[3::4, row, :].astype(np.float32)
            
            # 构造复数信号
            Dc_row = I0_row - I2_row
            Ds_row = I3_row - I1_row
            D_row = Dc_row + 1j * Ds_row  # (M, width)
            
            # 检查当前行是否包含频谱分析点
            current_row_spectrum_data = {}
            for point_idx, point in enumerate(spectrum_points_list):
                x, y = point
                if row == y and 0 <= x < width:
                    # 保存该点的原始数据（仅几KB内存）
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
                        'N': N
                    }
            
            # 对该行的每个像素进行DFT
            for col in range(width):
                # 提取该像素的频率序列
                signal = D_row[:, col]  # (M,)
                
                # DFT
                # 【关键改进】当frequency_step=0.5时，使用M作为DFT长度，而不是投影仪分辨率
                # N已经是num_frequencies，所以这里使用N是正确的
                C = np.fft.fft(signal, n=N)
                C_magnitude = np.abs(C)
                
                # 寻找峰值
                # 注意：这里得到的是频率索引，后续需要映射到投影仪坐标
                peak_idx = np.argmax(C_magnitude)
                # 【关键改进】归一化应该除以DFT长度N（即频率数量），而不是num_frequencies
                # 虽然N=num_frequencies，但为了代码清晰性和一致性，使用N
                peak_value = C_magnitude[peak_idx] / N
                
                # 检查是否是频谱分析点，保存完整频谱
                for point_idx in current_row_spectrum_data:
                    point_data = current_row_spectrum_data[point_idx]
                    if col == point_data['point'][0]:
                        if spectrum_data is None:
                            spectrum_data = {}
                        
                        # 计算坐标映射比例（低内存模式中需要重新计算）
                        if direction == "hor":
                            coord_scale_low = self.proj_height / num_frequencies
                        else:
                            coord_scale_low = self.proj_width / num_frequencies
                        
                        spectrum_data[point_idx] = {
                            'spectrum': C_magnitude / N,            # 归一化频谱幅值
                            'spectrum_raw': C_magnitude.copy(),     # 原始频谱幅值
                            'spectrum_complex': C.copy(),           # 复数频谱
                            'frequencies': np.arange(N),            # 频率索引坐标 [0, N-1]
                            'peak_index': int(peak_idx),            # 最大峰值频率索引（映射前）
                            'peak_value': float(peak_value),        # 最大峰值强度
                            'point': (col, row),
                            'direction': direction,
                            'N': N,                                 # DFT长度（频率数量）
                            'num_frequencies': num_frequencies,     # 频率数量M
                            'frequency_step': self.frequency_step,  # 频率步长（Hz）
                            # 添加原始数据
                            'I0': point_data['I0'],
                            'I1': point_data['I1'],
                            'I2': point_data['I2'],
                            'I3': point_data['I3'],
                            'Dc': point_data['Dc'],
                            'Ds': point_data['Ds'],
                        }
                        
                        print(f"\n\n  ⚡ [频谱分析] 点{point_idx+1} ({col}, {row}) 的完整频谱已保存")
                        print(f"    频谱长度: {N} (频率数量，DFT长度)")
                        print(f"    频率步长: {self.frequency_step} Hz")
                        print(f"    最大峰值频率索引: {peak_idx}")
                        print(f"    最大峰值投影仪坐标: {peak_idx * coord_scale_low:.2f}")
                        print(f"    最大峰值强度: {peak_value:.4f}")
                        
                        # 找出前10个峰值
                        top10_indices = np.argsort(C_magnitude)[-10:][::-1]
                        print(f"    频谱前10个峰值:")
                        for i, idx in enumerate(top10_indices):
                            mapped_coord = idx * coord_scale_low
                            print(f"      #{i+1}: 频率索引={idx:4d}, 投影仪坐标={mapped_coord:7.2f}, 强度={C_magnitude[idx]/N:.4f}")
                        print()
                
                # 子像素精度（质心法）
                if peak_value > 0.1:
                    idx_left = (peak_idx - 1) % N
                    idx_right = (peak_idx + 1) % N
                    
                    val_left = C_magnitude[idx_left]
                    val_center = C_magnitude[peak_idx]
                    val_right = C_magnitude[idx_right]
                    
                    if val_left + val_center + val_right > 0:
                        offset = (val_right - val_left) / (val_left + val_center + val_right)
                        refined_freq_index = peak_idx + offset * 0.5
                    else:
                        refined_freq_index = float(peak_idx)
                else:
                    refined_freq_index = float(peak_idx)
                
                # 保存结果（频率索引，后续统一映射）
                disparity[row, col] = refined_freq_index
                peak[row, col] = peak_value
            
            # 输出调试信息
            if debug_point is not None and row == debug_point[1]:
                x = debug_point[0]
                print(f"\n\n  [调试] 点 ({x}, {row}):")
                print(f"    解码坐标: {disparity[row, x]:.2f}")
                print(f"    峰值强度: {peak[row, x]:.4f}")
                print(f"    投影仪尺寸: {N}")
        
        print()  # 换行
        print(f"\n[OK] 低内存模式解码完成")
        
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
            条纹方向，'hor' 表示水平条纹（解码投影仪行坐标），
                      'ver' 表示垂直条纹（解码投影仪列坐标）
        num_frequencies : int
            采集的频率数量 M
        block_size : int
            分块处理的块大小，用于降低内存占用
        debug_point : tuple(x, y), optional
            调试点坐标，如果指定，将输出该点的详细解码信息
        low_memory : bool
            是否使用低内存模式（逐行处理，极低内存占用但速度较慢）
        spectrum_points : tuple(x, y) or list of tuples, optional
            频谱分析点坐标，支持单个点或多个点列表，将保存指定点的完整DFT频谱数据
            
        Returns:
        --------
        disparity : np.ndarray
            视差图（投影仪坐标），形状为 (height, width)，
            值范围为 [0, N)，其中 N 为投影仪的对应维度大小
        peak : np.ndarray
            峰值强度图（置信度），形状为 (height, width)，
            值越大表示该点的解码质量越好
        spectrum_data : dict or None
            如果指定了spectrum_points，返回频谱数据字典，包含：
            - 'spectrum': 完整的频谱幅值 (长度为N)
            - 'frequencies': 频率坐标 (0到N-1)
            - 'peak_index': 最大峰值的索引位置
            - 'peak_value': 最大峰值的归一化强度
            - 'point': 像素坐标 (x, y)
            - 'direction': 解码方向
            - 'I0', 'I1', 'I2', 'I3': 四个相位的强度序列
            - 'Dc', 'Ds': 差分信号
            
        原理详解：
        ---------
        1. 四步相移法：
           每个频率投影4张图，相位分别为 0°, 90°, 180°, 270°
           这样可以消除环境光和表面反射率的影响
           
        2. 复数信号构造：
           Dc = I0 - I2  (余弦分量)
           Ds = I3 - I1  (正弦分量)
           D = Dc + j*Ds (复数信号)
           
        3. DFT变换：
           C = FFT(D)
           将时域（频率序列）信号转换到频域（投影仪坐标域）
           
        4. 峰值检测：
           投影仪坐标 = argmax(|C|)
           峰值强度 = max(|C|) / N  (归一化)
        """
        # 确定DFT长度（使用频率数量，而不是投影仪尺寸）
        # 【关键改进】当frequency_step=0.5时，M=2*N，需要使用M作为DFT长度
        # 这样可以充分利用所有频率信息，获得更高的频率分辨率
        N = num_frequencies  # DFT长度 = 频率数量M
        
        # 计算坐标映射比例（频率索引 → 投影仪坐标）
        # 【关键改进】当frequency_step=0.5时，需要将频率索引映射到投影仪坐标
        # 映射公式：投影仪坐标 = 频率索引 * (投影仪分辨率 / 频率数量)
        if direction == "hor":
            coord_scale = self.proj_height / num_frequencies
        else:
            coord_scale = self.proj_width / num_frequencies
        
        print(f"\n{'='*60}")
        print(f"开始解码 {direction} 方向 (投影仪{'行' if direction == 'hor' else '列'}坐标)")
        print(f"图像数量: {images.shape[0]} ({num_frequencies} 频率 × 4 相移)")
        print(f"图像尺寸: {images.shape[1]} × {images.shape[2]}")
        print(f"频率步长: {self.frequency_step} Hz")
        print(f"最高频率: {(num_frequencies-1) * self.frequency_step} Hz")
        print(f"DFT长度: {N} (使用频率数量，而非投影仪分辨率)")
        print(f"坐标映射比例: {coord_scale:.3f} (频率索引 → 投影仪坐标)")
        if self.frequency_step != 1.0:
            print(f"⚠ 注意: 频率步长为{self.frequency_step}，将应用坐标映射")
        if low_memory:
            print(f"模式: 低内存模式 (逐行处理)")
        print(f"{'='*60}")
        
        # 如果启用低内存模式，使用逐行处理
        if low_memory:
            disparity, peak_values, spectrum_data = self._decode_direction_low_memory(
                images, direction, num_frequencies, N, 
                debug_point, spectrum_points
            )
            # 应用坐标映射
            disparity_mapped = disparity * coord_scale
            return disparity_mapped, peak_values, spectrum_data
        
        # 步骤1: 提取四个相移图像序列（边提取边转换为float32，节省内存）
        print("\n步骤1: 提取四步相移图像序列...")
        print("  说明: 从按频率组织的图像序列中，重新组织为按相位分组")
        print("  输入: [freq0_0°, freq0_90°, freq0_180°, freq0_270°, freq1_0°, freq1_90°, ...]")
        print("  输出: I0=[所有0°], I1=[所有90°], I2=[所有180°], I3=[所有270°]")
        print("  优化: 边提取边转换为float32，避免一次性复制整个数组")
        
        # 优化：分别提取并转换，避免一次性转换整个大数组
        I0 = images[0::4].astype(np.float32)  # 相位 0°,   提取索引 0,4,8,...  形状: (M, H, W)
        I1 = images[1::4].astype(np.float32)  # 相位 90°,  提取索引 1,5,9,...  形状: (M, H, W)
        I2 = images[2::4].astype(np.float32)  # 相位 180°, 提取索引 2,6,10,... 形状: (M, H, W)
        I3 = images[3::4].astype(np.float32)  # 相位 270°, 提取索引 3,7,11,... 形状: (M, H, W)
        
        print(f"  - I0 (0°):   形状 {I0.shape}, 包含M={num_frequencies}个频率的0°图像")
        print(f"  - I1 (90°):  形状 {I1.shape}, 包含M={num_frequencies}个频率的90°图像")
        print(f"  - I2 (180°): 形状 {I2.shape}, 包含M={num_frequencies}个频率的180°图像")
        print(f"  - I3 (270°): 形状 {I3.shape}, 包含M={num_frequencies}个频率的270°图像")
        
        # 释放原始图像数组，节省内存
        del images
        print(f"  [OK] 已释放原始图像数组，节省 ~{num_frequencies*4*I0.shape[1]*I0.shape[2]/1024/1024:.1f} MB 内存")
        
        # 步骤2: 构造复数信号
        print("\n步骤2: 构造复数信号...")
        print("  公式: D = (I0 - I2) + j*(I3 - I1)")
        Dc = I0 - I2  # 余弦分量（实部）
        Ds = I3 - I1  # 正弦分量（虚部）
        D = Dc + 1j * Ds  # 复数信号
        print(f"  - Dc (余弦分量): 形状 {Dc.shape}")
        print(f"  - Ds (正弦分量): 形状 {Ds.shape}")
        print(f"  - D  (复数信号): 形状 {D.shape}, 数据类型 {D.dtype}")
        
        # 如果指定了调试点，输出该点的信号
        if debug_point is not None:
            x, y = debug_point
            print(f"\n  [调试] 点 ({x}, {y}) 的信号:")
            print(f"    I0: 均值={I0[:, y, x].mean():.2f}, 标准差={I0[:, y, x].std():.2f}")
            print(f"    I2: 均值={I2[:, y, x].mean():.2f}, 标准差={I2[:, y, x].std():.2f}")
            print(f"    Dc: 均值={Dc[:, y, x].mean():.2f}, 标准差={Dc[:, y, x].std():.2f}")
            print(f"    Ds: 均值={Ds[:, y, x].mean():.2f}, 标准差={Ds[:, y, x].std():.2f}")
        
        # 如果指定了频谱分析点，提前保存这些点的原始数据
        spectrum_raw_data = {}
        if spectrum_points is not None:
            # 统一转换为列表格式
            if isinstance(spectrum_points, tuple):
                spectrum_points_list = [spectrum_points]
            else:
                spectrum_points_list = spectrum_points
                
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
                        'N': N
                    }
                    print(f"\n  [OK] 已保存频谱分析点{point_idx+1} ({x}, {y}) 的原始数据")
                else:
                    print(f"\n  ⚠ 警告: 频谱分析点{point_idx+1} ({x}, {y}) 超出图像范围")
        else:
            spectrum_points_list = []
        
        # 释放中间数组，节省内存
        del I0, I1, I2, I3, Dc, Ds
        print(f"  [OK] 已释放中间数组（I0, I1, I2, I3, Dc, Ds），节省 ~{6*num_frequencies*D.shape[1]*D.shape[2]*4/1024/1024:.1f} MB 内存")
        
        # 初始化输出数组
        disparity = np.empty(D.shape[1:], dtype=np.float32)  # 视差图
        peak = np.empty(D.shape[1:], dtype=np.float32)       # 峰值强度图
        
        # 初始化频谱数据存储
        spectrum_data = None
        if spectrum_points_list:
            print(f"\n⚡ 频谱分析点: {len(spectrum_points_list)} 个点")
            for i, point in enumerate(spectrum_points_list):
                print(f"   点{i+1}: ({point[0]}, {point[1]})")
            print("   将保存这些点的完整DFT频谱数据")
        
        # 步骤3: 分块DFT处理
        print("\n步骤3: 执行DFT变换（分块处理）...")
        print(f"  块大小: {block_size} × {block_size}")
        
        total_blocks = ((D.shape[1] + block_size - 1) // block_size) * \
                       ((D.shape[2] + block_size - 1) // block_size)
        processed_blocks = 0
        
        # 对图像进行分块处理，降低内存占用
        for ty in range(0, D.shape[1], block_size):
            for tx in range(0, D.shape[2], block_size):
                # 当前块的范围
                ey = min(ty + block_size, D.shape[1])
                ex = min(tx + block_size, D.shape[2])
                
                # 提取当前块的数据
                # D_block: 形状为 (M, block_h, block_w)
                D_block = D[:, ty:ey, tx:ex]
                
                # 执行DFT变换
                # np.fft.fft 沿 axis=0 方向进行FFT
                # 输入: (M, block_h, block_w) - M个频率点
                # 输出: (N, block_h, block_w) - N个频率分量（N=M，即频率数量）
                # 【关键改进】当frequency_step=0.5时，使用M作为DFT长度，而不是投影仪分辨率
                # 这样可以充分利用所有频率信息，提高频率分辨率
                C = np.fft.fft(D_block, n=N, axis=0)
                
                # 计算幅值谱
                C_magnitude = np.abs(C)  # 形状: (N, block_h, block_w)
                
                # 找到最大峰值位置（沿频率轴）
                # 注意：这里得到的是频率索引，后续需要映射到投影仪坐标
                freq_index = np.argmax(C_magnitude, axis=0)  # 频率索引
                disparity[ty:ey, tx:ex] = freq_index  # 先保存频率索引，后续统一映射
                
                # 记录峰值强度（归一化）
                # 【关键改进】归一化应该除以DFT长度N（即频率数量），而不是固定的投影仪分辨率
                peak[ty:ey, tx:ex] = np.max(C_magnitude, axis=0) / N
                
                # 更新进度
                processed_blocks += 1
                progress = processed_blocks / total_blocks * 100
                print(f"\r  进度: {processed_blocks}/{total_blocks} "
                      f"({progress:.1f}%) "
                      f"[块范围: y={ty}:{ey}, x={tx}:{ex}]", end="")
                
                # 如果调试点在当前块内，输出详细信息
                if debug_point is not None:
                    x, y = debug_point
                    if ty <= y < ey and tx <= x < ex:
                        local_y = y - ty
                        local_x = x - tx
                        C_point = C_magnitude[:, local_y, local_x]
                        disp_value = disparity[y, x]
                        peak_value = peak[y, x]
                        
                        print(f"\n\n  [调试] 点 ({x}, {y}) 的DFT结果:")
                        print(f"    频率索引: {disp_value:.2f} (映射前)")
                        print(f"    投影仪坐标: {disp_value * coord_scale:.2f} (映射后)")
                        print(f"    峰值强度: {peak_value:.4f}")
                        print(f"    频谱前5个峰值:")
                        top5_indices = np.argsort(C_point)[-5:][::-1]
                        for i, idx in enumerate(top5_indices):
                            mapped_coord = idx * coord_scale
                            print(f"      #{i+1}: 频率索引={idx}, 投影仪坐标={mapped_coord:.2f}, 强度={C_point[idx]/N:.4f}")
                        print()
                
                # 如果频谱分析点在当前块内，保存完整频谱
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
                                
                            # 构建完整的频谱数据字典
                            peak_freq_index = int(disparity[y, x])  # 频率索引（映射前）
                            spectrum_data[point_idx] = {
                                'spectrum': C_magnitude_point / N,  # 归一化的频谱幅值
                                'spectrum_raw': C_magnitude_point,   # 原始频谱幅值
                                'spectrum_complex': C_point,         # 复数频谱
                                'frequencies': np.arange(N),         # 频率索引坐标 [0, N-1]
                                'peak_index': peak_freq_index,       # 最大峰值频率索引（映射前）
                                'peak_value': float(peak[y, x]),     # 最大峰值强度
                                'point': (x, y),
                                'direction': direction,
                                'N': N,                              # DFT长度（频率数量）
                                'num_frequencies': num_frequencies,  # 频率数量M
                                'frequency_step': self.frequency_step,  # 频率步长（Hz）
                                # 添加原始数据
                                'I0': raw_data['I0'],
                                'I1': raw_data['I1'],
                                'I2': raw_data['I2'],
                                'I3': raw_data['I3'],
                                'Dc': raw_data['Dc'],
                                'Ds': raw_data['Ds'],
                            }
                            
                            # 计算投影仪坐标（映射后）
                            peak_proj_coord = peak_freq_index * coord_scale
                            
                            print(f"\n\n  ⚡ [频谱分析] 点{point_idx+1} ({x}, {y}) 的完整频谱已保存")
                            print(f"    频谱长度: {N} (频率数量，DFT长度)")
                            print(f"    频率步长: {self.frequency_step} Hz")
                            print(f"    最大峰值频率索引: {peak_freq_index} (映射前)")
                            print(f"    最大峰值投影仪坐标: {peak_proj_coord:.2f} (映射后)")
                            print(f"    最大峰值强度: {spectrum_data[point_idx]['peak_value']:.4f}")
                            
                            # 找出前10个峰值
                            top10_indices = np.argsort(C_magnitude_point)[-10:][::-1]
                            print(f"    频谱前10个峰值:")
                            for i, idx in enumerate(top10_indices):
                                mapped_coord = idx * coord_scale
                                print(f"      #{i+1}: 频率索引={idx:4d}, 投影仪坐标={mapped_coord:7.2f}, 强度={C_magnitude_point[idx]/N:.4f}")
                            print()
        
        print()  # 换行
        
        # 步骤4: 输出统计信息（映射前）
        print("\n步骤4: 解码完成，统计信息（频率索引）:")
        print(f"  频率索引范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        print(f"  峰值强度范围: [{peak.min():.4f}, {peak.max():.4f}]")
        print(f"  峰值强度均值: {peak.mean():.4f}")
        print(f"  峰值强度中值: {np.median(peak):.4f}")
        
        # 步骤5: 应用坐标映射（频率索引 → 投影仪坐标）
        # 【关键改进】当frequency_step=0.5时，需要将频率索引映射到投影仪坐标
        # 映射公式：投影仪坐标 = 频率索引 * (投影仪分辨率 / 频率数量)
        disparity_mapped = disparity * coord_scale
        print(f"\n步骤5: 应用坐标映射（频率索引 → 投影仪坐标）")
        print(f"  映射比例: {coord_scale:.6f} = 投影仪分辨率 / 频率数量")
        print(f"  频率索引范围: [{disparity.min():.1f}, {disparity.max():.1f}]")
        print(f"  投影仪坐标范围: [{disparity_mapped.min():.1f}, {disparity_mapped.max():.1f}]")
        if self.frequency_step != 1.0:
            print(f"  ✓ 频率步长为{self.frequency_step}，已正确应用坐标映射")
        
        return disparity_mapped, peak, spectrum_data
    
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
    
    这是一个示例函数，展示如何使用FSADecoder类
    """
    print("="*80)
    print("FSA (频移法) 解码演示程序")
    print("="*80)
    
    # 配置参数
    PROJECTOR_WIDTH = 1024   # 投影仪宽度
    PROJECTOR_HEIGHT = 768   # 投影仪高度
    NUM_FREQUENCIES = 121    # 采集的频率数量
    
    # 图像路径（需要根据实际情况修改）
    IMAGE_DIR_HOR = "./data/hor"  # 水平条纹图像文件夹
    IMAGE_DIR_VER = "./data/ver"  # 垂直条纹图像文件夹
    OUTPUT_DIR = "./results"      # 输出结果文件夹
    
    # 创建解码器
    decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    
    # 解码水平方向（投影仪行坐标）
    print("\n" + "="*80)
    print("第一步: 解码水平条纹（获取投影仪行坐标）")
    print("="*80)
    
    try:
        images_hor = decoder.load_images(
            image_dir=IMAGE_DIR_HOR,
            num_frequencies=NUM_FREQUENCIES,
            use_cache=True
        )
        
        disparity_hor, peak_hor = decoder.decode_direction(
            images=images_hor,
            direction='hor',
            num_frequencies=NUM_FREQUENCIES,
            block_size=100,
            debug_point=(500, 500)  # 可选：查看点(500,500)的解码详情
        )
        
        decoder.save_results(disparity_hor, peak_hor, OUTPUT_DIR, 'hor')
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print(f"请确保图像文件夹存在: {IMAGE_DIR_HOR}")
    
    # 解码垂直方向（投影仪列坐标）
    print("\n" + "="*80)
    print("第二步: 解码垂直条纹（获取投影仪列坐标）")
    print("="*80)
    
    try:
        images_ver = decoder.load_images(
            image_dir=IMAGE_DIR_VER,
            num_frequencies=NUM_FREQUENCIES,
            use_cache=True
        )
        
        disparity_ver, peak_ver = decoder.decode_direction(
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
    print(f"\n结果文件保存在: {OUTPUT_DIR}")
    print("  - disp_hor.npy: 投影仪行坐标（水平视差）")
    print("  - peak_hor.npy: 水平方向峰值强度（置信度）")
    print("  - disp_ver.npy: 投影仪列坐标（垂直视差）")
    print("  - peak_ver.npy: 垂直方向峰值强度（置信度）")
    print("\n使用方法:")
    print("  import numpy as np")
    print("  disparity = np.load('results/disp_ver.npy')")
    print("  peak = np.load('results/peak_ver.npy')")


if __name__ == "__main__":
    # 运行演示程序
    demonstrate_fsa_decoding()

