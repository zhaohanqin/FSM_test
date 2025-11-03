"""
FSA (Frequency Shift Analysis) 频移法解码核心模块

本模块实现了频移法的核心DFT解码算法，用于从采集的条纹图像中
解码出相机像素与投影仪像素之间的对应关系。

主要功能：
1. 读取四步相移条纹图像序列
2. 使用DFT（离散傅里叶变换）解码投影仪坐标
3. 输出视差图和置信度图

作者：从decode_fsa.py提取
日期：2025-11-01
"""

import numpy as np
import os
from typing import Tuple, Optional


class FSADecoder:
    """
    频移法解码器
    
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
    """
    
    def __init__(self, projector_width: int, projector_height: int):
        """
        初始化FSA解码器
        
        Parameters:
        -----------
        projector_width : int
            投影仪的宽度（像素数）
        projector_height : int
            投影仪的高度（像素数）
        """
        self.proj_width = projector_width
        self.proj_height = projector_height
        
    def verify_image_sequence(self, images: np.ndarray, num_frequencies: int, sample_points: int = 5):
        """
        验证图像序列的组织方式是否正确
        
        此函数检查相移图像之间的差异，帮助判断图像序列是否按照正确的顺序组织。
        对于正确的四步相移序列，相邻相位之间应该有明显的差异。
        
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
        
        print(f"✓ 图像数量正确: {images.shape[0]} 张")
        
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
                print(" ✓")
        
        if not has_variation:
            print("\n⚠ 警告: 某些频率的相位图像差异很小，请检查图像顺序！")
            return False
        
        print("\n✓ 图像序列验证通过！")
        print("="*60)
        return True
    
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
            self.verify_image_sequence(images, num_frequencies)
        
        return images
    
    def _decode_direction_low_memory(self,
                                     images: np.ndarray,
                                     direction: str,
                                     num_frequencies: int,
                                     N: int,
                                     debug_point: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        低内存模式解码（逐行处理）
        
        此函数通过逐行处理图像来最小化内存占用，适合内存受限的环境。
        虽然速度较慢，但可以在内存不足时完成解码任务。
        """
        height, width = images.shape[1], images.shape[2]
        
        print("\n使用低内存模式 - 逐行处理")
        print(f"  每次只处理1行，内存占用极低")
        print(f"  总行数: {height}")
        
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
            
            # 对该行的每个像素进行DFT
            for col in range(width):
                # 提取该像素的频率序列
                signal = D_row[:, col]  # (M,)
                
                # DFT
                C = np.fft.fft(signal, n=N)
                C_magnitude = np.abs(C)
                
                # 寻找峰值
                peak_idx = np.argmax(C_magnitude)
                peak_value = C_magnitude[peak_idx] / num_frequencies
                
                # 子像素精度（质心法）
                if peak_value > 0.1:
                    idx_left = (peak_idx - 1) % N
                    idx_right = (peak_idx + 1) % N
                    
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
                
                # 保存结果
                disparity[row, col] = peak_coord
                peak[row, col] = peak_value
            
            # 输出调试信息
            if debug_point is not None and row == debug_point[1]:
                x = debug_point[0]
                print(f"\n\n  [调试] 点 ({x}, {row}):")
                print(f"    解码坐标: {disparity[row, x]:.2f}")
                print(f"    峰值强度: {peak[row, x]:.4f}")
        
        print()  # 换行
        print(f"\n✓ 低内存模式解码完成")
        
        return disparity, peak
    
    def decode_direction(self,
                        images: np.ndarray,
                        direction: str,
                        num_frequencies: int,
                        block_size: int = 100,
                        debug_point: Optional[Tuple[int, int]] = None,
                        low_memory: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
            
        Returns:
        --------
        disparity : np.ndarray
            视差图（投影仪坐标），形状为 (height, width)，
            值范围为 [0, N)，其中 N 为投影仪的对应维度大小
        peak : np.ndarray
            峰值强度图（置信度），形状为 (height, width)，
            值越大表示该点的解码质量越好
            
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
        # 确定DFT长度（投影仪尺寸）
        N = self.proj_height if direction == "hor" else self.proj_width
        
        print(f"\n{'='*60}")
        print(f"开始解码 {direction} 方向 (投影仪{'行' if direction == 'hor' else '列'}坐标)")
        print(f"图像数量: {images.shape[0]} ({num_frequencies} 频率 × 4 相移)")
        print(f"图像尺寸: {images.shape[1]} × {images.shape[2]}")
        print(f"DFT长度: {N} (投影仪{'高度' if direction == 'hor' else '宽度'})")
        if low_memory:
            print(f"模式: 低内存模式 (逐行处理)")
        print(f"{'='*60}")
        
        # 如果启用低内存模式，使用逐行处理
        if low_memory:
            return self._decode_direction_low_memory(images, direction, num_frequencies, N, debug_point)
        
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
        print(f"  ✓ 已释放原始图像数组，节省 ~{num_frequencies*4*I0.shape[1]*I0.shape[2]/1024/1024:.1f} MB 内存")
        
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
        
        # 释放中间数组，节省内存
        del I0, I1, I2, I3, Dc, Ds
        print(f"  ✓ 已释放中间数组（I0, I1, I2, I3, Dc, Ds），节省 ~{6*num_frequencies*D.shape[1]*D.shape[2]*4/1024/1024:.1f} MB 内存")
        
        # 初始化输出数组
        disparity = np.empty(D.shape[1:], dtype=np.float32)  # 视差图
        peak = np.empty(D.shape[1:], dtype=np.float32)       # 峰值强度图
        
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
                # 输出: (N, block_h, block_w) - N个频率分量
                C = np.fft.fft(D_block, n=N, axis=0)
                
                # 计算幅值谱
                C_magnitude = np.abs(C)  # 形状: (N, block_h, block_w)
                
                # 找到最大峰值位置（沿频率轴）
                disparity[ty:ey, tx:ex] = np.argmax(C_magnitude, axis=0)
                
                # 记录峰值强度（归一化）
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
                        print(f"    投影仪坐标: {disp_value:.2f}")
                        print(f"    峰值强度: {peak_value:.4f}")
                        print(f"    频谱前5个峰值:")
                        top5_indices = np.argsort(C_point)[-5:][::-1]
                        for i, idx in enumerate(top5_indices):
                            print(f"      #{i+1}: 位置={idx}, 强度={C_point[idx]/N:.4f}")
                        print()
        
        print()  # 换行
        
        # 步骤4: 输出统计信息
        print("\n步骤4: 解码完成，统计信息:")
        print(f"  视差范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        print(f"  峰值强度范围: [{peak.min():.4f}, {peak.max():.4f}]")
        print(f"  峰值强度均值: {peak.mean():.4f}")
        print(f"  峰值强度中值: {np.median(peak):.4f}")
        
        return disparity, peak
    
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

