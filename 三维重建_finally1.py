"""
结构光三维重建系统（模块化版本 - 基于copy2原始实现）

功能特性：
✅ 三频外差相位解包裹（带去噪）
✅ 相位质量评估（调制度、平均强度）
✅ 基于标定参数的三维重建
✅ 基础点云过滤（3步流程）
✅ 点云颜色提取和着色（从四步相移图像生成无条纹纹理）
✅ 点云可视化和保存（支持彩色点云）

使用方法：
1. 准备标定文件：标定结果.txt
2. 准备条纹图像文件夹，包含24张图像：
   - h1.bmp ~ h12.bmp (水平条纹，3个频率×4步相移)
   - v1.bmp ~ v12.bmp (垂直条纹，3个频率×4步相移)
   - 注意：图像需要是彩色图像（用于颜色提取）
3. 运行此脚本

代码结构：
- CalibrationReader: 标定参数读取类
- ImageLoader: 图像读取类（支持彩色图像加载和纹理图像生成）
- ColorExtractor: 颜色提取类（从纹理图像提取颜色信息）
- PhaseCalculator: 相位计算类（包裹相位+质量评估）
- PhaseUnwrapper: 相位解包裹类（三频外差+去噪）
- PointCloudReconstructor: 点云重建类（支持颜色赋值）
- PointCloudFilter: 点云过滤类（基础3步过滤，保留颜色）
- ResultVisualizer: 结果可视化和保存类
- ReconstructionPipeline: 主流程协调器
"""

import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import re
import matplotlib
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文显示
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
except:
    try:
        font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        try:
            font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10)
            matplotlib.rcParams['font.sans-serif'] = ['SimSun']
        except:
            print("警告: 找不到中文字体，标题可能无法正确显示")
            font = None

matplotlib.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 标定参数读取类
# =============================================================================
class CalibrationReader:
    """标定参数读取类：从文本文件读取相机和投影仪的标定参数"""
    
    def __init__(self, filepath='./标定结果.txt'):
        """
        初始化并读取标定文件
        
        Args:
            filepath: 标定结果文件路径
        """
        print("="*70)
        print("【步骤0】读取标定参数")
        print("="*70)
        
        # 相机参数
        self.fcx, self.fcy = 0, 0
        self.cpointx, self.cpointy = 0, 0
        self.ck, self.cp = None, None
        
        # 投影仪参数
        self.fpx, self.fpy = 0, 0
        self.ppointx, self.ppointy = 0, 0
        self.pk, self.pp = None, None
        
        # 外参
        self.R, self.t = None, None
        self.F = None
        
        self._parse_file(filepath)
        print("  ✓ 标定参数读取成功！\n")

    def _parse_file(self, filepath):
        """解析标定文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"错误：找不到标定文件 {filepath}")

        lines_iter = iter(lines)
        for line in lines_iter:
            if "相机标定结果" in line:
                values = [float(next(lines_iter).strip()) for _ in range(8)]
                self.fcx, self.fcy = values[0], values[1]
                self.cpointx, self.cpointy = values[2], values[3]
                self.ck = np.array([values[4], values[5], 0.0])
                self.cp = np.array([values[6], values[7]])

            elif "dlp标定结果" in line:
                values = [float(next(lines_iter).strip()) for _ in range(8)]
                self.fpx, self.fpy = values[0], values[1]
                self.ppointx, self.ppointy = values[2], values[3]
                self.pk = np.array([values[4], values[5], 0.0])
                self.pp = np.array([values[6], values[7]])

            elif "旋转矩阵" in line:
                matrix_lines = [next(lines_iter).strip() for _ in range(3)]
                self.R = np.array([list(map(float, re.split(r'\s+', l.strip()))) for l in matrix_lines])

            elif "平移矩阵" in line:
                values = next(lines_iter).strip()
                self.t = np.array(list(map(float, re.split(r'\s+', values))))

            elif "基础矩阵" in line:
                matrix_lines = [next(lines_iter).strip() for _ in range(3)]
                self.F = np.array([list(map(float, re.split(r'\s+', l.strip()))) for l in matrix_lines])
                self.F = self.F / self.F[-1, -1]


# =============================================================================
# 2. 图像读取类
# =============================================================================
class ImageLoader:
    """图像读取类：从文件夹加载条纹图像"""
    
    @staticmethod
    def load_stripe_images(folder_path, image_prefix, image_ext='.bmp', num_images=12, 
                          load_color=False):
        """
        加载条纹图像
        
        Args:
            folder_path: 图像文件夹路径
            image_prefix: 图像前缀 ('h' 或 'v')
            image_ext: 图像扩展名
            num_images: 图像数量
            load_color: 是否加载彩色图像（用于颜色提取）
            
        Returns:
            images: 图像数组 (H, W, num_images) 或 (H, W, 3, num_images) 如果load_color=True
        """
        print(f"  正在读取 {image_prefix}1-{image_prefix}{num_images} 图像...")
        
        # 读取第一张图像以获取尺寸
        filename = os.path.join(folder_path, f"{image_prefix}1{image_ext}")
        if load_color:
            first_img = cv2.imread(filename, cv2.IMREAD_COLOR)
            if first_img is None:
                raise FileNotFoundError(f"无法读取图像: {filename}")
            h, w, c = first_img.shape
            images = np.zeros((h, w, c, num_images), dtype=np.uint8)
            
            # 读取所有图像（彩色）
            for i in range(num_images):
                filename = os.path.join(folder_path, f"{image_prefix}{i+1}{image_ext}")
                img = cv2.imread(filename, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"无法读取图像: {filename}")
                images[:, :, :, i] = img
            print(f"    ✓ 成功读取 {num_images} 张彩色图像，尺寸: {w} × {h} × {c}")
        else:
            first_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if first_img is None:
                raise FileNotFoundError(f"无法读取图像: {filename}")
            h, w = first_img.shape
            images = np.zeros((h, w, num_images), dtype=np.uint8)
            
            # 读取所有图像（灰度）
            for i in range(num_images):
                filename = os.path.join(folder_path, f"{image_prefix}{i+1}{image_ext}")
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"无法读取图像: {filename}")
                images[:, :, i] = img
            print(f"    ✓ 成功读取 {num_images} 张图像，尺寸: {w} × {h}")
        return images
    
    @staticmethod
    def generate_texture_image(folder_path, image_prefix='h', image_ext='.bmp', 
                              num_phase_images=4):
        """
        从四步相移图像生成无条纹纹理图像（用于颜色提取）
        
        Args:
            folder_path: 图像文件夹路径
            image_prefix: 图像前缀 ('h' 或 'v')，通常使用 'h'
            image_ext: 图像扩展名
            num_phase_images: 相移图像数量（通常为4）
            
        Returns:
            texture_image: 无条纹纹理图像 (H, W, 3)，BGR格式
        """
        print(f"  正在生成无条纹纹理图像（从{image_prefix}1-{image_prefix}{num_phase_images}）...")
        
        # 读取彩色图像
        texture_images = ImageLoader.load_stripe_images(
            folder_path, image_prefix, image_ext, num_phase_images, load_color=True
        )
        
        # 计算平均值（对4张相移图像取平均，消除条纹）
        # texture_images shape: (H, W, 3, num_phase_images)
        texture_image = np.mean(texture_images.astype(np.float32), axis=3).astype(np.uint8)
        
        print(f"    ✓ 无条纹纹理图像生成成功，尺寸: {texture_image.shape[1]} × {texture_image.shape[0]}")
        return texture_image


# =============================================================================
# 2.5. 颜色提取类
# =============================================================================
class ColorExtractor:
    """颜色提取类：从纹理图像中提取颜色信息"""
    
    def __init__(self, texture_image):
        """
        初始化颜色提取器
        
        Args:
            texture_image: 无条纹纹理图像 (H, W, 3)，BGR格式
        """
        self.texture_image = texture_image
        self.height, self.width = texture_image.shape[:2]
        print(f"  ✓ 颜色提取器初始化成功，纹理图像尺寸: {self.width} × {self.height}")
    
    def get_color(self, row, col):
        """
        从纹理图像中获取指定像素位置的颜色
        
        Args:
            row: 行索引（像素坐标）
            col: 列索引（像素坐标）
            
        Returns:
            color: RGB颜色值 [R, G, B]，范围 [0, 1]
        """
        # 边界检查
        row = int(np.clip(row, 0, self.height - 1))
        col = int(np.clip(col, 0, self.width - 1))
        
        # OpenCV使用BGR格式，需要转换为RGB
        bgr_color = self.texture_image[row, col]
        rgb_color = np.array([bgr_color[2], bgr_color[1], bgr_color[0]], dtype=np.float64)
        
        # 归一化到 [0, 1]
        rgb_color = rgb_color / 255.0
        
        return rgb_color
    
    def get_colors_batch(self, pixel_coords):
        """
        批量获取颜色（用于点云着色）
        
        Args:
            pixel_coords: 像素坐标数组 (N, 2)，每行为 [row, col]
            
        Returns:
            colors: RGB颜色数组 (N, 3)，范围 [0, 1]
        """
        colors = np.zeros((len(pixel_coords), 3), dtype=np.float64)
        
        for i, (row, col) in enumerate(pixel_coords):
            colors[i] = self.get_color(row, col)
        
        return colors


# =============================================================================
# 3. 相位计算类
# =============================================================================
class PhaseCalculator:
    """相位计算类：计算包裹相位和质量指标"""
    
    @staticmethod
    def calculate_wrapped_phase(phase_images):
        """
        计算包裹相位（四步相移法）并计算质量指标
        
        Args:
            phase_images: 形状为 (H, W, 4) 的四步相移图像
            
        Returns:
            wrapped_phase: 包裹相位图，范围 [0, 2π]
            modulation: 调制度图
            mean_intensity: 平均强度图
        """
        fenzi = phase_images[:, :, 3] - phase_images[:, :, 1]
        fenmu = phase_images[:, :, 0] - phase_images[:, :, 2]
        
        wrapped_phase = np.arctan2(fenzi, fenmu)
        wrapped_phase[wrapped_phase < 0] += 2 * np.pi
        
        # 计算质量指标
        Imax = np.max(phase_images, axis=2)
        Imin = np.min(phase_images, axis=2)
        Imean = np.mean(phase_images, axis=2)
        
        modulation = (Imax - Imin) / (Imax + Imin + 1e-6)
        mean_intensity = Imean
        
        return wrapped_phase, modulation, mean_intensity


# =============================================================================
# 4. 相位解包裹类
# =============================================================================
class PhaseUnwrapper:
    """相位解包裹类：三频外差法相位解包裹（带去噪）"""
    
    def __init__(self, frequencies=np.array([81, 72, 64])):
        """
        初始化
        
        Args:
            frequencies: 三个频率数组 [freq1, freq2, freq3]
        """
        self.frequencies = frequencies
        self.phase_calc = PhaseCalculator()
    
    def process_images(self, stripe_images):
        """
        处理条纹图像：计算包裹相位、质量指标和绝对相位
        
        Args:
            stripe_images: 条纹图像 (H, W, 12)
            
        Returns:
            absolute_phase: 绝对相位 (H, W)
            modulation: 调制度 (H, W)
            mean_intensity: 平均强度 (H, W)
        """
        # 1. 计算包裹相位
        wrapped_phases = np.zeros((stripe_images.shape[0], stripe_images.shape[1], 3))
        modulations = np.zeros((stripe_images.shape[0], stripe_images.shape[1], 3))
        mean_intensities = np.zeros((stripe_images.shape[0], stripe_images.shape[1], 3))
        
        for i in range(3):
            phase_imgs = stripe_images[:, :, i*4:(i+1)*4].astype(np.float64)
            wrapped_phases[:, :, i], modulations[:, :, i], mean_intensities[:, :, i] = \
                self.phase_calc.calculate_wrapped_phase(phase_imgs)
        
        # 2. 解包裹相位（三频外差）
        absolute_phase = self._unwrap_phase_heterodyne(wrapped_phases)
        
        # 3. 质量指标（取第一个频率的）
        modulation = modulations[:, :, 0]
        mean_intensity = mean_intensities[:, :, 0]
        
        return absolute_phase, modulation, mean_intensity
    
    def _unwrap_phase_heterodyne(self, wrapped_phases):
        """
        三频外差法解包裹相位（带去噪）
        
        Args:
            wrapped_phases: 包裹相位 (H, W, 3)
            
        Returns:
            absolute_phase: 绝对相位 (H, W)，范围 [0, 2π]
        """
        f = self.frequencies
        f12 = f[0] - f[1]
        f23 = f[1] - f[2]
        
        # 第一次外差
        phase_abs12, freq12 = self._heterodyne_phase(
            wrapped_phases[:, :, 0], f[0],
            wrapped_phases[:, :, 1], f[1]
        )
        
        phase_abs23, freq23 = self._heterodyne_phase(
            wrapped_phases[:, :, 1], f[1],
            wrapped_phases[:, :, 2], f[2]
        )
        
        # 第二次外差
        phase_abs123, freq_final = self._heterodyne_phase(
            phase_abs12, freq12, phase_abs23, freq23
        )
        
        # 第一次解包裹
        k1 = np.round((phase_abs123 * freq12 - phase_abs12) / (2 * np.pi))
        abs_phase12 = (k1 * 2 * np.pi + phase_abs12) / freq12
        
        k2 = np.round((phase_abs123 * freq23 - phase_abs23) / (2 * np.pi))
        abs_phase23 = (k2 * 2 * np.pi + phase_abs23) / freq23
        
        # 去噪
        abs_phase12 = self._denoise_unwrapped_phase(abs_phase12, phase_abs123, freq_final, freq12)
        abs_phase23 = self._denoise_unwrapped_phase(abs_phase23, phase_abs123, freq_final, freq23)
        
        # 第二次解包裹
        k3 = np.round((abs_phase12 * f[1] - wrapped_phases[:, :, 1]) / (2 * np.pi))
        abs_phase2_12 = (k3 * 2 * np.pi + wrapped_phases[:, :, 1]) / f[1]
        
        k4 = np.round((abs_phase23 * f[1] - wrapped_phases[:, :, 1]) / (2 * np.pi))
        abs_phase2_23 = (k4 * 2 * np.pi + wrapped_phases[:, :, 1]) / f[1]
        
        # 去噪
        abs_phase2_12 = self._denoise_unwrapped_phase(abs_phase2_12, abs_phase12, freq12, f[1])
        abs_phase2_23 = self._denoise_unwrapped_phase(abs_phase2_23, abs_phase23, freq23, f[1])
        
        # 最终平均
        absolute_phase = (abs_phase2_12 + abs_phase2_23) / 2
        
        # 归一化到 [0, 2π]
        if absolute_phase.max() > 2 * np.pi:
            absolute_phase = absolute_phase % (2 * np.pi)
        
        return absolute_phase

    @staticmethod
    def _heterodyne_phase(phase_i, freq_i, phase_j, freq_j):
        """双频外差算法"""
        freq_ij = freq_i - freq_j
        k = freq_i / freq_ij
        phase_ij = phase_i - phase_j
        phase_ij[phase_ij < 0] += 2 * np.pi
        period = np.round((k * phase_ij - phase_i) / (2 * np.pi))
        phase_abs = (period * 2 * np.pi + phase_i) / k
        return phase_abs, freq_ij
    
    @staticmethod
    def _denoise_unwrapped_phase(unwarp_phase, reference_phase, reference_freq, phase_freq):
        """
        对解包裹后的相位进行去噪处理
        
        Args:
            unwarp_phase: 解包裹后的相位
            reference_phase: 参考相位
            reference_freq: 参考相位频率
            phase_freq: 当前相位频率
            
        Returns:
            去噪后的相位
        """
        temp = phase_freq / reference_freq * reference_phase
        
        gauss_size = (3, 3)
        unwarp_phase_noise = unwarp_phase - cv2.GaussianBlur(unwarp_phase, gauss_size, 0)
        unwarp_reference_noise = temp - cv2.GaussianBlur(temp, gauss_size, 0)
        
        noise_ratio = np.abs(unwarp_phase_noise) / (np.abs(unwarp_reference_noise) + 0.001)
        order_flag = (np.abs(unwarp_phase_noise) - np.abs(unwarp_reference_noise) > 0.15) & (noise_ratio > 1.5)
        
        if np.sum(order_flag) > 0:
            unwarp_error = unwarp_phase[order_flag].copy()
            unwarp_error_direct = unwarp_phase_noise[order_flag]
            
            unwarp_error[unwarp_error_direct > 0] -= 2 * np.pi
            unwarp_error[unwarp_error_direct < 0] += 2 * np.pi
            
            unwarp_phase = unwarp_phase.copy()
            unwarp_phase[order_flag] = unwarp_error
            
            # 第二次去噪
            unwarp_phase_noise = unwarp_phase - cv2.GaussianBlur(unwarp_phase, gauss_size, 0)
            order_flag2 = np.abs(unwarp_phase_noise) > 0.2
            
            if np.sum(order_flag2) > 0:
                unwarp_error2 = unwarp_phase[order_flag2].copy()
                unwarp_error_direct2 = unwarp_phase_noise[order_flag2]
                
                unwarp_error2[unwarp_error_direct2 > 0] -= 2 * np.pi
                unwarp_error2[unwarp_error_direct2 < 0] += 2 * np.pi
                
                unwarp_phase[order_flag2] = unwarp_error2
        
        return unwarp_phase


# =============================================================================
# 5. 点云重建类
# =============================================================================
class PointCloudReconstructor:
    """点云重建类：基于相位和标定参数计算3D坐标"""
    
    def __init__(self, calibration: CalibrationReader, oversample=17):
        """
        初始化
        
        Args:
            calibration: 标定参数对象
            oversample: 过采样倍数（与FSM解码时的过采样倍数必须一致）
        """
        self.calib = calibration
        self.oversample = oversample
        
        # 质量过滤参数
        self.modulation_threshold = 0.15
        self.intensity_threshold = 30
    
    def _build_camera_matrix(self, fx, fy, cx, cy):
        """构建相机内参矩阵"""
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def _build_distortion_coeffs(self, k1, k2, p1, p2):
        """构建畸变系数向量"""
        return np.array([k1, k2, p1, p2, 0])
    
    def _triangulate_points_batch(self, camera_pixels, projector_pixels):
        """
        批量三角测量方法（使用OpenCV的cv2.triangulatePoints）
        
        Args:
            camera_pixels: 相机像素坐标，形状 (N, 2)
            projector_pixels: 投影仪像素坐标，形状 (N, 2)
        
        Returns:
            points_3d: 3D点坐标，形状 (N, 3)，在相机坐标系中
        """
        # 构建相机内参矩阵
        camera_matrix = self._build_camera_matrix(
            self.calib.fcx, self.calib.fcy,
            self.calib.cpointx, self.calib.cpointy
        )
        
        # 构建投影仪内参矩阵
        projector_matrix = self._build_camera_matrix(
            self.calib.fpx, self.calib.fpy,
            self.calib.ppointx, self.calib.ppointy
        )
        
        # 构建畸变系数
        camera_dist = self._build_distortion_coeffs(
            self.calib.ck[0], self.calib.ck[1],
            self.calib.cp[0], self.calib.cp[1]
        )
        
        projector_dist = self._build_distortion_coeffs(
            self.calib.pk[0], self.calib.pk[1],
            self.calib.pp[0], self.calib.pp[1]
        )
        
        # 去畸变
        camera_undistorted = cv2.undistortPoints(
            camera_pixels.astype(np.float32),
            camera_matrix,
            camera_dist,
            P=camera_matrix
        ).reshape(-1, 2)
        
        projector_undistorted = cv2.undistortPoints(
            projector_pixels.astype(np.float32),
            projector_matrix,
            projector_dist,
            P=projector_matrix
        ).reshape(-1, 2)
        
        # 构建投影矩阵
        # 相机投影矩阵（相机坐标系）
        P_camera = np.hstack([np.eye(3), np.zeros((3, 1))])
        P_camera = camera_matrix @ P_camera
        
        # 投影仪投影矩阵（投影仪坐标系，需要转换到相机坐标系）
        R = self.calib.R.astype(np.float64)
        t = self.calib.t.astype(np.float64).reshape(3, 1)
        P_projector = np.hstack([R, t])
        P_projector = projector_matrix @ P_projector
        
        # 三角测量
        points_4d = cv2.triangulatePoints(
            P_camera, P_projector,
            camera_undistorted.T, projector_undistorted.T
        )
        
        # 转换为3D坐标（齐次坐标转笛卡尔坐标）
        points_3d = (points_4d[:3] / points_4d[3]).T  # 形状: (N, 3)
        
        return points_3d
    
    def reconstruct(self, phase_h, phase_v, modulation, mean_intensity, z_min=0, z_max=2300,
                   color_extractor=None):
        """
        三维重建：从绝对相位计算点云
        
        Args:
            phase_h: 水平方向绝对相位 (H, W)
            phase_v: 垂直方向绝对相位 (H, W)
            modulation: 调制度 (H, W)
            mean_intensity: 平均强度 (H, W)
            z_min: 深度最小值
            z_max: 深度最大值
            color_extractor: 颜色提取器对象（可选，用于点云着色）
            
        Returns:
            raw_points: 原始点云 (N, 3)
            raw_colors: 点云颜色 (N, 3)，RGB格式，范围 [0, 1]（如果color_extractor提供）
            pixel_coords: 像素坐标 (N, 2)，每行为 [row, col]（如果color_extractor提供）
            valid_mask: 有效点掩码 (H, W)
        
        重要说明：
            - 过采样倍数 (self.oversample) 必须与FSM解码时的过采样倍数一致
            - 相位范围 [0, 2π] 对应投影仪分辨率 [0, 1920] 和 [0, 1080]
            - 投影仪像素坐标的范围应该与相机像素坐标范围相当
        """
        print("="*70)
        print("【步骤3】三维重建")
        print("="*70)
        
        row, col = phase_h.shape
        print(f"  图像尺寸: {col} × {row}")
        print(f"  过采样倍数: {self.oversample}")
        print(f"  质量过滤阈值: 调制度>{self.modulation_threshold}, 强度>{self.intensity_threshold}")
        print(f"  深度范围: {z_min}~{z_max} mm")
        if color_extractor is not None:
            print(f"  颜色提取: 启用")
        else:
            print(f"  颜色提取: 未启用")
        
        # 第一步：生成有效点的像素坐标和投影仪坐标
        proj_width = 1920
        proj_height = 1080
        
        cam_pixels_list = []
        proj_pixels_list = []
        pixel_coords_list = []  # 用于颜色提取
        valid_mask = np.zeros((row, col), dtype=bool)
        
        print("  - 第一步：生成像素坐标...")
        for i in range(row):
            for j in range(col):
                phi_h = phase_h[i, j]
                phi_v = phase_v[i, j]
                
                # 质量过滤（必须同时检查水平和垂直方向相位）
                if (phi_h <= 0 or np.isnan(phi_h) or 
                    phi_v <= 0 or np.isnan(phi_v) or
                    modulation[i, j] < self.modulation_threshold or
                    mean_intensity[i, j] < self.intensity_threshold):
                    continue
                
                # 相位到投影坐标
                up = phi_h / (2 * np.pi) * proj_width
                vp = phi_v / (2 * np.pi) * proj_height
                
                # 保存像素坐标
                cam_pixels_list.append([float(j), float(i)])  # (x, y) = (列, 行)
                proj_pixels_list.append([up, vp])
                pixel_coords_list.append([i, j])  # 用于颜色提取
                valid_mask[i, j] = True
        
        # 转换为numpy数组
        if len(cam_pixels_list) == 0:
            print("  - 错误：没有有效的点可以重建")
            return np.array([]), None, None, valid_mask
        
        camera_pixels = np.array(cam_pixels_list, dtype=np.float32)
        projector_pixels = np.array(proj_pixels_list, dtype=np.float32)
        
        print(f"  - 有效点数: {len(camera_pixels)}")
        print(f"  - 相机像素范围: X=[{camera_pixels[:, 0].min():.1f}, {camera_pixels[:, 0].max():.1f}], Y=[{camera_pixels[:, 1].min():.1f}, {camera_pixels[:, 1].max():.1f}]")
        print(f"  - 投影仪像素范围: X=[{projector_pixels[:, 0].min():.1f}, {projector_pixels[:, 0].max():.1f}], Y=[{projector_pixels[:, 1].min():.1f}, {projector_pixels[:, 1].max():.1f}]")
        
        # 第二步：批量三角测量
        print("  - 第二步：执行批量三角测量...")
        points_3d = self._triangulate_points_batch(camera_pixels, projector_pixels)
        
        # 第三步：深度范围过滤
        print("  - 第三步：深度范围过滤...")
        valid_depth_mask = (points_3d[:, 2] > z_min) & (points_3d[:, 2] < z_max)
        data = points_3d[valid_depth_mask]
        
        # 更新有效掩码和像素坐标
        valid_indices = np.where(valid_depth_mask)[0]
        new_valid_mask = np.zeros((row, col), dtype=bool)
        pixel_coords = []
        filtered_cam_pixels = []
        filtered_proj_pixels = []
        
        for idx in valid_indices:
            i, j = pixel_coords_list[idx]
            new_valid_mask[i, j] = True
            if color_extractor is not None:
                pixel_coords.append([i, j])
            filtered_cam_pixels.append(cam_pixels_list[idx])
            filtered_proj_pixels.append(proj_pixels_list[idx])
        
        valid_mask = new_valid_mask
        
        raw_points = np.array(data)
        
        # 提取颜色
        raw_colors = None
        pixel_coords_array = None
        if color_extractor is not None and len(pixel_coords) > 0:
            pixel_coords_array = np.array(pixel_coords)
            raw_colors = color_extractor.get_colors_batch(pixel_coords_array)
            print(f"  ✓ 颜色信息提取完成: {len(raw_colors)} 个点")
        
        print(f"  ✓ 初步过滤后点数量: {len(raw_points)}\n")
        
        # 保存相机像素坐标和投影仪像素坐标点集（只保存通过深度范围过滤的点）
        if len(filtered_cam_pixels) > 0:
            cam_pixels_array = np.array(filtered_cam_pixels, dtype=np.float32).reshape(-1, 1, 2)
            proj_pixels_array = np.array(filtered_proj_pixels, dtype=np.float32).reshape(-1, 1, 2)
            
            # 保存为NPY文件
            output_dir = os.path.dirname(os.path.abspath(__file__))
            cam_pixels_file = os.path.join(output_dir, 'cam_pixels_finally.npy')
            proj_pixels_file = os.path.join(output_dir, 'proj_pixels_finally.npy')
            
            np.save(cam_pixels_file, cam_pixels_array)
            np.save(proj_pixels_file, proj_pixels_array)
            
            print(f"  ✓ 像素坐标点集已保存:")
            print(f"    - {cam_pixels_file}")
            print(f"    - {proj_pixels_file}")
            print(f"    - 点数: {len(filtered_cam_pixels)}\n")
        
        return raw_points, raw_colors, pixel_coords_array, valid_mask


# =============================================================================
# 6. 点云过滤类
# =============================================================================
class PointCloudFilter:
    """点云过滤类：基础3步过滤流程"""
    
    def __init__(self):
        """初始化过滤参数"""
        # 统计滤波参数
        self.stat_nb_neighbors = 20
        self.stat_std_ratio = 2.0
        
        # 半径滤波参数
        self.radius_nb_points = 16
        self.radius = 10.0
    
    def filter(self, raw_points, raw_colors=None, voxel_size=1.0):
        """
        多级点云过滤
        
        Args:
            raw_points: 原始点云 (N, 3)
            raw_colors: 点云颜色 (N, 3)，RGB格式，范围 [0, 1]（可选）
            voxel_size: 体素下采样尺寸
            
        Returns:
            filtered_pcd: 过滤后的点云对象（包含颜色信息，如果提供）
        """
        print("="*70)
        print("【步骤4】点云过滤（基础模式 - 3步过滤）")
        print("="*70)
        
        initial_count = len(raw_points)
        print(f"  初始点云数量: {initial_count}")
        if raw_colors is not None:
            print(f"  颜色信息: 已包含")
        
        # 创建点云对象
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(raw_points)
        
        # 添加颜色信息（如果提供）
        if raw_colors is not None:
            if len(raw_colors) != len(raw_points):
                print(f"  ⚠️  警告: 颜色数量 ({len(raw_colors)}) 与点数量 ({len(raw_points)}) 不匹配，将不使用颜色")
            else:
                pcd_raw.colors = o3d.utility.Vector3dVector(raw_colors)
        
        # 步骤1: 统计离群点去除
        print(f"\n  [步骤1] 统计离群点去除 (nb_neighbors={self.stat_nb_neighbors})...")
        count_before = len(pcd_raw.points)
        pcd_filtered, _ = pcd_raw.remove_statistical_outlier(
            nb_neighbors=self.stat_nb_neighbors,
            std_ratio=self.stat_std_ratio
        )
        print(f"    统计滤波后: {len(pcd_filtered.points)} 点 (剔除 {count_before - len(pcd_filtered.points)})")
        
        # 步骤2: 半径离群点去除
        print(f"\n  [步骤2] 半径离群点去除 (radius={self.radius})...")
        count_before = len(pcd_filtered.points)
        pcd_filtered, _ = pcd_filtered.remove_radius_outlier(
            nb_points=self.radius_nb_points,
            radius=self.radius
        )
        print(f"    半径滤波后: {len(pcd_filtered.points)} 点 (剔除 {count_before - len(pcd_filtered.points)})")
        
        # 步骤3: 体素下采样
        print(f"\n  [步骤3] 体素下采样 (voxel_size={voxel_size})...")
        count_before = len(pcd_filtered.points)
        pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=voxel_size)
        print(f"    下采样后: {len(pcd_filtered.points)} 点 (减少 {count_before - len(pcd_filtered.points)})")
        
        # 检查是否保留了颜色
        has_colors = len(pcd_filtered.colors) > 0 if pcd_filtered.has_colors() else False
        if has_colors:
            print(f"    颜色信息: 已保留")
        
        print(f"\n  ✓ 点云过滤完成！")
        print(f"    过滤模式: 基础过滤（3步）")
        print(f"    最终点数: {len(pcd_filtered.points)} / {initial_count} "
              f"(保留率: {100*len(pcd_filtered.points)/initial_count:.1f}%)\n")
        
        return pcd_filtered


# =============================================================================
# 7. 结果可视化和保存类
# =============================================================================
class ResultVisualizer:
    """结果可视化和保存类：保存点云、显示结果"""
    
    @staticmethod
    def save_point_cloud(pcd, filename='pointCloud.ply'):
        """
        保存点云到文件
        
        Args:
            pcd: 点云对象
            filename: 输出文件名
        """
        print("="*70)
        print("【步骤5】保存结果")
        print("="*70)
        
        o3d.io.write_point_cloud(filename, pcd)
        print(f"  ✓ 点云已保存到: {filename}\n")
    
    @staticmethod
    def save_phase_maps(phase_h, phase_v, output_dir='./reconstruction_results', detected_points=None):
        """
        保存水平和垂直方向的绝对相位图（黑白和热力图两种格式）
        
        Args:
            phase_h: 水平方向绝对相位 (H, W)
            phase_v: 垂直方向绝对相位 (H, W)
            output_dir: 输出文件夹
            detected_points: 检测到的角点坐标 (N, 1, 2) 或 (N, 2)，用于在相位图上绘制
        """
        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)
        
        # 归一化相位数据到 0-255
        phase_h_norm = ((phase_h - phase_h.min()) / (phase_h.max() - phase_h.min() + 1e-8) * 255).astype(np.uint8)
        phase_v_norm = ((phase_v - phase_v.min()) / (phase_v.max() - phase_v.min() + 1e-8) * 255).astype(np.uint8)
        
        # 如果有角点信息，则在相位图上绘制角点
        if detected_points is not None and len(detected_points) > 0:
            # 转换为彩色图像
            phase_h_color = cv2.cvtColor(phase_h_norm, cv2.COLOR_GRAY2BGR)
            phase_v_color = cv2.cvtColor(phase_v_norm, cv2.COLOR_GRAY2BGR)
            
            # 绘制角点（绿色圆）
            for pt in detected_points:
                if len(pt.shape) > 1:
                    pt_coord = tuple(pt[0].astype(int))
                else:
                    pt_coord = tuple(pt.astype(int))
                cv2.circle(phase_h_color, pt_coord, 5, (0, 255, 0), -1)
                cv2.circle(phase_v_color, pt_coord, 5, (0, 255, 0), -1)
            
            # 保存黑白图（带角点）
            cv2.imwrite(os.path.join(output_dir, 'phase_horizontal_gray.png'), phase_h_color)
            cv2.imwrite(os.path.join(output_dir, 'phase_vertical_gray.png'), phase_v_color)
            print(f"  ✓ 黑白相位图已保存到: {output_dir}/phase_horizontal_gray.png（带角点）")
            print(f"  ✓ 黑白相位图已保存到: {output_dir}/phase_vertical_gray.png（带角点）")
            
            # 保存热力图（带角点）
            phase_h_heatmap = cv2.applyColorMap(phase_h_norm, cv2.COLORMAP_JET)
            phase_v_heatmap = cv2.applyColorMap(phase_v_norm, cv2.COLORMAP_JET)
            
            for pt in detected_points:
                if len(pt.shape) > 1:
                    pt_coord = tuple(pt[0].astype(int))
                else:
                    pt_coord = tuple(pt.astype(int))
                cv2.circle(phase_h_heatmap, pt_coord, 5, (0, 255, 0), -1)
                cv2.circle(phase_v_heatmap, pt_coord, 5, (0, 255, 0), -1)
            
            cv2.imwrite(os.path.join(output_dir, 'phase_horizontal_heatmap.png'), phase_h_heatmap)
            cv2.imwrite(os.path.join(output_dir, 'phase_vertical_heatmap.png'), phase_v_heatmap)
            print(f"  ✓ 热力图相位图已保存到: {output_dir}/phase_horizontal_heatmap.png（带角点）")
            print(f"  ✓ 热力图相位图已保存到: {output_dir}/phase_vertical_heatmap.png（带角点）")
        else:
            # 保存黑白图
            cv2.imwrite(os.path.join(output_dir, 'phase_horizontal_gray.png'), phase_h_norm)
            cv2.imwrite(os.path.join(output_dir, 'phase_vertical_gray.png'), phase_v_norm)
            print(f"  ✓ 黑白相位图已保存到: {output_dir}/phase_horizontal_gray.png")
            print(f"  ✓ 黑白相位图已保存到: {output_dir}/phase_vertical_gray.png")
            
            # 保存热力图
            phase_h_heatmap = cv2.applyColorMap(phase_h_norm, cv2.COLORMAP_JET)
            phase_v_heatmap = cv2.applyColorMap(phase_v_norm, cv2.COLORMAP_JET)
            
            cv2.imwrite(os.path.join(output_dir, 'phase_horizontal_heatmap.png'), phase_h_heatmap)
            cv2.imwrite(os.path.join(output_dir, 'phase_vertical_heatmap.png'), phase_v_heatmap)
            print(f"  ✓ 热力图相位图已保存到: {output_dir}/phase_horizontal_heatmap.png")
            print(f"  ✓ 热力图相位图已保存到: {output_dir}/phase_vertical_heatmap.png")
    
    @staticmethod
    def visualize_phase_maps(phase_h, phase_v, show=True):
        """
        可视化水平和垂直方向的绝对相位图
        
        Args:
            phase_h: 水平方向绝对相位 (H, W)
            phase_v: 垂直方向绝对相位 (H, W)
            show: 是否显示图像
        """
        if not show:
            return
        
        print("="*70)
        print("【可视化】绝对相位图")
        print("="*70)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 水平方向绝对相位
        im1 = axes[0].imshow(phase_h, cmap='gray')
        axes[0].set_title('水平方向绝对相位\n(从垂直条纹v1-v12得到)', fontsize=12, color='red')
        axes[0].set_xlabel('列（像素）')
        axes[0].set_ylabel('行（像素）')
        plt.colorbar(im1, ax=axes[0], label='相位值 (rad)')
        
        # 垂直方向绝对相位
        im2 = axes[1].imshow(phase_v, cmap='gray')
        axes[1].set_title('垂直方向绝对相位\n(从水平条纹h1-h12得到)', fontsize=12, color='red')
        axes[1].set_xlabel('列（像素）')
        axes[1].set_ylabel('行（像素）')
        plt.colorbar(im2, ax=axes[1], label='相位值 (rad)')
        
        plt.tight_layout()
        print(f"  ✓ 水平方向相位范围: [{phase_h.min():.3f}, {phase_h.max():.3f}] rad")
        print(f"  ✓ 垂直方向相位范围: [{phase_v.min():.3f}, {phase_v.max():.3f}] rad")
        print()
        plt.show()
    
    @staticmethod
    def visualize(pcd, valid_mask=None, phase_h=None, phase_v=None, 
                  show_phase=False, show_mask=True, show_point_cloud=True):
        """
        可视化结果
        
        Args:
            pcd: 点云对象
            valid_mask: 有效点掩码 (H, W)
            phase_h: 水平方向绝对相位 (H, W)
            phase_v: 垂直方向绝对相位 (H, W)
            show_phase: 是否显示绝对相位图
            show_mask: 是否显示掩码
            show_point_cloud: 是否显示点云
        """
        print("="*70)
        print("【步骤6】可视化结果")
        print("="*70)
        
        # 显示绝对相位图
        if show_phase and phase_h is not None and phase_v is not None:
            ResultVisualizer.visualize_phase_maps(phase_h, phase_v, show=True)
        
        if show_mask and valid_mask is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(valid_mask, cmap='gray')
            plt.title('有效点掩码', fontsize=14, color='red')
            plt.xlabel('列（像素）')
            plt.ylabel('行（像素）')
            plt.colorbar()
            plt.show()
        
        if show_point_cloud and pcd is not None:
            print("  - 打开3D点云可视化窗口...")
            o3d.visualization.draw_geometries([pcd], window_name='三维重建结果')


# =============================================================================
# 8. 主流程协调器
# =============================================================================
class ReconstructionPipeline:
    """主流程协调器：整合所有模块完成三维重建"""
    
    def __init__(self, config):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化各个模块
        self.calib_reader = CalibrationReader(config['calibration_file'])
        self.image_loader = ImageLoader()
        self.phase_unwrapper = PhaseUnwrapper(config['frequencies'])
        # 传递过采样倍数给重建器（必须与FSM解码时的过采样倍数一致）
        oversample = config.get('oversample', 17)
        self.reconstructor = PointCloudReconstructor(self.calib_reader, oversample=oversample)
        self.filter = PointCloudFilter()
        self.visualizer = ResultVisualizer()
        
        # 设置过滤参数
        self._configure_filters(config)
    
    def _configure_filters(self, config):
        """配置过滤参数"""
        # 质量过滤参数
        self.reconstructor.modulation_threshold = config.get('modulation_threshold', 0.2)
        self.reconstructor.intensity_threshold = config.get('intensity_threshold', 30)
        
        # 点云过滤参数
        self.filter.stat_nb_neighbors = config.get('stat_nb_neighbors', 30)
        self.filter.stat_std_ratio = config.get('stat_std_ratio', 2.0)
        self.filter.radius_nb_points = config.get('radius_nb_points', 10)
        self.filter.radius = config.get('radius', 10.0)
    
    def run(self):
        """执行完整的三维重建流程"""
        print("\n" + "="*70)
        print("           结构光三维重建系统（模块化版本）")
        print("="*70 + "\n")
        
        # 步骤1: 读取图像
        print("="*70)
        print("【步骤1】读取条纹图像")
        print("="*70)
        
        h_images = self.image_loader.load_stripe_images(
            self.config['fringe_folder'], 'h', '.bmp', 12
        )
        v_images = self.image_loader.load_stripe_images(
            self.config['fringe_folder'], 'v', '.bmp', 12
        )
        
        # 生成无条纹纹理图像（用于颜色提取）
        color_extractor = None
        if self.config.get('enable_color', False):
            texture_image = self.image_loader.generate_texture_image(
                self.config['fringe_folder'], 'h', '.bmp', 4
            )
            color_extractor = ColorExtractor(texture_image)
        
        print()
        
        # 步骤2: 相位处理
        print("="*70)
        print("【步骤2】相位处理（包裹相位+解包裹+去噪）")
        print("="*70)
        
        # ⚠️ 重要：条纹方向与相位方向的对应关系
        # - 水平条纹（h1-h12）→ 垂直方向的相位（沿列变化）
        # - 垂直条纹（v1-v12）→ 水平方向的相位（沿行变化）
        
        print("  处理水平条纹图像（h1-h12）→ 获取垂直方向相位...")
        phase_v, mod_h, int_h = self.phase_unwrapper.process_images(h_images)
        print(f"    ✓ 垂直绝对相位范围: [{phase_v.min():.3f}, {phase_v.max():.3f}]")
        
        print("  处理垂直条纹图像（v1-v12）→ 获取水平方向相位...")
        phase_h, mod_v, int_v = self.phase_unwrapper.process_images(v_images)
        print(f"    ✓ 水平绝对相位范围: [{phase_h.min():.3f}, {phase_h.max():.3f}]")
        print()
        
        # 步骤3: 三维重建
        # 注意：使用垂直条纹（v1-v12）的调制度和平均强度作为质量指标
        raw_points, raw_colors, pixel_coords, valid_mask = self.reconstructor.reconstruct(
            phase_h, phase_v, mod_v, int_v,
            z_min=self.config.get('z_min', 0),
            z_max=self.config.get('z_max', 2300),
            color_extractor=color_extractor
        )
        
        # 步骤4: 点云过滤
        filtered_pcd = self.filter.filter(
            raw_points,
            raw_colors,
            voxel_size=self.config.get('voxel_size', 1.0)
        )
        
        # 步骤5: 保存结果
        self.visualizer.save_point_cloud(
            filtered_pcd,
            filename=self.config.get('output_file', 'pointCloud.ply')
        )
        
        # 保存相位图（黑白和热力图两种格式，包含角点）
        output_dir = self.config.get('output_dir', './reconstruction_results')
        
        # 尝试读取标定过程中保存的角点信息
        detected_points = None
        try:
            if os.path.exists('detected_points.npy'):
                detected_points = np.load('detected_points.npy')
                print(f"  ✓ 已读取标定角点: detected_points.npy")
        except Exception as e:
            print(f"  ⚠️  无法读取角点信息: {e}")
        
        self.visualizer.save_phase_maps(phase_h, phase_v, output_dir=output_dir, detected_points=detected_points)
        
        # 步骤6: 可视化
        self.visualizer.visualize(
            filtered_pcd, valid_mask, phase_h, phase_v,
            show_phase=self.config.get('show_phase_maps', False),
            show_mask=self.config.get('show_mask', True),
            show_point_cloud=self.config.get('show_point_cloud', True)
        )
        
        print("="*70)
        print("✅ 三维重建完成！")
        print("="*70)
        
        return {
            'filtered_point_cloud': filtered_pcd,
            'raw_points': raw_points,
            'valid_mask': valid_mask,
            'phase_h': phase_h,
            'phase_v': phase_v
        }


# =============================================================================
# 9. 主程序入口
# =============================================================================
def main():
    """主程序入口"""
        
        # =========================================================================
    # 配置参数
        # =========================================================================
    config = {
        # === 文件路径 ===
        'calibration_file': './标定结果.txt',              # 标定文件路径
        'fringe_folder': r'E:\images\images_PSM_81_72_64\06_luminiance200_81_object_grouped\1',      # 条纹图像文件夹（包含h1-h12和v1-v12）
        'output_file': 'pointCloud.ply',                 # 输出点云文件名
        'output_dir': './reconstruction_results',        # 输出文件夹（保存点云和相位图）
        
        # === 相位解包裹参数 ===
        'frequencies': np.array([81, 72, 64]),           # 三个频率
        'oversample': 17,                                # 过采样倍数（必须与FSM解码时的过采样倍数一致）
        
        # === 三维重建参数 ===
        'z_min': 0,                                      # 深度最小值（mm）
        'z_max': 2300,                                   # 深度最大值（mm）
        
        # === 质量过滤参数 ===
        'modulation_threshold': 0.2,                     # 调制度阈值（0-1），值越大过滤越多
        'intensity_threshold': 30,                       # 强度阈值（0-255），值越大过滤越多
        
        # === 点云过滤参数 ===
        'stat_nb_neighbors': 30,                         # 统计滤波邻居数（值越大越严格，推荐20-50）
        'stat_std_ratio': 2.0,                          # 标准差倍数（值越小越严格，推荐1.0-3.0）
        'radius_nb_points': 10,                         # 半径滤波最少邻居数（值越大越严格，推荐5-20）
        'radius': 10.0,                                 # 半径滤波范围mm（值越小越严格，推荐5-20）
        'voxel_size': 1.0,                              # 体素下采样尺寸mm（值越大点越少，推荐0.5-2.0）
        
        # === 颜色提取参数 ===
        'enable_color': True,                           # 是否启用颜色提取（从h1-h4生成无条纹纹理）
        
        # === 可视化参数 ===
        'show_phase_maps': True,                        # 显示绝对相位图（水平和垂直方向）
        'show_mask': True,                              # 显示有效点掩码
        'show_point_cloud': True,                       # 显示3D点云
    }
    
    # =========================================================================
    # 执行三维重建
    # =========================================================================
    try:
        pipeline = ReconstructionPipeline(config)
        results = pipeline.run()
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到错误: {e}")
        print("\n请检查：")
        print("  1. 标定文件是否存在: 标定结果.txt")
        print("  2. 图像文件夹是否正确，并包含h1-h12和v1-v12图像")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()