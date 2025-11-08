# -*- coding: utf-8 -*-
"""
FSA三维重建核心模块

本模块实现了基于频移法（FSA）的三维点云重建功能。
通过结合相机和投影仪的标定参数以及解码得到的视差图，
重建出三维点云数据。

主要功能：
1. 读取相机和投影仪的标定结果
2. 从视差图重建三维点云
3. 点云可视化和保存

作者：基于 reconstruct_fsa.py 提取
日期：2025-11-07
"""

import numpy as np
import os
from typing import Tuple, Optional, Dict
import cv2


class CalibrationReader:
    """
    标定结果读取器
    
    用于从标定结果文件中读取相机和投影仪的标定参数
    """
    
    @staticmethod
    def read_calibration_file(calib_file_path: str) -> Dict:
        """
        读取标定结果文件
        
        Parameters:
        -----------
        calib_file_path : str
            标定结果文件路径
            
        Returns:
        --------
        calib_data : dict
            包含相机和投影仪标定参数的字典
        """
        print(f"\n读取标定文件: {calib_file_path}")
        
        with open(calib_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        calib_data = {}
        
        # 解析相机参数
        cam_params = []
        for i in range(4, 12):  # 第5-12行是相机参数
            cam_params.append(float(lines[i].strip()))
        
        calib_data['camera'] = {
            'fx': cam_params[0],
            'fy': cam_params[1],
            'cx': cam_params[2],
            'cy': cam_params[3],
            'k1': cam_params[4],
            'k2': cam_params[5],
            'p1': cam_params[6],
            'p2': cam_params[7]
        }
        
        # 解析投影仪参数
        proj_params = []
        for i in range(17, 25):  # 第18-25行是投影仪参数
            proj_params.append(float(lines[i].strip()))
        
        calib_data['projector'] = {
            'fx': proj_params[0],
            'fy': proj_params[1],
            'cx': proj_params[2],
            'cy': proj_params[3],
            'k1': proj_params[4],
            'k2': proj_params[5],
            'p1': proj_params[6],
            'p2': proj_params[7]
        }
        
        # 解析旋转矩阵（3x3）
        rotation_matrix = []
        for i in range(27, 30):  # 第28-30行是旋转矩阵
            row = [float(x) for x in lines[i].strip().split()]
            rotation_matrix.append(row)
        calib_data['rotation_matrix'] = np.array(rotation_matrix)
        
        # 解析平移矩阵（1x3）
        translation = [float(x) for x in lines[32].strip().split()]
        calib_data['translation'] = np.array(translation)
        
        print("[OK] 标定文件读取成功")
        print(f"  相机参数: fx={calib_data['camera']['fx']:.2f}, fy={calib_data['camera']['fy']:.2f}")
        print(f"  投影仪参数: fx={calib_data['projector']['fx']:.2f}, fy={calib_data['projector']['fy']:.2f}")
        print(f"  平移向量: {calib_data['translation']}")
        
        return calib_data
    
    @staticmethod
    def save_pyaoi_calibration(calib_data: Dict, 
                               camera_output_path: str,
                               projector_output_path: str,
                               image_size: Tuple[int, int],
                               projector_size: Tuple[int, int] = (1920, 1080)):
        """
        将标定数据保存为.npz格式的标定文件
        
        Parameters:
        -----------
        calib_data : dict
            标定数据字典
        camera_output_path : str
            相机标定文件输出路径
        projector_output_path : str
            投影仪标定文件输出路径
        image_size : tuple(width, height)
            相机图像尺寸
        projector_size : tuple(width, height)
            投影仪分辨率，默认为(1920, 1080)
        """
        # 创建相机标定数据
        cam = calib_data['camera']
        camera_matrix = np.array([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ])
        camera_dist = np.array([cam['k1'], cam['k2'], cam['p1'], cam['p2'], 0])
        
        # 保存相机标定
        np.savez(camera_output_path,
                 camera_matrix=camera_matrix,
                 dist_coeffs=camera_dist,
                 image_size=np.array(image_size),
                 rotation_vectors=np.array([]),
                 translation_vectors=np.array([]))
        
        print(f"[OK] 相机标定文件已保存: {camera_output_path}")
        
        # 创建投影仪标定数据
        proj = calib_data['projector']
        projector_matrix = np.array([
            [proj['fx'], 0, proj['cx']],
            [0, proj['fy'], proj['cy']],
            [0, 0, 1]
        ])
        projector_dist = np.array([proj['k1'], proj['k2'], proj['p1'], proj['p2'], 0])
        
        # 投影仪的外参（相对于相机的位姿）
        R = calib_data['rotation_matrix']
        T = calib_data['translation'].reshape(3, 1)
        
        # 保存投影仪标定
        np.savez(projector_output_path,
                 camera_matrix=projector_matrix,
                 dist_coeffs=projector_dist,
                 image_size=np.array(projector_size),
                 rotation_matrix=R,
                 translation_vector=T)
        
        print(f"[OK] 投影仪标定文件已保存: {projector_output_path}")


class FSAReconstructor:
    """
    FSA三维重建器
    
    原理说明：
    --------
    通过相机和投影仪的标定参数，以及解码得到的视差图，
    可以重建出物体表面的三维点云。
    
    重建过程：
    1. 对于相机图像中的每个像素 (xc, yc)
    2. 通过视差图得到对应的投影仪坐标 (xp, yp)
    3. 利用三角测量原理计算该点的三维坐标 (X, Y, Z)
    """
    
    def __init__(self, 
                 camera_calib_path: str,
                 projector_calib_path: str,
                 projector_width: int = 1920,
                 projector_height: int = 1080):
        """
        初始化三维重建器
        
        Parameters:
        -----------
        camera_calib_path : str
            相机标定文件路径（.npz格式）
        projector_calib_path : str
            投影仪标定文件路径（.npz格式）
        projector_width : int
            投影仪宽度
        projector_height : int
            投影仪高度
        """
        self.proj_width = projector_width
        self.proj_height = projector_height
                # 加载标定数据
        print("\n加载标定数据...")
        self._load_calibration(camera_calib_path, projector_calib_path)
    
    def _load_calibration(self, camera_calib_path: str, projector_calib_path: str):
        """加载标定数据"""
        # 加载相机标定
        cam_data = np.load(camera_calib_path)
        self.camera_matrix = cam_data['camera_matrix']
        self.camera_dist = cam_data['dist_coeffs']
        
        # 加载投影仪标定
        proj_data = np.load(projector_calib_path)
        self.projector_matrix = proj_data['camera_matrix']
        self.projector_dist = proj_data['dist_coeffs']
        self.R = proj_data['rotation_matrix']
        self.T = proj_data['translation_vector']
        
        print("[OK] 标定数据加载成功")
    
    def _reconstruct_pointcloud(self,
                                   disparity_hor: np.ndarray,
                                   disparity_ver: np.ndarray,
                                   white_pattern: np.ndarray,
                                   peak_limit: float = 0.8,
                                   peak_hor: Optional[np.ndarray] = None,
                                   peak_ver: Optional[np.ndarray] = None,
                                   oversample: int = 4,
                                   min_distance: Optional[float] = None,
                                   max_distance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        三维重建实现
        
        原理：
        1. 对于相机像素 (uc, vc)，通过视差图得到投影仪坐标 (up, vp)
        2. 对相机和投影仪像素进行去畸变
        3. 使用三角测量计算3D坐标
        """
        print("\n" + "="*60)
        print("开始三维重建")
        print("="*60)
        
        import time
        tstart = time.time()
        
        # 1. 处理视差图
        disparity_hor_int = np.round(disparity_hor * oversample).astype(np.int32)
        disparity_ver_int = np.round(disparity_ver * oversample).astype(np.int32)
        
        # 2. 计算有效掩码
        valid_mask = np.ones(disparity_hor.shape, dtype=bool)
        
        # 诊断信息：视差图统计
        print("\n视差图统计信息:")
        print(f"  水平视差范围: [{disparity_hor_int.min()}, {disparity_hor_int.max()}]")
        print(f"  垂直视差范围: [{disparity_ver_int.min()}, {disparity_ver_int.max()}]")
        print(f"  投影仪分辨率: {self.proj_width} x {self.proj_height}")
        print(f"  过采样后范围: [0, {self.proj_width * oversample}) x [0, {self.proj_height * oversample})")
        
        # 检查视差图覆盖的投影仪范围（过滤前）
        valid_for_coverage = (disparity_hor_int >= 0) & (disparity_ver_int >= 0)
        if valid_for_coverage.sum() > 0:
            disp_hor_coverage = disparity_hor_int[valid_for_coverage] / oversample
            disp_ver_coverage = disparity_ver_int[valid_for_coverage] / oversample
            print(f"\n视差图覆盖的投影仪范围（过滤前）:")
            print(f"  投影仪X范围: [{disp_ver_coverage.min():.1f}, {disp_ver_coverage.max():.1f}]")
            print(f"  投影仪Y范围: [{disp_hor_coverage.min():.1f}, {disp_hor_coverage.max():.1f}]")
            coverage_before_x = (disp_ver_coverage.max() - disp_ver_coverage.min()) / self.proj_width * 100
            coverage_before_y = (disp_hor_coverage.max() - disp_hor_coverage.min()) / self.proj_height * 100
            print(f"  覆盖率: X方向 {coverage_before_x:.1f}%, Y方向 {coverage_before_y:.1f}%")
            
            if coverage_before_x < 80 or coverage_before_y < 80:
                print(f"\n[!] 关键问题: 视差图本身就没有覆盖整个投影仪画面！")
                print(f"  这是数据采集或解码阶段的问题，不是重建算法的问题。")
                print(f"  可能原因:")
                print(f"    1. 相机视野只看到了投影仪画面的一部分")
                print(f"    2. 物体只占投影仪画面的一部分（如果是扫描物体，这是正常的）")
                print(f"    3. FSA解码时某些区域解码失败")
                print(f"    4. 投影仪和相机的相对位置导致视野受限")
        
        # 视差范围过滤（先做，以便统计）
        valid_mask_range = (disparity_hor_int >= 0) & (disparity_hor_int < self.proj_height * oversample)
        valid_mask_range = valid_mask_range & (disparity_ver_int >= 0) & (disparity_ver_int < self.proj_width * oversample)
        print(f"\n视差范围过滤:")
        print(f"  有效点数量: {valid_mask_range.sum()} / {valid_mask_range.size} ({valid_mask_range.sum()/valid_mask_range.size*100:.2f}%)")
        
        valid_mask = valid_mask & valid_mask_range
        
        # 峰值过滤
        if peak_hor is not None and peak_ver is not None:
            peak_min = np.minimum(peak_hor, peak_ver)
            print(f"\n峰值图统计信息:")
            print(f"  峰值范围: [{peak_min.min():.3f}, {peak_min.max():.3f}]")
            print(f"  峰值均值: {peak_min.mean():.3f}")
            print(f"  峰值中位数: {np.median(peak_min):.3f}")
            
            # 统计不同阈值下的有效点数量
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            print(f"\n不同峰值阈值下的有效点数量:")
            for thresh in thresholds:
                count = (peak_min >= thresh).sum()
                pct = count / peak_min.size * 100
                marker = " ← 当前" if abs(thresh - peak_limit) < 0.01 else ""
                print(f"  {thresh:.1f}: {count:,} ({pct:.2f}%){marker}")
            
            valid_mask_peak = (peak_min >= peak_limit)
            print(f"\n使用峰值阈值: {peak_limit}")
            print(f"峰值过滤后有效点: {valid_mask_peak.sum()} / {valid_mask_peak.size} ({valid_mask_peak.sum()/valid_mask_peak.size*100:.2f}%)")
            
            valid_mask = valid_mask & valid_mask_peak
        else:
            print("\n[!] 警告: 未提供峰值图，无法进行峰值过滤")
        
        # 获取有效像素的索引
        valid_pixels = np.where(valid_mask)
        num_valid = len(valid_pixels[0])
        
        if num_valid == 0:
            raise ValueError("没有有效的点可以重建")
        
        print(f"\n最终有效像素数量: {num_valid:,} / {valid_mask.size:,} ({num_valid/valid_mask.size*100:.2f}%)")
        
        # 分析有效点的分布
        if num_valid > 0:
            valid_y = valid_pixels[0]
            valid_x = valid_pixels[1]
            print(f"\n相机图像中的有效点分布:")
            print(f"  X范围: [{valid_x.min()}, {valid_x.max()}] (图像宽度: {disparity_hor.shape[1]})")
            print(f"  Y范围: [{valid_y.min()}, {valid_y.max()}] (图像高度: {disparity_hor.shape[0]})")
            
            # 检查是否有大片区域被过滤
            coverage_x = (valid_x.max() - valid_x.min() + 1) / disparity_hor.shape[1] * 100
            coverage_y = (valid_y.max() - valid_y.min() + 1) / disparity_hor.shape[0] * 100
            print(f"  相机图像有效区域覆盖率: X方向 {coverage_x:.1f}%, Y方向 {coverage_y:.1f}%")
            print(f"  (注: 此覆盖率表示相机图像中有有效点的区域占比，")
            print(f"   如果物体只占画面一部分，此值较低是正常的)")
            
            if coverage_x < 50 or coverage_y < 50:
                print("\n[!] 提示: 相机图像中有效点覆盖范围较小，可能原因:")
                print("  1. 峰值阈值过高，建议降低 peak_limit (如 0.6-0.7)")
                print("  2. 视差图质量不佳，某些区域解码失败")
                print("  3. 物体只占相机画面的一部分（这是正常的）")
                print("  4. 相机视野限制，只看到了部分投影区域")
        
        # 3. 准备相机和投影仪的像素坐标
        camera_pixels = np.column_stack([valid_pixels[1], valid_pixels[0]])  # (x, y) 格式
        projector_pixels = np.column_stack([
            disparity_ver_int[valid_mask] / oversample,  # 投影仪x坐标
            disparity_hor_int[valid_mask] / oversample    # 投影仪y坐标
        ])
        
        # 诊断：检查投影仪坐标的分布
        print(f"\n投影仪坐标分布:")
        print(f"  投影仪X范围: [{projector_pixels[:, 0].min():.1f}, {projector_pixels[:, 0].max():.1f}] (投影仪宽度: {self.proj_width})")
        print(f"  投影仪Y范围: [{projector_pixels[:, 1].min():.1f}, {projector_pixels[:, 1].max():.1f}] (投影仪高度: {self.proj_height})")
        
        proj_coverage_x = (projector_pixels[:, 0].max() - projector_pixels[:, 0].min()) / self.proj_width * 100
        proj_coverage_y = (projector_pixels[:, 1].max() - projector_pixels[:, 1].min()) / self.proj_height * 100
        print(f"  投影仪画面覆盖率: X方向 {proj_coverage_x:.1f}%, Y方向 {proj_coverage_y:.1f}%")
        print(f"  (注: 此覆盖率表示视差图覆盖的投影仪画面范围，")
        print(f"   接近100%说明视差图覆盖了几乎整个投影仪范围，这是正常的)")
        
        if proj_coverage_x < 80 or proj_coverage_y < 80:
            print("\n[!] 警告: 投影仪坐标覆盖范围不足！")
            print("  可能原因:")
            print("  1. 视差图本身就没有覆盖整个投影仪范围")
            print("  2. 相机视野只看到了投影仪画面的一部分")
            print("  3. 物体只占投影仪画面的一部分（如果是扫描物体，这是正常的）")
            print("  4. 视差图解码时某些区域失败")
            
            # 检查视差图的原始分布（在过滤前）
            print(f"\n视差图原始分布（过滤前）:")
            valid_before_filter = (disparity_hor_int >= 0) & (disparity_ver_int >= 0)
            if valid_before_filter.sum() > 0:
                disp_hor_valid = disparity_hor_int[valid_before_filter] / oversample
                disp_ver_valid = disparity_ver_int[valid_before_filter] / oversample
                print(f"  水平视差范围: [{disp_hor_valid.min():.1f}, {disp_hor_valid.max():.1f}]")
                print(f"  垂直视差范围: [{disp_ver_valid.min():.1f}, {disp_ver_valid.max():.1f}]")
                
                orig_coverage_x = (disp_ver_valid.max() - disp_ver_valid.min()) / self.proj_width * 100
                orig_coverage_y = (disp_hor_valid.max() - disp_hor_valid.min()) / self.proj_height * 100
                print(f"  原始覆盖率: X方向 {orig_coverage_x:.1f}%, Y方向 {orig_coverage_y:.1f}%")
                
                if orig_coverage_x < 80 or orig_coverage_y < 80:
                    print("\n  → 视差图本身就没有覆盖整个投影仪范围")
                    print("     这是数据采集或解码的问题，不是重建算法的问题")
                else:
                    print("\n  → 视差图原始覆盖范围正常，但过滤后范围缩小")
                    print("     建议降低峰值阈值或检查过滤条件")
        
        # 4. 去畸变
        print("去畸变处理...")
        camera_undistorted = cv2.undistortPoints(
            camera_pixels.astype(np.float32),
            self.camera_matrix,
            self.camera_dist,
            P=self.camera_matrix
        ).reshape(-1, 2)
        
        projector_undistorted = cv2.undistortPoints(
            projector_pixels.astype(np.float32),
            self.projector_matrix,
            self.projector_dist,
            P=self.projector_matrix
        ).reshape(-1, 2)
        
        # 5. 构建投影矩阵
        # 相机投影矩阵（相机坐标系）
        P_camera = np.hstack([np.eye(3), np.zeros((3, 1))])
        P_camera = self.camera_matrix @ P_camera
        
        # 投影仪投影矩阵（投影仪坐标系，需要转换到相机坐标系）
        # 投影仪相对于相机的位姿：R, T
        P_projector = np.hstack([self.R, self.T])
        P_projector = self.projector_matrix @ P_projector
        
        # 6. 三角测量
        print("执行三角测量...")
        points_4d = cv2.triangulatePoints(
            P_camera, P_projector,
            camera_undistorted.T, projector_undistorted.T
        )
        
        # 转换为3D坐标（齐次坐标转笛卡尔坐标）
        points_3d = (points_4d[:3] / points_4d[3]).T  # 形状: (N, 3)
        
        # 7. 距离过滤
        if min_distance is not None or max_distance is not None:
            distances = np.linalg.norm(points_3d, axis=1)
            if min_distance is not None:
                valid_mask_dist = distances >= min_distance
                points_3d = points_3d[valid_mask_dist]
                valid_pixels = (valid_pixels[0][valid_mask_dist], valid_pixels[1][valid_mask_dist])
            if max_distance is not None:
                valid_mask_dist = distances <= max_distance
                points_3d = points_3d[valid_mask_dist]
                valid_pixels = (valid_pixels[0][valid_mask_dist], valid_pixels[1][valid_mask_dist])
        
        # 8. 提取颜色
        if len(white_pattern.shape) == 2:  # 灰度图
            point_color = white_pattern[valid_pixels[0], valid_pixels[1]]
            point_color = np.column_stack([point_color, point_color, point_color])
        else:  # 彩色图
            point_color = white_pattern[valid_pixels[0], valid_pixels[1], :]
            if point_color.shape[1] == 3:
                point_color = point_color[:, ::-1]  # BGR转RGB
        
        # 9. 转换坐标系：从相机坐标系转换到投影仪坐标系
        # 投影仪坐标系 = R^T @ (相机坐标系 - T)
        points_3d_camera = points_3d
        points_3d_projector = (self.R.T @ (points_3d_camera - self.T.flatten()).T).T
        
        elapsed = time.time() - tstart
        print(f"\n[OK] 重建完成（耗时: {elapsed:.2f}秒）")
        print(f"  点云数量: {len(points_3d_projector):,}")
        print(f"  点云范围:")
        print(f"    X: [{points_3d_projector[:, 0].min():.2f}, {points_3d_projector[:, 0].max():.2f}] mm")
        print(f"    Y: [{points_3d_projector[:, 1].min():.2f}, {points_3d_projector[:, 1].max():.2f}] mm")
        print(f"    Z: [{points_3d_projector[:, 2].min():.2f}, {points_3d_projector[:, 2].max():.2f}] mm")
        print("="*60)
        
        return points_3d_projector, point_color
    
    def reconstruct_pointcloud(self,
                              disparity_hor: np.ndarray,
                              disparity_ver: np.ndarray,
                              white_pattern: np.ndarray,
                              peak_limit: float = 0.8,
                              peak_hor: Optional[np.ndarray] = None,
                              peak_ver: Optional[np.ndarray] = None,
                              oversample: int = 4,
                              min_distance: Optional[float] = None,
                              max_distance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从视差图重建三维点云
        
        Parameters:
        -----------
        disparity_hor : np.ndarray
            水平视差图（投影仪行坐标）
        disparity_ver : np.ndarray
            垂直视差图（投影仪列坐标）
        white_pattern : np.ndarray
            白光图像（用于提取点云颜色）
        peak_limit : float
            峰值阈值，低于此值的点将被过滤
        peak_hor : np.ndarray, optional
            水平方向的峰值强度图
        peak_ver : np.ndarray, optional
            垂直方向的峰值强度图
        oversample : int
            过采样倍数（视差图的分辨率倍数）
        min_distance : float, optional
            最小距离限制
        max_distance : float, optional
            最大距离限制
            
        Returns:
        --------
        point_cloud : np.ndarray
            点云数据，形状为 (N, 3)，单位为毫米
        point_color : np.ndarray
            点云颜色，形状为 (N, 3)，RGB格式，范围 [0, 255]
        """
        # 直接使用重建方法
        return self._reconstruct_pointcloud(
            disparity_hor, disparity_ver, white_pattern,
            peak_limit, peak_hor, peak_ver, oversample,
            min_distance, max_distance
        )
    
    def save_pointcloud(self,
                       point_cloud: np.ndarray,
                       point_color: np.ndarray,
                       output_path: str):
        """
        保存点云为PLY格式
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            点云数据
        point_color : np.ndarray
            点云颜色
        output_path : str
            输出文件路径
        """
        try:
            import open3d as o3d
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.colors = o3d.utility.Vector3dVector(point_color / 255.0)
            
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"\n[OK] 点云已保存: {output_path}")
            
        except ImportError:
            print("[!] 警告: 未安装open3d库，使用简化的PLY保存方法")
            self._save_ply_simple(point_cloud, point_color, output_path)
    
    def _save_ply_simple(self,
                        point_cloud: np.ndarray,
                        point_color: np.ndarray,
                        output_path: str):
        """简化的PLY保存方法（不依赖open3d）"""
        with open(output_path, 'w') as f:
            # 写入PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 写入点云数据
            for i in range(len(point_cloud)):
                x, y, z = point_cloud[i]
                r, g, b = point_color[i].astype(np.uint8)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        print(f"\n[OK] 点云已保存（简化格式）: {output_path}")
    
    def visualize_pointcloud(self,
                            point_cloud: np.ndarray,
                            point_color: np.ndarray,
                            window_name: str = "FSA Point Cloud"):
        """
        使用Open3D可视化点云
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            点云数据
        point_color : np.ndarray
            点云颜色
        window_name : str
            窗口名称
        """
        try:
            import open3d as o3d
            
            print(f"\n启动Open3D可视化...")
            print(f"  点云数量: {len(point_cloud):,}")
            
            # 检查点云数据
            if len(point_cloud) == 0:
                print("❌ 错误: 点云为空，无法可视化")
                return
            
            # 计算点云统计信息
            center = point_cloud.mean(axis=0)
            min_bound = point_cloud.min(axis=0)
            max_bound = point_cloud.max(axis=0)
            extent = max_bound - min_bound
            max_extent = extent.max()
            
            print(f"  点云中心: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm")
            print(f"  点云范围: X[{min_bound[0]:.2f}, {max_bound[0]:.2f}], "
                  f"Y[{min_bound[1]:.2f}, {max_bound[1]:.2f}], "
                  f"Z[{min_bound[2]:.2f}, {max_bound[2]:.2f}] mm")
            print(f"  最大尺寸: {max_extent:.2f} mm")
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            # 处理颜色
            if point_color is not None and len(point_color) > 0:
                # 确保颜色值在[0, 1]范围内
                if point_color.max() > 1.0:
                    point_color_normalized = point_color / 255.0
                else:
                    point_color_normalized = point_color
                pcd.colors = o3d.utility.Vector3dVector(point_color_normalized)
                color_min = point_color_normalized.min()
                color_max = point_color_normalized.max()
                print(f"  颜色范围: [{color_min:.2f}, {color_max:.2f}]")
                
                # 如果颜色都是白色或接近白色，使用红色以便可见
                if color_min > 0.9 and color_max > 0.9:
                    print("  警告: 点云颜色接近白色，可能不易在白色背景下看到")
                    print("  建议: 如果看不到点云，尝试旋转视角或检查点云数据")
            else:
                # 如果没有颜色，使用默认颜色（红色）
                print("  警告: 点云没有颜色信息，使用默认红色")
                pcd.paint_uniform_color([1.0, 0.0, 0.0])
            
            # 如果点云数量太大，进行下采样以提高性能
            original_count = len(pcd.points)
            if original_count > 1000000:  # 如果超过100万个点
                print(f"  点云数量较大 ({original_count:,})，进行下采样以提高渲染性能...")
                # 使用体素下采样
                voxel_size = max_extent / 500  # 体素大小为点云最大尺寸的1/500
                pcd_downsampled = pcd.voxel_down_sample(voxel_size)
                pcd = pcd_downsampled
                print(f"  下采样后点云数量: {len(pcd.points):,}")
            
            # 验证点云是否有效
            if len(pcd.points) == 0:
                print("❌ 错误: 点云为空，无法可视化")
                return
            
            print(f"  准备显示 {len(pcd.points):,} 个点")
            
            # 创建可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, width=1280, height=720, visible=True)
            vis.add_geometry(pcd)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            render_option.point_size = 2.0  # 点大小
            render_option.show_coordinate_frame = True  # 显示坐标轴
            
            # 获取视图控制并设置相机视角
            view_control = vis.get_view_control()
            
            # 更新一次渲染以确保几何体已添加
            vis.poll_events()
            vis.update_renderer()
            
            # 设置相机参数以包含整个点云
            # 使用相机参数来设置合适的视角
            ctr = vis.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            
            # 计算合适的相机位置（从斜上方观察点云）
            # 点云中心在center，我们让相机从斜上方看向中心
            camera_distance = max_extent * 3.0  # 相机距离
            
            # 计算相机位置（从斜上方观察）
            # 使用球坐标系：从(θ, φ)方向观察点云中心
            theta = np.pi / 4  # 45度
            phi = np.pi / 6    # 30度
            eye_x = center[0] + camera_distance * np.sin(phi) * np.cos(theta)
            eye_y = center[1] + camera_distance * np.sin(phi) * np.sin(theta)
            eye_z = center[2] + camera_distance * np.cos(phi)
            eye = np.array([eye_x, eye_y, eye_z])
            
            # 构建look-at矩阵
            forward = center - eye
            forward = forward / np.linalg.norm(forward)
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 构建外参矩阵（世界坐标系到相机坐标系）
            extrinsic = np.eye(4)
            extrinsic[0, :3] = right
            extrinsic[1, :3] = -up  # Open3D使用-Y向上
            extrinsic[2, :3] = -forward
            extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye
            
            cam.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(cam)
            
            # 设置合适的缩放
            ctr.set_zoom(0.7)
            
            # 再次更新渲染
            vis.poll_events()
            vis.update_renderer()
            
            print("\n可视化操作提示:")
            print("  - 鼠标左键拖动: 旋转视角")
            print("  - 鼠标右键拖动: 平移视角")
            print("  - 滚轮: 缩放")
            print("  - 按 H 键: 显示帮助")
            print("  - 按 Q 键或关闭窗口: 退出")
            print("  - 按 R 键: 重置视角")
            print("\n正在显示点云窗口...")
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
            print("✓ 可视化窗口已关闭")
            
        except ImportError:
            print("[ERROR] 错误: 未安装open3d库，无法进行可视化")
            print("   请运行: pip install open3d")
        except Exception as e:
            print(f"❌ 可视化错误: {e}")
            import traceback
            traceback.print_exc()
    


def demonstrate_reconstruction():
    """
    演示三维重建的完整流程
    """
    print("="*80)
    print("FSA三维重建演示程序")
    print("="*80)
    
    # 示例：如何使用重建模块
    
    print("\n第一步：读取标定文件并转换格式")
    print("-" * 60)
    
    # 读取标定结果
    calib_data = CalibrationReader.read_calibration_file("标定结果.txt")
    
    # 假设相机图像尺寸（需要根据实际情况修改）
    camera_image_size = (2448, 2048)  # (width, height)
    projector_size = (1920, 1080)     # (width, height)
    
    # 保存为pyaoi格式
    CalibrationReader.save_pyaoi_calibration(
        calib_data,
        camera_output_path="camera_calib.npz",
        projector_output_path="projector_calib.npz",
        image_size=camera_image_size,
        projector_size=projector_size
    )
    
    print("\n第二步：加载视差图")
    print("-" * 60)
    
    # 这里需要根据实际路径修改
    # 假设视差图已经通过FSA解码得到
    print("提示：请确保已经运行FSA解码，生成了视差图文件")
    print("  - disp_hor.npy: 水平视差图")
    print("  - disp_ver.npy: 垂直视差图")
    print("  - peak_hor.npy: 水平峰值图")
    print("  - peak_ver.npy: 垂直峰值图")
    
    print("\n第三步：执行三维重建")
    print("-" * 60)
    print("示例代码:")
    print("""
    # 创建重建器
    reconstructor = FSAReconstructor(
        camera_calib_path="camera_calib.npz",
        projector_calib_path="projector_calib.npz",
        projector_width=1920,
        projector_height=1080
    )
    
    # 加载视差图
    disparity_hor = np.load("results/disp_hor.npy")
    disparity_ver = np.load("results/disp_ver.npy")
    peak_hor = np.load("results/peak_hor.npy")
    peak_ver = np.load("results/peak_ver.npy")
    
    # 加载白光图像
    white_pattern = cv2.imread("data/hor/0.png", cv2.IMREAD_GRAYSCALE)
    
    # 重建点云
    point_cloud, point_color = reconstructor.reconstruct_pointcloud(
        disparity_hor=disparity_hor,
        disparity_ver=disparity_ver,
        white_pattern=white_pattern,
        peak_limit=0.8,
        peak_hor=peak_hor,
        peak_ver=peak_ver,
        oversample=4
    )
    
    # 保存点云
    reconstructor.save_pointcloud(
        point_cloud, 
        point_color, 
        "pointcloud.ply"
    )
    
    # 可视化点云
    reconstructor.visualize_pointcloud(point_cloud, point_color)
    """)


if __name__ == "__main__":
    demonstrate_reconstruction()

