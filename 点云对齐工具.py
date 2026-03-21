# -*- coding: utf-8 -*-
"""
频移法（FSM）与相移法（PSM）点云坐标对齐工具

问题背景：
---------
频移法（FSM_reconstruct_core.py）和相移法（三维重建_finally1.py）在重建流程中使用了相同的
标定参数和三角测量方法（cv2.triangulatePoints），但最终输出的点云坐标系不同：

  - 相移法（PSM）：输出点云在 **相机坐标系** 下
    → triangulatePoints 得到的结果直接输出，不做坐标变换

  - 频移法（FSM）：输出点云在 **投影仪坐标系** 下
    → triangulatePoints 得到的结果经过了一次额外的刚体变换：
      P_proj = R^T @ (P_cam - T)
    将点从相机坐标系转换到了投影仪坐标系

这个差异导致两个点云在三维空间中产生了前后偏移（偏移量近似等于平移向量T的模长）。

本脚本功能：
-----------
1. 读取频移法和相移法各自生成的 PLY 点云文件
2. 将频移法点云从投影仪坐标系变换回相机坐标系（或反过来）
3. 输出对齐后的点云

使用方法：
---------
方式一：将频移法点云（投影仪坐标系）变换到相机坐标系，与相移法对齐
    python align_pointclouds.py --fsm pointcloud_fsa.ply --psm pointCloud.ply --calib 标定结果.txt --mode fsm_to_camera

方式二：将相移法点云（相机坐标系）变换到投影仪坐标系，与频移法对齐
    python align_pointclouds.py --fsm pointcloud_fsa.ply --psm pointCloud.ply --calib 标定结果.txt --mode psm_to_projector

方式三：同时输出两种对齐后的结果
    python align_pointclouds.py --fsm pointcloud_fsa.ply --psm pointCloud.ply --calib 标定结果.txt --mode both

作者：基于 FSM_reconstruct_core.py 和 三维重建_finally1.py 的坐标系分析
日期：2025
"""

import numpy as np
import argparse
import os
import re
import sys


# =============================================================================
# 1. 标定参数读取
# =============================================================================
def read_calibration(filepath):
    """
    读取标定结果文件，提取旋转矩阵 R 和平移向量 T
    
    这两个参数定义了投影仪相对于相机的外参：
      P_projector = R @ P_camera + T
    即：相机坐标系下的点 P_camera，经过 R, T 变换后得到投影仪坐标系下的点。
    
    但在 FSM_reconstruct_core.py 中使用的是反向变换：
      P_projector_frame = R^T @ (P_camera_frame - T)
    
    Parameters:
        filepath: 标定结果文件路径
        
    Returns:
        R: 旋转矩阵 (3x3)
        T: 平移向量 (3,)
    """
    print(f"读取标定文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    R = None
    T = None
    
    lines_iter = iter(lines)
    for line in lines_iter:
        if "旋转矩阵" in line:
            matrix_lines = [next(lines_iter).strip() for _ in range(3)]
            R = np.array([list(map(float, re.split(r'\s+', l.strip()))) for l in matrix_lines])
        elif "平移矩阵" in line:
            values = next(lines_iter).strip()
            T = np.array(list(map(float, re.split(r'\s+', values))))
    
    if R is None or T is None:
        raise ValueError("标定文件中未找到旋转矩阵或平移向量")
    
    print(f"  旋转矩阵 R:\n{R}")
    print(f"  平移向量 T: {T}")
    print(f"  平移向量模长: {np.linalg.norm(T):.2f} mm")
    print(f"  → 这个模长近似等于两个点云之间的偏移距离\n")
    
    return R, T


# =============================================================================
# 2. PLY 文件读写（不依赖 open3d）
# =============================================================================
def read_ply(filepath):
    """
    读取 PLY 点云文件
    
    Returns:
        points: 点坐标 (N, 3)
        colors: 颜色 (N, 3) 或 None
    """
    print(f"读取点云: {filepath}")
    
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        print(f"  点数: {len(points)}")
        if len(points) > 0:
            print(f"  X 范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y 范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z 范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        return points, colors
        
    except ImportError:
        print("  (未安装 open3d，使用简易 PLY 读取)")
        return _read_ply_simple(filepath)


def _read_ply_simple(filepath):
    """简易 PLY 读取（纯文本 ASCII 格式）"""
    with open(filepath, 'r') as f:
        header_lines = []
        while True:
            line = f.readline().strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # 解析顶点数
        num_vertices = 0
        has_color = False
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if 'red' in line or 'green' in line or 'blue' in line:
                has_color = True
        
        points = []
        colors = []
        for _ in range(num_vertices):
            parts = f.readline().strip().split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_color and len(parts) >= 6:
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    
    points = np.array(points)
    colors = np.array(colors) / 255.0 if colors else None
    
    print(f"  点数: {len(points)}")
    return points, colors


def write_ply(filepath, points, colors=None):
    """保存 PLY 点云文件"""
    print(f"保存点云: {filepath}")
    
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filepath, pcd)
        
    except ImportError:
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if colors is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                x, y, z = points[i]
                if colors is not None:
                    r, g, b = (colors[i] * 255).astype(np.uint8) if colors[i].max() <= 1.0 else colors[i].astype(np.uint8)
                    f.write(f"{x} {y} {z} {r} {g} {b}\n")
                else:
                    f.write(f"{x} {y} {z}\n")
    
    print(f"  已保存 {len(points)} 个点\n")


# =============================================================================
# 3. 坐标变换函数
# =============================================================================
def projector_to_camera(points, R, T):
    """
    将投影仪坐标系下的点变换到相机坐标系
    
    FSM 代码中从相机到投影仪的变换为：
        P_proj = R^T @ (P_cam - T)
    
    其逆变换（投影仪到相机）为：
        P_cam = R @ P_proj + T
    
    Parameters:
        points: (N, 3) 投影仪坐标系下的点
        R: (3, 3) 旋转矩阵
        T: (3,) 平移向量
        
    Returns:
        (N, 3) 相机坐标系下的点
    """
    # P_cam = R @ P_proj + T
    return (R @ points.T).T + T.flatten()


def camera_to_projector(points, R, T):
    """
    将相机坐标系下的点变换到投影仪坐标系
    
    这是 FSM_reconstruct_core.py 第1028行的变换：
        P_proj = R^T @ (P_cam - T)
    
    Parameters:
        points: (N, 3) 相机坐标系下的点
        R: (3, 3) 旋转矩阵
        T: (3,) 平移向量
        
    Returns:
        (N, 3) 投影仪坐标系下的点
    """
    # P_proj = R^T @ (P_cam - T)
    return (R.T @ (points - T.flatten()).T).T


# =============================================================================
# 4. 主流程
# =============================================================================
def align_pointclouds(fsm_ply, psm_ply, calib_file, mode='fsm_to_camera', output_dir='.'):
    """
    对齐频移法和相移法的点云
    
    Parameters:
        fsm_ply: 频移法点云路径（投影仪坐标系）
        psm_ply: 相移法点云路径（相机坐标系）
        calib_file: 标定文件路径
        mode: 对齐模式
            'fsm_to_camera'    - 将 FSM 点云从投影仪坐标系变换到相机坐标系
            'psm_to_projector' - 将 PSM 点云从相机坐标系变换到投影仪坐标系
            'both'             - 同时输出两种结果
        output_dir: 输出目录
    """
    print("=" * 70)
    print("频移法 / 相移法 点云坐标对齐工具")
    print("=" * 70)
    print()
    
    # 读取标定参数
    R, T = read_calibration(calib_file)
    
    # 读取点云
    print("-" * 70)
    fsm_points, fsm_colors = read_ply(fsm_ply)
    print()
    psm_points, psm_colors = read_ply(psm_ply)
    print()
    
    # 诊断：显示两个点云质心的偏差
    if len(fsm_points) > 0 and len(psm_points) > 0:
        fsm_center = fsm_points.mean(axis=0)
        psm_center = psm_points.mean(axis=0)
        print("-" * 70)
        print("对齐前诊断:")
        print(f"  FSM 点云质心（投影仪坐标系）: [{fsm_center[0]:.2f}, {fsm_center[1]:.2f}, {fsm_center[2]:.2f}]")
        print(f"  PSM 点云质心（相机坐标系）  : [{psm_center[0]:.2f}, {psm_center[1]:.2f}, {psm_center[2]:.2f}]")
        print(f"  质心间距: {np.linalg.norm(fsm_center - psm_center):.2f} mm")
        print(f"  平移向量 T 模长: {np.linalg.norm(T):.2f} mm")
        print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行对齐
    if mode in ('fsm_to_camera', 'both'):
        print("=" * 70)
        print("模式: 将 FSM 点云从投影仪坐标系 → 相机坐标系")
        print("  变换公式: P_cam = R @ P_proj + T")
        print("=" * 70)
        
        fsm_in_camera = projector_to_camera(fsm_points, R, T)
        
        out_path = os.path.join(output_dir, 'fsm_aligned_camera.ply')
        write_ply(out_path, fsm_in_camera, fsm_colors)
        
        # 诊断对齐结果
        if len(psm_points) > 0:
            fsm_cam_center = fsm_in_camera.mean(axis=0)
            psm_center = psm_points.mean(axis=0)
            print(f"  对齐后 FSM 质心: [{fsm_cam_center[0]:.2f}, {fsm_cam_center[1]:.2f}, {fsm_cam_center[2]:.2f}]")
            print(f"  PSM 质心:         [{psm_center[0]:.2f}, {psm_center[1]:.2f}, {psm_center[2]:.2f}]")
            print(f"  对齐后质心间距: {np.linalg.norm(fsm_cam_center - psm_center):.2f} mm")
            print()
        
        # 同时拷贝 PSM 点云到输出目录（方便对比）
        psm_copy_path = os.path.join(output_dir, 'psm_original_camera.ply')
        write_ply(psm_copy_path, psm_points, psm_colors)
        
        print(f"★ 对齐完成！请将以下两个文件加载到同一个点云查看器中对比：")
        print(f"  1. {out_path}  （FSM，已变换到相机坐标系）")
        print(f"  2. {psm_copy_path}  （PSM，原始相机坐标系）")
        print()
    
    if mode in ('psm_to_projector', 'both'):
        print("=" * 70)
        print("模式: 将 PSM 点云从相机坐标系 → 投影仪坐标系")
        print("  变换公式: P_proj = R^T @ (P_cam - T)")
        print("=" * 70)
        
        psm_in_projector = camera_to_projector(psm_points, R, T)
        
        out_path = os.path.join(output_dir, 'psm_aligned_projector.ply')
        write_ply(out_path, psm_in_projector, psm_colors)
        
        # 诊断
        if len(fsm_points) > 0:
            psm_proj_center = psm_in_projector.mean(axis=0)
            fsm_center = fsm_points.mean(axis=0)
            print(f"  对齐后 PSM 质心: [{psm_proj_center[0]:.2f}, {psm_proj_center[1]:.2f}, {psm_proj_center[2]:.2f}]")
            print(f"  FSM 质心:         [{fsm_center[0]:.2f}, {fsm_center[1]:.2f}, {fsm_center[2]:.2f}]")
            print(f"  对齐后质心间距: {np.linalg.norm(psm_proj_center - fsm_center):.2f} mm")
            print()
        
        # 同时拷贝 FSM 点云到输出目录
        fsm_copy_path = os.path.join(output_dir, 'fsm_original_projector.ply')
        write_ply(fsm_copy_path, fsm_points, fsm_colors)
        
        print(f"★ 对齐完成！请将以下两个文件加载到同一个点云查看器中对比：")
        print(f"  1. {fsm_copy_path}  （FSM，原始投影仪坐标系）")
        print(f"  2. {out_path}  （PSM，已变换到投影仪坐标系）")
        print()
    
    print("=" * 70)
    print("完成！")
    print("=" * 70)


# =============================================================================
# 5. 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="频移法（FSM）与相移法（PSM）点云坐标对齐工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 将 FSM 点云变换到相机坐标系（推荐）
  python align_pointclouds.py --fsm pointcloud_fsa3.ply --psm pointCloud.ply --calib 标定结果.txt

  # 将 PSM 点云变换到投影仪坐标系
  python align_pointclouds.py --fsm pointcloud_fsa.ply --psm pointCloud.ply --calib 标定结果.txt --mode psm_to_projector

  # 同时输出两种对齐结果
  python align_pointclouds.py --fsm pointcloud_fsa.ply --psm pointCloud.ply --calib 标定结果.txt --mode both

原因说明:
  频移法代码（FSM_reconstruct_core.py 第1025-1028行）在三角测量之后，
  额外执行了一次从相机坐标系到投影仪坐标系的刚体变换：
    points_3d_projector = (R^T @ (points_3d_camera - T)).T
  而相移法代码（三维重建_finally1.py）直接输出三角测量的结果（相机坐标系）。
  这导致两个点云的原点和坐标轴定义不同，产生了系统性的前后偏移。
"""
    )
    
    parser.add_argument('--fsm', required=True, help='频移法点云文件路径（PLY格式）')
    parser.add_argument('--psm', required=True, help='相移法点云文件路径（PLY格式）')
    parser.add_argument('--calib', required=True, help='标定结果文件路径')
    parser.add_argument('--mode', default='fsm_to_camera',
                        choices=['fsm_to_camera', 'psm_to_projector', 'both'],
                        help='对齐模式（默认: fsm_to_camera）')
    parser.add_argument('--output', default='./aligned_output', help='输出目录（默认: ./aligned_output）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for path, name in [(args.fsm, 'FSM点云'), (args.psm, 'PSM点云'), (args.calib, '标定文件')]:
        if not os.path.exists(path):
            print(f"错误: {name}文件不存在: {path}")
            sys.exit(1)
    
    align_pointclouds(
        fsm_ply=args.fsm,
        psm_ply=args.psm,
        calib_file=args.calib,
        mode=args.mode,
        output_dir=args.output
    )
