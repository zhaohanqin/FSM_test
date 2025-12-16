# -*- coding: utf-8 -*-
"""
FSM三维重建脚本 - 改进版本

该脚本用于：
1. 从.npy文件加载FSM解码后的视差图和峰值图
2. 从标定结果文件加载相机和投影仪标定参数
3. 使用三角测量方法进行三维重建
4. 生成点云并进行可视化
5. 支持基于峰值阈值的点云质量过滤

主要改进：
- 移除所有pyaoi依赖
- 直接从.npy文件读取数据
- 从标定结果文本文件解析标定参数
- 使用numpy进行三角测量计算
- 使用Open3D进行点云可视化和保存
"""

# %%
# 导入必要的库

import sys
import numpy as np                                    # 数值计算库
import cv2                                            # OpenCV图像处理库
import os                                             # 操作系统接口
import time                                           # 时间处理库
import matplotlib.pyplot as plt                       # 绘图库
import open3d as o3d                                  # Open3D点云处理库

# 配置 stdout 编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
	sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# 辅助函数
# ============================================================================

def parse_calibration_file(calib_file_path):
	"""
	从标定结果文本文件中解析标定参数
	
	参数：
		calib_file_path: 标定结果文件路径
	
	返回值：
		dict: 包含以下键的字典
			- cam_fx, cam_fy, cam_cx, cam_cy: 相机内参
			- cam_k1, cam_k2, cam_p1, cam_p2: 相机畸变系数
			- proj_fx, proj_fy, proj_cx, proj_cy: 投影仪内参
			- proj_k1, proj_k2, proj_p1, proj_p2: 投影仪畸变系数
			- R: 旋转矩阵 (3, 3)
			- t: 平移向量 (3,)
	"""
	calib_data = {}
	
	with open(calib_file_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	
	# 解析相机标定参数
	cam_params = []
	for i, line in enumerate(lines):
		if '相机标定结果' in line:
			# 读取后面的8行（fx, fy, cx, cy, k1, k2, p1, p2）
			for j in range(1, 9):
				cam_params.append(float(lines[i + j].strip()))
			break
	
	calib_data['cam_fx'] = cam_params[0]
	calib_data['cam_fy'] = cam_params[1]
	calib_data['cam_cx'] = cam_params[2]
	calib_data['cam_cy'] = cam_params[3]
	calib_data['cam_k1'] = cam_params[4]
	calib_data['cam_k2'] = cam_params[5]
	calib_data['cam_p1'] = cam_params[6]
	calib_data['cam_p2'] = cam_params[7]
	
	# 解析投影仪标定参数
	proj_params = []
	for i, line in enumerate(lines):
		if 'dlp标定结果' in line:
			# 读取后面的8行
			for j in range(1, 9):
				proj_params.append(float(lines[i + j].strip()))
			break
	
	calib_data['proj_fx'] = proj_params[0]
	calib_data['proj_fy'] = proj_params[1]
	calib_data['proj_cx'] = proj_params[2]
	calib_data['proj_cy'] = proj_params[3]
	calib_data['proj_k1'] = proj_params[4]
	calib_data['proj_k2'] = proj_params[5]
	calib_data['proj_p1'] = proj_params[6]
	calib_data['proj_p2'] = proj_params[7]
	
	# 解析旋转矩阵
	R = []
	for i, line in enumerate(lines):
		if '旋转矩阵：' in line:
			for j in range(1, 4):
				row = [float(x) for x in lines[i + j].split()]
				R.append(row)
			break
	calib_data['R'] = np.array(R)
	
	# 解析平移向量
	for i, line in enumerate(lines):
		if '平移矩阵：' in line:
			t = [float(x) for x in lines[i + 1].split()]
			calib_data['t'] = np.array(t)
			break
	
	return calib_data


def build_camera_matrix(fx, fy, cx, cy):
	"""
	构建相机内参矩阵
	
	参数：
		fx, fy: 焦距
		cx, cy: 主点
	
	返回值：
		camera_matrix: 3x3内参矩阵
	"""
	return np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	])


def build_distortion_coeffs(k1, k2, p1, p2):
	"""
	构建畸变系数向量
	
	参数：
		k1, k2: 径向畸变系数
		p1, p2: 切向畸变系数
	
	返回值：
		dist_coeffs: 5元素畸变系数向量
	"""
	return np.array([k1, k2, p1, p2, 0])


def triangulate_points_batch(camera_pixels, projector_pixels, calib_data):
	"""
	批量三角测量方法（使用OpenCV的cv2.triangulatePoints）
	
	参数：
		camera_pixels: 相机像素坐标，形状 (N, 2)
		projector_pixels: 投影仪像素坐标，形状 (N, 2)
		calib_data: 标定参数字典
	
	返回值：
		points_3d: 3D点坐标，形状 (N, 3)，在相机坐标系中
	"""
	# 构建相机内参矩阵
	camera_matrix = build_camera_matrix(
		calib_data['cam_fx'], calib_data['cam_fy'],
		calib_data['cam_cx'], calib_data['cam_cy']
	)
	
	# 构建投影仪内参矩阵
	projector_matrix = build_camera_matrix(
		calib_data['proj_fx'], calib_data['proj_fy'],
		calib_data['proj_cx'], calib_data['proj_cy']
	)
	
	# 构建畸变系数
	camera_dist = build_distortion_coeffs(
		calib_data['cam_k1'], calib_data['cam_k2'],
		calib_data['cam_p1'], calib_data['cam_p2']
	)
	
	projector_dist = build_distortion_coeffs(
		calib_data['proj_k1'], calib_data['proj_k2'],
		calib_data['proj_p1'], calib_data['proj_p2']
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
	R = calib_data['R'].astype(np.float64)
	t = calib_data['t'].astype(np.float64).reshape(3, 1)
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


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":

	# ========================================================================
	# 第一部分：参数配置
	# ========================================================================

	# 文件路径配置
	npy_data_dir = r"E:\code\Structured_light_for_3DScanning\FSM\results\fsa_decode"  # .npy文件目录
	calib_file = r"E:\code\Structured_light_for_3DScanning\FSM\标定结果.txt"  # 标定结果文件
	output_dir = r"E:\code\Structured_light_for_3DScanning\FSM\results\reconstruction"  # 输出目录

	# 创建输出目录
	os.makedirs(output_dir, exist_ok=True)

	# 参数配置
	oversample = 17  # 过采样倍数（与解码时的过采样倍数必须一致）
	peak_limit = 10.0  # 峰值阈值（注意：峰值范围是0-200，不是0-1），用于过滤低质量的点

	print("=" * 70)
	print("FSM三维重建 - 改进版本")
	print("=" * 70)

	# ========================================================================
	# 第二部分：数据源选择
	# ========================================================================

	print("\n[0/5] 数据源选择...")
	
	# 检查是否存在点集文件
	cam_pixels_file = r"E:\code\Structured_light_for_3DScanning\FSM\cam_pixels_finally.npy"
	proj_pixels_file = r"E:\code\Structured_light_for_3DScanning\FSM\proj_pixels_finally.npy"
	
	files_exist = os.path.exists(cam_pixels_file) and os.path.exists(proj_pixels_file)
	
	if files_exist:
		print(f"✓ 检测到点集文件:")
		print(f"  - {cam_pixels_file}")
		print(f"  - {proj_pixels_file}")
		print()
		print("请选择数据源:")
		print("  1. 使用保存的点集文件 (推荐)")
		print("  2. 使用解码结果 (disp_hor.npy, disp_ver.npy等)")
		print()
		choice = input("请输入选择 (1 或 2，默认为 1): ").strip()
		use_saved_pixels = (choice != '2')
	else:
		print(f"未检测到点集文件，将使用解码结果")
		use_saved_pixels = False

	# ========================================================================
	# 第三部分：加载数据
	# ========================================================================

	print("\n[1/5] 加载数据...")
	tstart = time.time()

	if use_saved_pixels:
		# 加载保存的点集文件
		print("从点集文件加载...")
		cam_pixels = np.load(cam_pixels_file)  # 形状: (N, 1, 2)
		proj_pixels = np.load(proj_pixels_file)  # 形状: (N, 1, 2)
		
		# 转换为 (N, 2) 格式
		camera_pixels = cam_pixels.reshape(-1, 2)
		projector_pixels = proj_pixels.reshape(-1, 2)
		
		print(f"  - 已加载相机像素坐标: {camera_pixels.shape}")
		print(f"  - 已加载投影仪像素坐标: {projector_pixels.shape}")
		print(f"  - 点数: {len(camera_pixels)}")
		
		# 创建有效性掩码（所有点都有效）
		invalid_mask = np.zeros(camera_pixels.shape[0], dtype=bool)
		valid_count = len(camera_pixels)
		total_count = len(camera_pixels)
		
	else:
		# 从.npy文件加载视差图和峰值图
		disparity_hor = np.load(os.path.join(npy_data_dir, "disp_hor.npy"))  # 水平视差图（行方向）
		disparity_ver = np.load(os.path.join(npy_data_dir, "disp_ver.npy"))  # 竖直视差图（列方向）
		peak_hor = np.load(os.path.join(npy_data_dir, "peak_hor.npy"))  # 水平方向峰值
		peak_ver = np.load(os.path.join(npy_data_dir, "peak_ver.npy"))  # 竖直方向峰值

		# 视差图缩放：乘以过采样倍数以恢复原始分辨率
		disparity_hor = np.round(disparity_hor * oversample).astype(np.int32)
		disparity_ver = np.round(disparity_ver * oversample).astype(np.int32)

		# 计算综合置信度（取两个方向的最小值）
		peak_min = np.minimum(peak_hor, peak_ver)

		# 生成有效性掩码
		invalid_mask = (peak_min < peak_limit)
		valid_count = np.sum(~invalid_mask)
		total_count = invalid_mask.size

		print(f"  - 图像分辨率: {disparity_hor.shape}")
		print(f"  - 视差图范围:")
		print(f"    * disp_hor: [{disparity_hor.min()}, {disparity_hor.max()}]")
		print(f"    * disp_ver: [{disparity_ver.min()}, {disparity_ver.max()}]")
		print(f"  - 峰值范围:")
		print(f"    * peak_hor: [{peak_hor.min():.2f}, {peak_hor.max():.2f}]")
		print(f"    * peak_ver: [{peak_ver.min():.2f}, {peak_ver.max():.2f}]")
		print(f"  - 有效点数: {valid_count} / {total_count} ({100*valid_count/total_count:.2f}%)")
		print(f"  - 加载时间: {time.time() - tstart:.2f}s")

	# ========================================================================
	# 第四部分：加载标定参数
	# ========================================================================

	print("\n[2/5] 加载标定参数...")
	tstart = time.time()

	calib_data = parse_calibration_file(calib_file)

	print(f"  - 相机内参: fx={calib_data['cam_fx']:.2f}, fy={calib_data['cam_fy']:.2f}")
	print(f"              cx={calib_data['cam_cx']:.2f}, cy={calib_data['cam_cy']:.2f}")
	print(f"  - 投影仪内参: fx={calib_data['proj_fx']:.2f}, fy={calib_data['proj_fy']:.2f}")
	print(f"                cx={calib_data['proj_cx']:.2f}, cy={calib_data['proj_cy']:.2f}")
	print(f"  - 加载时间: {time.time() - tstart:.2f}s")

	# ========================================================================
	# 第四部分：三维重建
	# ========================================================================

	print("\n[3/5] 进行三维重建...")
	tstart = time.time()

	if use_saved_pixels:
		# 使用保存的点集文件
		num_valid = len(camera_pixels)
		print(f"  - 有效点数: {num_valid}")
		print(f"  - 相机像素范围: X=[{camera_pixels[:, 0].min():.1f}, {camera_pixels[:, 0].max():.1f}], Y=[{camera_pixels[:, 1].min():.1f}, {camera_pixels[:, 1].max():.1f}]")
		print(f"  - 投影仪像素范围: X=[{projector_pixels[:, 0].min():.1f}, {projector_pixels[:, 0].max():.1f}], Y=[{projector_pixels[:, 1].min():.1f}, {projector_pixels[:, 1].max():.1f}]")
		
		# 批量进行三角测量
		print("  - 执行批量三角测量...")
		points_3d = triangulate_points_batch(camera_pixels, projector_pixels, calib_data)
		
		# 使用均匀颜色（因为没有峰值信息）
		point_colors = np.full((len(points_3d), 3), 128, dtype=np.uint8)
		
	else:
		# 从视差图提取像素坐标
		# 获取有效像素的索引
		valid_y, valid_x = np.where(~invalid_mask)
		num_valid = len(valid_y)

		if num_valid == 0:
			print("  - 错误：没有有效的点可以重建")
			exit(1)

		print(f"  - 有效点数: {num_valid}")

		# 准备相机和投影仪的像素坐标
		# 注意：相机坐标是 (x, y) = (列, 行)
		# 投影仪坐标也应该是 (x, y) = (列, 行)
		# disp_hor 是水平视差，对应y方向（行）
		# disp_ver 是竖直视差，对应x方向（列）
		camera_pixels = np.column_stack([valid_x, valid_y])  # (x, y) 格式
		projector_pixels = np.column_stack([
			disparity_ver[valid_y, valid_x] / oversample,  # 投影仪x坐标（列）
			disparity_hor[valid_y, valid_x] / oversample   # 投影仪y坐标（行）
		])

		print(f"  - 相机像素范围: X=[{valid_x.min()}, {valid_x.max()}], Y=[{valid_y.min()}, {valid_y.max()}]")
		print(f"  - 投影仪像素范围: X=[{projector_pixels[:, 0].min():.1f}, {projector_pixels[:, 0].max():.1f}], Y=[{projector_pixels[:, 1].min():.1f}, {projector_pixels[:, 1].max():.1f}]")

		# 批量进行三角测量
		print("  - 执行批量三角测量...")
		points_3d = triangulate_points_batch(camera_pixels, projector_pixels, calib_data)

		# 注意：cv2.triangulatePoints 返回的是相机坐标系中的点
		# 直接使用相机坐标系的点，不需要进行坐标系转换
		# 如果需要转换到投影仪坐标系，使用以下公式：
		# points_3d_projector = (R.T @ (points_3d - t).T).T
		# 但这会导致条纹状分层，所以保持在相机坐标系中

		# 准备颜色信息
		peak_min_valid = peak_min[valid_y, valid_x]
		peak_min_max = np.max(peak_min_valid) if len(peak_min_valid) > 0 else 1.0
		peak_min_min = np.min(peak_min_valid) if len(peak_min_valid) > 0 else 0.0

		# 线性映射峰值到[0, 255]范围
		if peak_min_max > peak_min_min:
			normalized_peak = (peak_min_valid - peak_min_min) / (peak_min_max - peak_min_min)
		else:
			normalized_peak = np.full_like(peak_min_valid, 0.5)

		intensity = np.clip(normalized_peak * 255, 0, 255).astype(np.uint8)
		point_colors = np.column_stack([intensity, intensity, intensity])

	point_cloud = points_3d
	print(f"  - 重建点数: {len(point_cloud)}")
	print(f"  - 重建时间: {time.time() - tstart:.2f}s")

	# ========================================================================
	# 第五部分：点云可视化和保存
	# ========================================================================

	print("\n[4/5] 保存点云...")
	tstart = time.time()

	# 创建Open3D点云对象
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(point_cloud)
	pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)

	# 保存为PLY格式
	output_file = os.path.join(output_dir, f"pointcloud_reconstructed_{peak_limit:.2f}.ply")
	o3d.io.write_point_cloud(output_file, pcd)

	print(f"  - 保存位置: {output_file}")
	print(f"  - 保存时间: {time.time() - tstart:.2f}s")

	# ========================================================================
	# 第六部分：点云可视化
	# ========================================================================

	print("\n[5/5] 点云可视化...")
	print("  - 正在打开Open3D可视化窗口...")
	print("  - 操作说明:")
	print("    * 鼠标左键: 旋转")
	print("    * 鼠标右键: 平移")
	print("    * 鼠标滚轮: 缩放")
	print("    * 按 Q 关闭窗口")

	# 使用Open3D可视化点云
	o3d.visualization.draw_geometries([pcd])

	print("\n" + "=" * 70)
	print("重建完成！")
	print("=" * 70)
