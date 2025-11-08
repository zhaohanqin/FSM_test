"""
FSA三维重建脚本

直接运行此脚本即可完成三维重建：
1. 读取标定结果
2. 加载解码后的视差图
3. 执行三维重建
4. 保存和可视化点云

使用前提：
- 已完成相机和投影仪的标定（标定结果.txt）
- 已完成FSA解码，获得视差图（disp_hor.npy, disp_ver.npy等）
"""

import numpy as np
import cv2
import os
from fsa_reconstruct_core import CalibrationReader, FSAReconstructor


if __name__ == "__main__":
    print("="*80)
    print("FSA 三维重建")
    print("="*80)
    
    # ========== 配置参数 ==========
    # 尝试从 constants.py 导入路径配置（如果存在）
    try:
        from constants import fsa_result_path, fsa_data_path, W, H, oversample, peak_limit
        RESULT_DIR = fsa_result_path
        DATA_PATH = fsa_data_path
        PROJECTOR_WIDTH = W
        PROJECTOR_HEIGHT = H
        OVERSAMPLE = oversample
        PEAK_LIMIT = peak_limit
        print(f"✓ 从 constants.py 加载配置")
    except ImportError:
        # 使用默认配置
        RESULT_DIR = "results/fsa_decode"  # 存放解码结果的文件夹
        DATA_PATH = "data"
        PROJECTOR_WIDTH = 1920
        PROJECTOR_HEIGHT = 1080
        OVERSAMPLE = 4
        PEAK_LIMIT = 0.8
        print("⚠ 未找到 constants.py，使用默认配置")
    
    # 标定文件路径
    CALIBRATION_FILE = "标定结果.txt"
    
    # 视差图文件路径（FSA解码结果）
    DISPARITY_HOR_FILE = os.path.join(RESULT_DIR, "disp_hor.npy")
    DISPARITY_VER_FILE = os.path.join(RESULT_DIR, "disp_ver.npy")
    PEAK_HOR_FILE = os.path.join(RESULT_DIR, "peak_hor.npy")
    PEAK_VER_FILE = os.path.join(RESULT_DIR, "peak_ver.npy")
    
    # 白光图像路径（用于提取点云颜色）
    # 暂时不使用颜色，注释掉此部分
    # # 尝试多个可能的路径
    # possible_white_patterns = [
    #     os.path.join(DATA_PATH, "hor", "0.png"),
    #     os.path.join(DATA_PATH, "hor", "0.bmp"),
    #     os.path.join("data", "hor", "0.png"),
    #     os.path.join(RESULT_DIR, "white_pattern.png"),
    #     os.path.join(RESULT_DIR, "white_pattern.bmp"),
    # ]
    # 
    # WHITE_PATTERN_FILE = None
    # for pattern_path in possible_white_patterns:
    #     if os.path.exists(pattern_path):
    #         WHITE_PATTERN_FILE = pattern_path
    #         break
    
    # 相机和投影仪参数
    try:
        from constants import CAMERA_IMAGE_SIZE
    except ImportError:
        CAMERA_IMAGE_SIZE = (2448, 2048)  # 相机图像尺寸 (宽, 高) - 请根据实际情况修改
    
    PROJECTOR_SIZE = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)  # 投影仪分辨率 (宽, 高)
    
    # 输出文件
    OUTPUT_PLY = "pointcloud_fsa.ply"
    
    # ========== 步骤1: 读取标定文件并转换格式 ==========
    print("\n" + "="*80)
    print("步骤1: 读取标定文件")
    print("="*80)
    
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ 错误: 标定文件不存在: {CALIBRATION_FILE}")
        print("   请确保标定文件存在")
        exit(1)
    
    # 读取标定结果
    calib_data = CalibrationReader.read_calibration_file(CALIBRATION_FILE)
    
    # 转换并保存为pyaoi格式
    camera_calib_path = "camera_calib.npz"
    projector_calib_path = "projector_calib.npz"
    
    CalibrationReader.save_pyaoi_calibration(
        calib_data,
        camera_output_path=camera_calib_path,
        projector_output_path=projector_calib_path,
        image_size=CAMERA_IMAGE_SIZE,
        projector_size=PROJECTOR_SIZE
    )
    
    # ========== 步骤2: 加载视差图 ==========
    print("\n" + "="*80)
    print("步骤2: 加载视差图")
    print("="*80)
    
    # 检查文件是否存在
    required_files = {
        "水平视差图": DISPARITY_HOR_FILE,
        "垂直视差图": DISPARITY_VER_FILE,
        "水平峰值图": PEAK_HOR_FILE,
        "垂直峰值图": PEAK_VER_FILE,
    }
    
    missing_files = {name: path for name, path in required_files.items() if not os.path.exists(path)}
    
    if missing_files:
        print("❌ 错误: 以下文件不存在:")
        for name, path in missing_files.items():
            print(f"   - {name}: {path}")
        print("\n提示: 请先运行FSA解码程序生成视差图")
        exit(1)
    
    # 检查白光图像（暂时不使用颜色，注释掉此部分）
    # if WHITE_PATTERN_FILE is None or not os.path.exists(WHITE_PATTERN_FILE):
    #     print("⚠ 警告: 未找到白光图像，将使用灰度图代替")
    #     print("  尝试过的路径:")
    #     for pattern_path in possible_white_patterns:
    #         print(f"   - {pattern_path}")
    #     # 创建一个全白图像作为替代
    #     # 我们需要先加载一个视差图来获取图像尺寸
    #     temp_disp = np.load(DISPARITY_HOR_FILE)
    #     white_pattern = np.ones(temp_disp.shape, dtype=np.uint8) * 128
    #     WHITE_PATTERN_FILE = "temp_white_pattern.png"
    #     cv2.imwrite(WHITE_PATTERN_FILE, white_pattern)
    #     print(f"  已创建临时白光图像: {WHITE_PATTERN_FILE}")
    # else:
    #     print(f"✓ 找到白光图像: {WHITE_PATTERN_FILE}")
    
    # 加载视差图
    print("加载视差图...")
    disparity_hor = np.load(DISPARITY_HOR_FILE)
    disparity_ver = np.load(DISPARITY_VER_FILE)
    peak_hor = np.load(PEAK_HOR_FILE)
    peak_ver = np.load(PEAK_VER_FILE)
    
    print(f"✓ 水平视差图: {disparity_hor.shape}")
    print(f"✓ 垂直视差图: {disparity_ver.shape}")
    print(f"✓ 水平峰值图: {peak_hor.shape}")
    print(f"✓ 垂直峰值图: {peak_ver.shape}")
    
    # 加载白光图像（暂时不使用颜色，注释掉此部分）
    # print(f"\n加载白光图像: {WHITE_PATTERN_FILE}")
    # white_pattern = cv2.imread(WHITE_PATTERN_FILE, cv2.IMREAD_GRAYSCALE)
    # if white_pattern is None:
    #     # 如果OpenCV无法读取（可能是中文路径问题），尝试使用numpy
    #     try:
    #         img_array = np.fromfile(WHITE_PATTERN_FILE, dtype=np.uint8)
    #         white_pattern = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    #     except Exception as e:
    #         print(f"❌ 错误: 无法读取白光图像: {e}")
    #         exit(1)
    # 
    # print(f"✓ 白光图像: {white_pattern.shape}")
    
    # 创建一个简单的占位符图像（用于重建函数，但不提取颜色）
    print("\n创建占位符图像（不提取颜色）...")
    white_pattern = np.ones(disparity_hor.shape, dtype=np.uint8) * 255  # 全白图像
    print(f"✓ 占位符图像: {white_pattern.shape}")
    
    # ========== 步骤3: 创建重建器 ==========
    print("\n" + "="*80)
    print("步骤3: 创建三维重建器")
    print("="*80)
    
    try:
        reconstructor = FSAReconstructor(
            camera_calib_path=camera_calib_path,
            projector_calib_path=projector_calib_path,
            projector_width=PROJECTOR_SIZE[0],
            projector_height=PROJECTOR_SIZE[1]
        )
    except Exception as e:
        print(f"❌ 错误: 创建重建器失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # ========== 步骤4: 执行三维重建 ==========
    print("\n" + "="*80)
    print("步骤4: 执行三维重建")
    print("="*80)
    
    try:
        point_cloud, point_color = reconstructor.reconstruct_pointcloud(
            disparity_hor=disparity_hor,
            disparity_ver=disparity_ver,
            white_pattern=white_pattern,
            peak_limit=PEAK_LIMIT,
            peak_hor=peak_hor,
            peak_ver=peak_ver,
            oversample=OVERSAMPLE,
            min_distance=None,  # 可选：设置距离范围过滤
            max_distance=None
        )
    except Exception as e:
        print(f"❌ 重建失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # ========== 步骤5: 保存点云 ==========
    print("\n" + "="*80)
    print("步骤5: 保存点云")
    print("="*80)
    
    reconstructor.save_pointcloud(
        point_cloud,
        point_color,
        OUTPUT_PLY
    )
    
    # 输出点云统计信息
    print("\n点云统计信息:")
    print(f"  点云数量: {len(point_cloud):,}")
    print(f"  点云尺寸范围:")
    print(f"    X轴: [{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}] mm")
    print(f"    Y轴: [{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}] mm")
    print(f"    Z轴: [{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}] mm")
    
    # ========== 步骤6: 可视化点云 ==========
    print("\n" + "="*80)
    print("步骤6: 可视化点云")
    print("="*80)
    
    try:
        reconstructor.visualize_pointcloud(
            point_cloud,
            point_color,
            window_name="FSA 三维重建结果"
        )
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        print("\n提示: 请确保已安装Open3D库")
        print("  pip install open3d")
    
    print("\n" + "="*80)
    print("重建完成！")
    print("="*80)
    print(f"点云文件已保存: {OUTPUT_PLY}")
    print("\n提示: 可以使用以下软件打开PLY文件:")
    print("  - CloudCompare (推荐)")
    print("  - MeshLab")
    print("  - Open3D")

