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
from FSM_reconstruct_core import FSAReconstructor


if __name__ == "__main__":
    """
    ============================================================================
    📋 配置说明：所有可配置参数都在下方，修改后直接运行即可
    ============================================================================
    """
    
    # ========================================================================
    # 🔧 配置参数区域 - 在这里修改所有参数
    # ========================================================================
    
    # -------------------- 1. 标定文件路径 --------------------
    CALIBRATION_FILE = "标定结果.txt"  # 相机-投影仪标定结果文件
    
    # -------------------- 2. 相机参数 --------------------
    CAMERA_IMAGE_SIZE = (2448, 2048)   # 相机图像尺寸 (宽, 高)
    
    # -------------------- 3. 投影仪参数 --------------------
    PROJECTOR_WIDTH = 1920             # 投影仪宽度（像素）
    PROJECTOR_HEIGHT = 1080            # 投影仪高度（像素）
    
    # -------------------- 4. 单方向模式配置 --------------------
    # 新功能：支持单方向输入，只需提供一个方向的视差图
    # 通过基础矩阵自动计算另一方向
    USE_SINGLE_DIRECTION = True  # 是否使用单方向模式（通过基础矩阵计算另一方向）
    PRIMARY_DIRECTION = 'vertical'  # 单方向模式下的主方向：'horizontal' 或 'vertical'
                                     # 建议使用 'vertical'，因为垂直方向包含更多深度信息
                                       # 仅在 USE_SINGLE_DIRECTION=True 时有效
    # 说明：
    #   - USE_SINGLE_DIRECTION=False: 传统双方向模式，需要水平和垂直两个视差图
    #   - USE_SINGLE_DIRECTION=True:  单方向模式，只需一个方向的视差图
    #     * PRIMARY_DIRECTION='horizontal': 只需 disp_hor.npy 和 peak_hor.npy
    #     * PRIMARY_DIRECTION='vertical':   只需 disp_ver.npy 和 peak_ver.npy
    
    # -------------------- 5. 数据路径配置 --------------------
    RESULT_DIR = "results/fsa_decode"  # 存放解码结果的文件夹
    
    # -------------------- 6. 重建参数 --------------------
    OVERSAMPLE = 4                     # 过采样倍数（与解码时一致）
    PEAK_LIMIT = 1                   # 峰值阈值（范围：0-1）
                                       # 说明：越大过滤越严格，建议值：
                                       #   - 0.5: 包含中等质量以上的点
                                       #   - 0.8: 只保留高质量的点（推荐）
                                       #   - 1.0: 只保留最高质量的点
    
    # -------------------- 7. 白光图像路径（可选，用于点云颜色）--------------------
    # 用途：提供一张白光照明下拍摄的图像，用于给点云赋予真实颜色
    # 格式：图像文件路径（字符串）或 None
    # 
    # 示例：
    #   WHITE_IMAGE_PATH = "white_light.png"           # 使用白光图像的颜色
    #   WHITE_IMAGE_PATH = r"F:\data\white.png"       # 使用绝对路径
    #   WHITE_IMAGE_PATH = None                        # 不使用颜色，点云为灰色
    #
    WHITE_IMAGE_PATH = None            # 设为None则点云使用默认灰色
    
    # -------------------- 8. 距离过滤参数（可选）--------------------
    MIN_DISTANCE = 500                # 最小距离限制（mm），None表示不限制
    MAX_DISTANCE = 2000                # 最大距离限制（mm），None表示不限制
    # 示例：
    #   MIN_DISTANCE = 500             # 过滤掉距离小于500mm的点
    #   MAX_DISTANCE = 2000            # 过滤掉距离大于2000mm的点
    
    # -------------------- 9. 输入文件路径（自动生成）--------------------
    # 视差图文件路径（FSA解码结果）
    DISPARITY_HOR_FILE = os.path.join(RESULT_DIR, "disp_hor.npy")
    DISPARITY_VER_FILE = os.path.join(RESULT_DIR, "disp_ver.npy")
    PEAK_HOR_FILE = os.path.join(RESULT_DIR, "peak_hor.npy")
    PEAK_VER_FILE = os.path.join(RESULT_DIR, "peak_ver.npy")
    
    # -------------------- 10. 输出文件路径 --------------------
    OUTPUT_PLY = "pointcloud_fsa.ply"  # 输出点云文件名
    
    # ========================================================================
    # ✅ 配置参数区域结束 - 下方为程序执行代码，一般不需要修改
    # ========================================================================
    
    print("="*80)
    print("FSA 三维重建")
    print("="*80)
    
    # 打印配置信息
    print("\n📋 配置参数:")
    print(f"  标定文件: {CALIBRATION_FILE}")
    print(f"  相机图像尺寸: {CAMERA_IMAGE_SIZE[0]} x {CAMERA_IMAGE_SIZE[1]}")
    print(f"  投影仪分辨率: {PROJECTOR_WIDTH} x {PROJECTOR_HEIGHT}")
    print(f"  解码结果目录: {RESULT_DIR}")
    print(f"  重建模式: {'\u5355\u65b9\u5411 (' + PRIMARY_DIRECTION + ')' if USE_SINGLE_DIRECTION else '\u53cc\u65b9\u5411'}")
    print(f"  过采样倍数: {OVERSAMPLE}")
    print(f"  峰值阈值: {PEAK_LIMIT}")
    print(f"  白光图像: {WHITE_IMAGE_PATH if WHITE_IMAGE_PATH else '未配置（点云将为灰色）'}")
    if MIN_DISTANCE is not None or MAX_DISTANCE is not None:
        print(f"  距离过滤: [{MIN_DISTANCE or '无限制'}, {MAX_DISTANCE or '无限制'}] mm")
    print(f"  输出文件: {OUTPUT_PLY}")
    
    # ========== 步骤1: 检查标定文件 ==========
    print("\n" + "="*80)
    print("步骤1: 检查标定文件")
    print("="*80)
    
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ 错误: 标定文件不存在: {CALIBRATION_FILE}")
        print("   请确保标定文件存在")
        exit(1)
    
    print(f"✓ 标定文件存在: {CALIBRATION_FILE}")
    
    # ========== 步骤2: 加载视差图 ==========
    print("\n" + "="*80)
    print("步骤2: 加载视差图")
    print("="*80)
    
    # 检查文件是否存在（根据模式不同）
    if USE_SINGLE_DIRECTION:
        # 单方向模式：只需要一个方向的文件
        if PRIMARY_DIRECTION == 'horizontal':
            required_files = {
                "水平视差图": DISPARITY_HOR_FILE,
                "水平峰值图": PEAK_HOR_FILE,
            }
            print("\n单方向模式：使用水平方向数据，将通过基础矩阵计算垂直方向")
        else:  # vertical
            required_files = {
                "垂直视差图": DISPARITY_VER_FILE,
                "垂直峰值图": PEAK_VER_FILE,
            }
            print("\n单方向模式：使用垂直方向数据，将通过基础矩阵计算水平方向")
    else:
        # 双方向模式：需要两个方向的文件
        required_files = {
            "水平视差图": DISPARITY_HOR_FILE,
            "垂直视差图": DISPARITY_VER_FILE,
            "水平峰值图": PEAK_HOR_FILE,
            "垂直峰值图": PEAK_VER_FILE,
        }
        print("\n双方向模式：需要水平和垂直两个方向的数据")
    
    missing_files = {name: path for name, path in required_files.items() if not os.path.exists(path)}
    
    if missing_files:
        print("❌ 错误: 以下文件不存在:")
        for name, path in missing_files.items():
            print(f"   - {name}: {path}")
        print("\n提示: 请先运行FSA解码程序生成视差图")
        exit(1)
    
    # 检查并加载白光图像（用于点云颜色）
    white_pattern = None
    if WHITE_IMAGE_PATH is not None:
        if os.path.exists(WHITE_IMAGE_PATH):
            print(f"\n加载白光图像: {WHITE_IMAGE_PATH}")
            try:
                # 尝试使用OpenCV读取
                white_pattern = cv2.imread(WHITE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
                if white_pattern is None:
                    # 如果OpenCV无法读取（可能是中文路径问题），尝试使用numpy
                    img_array = np.fromfile(WHITE_IMAGE_PATH, dtype=np.uint8)
                    white_pattern = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                if white_pattern is not None:
                    print(f"✓ 白光图像加载成功: {white_pattern.shape}")
                    print("  → 点云将使用图像颜色")
                else:
                    print(f"❌ 错误: 无法读取白光图像: {WHITE_IMAGE_PATH}")
                    print("  → 点云将使用默认灰色")
            except Exception as e:
                print(f"❌ 错误: 加载白光图像失败: {e}")
                print("  → 点云将使用默认灰色")
                white_pattern = None
        else:
            print(f"\n⚠ 警告: 白光图像文件不存在: {WHITE_IMAGE_PATH}")
            print("  → 点云将使用默认灰色")
    else:
        print("\n⊙ 未配置白光图像路径 (WHITE_IMAGE_PATH = None)")
        print("  → 点云将使用默认灰色")
    
    # 单方向模式需要先创建重建器（获取基础矩阵）
    reconstructor = None
    if USE_SINGLE_DIRECTION:
        print("\n" + "="*80)
        print("步骤2.5: 创建三维重建器（单方向模式需要基础矩阵）")
        print("="*80)
        
        try:
            reconstructor = FSAReconstructor(
                calibration_file=CALIBRATION_FILE,
                camera_image_size=CAMERA_IMAGE_SIZE,
                projector_width=PROJECTOR_WIDTH,
                projector_height=PROJECTOR_HEIGHT
            )
        except Exception as e:
            print(f"❌ 错误: 创建重建器失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    # 加载视差图（根据模式不同）
    print("加载视差图...")
    
    if USE_SINGLE_DIRECTION:
        # 单方向模式
        from FSM_reconstruct_core import CorrespondenceReader
        
        if PRIMARY_DIRECTION == 'horizontal':
            disparity_hor = np.load(DISPARITY_HOR_FILE)
            peak_hor = np.load(PEAK_HOR_FILE)
            print(f"✓ 水平视差图: {disparity_hor.shape}")
            print(f"✓ 水平峰值图: {peak_hor.shape}")
            
            # 使用基础矩阵计算垂直视差图（改进版：启用去畸变和峰值加权）
            print("\n使用基础矩阵计算垂直视差图...")
            disparity_ver = CorrespondenceReader.compute_missing_direction(
                disparity_hor,
                reconstructor.fundamental_matrix,
                'horizontal',
                PROJECTOR_WIDTH,
                PROJECTOR_HEIGHT,
                camera_matrix=reconstructor.camera_matrix,
                camera_dist=reconstructor.camera_dist,
                projector_matrix=reconstructor.projector_matrix,
                projector_dist=reconstructor.projector_dist,
                use_undistortion=False,  # 暂时禁用去畸变（归一化坐标方案需要改进）
                peak_map=peak_hor  # 使用峰值图进行加权优化
            )
            peak_ver = None  # 单方向模式不需要另一方向的峰值图
            print(f"✓ 垂直视差图（计算得到）: {disparity_ver.shape}")
        else:  # vertical
            disparity_ver = np.load(DISPARITY_VER_FILE)
            peak_ver = np.load(PEAK_VER_FILE)
            print(f"✓ 垂直视差图: {disparity_ver.shape}")
            print(f"✓ 垂直峰值图: {peak_ver.shape}")
            
            # 使用基础矩阵计算水平视差图（改进版：启用去畸变和峰值加权）
            print("\n使用基础矩阵计算水平视差图...")
            disparity_hor = CorrespondenceReader.compute_missing_direction(
                disparity_ver,
                'vertical',
                reconstructor.fundamental_matrix,
                PROJECTOR_WIDTH,
                PROJECTOR_HEIGHT,
                camera_matrix=reconstructor.camera_matrix,
                camera_dist=reconstructor.camera_dist,
                projector_matrix=reconstructor.projector_matrix,
                projector_dist=reconstructor.projector_dist,
                use_undistortion=True,  # 启用去畸变（使用修正后的像素坐标计算）
                peak_map=peak_ver,  # 使用峰值图进行加权优化
                peak_threshold=0.6  # 峰值阈值：降低阈值以获得更多有效点
            )
            peak_hor = None  # 单方向模式不需要另一方向的峰值图
            print(f"✓ 水平视差图（计算得到）: {disparity_hor.shape}")
    else:
        # 双方向模式
        disparity_hor = np.load(DISPARITY_HOR_FILE)
        disparity_ver = np.load(DISPARITY_VER_FILE)
        peak_hor = np.load(PEAK_HOR_FILE)
        peak_ver = np.load(PEAK_VER_FILE)
        
        print(f"✓ 水平视差图: {disparity_hor.shape}")
        print(f"✓ 垂直视差图: {disparity_ver.shape}")
        print(f"✓ 水平峰值图: {peak_hor.shape}")
        print(f"✓ 垂直峰值图: {peak_ver.shape}")
    
    # 如果没有白光图像，创建占位符图像
    if white_pattern is None:
        print("\n创建占位符图像（不提取颜色）...")
        white_pattern = np.ones(disparity_hor.shape, dtype=np.uint8) * 128  # 灰度占位符
        print(f"✓ 占位符图像: {white_pattern.shape}")
    
    # ========== 步骤3: 创建重建器 ==========
    # 如果是双方向模式，在这里创建重建器；单方向模式已在步骤2.5创建
    if reconstructor is None:
        print("\n" + "="*80)
        print("步骤3: 创建三维重建器")
        print("="*80)
        
        try:
            reconstructor = FSAReconstructor(
                calibration_file=CALIBRATION_FILE,
                camera_image_size=CAMERA_IMAGE_SIZE,
                projector_width=PROJECTOR_WIDTH,
                projector_height=PROJECTOR_HEIGHT
            )
        except Exception as e:
            print(f"❌ 错误: 创建重建器失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        print("\n" + "="*80)
        print("步骤3: 三维重建器已就绪（已在步骤2.5创建）")
        print("="*80)
    
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
            min_distance=MIN_DISTANCE,  # 使用配置区域的参数
            max_distance=MAX_DISTANCE
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

