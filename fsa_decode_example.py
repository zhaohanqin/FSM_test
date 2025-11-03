"""
FSA解码使用示例

本文件展示如何使用 fsa_decode_core.py 中的FSADecoder类
对实际采集的条纹图像进行解码

支持的解码模式：
---------------
1. 解码两个方向（both）：获得完整的2D像素对应关系
   - 输入：水平条纹 + 垂直条纹
   - 输出：投影仪X坐标 + Y坐标
   
2. 仅解码垂直方向（ver）：获得投影仪列坐标
   - 输入：垂直条纹
   - 输出：投影仪X坐标
   
3. 仅解码水平方向（hor）：获得投影仪行坐标
   - 输入：水平条纹
   - 输出：投影仪Y坐标

使用前请确保：
1. 已采集好条纹图像（根据需求采集一个或两个方向）
2. 图像按顺序命名（如 0.png, 1.png, 2.png, ...）
3. 已知投影仪分辨率和采集的频率数量
4. 修改脚本中的 DECODE_DIRECTION 参数

图像序列的组织方式：
-------------------
图像应该按照"频率优先，相位顺序"的方式组织：
- 0.png, 1.png, 2.png, 3.png → 频率0的 (0°, 90°, 180°, 270°)
- 4.png, 5.png, 6.png, 7.png → 频率1的 (0°, 90°, 180°, 270°)
- 8.png, 9.png, 10.png, 11.png → 频率2的 (0°, 90°, 180°, 270°)
- ... 以此类推

例如，如果采集121个频率（NUM_FREQUENCIES = 121），总共需要：
  121 × 4 = 484 张图像（0.png 到 483.png）

每个频率的4张图像按顺序为：0°、90°、180°、270° 的相移
"""

import numpy as np
import os
from fsa_decode_core import FSADecoder


def decode_fsa_images():
    """
    使用FSADecoder解码实际采集的图像
    """
    
    # ==================== 配置参数 ====================
    # 投影仪参数
    PROJECTOR_WIDTH = 1920
    PROJECTOR_HEIGHT = 1080
    
    # FSA采集参数
    NUM_FREQUENCIES = 121  # 采集的频率数量（M值）
    # 注意：总图像数 = NUM_FREQUENCIES × 4 (四步相移)
    
    # ==================== 解码方向选择 ====================
    # 控制解码哪个方向：
    # - "both": 解码两个方向（水平+垂直），获得完整的2D对应关系
    # - "ver": 仅解码垂直条纹（获取投影仪列坐标/X坐标）
    # - "hor": 仅解码水平条纹（获取投影仪行坐标/Y坐标）
    DECODE_DIRECTION = "ver"  # 可选: "both", "ver", "hor"
    # ====================================================
    
    # ==================== 内存优化说明 ====================
    # 💡 如果遇到内存不足错误，请查看下方的 LOW_MEMORY_MODE 参数
    # 低内存模式可以将内存占用从 12GB 降低到 100MB
    # ====================================================
    
    # ==================== 图像路径配置 ====================
    # 直接输入图像文件夹的绝对路径（请根据实际情况修改）
    # 
    # 方式1：直接指定完整路径（推荐）
    IMAGE_DIR_HOR = r"E:\code\images\data\fsa_test\hor"  # 水平条纹图像文件夹
    IMAGE_DIR_VER = r"E:\code\images\data\fsa_test\ver"  # 垂直条纹图像文件夹
    OUTPUT_DIR = r"E:\在校学习\fsa\results\fsa_decode"    # 输出结果文件夹
    
    # 提示：如果路径包含中文，确保使用 r"..." 原始字符串
    # =====================================================
    
    # 处理参数
    BLOCK_SIZE = 2  # 分块处理大小（低内存模式下不使用）
    USE_CACHE = False  # 是否使用缓存（禁用缓存可避免大文件保存导致的内存错误）
    VERIFY_SEQUENCE = True  # 是否验证图像序列顺序
    
    # ==================== 内存模式选择 ====================
    # 🔥 LOW_MEMORY_MODE: 低内存模式开关
    # 
    # True  (低内存模式):
    #   - 内存占用: 约 100MB
    #   - 速度: 较慢（约需 10-30 分钟）
    #   - 适用: 内存 < 12GB 的系统
    #   - 推荐: 如果遇到内存错误，请设为 True
    #
    # False (标准模式):
    #   - 内存占用: 约 12GB
    #   - 速度: 快（约需 1-3 分钟）
    #   - 适用: 内存 ≥ 16GB 的系统
    #
    LOW_MEMORY_MODE = True  # 🔴 内存不足时必须设为 True
    # ====================================================
    
    # 调试参数（可选）
    DEBUG_POINT = (900, 495)  # 查看特定点的解码详情，设为None则不输出
    
    # ==================================================
    
    # 验证配置
    if DECODE_DIRECTION not in ["both", "ver", "hor"]:
        print(f"❌ 错误: DECODE_DIRECTION 参数无效: '{DECODE_DIRECTION}'")
        print("   有效值: 'both', 'ver', 'hor'")
        return None
    
    print("="*80)
    print("FSA 解码程序")
    print("="*80)
    print(f"投影仪分辨率: {PROJECTOR_WIDTH} × {PROJECTOR_HEIGHT}")
    print(f"采集频率数: {NUM_FREQUENCIES}")
    print(f"总图像数: {NUM_FREQUENCIES * 4} (4步相移)")
    print(f"解码方向: {DECODE_DIRECTION.upper()}")
    if DECODE_DIRECTION == "both":
        print("  → 解码两个方向（水平+垂直），获得完整2D对应关系")
    elif DECODE_DIRECTION == "ver":
        print("  → 仅解码垂直条纹，获得投影仪列坐标（X坐标）")
    elif DECODE_DIRECTION == "hor":
        print("  → 仅解码水平条纹，获得投影仪行坐标（Y坐标）")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建解码器实例
    decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    
    # ==================== 解码水平方向 ====================
    if DECODE_DIRECTION in ["both", "hor"]:
        print("\n" + "█"*80)
        print("█ 解码水平条纹（获取投影仪行坐标 / Y坐标）")
        print("█"*80)
    
    if DECODE_DIRECTION in ["both", "hor"] and os.path.exists(IMAGE_DIR_HOR):
        try:
            # 加载图像
            images_hor = decoder.load_images(
                image_dir=IMAGE_DIR_HOR,
                num_frequencies=NUM_FREQUENCIES,
                image_pattern="{:d}.png",
                use_cache=USE_CACHE,
                verify_sequence=VERIFY_SEQUENCE
            )
            
            # 执行解码
            disparity_hor, peak_hor = decoder.decode_direction(
                images=images_hor,
                direction='hor',
                num_frequencies=NUM_FREQUENCIES,
                block_size=BLOCK_SIZE,
                debug_point=DEBUG_POINT,
                low_memory=LOW_MEMORY_MODE
            )
            
            # 保存结果
            decoder.save_results(disparity_hor, peak_hor, OUTPUT_DIR, 'hor')
            
            # 显示质量统计
            print("\n质量评估（水平方向）:")
            print(f"  高质量像素 (peak > 1.0): {np.sum(peak_hor > 1.0)} "
                  f"({np.sum(peak_hor > 1.0) / peak_hor.size * 100:.2f}%)")
            print(f"  中等质量像素 (0.5 < peak ≤ 1.0): {np.sum((peak_hor > 0.5) & (peak_hor <= 1.0))} "
                  f"({np.sum((peak_hor > 0.5) & (peak_hor <= 1.0)) / peak_hor.size * 100:.2f}%)")
            print(f"  低质量像素 (peak ≤ 0.5): {np.sum(peak_hor <= 0.5)} "
                  f"({np.sum(peak_hor <= 0.5) / peak_hor.size * 100:.2f}%)")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    elif DECODE_DIRECTION in ["both", "hor"]:
        print(f"\n⚠ 警告: 图像文件夹不存在: {IMAGE_DIR_HOR}")
        print("跳过水平方向解码")
    elif DECODE_DIRECTION == "ver":
        print("\n⊙ 根据配置，跳过水平方向解码（仅解码垂直方向）")
    
    # ==================== 解码垂直方向 ====================
    if DECODE_DIRECTION in ["both", "ver"]:
        print("\n" + "█"*80)
        print("█ 解码垂直条纹（获取投影仪列坐标 / X坐标）")
        print("█"*80)
    
    if DECODE_DIRECTION in ["both", "ver"] and os.path.exists(IMAGE_DIR_VER):
        try:
            # 加载图像
            images_ver = decoder.load_images(
                image_dir=IMAGE_DIR_VER,
                num_frequencies=NUM_FREQUENCIES,
                image_pattern="{:d}.png",
                use_cache=USE_CACHE,
                verify_sequence=VERIFY_SEQUENCE
            )
            
            # 执行解码
            disparity_ver, peak_ver = decoder.decode_direction(
                images=images_ver,
                direction='ver',
                num_frequencies=NUM_FREQUENCIES,
                block_size=BLOCK_SIZE,
                debug_point=DEBUG_POINT,
                low_memory=LOW_MEMORY_MODE
            )
            
            # 保存结果
            decoder.save_results(disparity_ver, peak_ver, OUTPUT_DIR, 'ver')
            
            # 显示质量统计
            print("\n质量评估（垂直方向）:")
            print(f"  高质量像素 (peak > 1.0): {np.sum(peak_ver > 1.0)} "
                  f"({np.sum(peak_ver > 1.0) / peak_ver.size * 100:.2f}%)")
            print(f"  中等质量像素 (0.5 < peak ≤ 1.0): {np.sum((peak_ver > 0.5) & (peak_ver <= 1.0))} "
                  f"({np.sum((peak_ver > 0.5) & (peak_ver <= 1.0)) / peak_ver.size * 100:.2f}%)")
            print(f"  低质量像素 (peak ≤ 0.5): {np.sum(peak_ver <= 0.5)} "
                  f"({np.sum(peak_ver <= 0.5) / peak_ver.size * 100:.2f}%)")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    elif DECODE_DIRECTION in ["both", "ver"]:
        print(f"\n⚠ 警告: 图像文件夹不存在: {IMAGE_DIR_VER}")
        print("跳过垂直方向解码")
    elif DECODE_DIRECTION == "hor":
        print("\n⊙ 根据配置，跳过垂直方向解码（仅解码水平方向）")
    
    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✓ 解码完成！")
    print("="*80)
    print(f"\n结果保存位置: {OUTPUT_DIR}")
    print(f"解码方向: {DECODE_DIRECTION.upper()}")
    print("\n生成的文件:")
    
    # 根据解码方向确定期望的文件
    result_files = []
    if DECODE_DIRECTION in ["both", "hor"]:
        result_files.extend([
            ("disp_hor.npy", "投影仪行坐标 / Y坐标（水平视差图）"),
            ("peak_hor.npy", "水平方向峰值强度（质量/置信度）"),
        ])
    if DECODE_DIRECTION in ["both", "ver"]:
        result_files.extend([
            ("disp_ver.npy", "投影仪列坐标 / X坐标（垂直视差图）"),
            ("peak_ver.npy", "垂直方向峰值强度（质量/置信度）"),
        ])
    
    for filename, description in result_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            print(f"  ✓ {filename:20s} - {description} ({file_size:.2f} MB)")
        else:
            print(f"  ✗ {filename:20s} - 未生成")
    
    print("\n后续步骤:")
    if DECODE_DIRECTION == "both":
        print("  ✓ 已获得完整的2D像素对应关系（投影仪X和Y坐标）")
        print("  1. 使用 fsa_visualize.py 可视化解码结果")
        print("  2. 使用峰值图过滤低质量像素点")
        print("  3. 结合双目标定参数进行三维重建")
    elif DECODE_DIRECTION == "ver":
        print("  ⊙ 仅获得投影仪列坐标（X坐标）")
        print("  ! 如需完整的2D对应关系，还需要采集并解码水平条纹（Y坐标）")
        print("  1. 可视化当前的列坐标结果")
        print("  2. 使用峰值图评估解码质量")
    elif DECODE_DIRECTION == "hor":
        print("  ⊙ 仅获得投影仪行坐标（Y坐标）")
        print("  ! 如需完整的2D对应关系，还需要采集并解码垂直条纹（X坐标）")
        print("  1. 可视化当前的行坐标结果")
        print("  2. 使用峰值图评估解码质量")
    
    return OUTPUT_DIR


def load_and_inspect_results(result_dir: str):
    """
    加载并检查解码结果
    
    Parameters:
    -----------
    result_dir : str
        结果文件夹路径
    """
    print("\n" + "="*80)
    print("检查解码结果")
    print("="*80)
    
    # 检查并加载结果文件
    files_to_check = ['disp_hor.npy', 'peak_hor.npy', 'disp_ver.npy', 'peak_ver.npy']
    results = {}
    
    for filename in files_to_check:
        filepath = os.path.join(result_dir, filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            results[filename] = data
            print(f"\n✓ {filename}:")
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  值范围: [{data.min():.2f}, {data.max():.2f}]")
            print(f"  均值: {data.mean():.2f}")
            print(f"  标准差: {data.std():.2f}")
        else:
            print(f"\n✗ {filename}: 文件不存在")
    
    # 如果两个方向都解码成功，可以构建完整的对应关系
    if 'disp_hor.npy' in results and 'disp_ver.npy' in results:
        print("\n" + "-"*80)
        print("相机-投影仪像素对应关系已建立")
        print("-"*80)
        print("示例：对于相机像素 (xc, yc)，对应的投影仪像素为:")
        print("  xp = disparity_ver[yc, xc]")
        print("  yp = disparity_hor[yc, xc]")
        print("  置信度 = min(peak_hor[yc, xc], peak_ver[yc, xc])")
        
        # 示例点
        yc, xc = 500, 500
        if yc < results['disp_hor.npy'].shape[0] and xc < results['disp_hor.npy'].shape[1]:
            yp = results['disp_hor.npy'][yc, xc]
            xp = results['disp_ver.npy'][yc, xc]
            conf_h = results['peak_hor.npy'][yc, xc]
            conf_v = results['peak_ver.npy'][yc, xc]
            print(f"\n示例: 相机像素 ({xc}, {yc}) → 投影仪像素 ({xp:.2f}, {yp:.2f})")
            print(f"      置信度: 水平={conf_h:.4f}, 垂直={conf_v:.4f}")


if __name__ == "__main__":
    # 执行解码
    result_dir = decode_fsa_images()
    
    # 检查结果
    if result_dir and os.path.exists(result_dir):
        load_and_inspect_results(result_dir)
    
    print("\n" + "="*80)
    print("程序结束")
    print("="*80)

