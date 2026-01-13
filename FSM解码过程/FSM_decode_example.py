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
from FSM_decode_core import FSADecoder


def save_spectrum_data(spectrum_data, output_dir, direction, spectrum_points):
    """
    保存多点频谱数据到不同的文件夹
    
    Parameters:
    -----------
    spectrum_data : dict or None
        频谱数据字典，键为点索引，值为频谱数据
    output_dir : str
        输出目录
    direction : str
        方向标识 ('hor' 或 'ver')
    spectrum_points : tuple, list or None
        原始的频谱点配置
    """
    if spectrum_data is None:
        print(f"\n⊙ {direction.upper()}方向: 无频谱数据需要保存")
        return
    
    # 统一转换为列表格式
    if isinstance(spectrum_points, tuple):
        points_list = [spectrum_points]
    else:
        points_list = spectrum_points
    
    print(f"\n✓ 保存{direction.upper()}方向频谱数据:")
    
    for point_idx, point_data in spectrum_data.items():
        point = point_data['point']
        
        # 为每个点创建独立的文件夹
        point_dir = os.path.join(output_dir, f"spectrum_point_{point[0]}_{point[1]}_{direction}")
        os.makedirs(point_dir, exist_ok=True)
        
        # 保存频谱数据
        spectrum_path = os.path.join(point_dir, f"spectrum_{direction}.npy")
        np.save(spectrum_path, point_data)
        
        print(f"  点{point_idx+1} ({point[0]}, {point[1]}): 数据已保存到 {point_dir}")
    
    print(f"  共保存了 {len(spectrum_data)} 个点的{direction.upper()}方向频谱数据")


def decode_fsa_images(projector_width, projector_height, num_frequencies, decode_direction,
                      image_dir_hor, image_dir_ver, output_dir, block_size, use_cache,
                      verify_sequence, low_memory_mode, spectrum_points, frequency_step=1.0):
    """
    使用FSADecoder解码实际采集的图像
    
    Parameters:
    -----------
    projector_width : int
        投影仪宽度（像素）
    projector_height : int
        投影仪高度（像素）
    num_frequencies : int
        采集的频率数量（M值）
    decode_direction : str
        解码方向: "both", "ver", "hor"
    image_dir_hor : str
        水平条纹图像文件夹路径
    image_dir_ver : str
        垂直条纹图像文件夹路径
    output_dir : str
        输出结果文件夹路径
    block_size : int
        分块处理大小
    use_cache : bool
        是否使用缓存
    verify_sequence : bool
        是否验证图像序列顺序
    low_memory_mode : bool
        是否使用低内存模式
    spectrum_points : tuple, list of tuples, or None
        频谱分析点坐标，支持单个点 (x, y) 或多个点列表 [(x1, y1), (x2, y2), ...]
    frequency_step : float
        频率步长（Hz），默认为1.0
    """
    
    # 使用传入的参数
    PROJECTOR_WIDTH = projector_width
    PROJECTOR_HEIGHT = projector_height
    NUM_FREQUENCIES = num_frequencies
    DECODE_DIRECTION = decode_direction
    IMAGE_DIR_HOR = image_dir_hor
    IMAGE_DIR_VER = image_dir_ver
    OUTPUT_DIR = output_dir
    BLOCK_SIZE = block_size
    USE_CACHE = use_cache
    VERIFY_SEQUENCE = verify_sequence
    LOW_MEMORY_MODE = low_memory_mode
    SPECTRUM_POINTS = spectrum_points
    FREQUENCY_STEP = frequency_step
    
    # 验证配置
    if DECODE_DIRECTION not in ["both", "ver", "hor"]:
        print(f"[ERROR] 错误: DECODE_DIRECTION 参数无效: '{DECODE_DIRECTION}'")
        print("   有效值: 'both', 'ver', 'hor'")
        return None
    
    print("="*80)
    print("FSA 解码程序")
    print("="*80)
    print(f"投影仪分辨率: {PROJECTOR_WIDTH} × {PROJECTOR_HEIGHT}")
    print(f"采集频率数: {NUM_FREQUENCIES}")
    print(f"频率步长: {FREQUENCY_STEP} Hz")
    print(f"最高频率: {(NUM_FREQUENCIES-1) * FREQUENCY_STEP} Hz")
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
    decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT, FREQUENCY_STEP)
    
    # ==================== 解码水平方向 ====================
    if DECODE_DIRECTION in ["both", "hor"]:
        print("\n" + "="*80)
        print("解码水平条纹（获取投影仪行坐标 / Y坐标）")
        print("="*80)
    
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
            disparity_hor, peak_hor, spectrum_hor = decoder.decode_direction(
                images=images_hor,
                direction='hor',
                num_frequencies=NUM_FREQUENCIES,
                block_size=BLOCK_SIZE,
                low_memory=LOW_MEMORY_MODE,
                spectrum_points=SPECTRUM_POINTS 
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
        print("\n" + "="*80)
        print("解码垂直条纹（获取投影仪列坐标 / X坐标）")
        print("="*80)
    
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
            
            # 执行解码 - 注意：both模式下也要生成频谱
            disparity_ver, peak_ver, spectrum_ver = decoder.decode_direction(
                images=images_ver,
                direction='ver',
                num_frequencies=NUM_FREQUENCIES,
                block_size=BLOCK_SIZE,
                low_memory=LOW_MEMORY_MODE,
                spectrum_points=SPECTRUM_POINTS  # 总是传入，不管是什么模式
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
    
    # ==================== 保存频谱数据 ====================
    spectrum_data_list = []  # 存储所有方向的频谱数据
    
    # 保存水平方向频谱数据
    if DECODE_DIRECTION in ["both", "hor"] and 'spectrum_hor' in locals() and spectrum_hor is not None:
        save_spectrum_data(spectrum_hor, OUTPUT_DIR, 'hor', SPECTRUM_POINTS)
        spectrum_data_list.append(spectrum_hor)
        # 生成可视化图像
        visualize_multi_point_spectrum(spectrum_hor, OUTPUT_DIR)
    
    # 保存垂直方向频谱数据
    if DECODE_DIRECTION in ["both", "ver"] and 'spectrum_ver' in locals() and spectrum_ver is not None:
        save_spectrum_data(spectrum_ver, OUTPUT_DIR, 'ver', SPECTRUM_POINTS)
        spectrum_data_list.append(spectrum_ver)
        # 生成可视化图像
        visualize_multi_point_spectrum(spectrum_ver, OUTPUT_DIR)
    
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
    
    return OUTPUT_DIR, spectrum_data_list


def visualize_spectrum(spectrum_data: dict, save_path: str = None):
    """
    可视化频谱数据，展示FSA解码的完整过程
    
    Parameters:
    -----------
    spectrum_data : dict
        频谱数据字典（从decode_direction返回）
    save_path : str, optional
        保存图像的路径
    
    注意：如果出现 matplotlib 导入错误，请运行：
    pip install --upgrade matplotlib scikit-image
    或降级 NumPy：
    pip install "numpy<2"
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\n[ERROR] 频谱可视化错误：无法导入 matplotlib")
        print(f"错误原因：{str(e)[:100]}")
        print("\n解决方案：")
        print("  1. 降级 NumPy：pip install 'numpy<2'")
        print("  2. 升级 matplotlib：pip install --upgrade matplotlib")
        print("  3. 重新安装依赖：pip install --upgrade scikit-image pillow")
        return None
    
    # ==================== 字体配置（支持中文和数学符号） ====================
    # 设置中文字体（自动选择系统可用的字体）
    import matplotlib.font_manager as fm
    
    # 获取系统中可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 优先级字体列表（按优先级排序）
    font_candidates = [
        'SimHei',           # Windows 中文
        'Microsoft YaHei',  # Windows 微软雅黑
        'STHeiti',          # macOS
        'DejaVu Sans',      # 备选（英文字体）
        'Arial Unicode MS', # 备选
        'WenQuanYi Zen Hei'  # Linux
    ]
    
    # 选择第一个可用的字体
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            print(f"  ✓ 已选择字体: {font}")
            break
    
    if selected_font is None:
        selected_font = 'DejaVu Sans'
        print(f"  ⚠ 未找到中文字体，使用备选字体: {selected_font}")
    
    # 配置 matplotlib 字体和显示参数
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9  # 字体大小
    
    # 关闭字体警告（避免大量 UserWarning）
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print("\n" + "="*80)
    print("频谱可视化分析")
    print("="*80)
    
    point = spectrum_data['point']
    direction = spectrum_data['direction']
    N = spectrum_data['N']
    M = spectrum_data['num_frequencies']
    frequency_step = spectrum_data.get('frequency_step', 1.0)  # 获取频率步长，默认为1.0
    
    print(f"分析点: ({point[0]}, {point[1]})")
    print(f"解码方向: {'水平 (投影仪行坐标)' if direction == 'hor' else '垂直 (投影仪列坐标)'}")
    print(f"频率数量: {M}")
    print(f"频率步长: {frequency_step} Hz")
    print(f"最高频率: {(M-1) * frequency_step} Hz")
    print(f"DFT长度: {N} (投影仪{'高度' if direction == 'hor' else '宽度'})")
    print(f"最大峰值位置: {spectrum_data['peak_index']}")
    print(f"最大峰值强度: {spectrum_data['peak_value']:.4f}")
    
    # 创建图形 (2行4列布局，但只用7个子图)
    fig = plt.figure(figsize=(20, 10))
    
    # 计算实际频率轴
    freq_axis = np.arange(M) * frequency_step  # 实际频率值
    
    # 1. 原始强度序列
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(freq_axis, spectrum_data['I0'], 'o-', label='I0 (0°)', linewidth=1.5, markersize=3)
    ax1.plot(freq_axis, spectrum_data['I1'], 's-', label='I1 (90°)', linewidth=1.5, markersize=3)
    ax1.plot(freq_axis, spectrum_data['I2'], '^-', label='I2 (180°)', linewidth=1.5, markersize=3)
    ax1.plot(freq_axis, spectrum_data['I3'], 'd-', label='I3 (270°)', linewidth=1.5, markersize=3)
    ax1.set_xlabel('频率 (Hz)')
    ax1.set_ylabel('强度值')
    ax1.set_title(f'1. 原始强度序列\n(M={M}个频率，步长={frequency_step}Hz)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 差分信号
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(freq_axis, spectrum_data['Dc'], 'o-', label='Dc = I0 - I2', linewidth=1.5, markersize=3)
    ax2.plot(freq_axis, spectrum_data['Ds'], 's-', label='Ds = I3 - I1', linewidth=1.5, markersize=3)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('差分值')
    ax2.set_title('2. 差分信号\n(消除直流分量和反射率)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. 复数信号的幅值
    ax3 = plt.subplot(2, 4, 3)
    D_magnitude = np.abs(spectrum_data['Dc'] + 1j * spectrum_data['Ds'])
    ax3.plot(freq_axis, D_magnitude, 'o-', color='purple', linewidth=1.5, markersize=3)
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('|D| (复数信号幅值)')
    ax3.set_title('3. 复数信号幅值\nD = Dc + j·Ds')
    ax3.grid(True, alpha=0.3)
    
    # 4. 完整频谱（线性坐标）
    ax4 = plt.subplot(2, 4, 4)
    spectrum = spectrum_data['spectrum']
    frequencies = spectrum_data['frequencies']
    ax4.plot(frequencies, spectrum, '-', linewidth=1, color='blue', alpha=0.7)
    ax4.axvline(x=spectrum_data['peak_index'], color='red', linestyle='--', 
                linewidth=2, label=f"峰值位置={spectrum_data['peak_index']}")
    ax4.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标")
    ax4.set_ylabel('归一化幅值')
    ax4.set_title(f'4. 完整DFT频谱（线性坐标）\n峰值强度={spectrum_data["peak_value"]:.4f}')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. 完整频谱（对数坐标）
    ax5 = plt.subplot(2, 4, 5)
    spectrum_log = np.log10(spectrum + 1e-10)
    ax5.plot(frequencies, spectrum_log, '-', linewidth=1, color='green', alpha=0.7)
    ax5.axvline(x=spectrum_data['peak_index'], color='red', linestyle='--', 
                linewidth=2, label=f"峰值位置={spectrum_data['peak_index']}")
    ax5.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标")
    ax5.set_ylabel('log₁₀(幅值)')
    ax5.set_title('5. 完整DFT频谱（对数坐标）\n更清晰地显示次要峰值')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. 局部放大（峰值附近）
    ax6 = plt.subplot(2, 4, 6)
    peak_idx = spectrum_data['peak_index']
    half_width = 100
    start = max(0, peak_idx - half_width)
    end = min(N, peak_idx + half_width)
    ax6.plot(frequencies[start:end], spectrum[start:end], 'o-', 
             linewidth=2, markersize=4, color='purple')
    ax6.axvline(x=peak_idx, color='red', linestyle='--', linewidth=2, 
                label=f'峰值位置={peak_idx}')
    ax6.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标")
    ax6.set_ylabel('归一化幅值')
    ax6.set_title(f'6. 峰值附近局部放大\n范围: [{start}, {end}]')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    # 7. 前20个峰值分析（改进版：折线图显示投影仪坐标与峰值强度的关系）
    ax7 = plt.subplot(2, 4, 7)
    top20_indices = np.argsort(spectrum)[-20:][::-1]
    top20_values = spectrum[top20_indices]
    
    # 绘制折线图
    ax7.plot(top20_indices, top20_values, 'o-', linewidth=2, markersize=6, color='blue', alpha=0.7)
    
    # 高亮显示最大峰值（红色）
    ax7.plot(top20_indices[0], top20_values[0], 'o', markersize=10, color='red', 
             label=f'最大峰值 (位置={top20_indices[0]}, 强度={top20_values[0]:.2f})')
    
    # 在每个点旁边显示纵坐标值（简化版，不用框）
    for i in range(min(10, len(top20_indices))):
        # 偏移量，避免文字重叠
        y_offset = 0.15 if i % 2 == 0 else -0.25
        ax7.text(top20_indices[i], top20_values[i] + y_offset, f'{top20_values[i]:.2f}', 
                fontsize=8, ha='center', va='center', 
                color='red' if i == 0 else 'darkblue', fontweight='bold')
    
    ax7.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=11)
    ax7.set_ylabel('归一化幅值', fontsize=11)
    ax7.set_title('7. 前20个峰值分析（按强度排序）\n(投影仪坐标 vs 峰值强度)', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. 峰值强度随坐标变化（新增：按投影仪坐标排序）
    ax8 = plt.subplot(2, 4, 8)
    
    # 按照投影仪坐标从小到大排序
    sorted_order = np.argsort(top20_indices)
    sorted_indices = top20_indices[sorted_order]
    sorted_values = top20_values[sorted_order]
    
    # 找到最大峰值在排序后的位置
    max_peak_pos = np.where(sorted_indices == top20_indices[0])[0][0]
    
    # 绘制折线图
    ax8.plot(sorted_indices, sorted_values, 'o-', linewidth=2, markersize=6, color='blue', alpha=0.7, label='峰值强度')
    
    # 高亮显示最大峰值（红色）
    ax8.plot(sorted_indices[max_peak_pos], sorted_values[max_peak_pos], 'o', markersize=10, color='red', 
             label=f'最大峰值 (位置={sorted_indices[max_peak_pos]}, 强度={sorted_values[max_peak_pos]:.2f})', zorder=5)
    
    # 在关键点旁边显示数值（简化版）
    for i in range(0, len(sorted_indices), max(1, len(sorted_indices)//8)):  # 只显示部分点，避免拥挤
        y_offset = 0.15 if i % 2 == 0 else -0.25
        color = 'red' if i == max_peak_pos else 'darkblue'
        ax8.text(sorted_indices[i], sorted_values[i] + y_offset, f'{sorted_values[i]:.2f}', 
                fontsize=8, ha='center', va='center', color=color, fontweight='bold')
    
    ax8.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=11)
    ax8.set_ylabel('归一化幅值', fontsize=11)
    ax8.set_title('8. 峰值强度随坐标变化（按坐标排序）\n显示峰值的空间分布', fontsize=12, fontweight='bold')
    ax8.legend(loc='best', fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # 添加峰值大小信息到总标题
    peak_value = spectrum_data['peak_value']
    peak_index = spectrum_data['peak_index']
    plt.suptitle(f'FSA频谱分析 - 像素({point[0]}, {point[1]}) - {direction.upper()}方向\n峰值位置: {peak_index}, 峰值强度(质量): {peak_value:.4f}', 
                 fontsize=16, fontweight='bold')
    
    # ==================== 保存图像到像素点文件夹 ====================
    if save_path:
        # 获取保存目录和基础文件名
        pixel_save_dir = os.path.dirname(save_path)
        base_name = os.path.basename(save_path).replace('.png', '')
        pixel_folder_name = os.path.basename(pixel_save_dir)
        
        # 保存主要的8子图汇总图像
        plt.tight_layout(pad=2.0)  # 使用固定的padding
        plt.savefig(save_path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
        print(f"\n✓ 完整频谱图像已保存到: {pixel_folder_name}/{os.path.basename(save_path)}")
        
        print(f"\n正在保存各个子图到文件夹: {pixel_folder_name}...")
        
        # 定义9个子图的绘制函数
        def save_subplot_1():
            """1. 原始强度序列"""
            fig_sub = plt.figure(figsize=(12, 8))
            plt.plot(freq_axis, spectrum_data['I0'], 'o-', label='I0 (0°)', linewidth=2, markersize=4)
            plt.plot(freq_axis, spectrum_data['I1'], 's-', label='I1 (90°)', linewidth=2, markersize=4)
            plt.plot(freq_axis, spectrum_data['I2'], '^-', label='I2 (180°)', linewidth=2, markersize=4)
            plt.plot(freq_axis, spectrum_data['I3'], 'd-', label='I3 (270°)', linewidth=2, markersize=4)
            plt.xlabel('频率 (Hz)', fontsize=12)
            plt.ylabel('强度值', fontsize=12)
            plt.title(f'原始强度序列 (M={M}个频率，步长={frequency_step}Hz)', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.3)
            path = os.path.join(pixel_save_dir, f'{base_name}_01_原始强度序列.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_2():
            """2. 差分信号"""
            fig_sub = plt.figure(figsize=(12, 8))
            plt.plot(freq_axis, spectrum_data['Dc'], 'o-', label='Dc = I0 - I2', linewidth=2, markersize=4)
            plt.plot(freq_axis, spectrum_data['Ds'], 's-', label='Ds = I3 - I1', linewidth=2, markersize=4)
            plt.xlabel('频率 (Hz)', fontsize=12)
            plt.ylabel('差分值', fontsize=12)
            plt.title('差分信号 (消除直流分量和反射率)', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.3)
            path = os.path.join(pixel_save_dir, f'{base_name}_02_差分信号.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_3():
            """3. 复数信号幅值"""
            fig_sub = plt.figure(figsize=(12, 8))
            D_magnitude = np.abs(spectrum_data['Dc'] + 1j * spectrum_data['Ds'])
            plt.plot(freq_axis, D_magnitude, 'o-', color='purple', linewidth=2, markersize=4)
            plt.xlabel('频率 (Hz)', fontsize=12)
            plt.ylabel('|D| (复数信号幅值)', fontsize=12)
            plt.title('复数信号幅值 (D = Dc + j·Ds)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            path = os.path.join(pixel_save_dir, f'{base_name}_03_复数信号幅值.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_4():
            """4. 完整频谱（线性）"""
            fig_sub = plt.figure(figsize=(12, 8))
            plt.plot(frequencies, spectrum, '-', linewidth=1.5, color='blue', alpha=0.7)
            plt.axvline(x=spectrum_data['peak_index'], color='red', linestyle='--', 
                       linewidth=2.5, label=f"峰值位置={spectrum_data['peak_index']}")
            plt.xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=12)
            plt.ylabel('归一化幅值', fontsize=12)
            plt.title(f'完整DFT频谱-线性坐标 (峰值强度={spectrum_data["peak_value"]:.4f})', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.3)
            path = os.path.join(pixel_save_dir, f'{base_name}_04_完整频谱_线性.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_5():
            """5. 完整频谱（对数）"""
            fig_sub = plt.figure(figsize=(12, 8))
            spectrum_log = np.log10(spectrum + 1e-10)
            plt.plot(frequencies, spectrum_log, '-', linewidth=1.5, color='green', alpha=0.7)
            plt.axvline(x=spectrum_data['peak_index'], color='red', linestyle='--', 
                       linewidth=2.5, label=f"峰值位置={spectrum_data['peak_index']}")
            plt.xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=12)
            plt.ylabel('log₁₀(幅值)', fontsize=12)
            plt.title('完整DFT频谱-对数坐标 (更清晰显示次要峰值)', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.3)
            path = os.path.join(pixel_save_dir, f'{base_name}_05_完整频谱_对数.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_6():
            """6. 峰值附近局部放大"""
            fig_sub = plt.figure(figsize=(12, 8))
            peak_idx = spectrum_data['peak_index']
            half_width = 100
            start = max(0, peak_idx - half_width)
            end = min(N, peak_idx + half_width)
            plt.plot(frequencies[start:end], spectrum[start:end], 'o-', 
                    linewidth=2.5, markersize=5, color='purple')
            plt.axvline(x=peak_idx, color='red', linestyle='--', linewidth=2.5, 
                       label=f'峰值位置={peak_idx}')
            plt.xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=12)
            plt.ylabel('归一化幅值', fontsize=12)
            plt.title(f'峰值附近局部放大 (范围: [{start}, {end}])', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.3)
            path = os.path.join(pixel_save_dir, f'{base_name}_06_峰值局部放大.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_7():
            """7. 前20个峰值分析（按强度排序）"""
            fig_sub = plt.figure(figsize=(12, 8))
            top20_indices = np.argsort(spectrum)[-20:][::-1]
            top20_values = spectrum[top20_indices]
            
            # 绘制折线图
            plt.plot(top20_indices, top20_values, 'o-', linewidth=2.5, markersize=8, 
                    color='blue', alpha=0.7, label='峰值强度')
            
            # 高亮显示最大峰值（红色，与其他点相同大小）
            plt.plot(top20_indices[0], top20_values[0], 'o', markersize=8, color='red', 
                    zorder=5, label=f'最大峰值 (位置={top20_indices[0]}, 强度={top20_values[0]:.2f})')
            
            plt.xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=13, fontweight='bold')
            plt.ylabel('归一化幅值', fontsize=13, fontweight='bold')
            plt.title(f'前20个峰值分析（按强度排序）\n像素({point[0]}, {point[1]}) - {direction.upper()}方向', 
                     fontsize=15, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.4, linestyle='--')
            
            path = os.path.join(pixel_save_dir, f'{base_name}_07_峰值分析_按强度.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        def save_subplot_8():
            """8. 峰值强度随坐标变化（按坐标排序）"""
            fig_sub = plt.figure(figsize=(12, 8))
            top20_indices = np.argsort(spectrum)[-20:][::-1]
            top20_values = spectrum[top20_indices]
            
            # 按照投影仪坐标从小到大排序
            sorted_order = np.argsort(top20_indices)
            sorted_indices = top20_indices[sorted_order]
            sorted_values = top20_values[sorted_order]
            
            # 找到最大峰值在排序后的位置
            max_peak_pos = np.where(sorted_indices == top20_indices[0])[0][0]
            
            # 绘制折线图
            plt.plot(sorted_indices, sorted_values, 'o-', linewidth=2.5, markersize=8, 
                    color='blue', alpha=0.7, label='峰值强度')
            
            # 高亮显示最大峰值（红色，与其他点相同大小）
            plt.plot(sorted_indices[max_peak_pos], sorted_values[max_peak_pos], 'o', markersize=8, color='red', 
                    zorder=5, label=f'最大峰值 (位置={sorted_indices[max_peak_pos]}, 强度={sorted_values[max_peak_pos]:.2f})')
            
            plt.xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标", fontsize=13, fontweight='bold')
            plt.ylabel('归一化幅值', fontsize=13, fontweight='bold')
            plt.title(f'峰值强度随坐标变化（按坐标排序）\n像素({point[0]}, {point[1]}) - {direction.upper()}方向', 
                     fontsize=15, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.4, linestyle='--')
            
            path = os.path.join(pixel_save_dir, f'{base_name}_08_峰值分析_按坐标.png')
            plt.tight_layout(pad=1.5)
            plt.savefig(path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
            plt.close(fig_sub)
            return path
        
        # 执行保存（保存8张子图）
        saved_files = []
        try:
            saved_files.append(('原始强度序列', save_subplot_1()))
            saved_files.append(('差分信号', save_subplot_2()))
            saved_files.append(('复数信号幅值', save_subplot_3()))
            saved_files.append(('完整频谱_线性', save_subplot_4()))
            saved_files.append(('完整频谱_对数', save_subplot_5()))
            saved_files.append(('峰值局部放大', save_subplot_6()))
            saved_files.append(('峰值分析_按强度', save_subplot_7()))
            saved_files.append(('峰值分析_按坐标', save_subplot_8()))
            
            print(f"\n✓ 已保存汇总图和 {len(saved_files)} 张独立子图到文件夹 {pixel_folder_name}:")
            # 显示汇总图
            summary_file_size = os.path.getsize(save_path) / 1024  # KB
            print(f"  ✓ {'8子图汇总':16s} - {os.path.basename(save_path):50s} ({summary_file_size:.1f} KB)")
            # 显示各个子图
            for name, path in saved_files:
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"  ✓ {name:16s} - {os.path.basename(path):50s} ({file_size:.1f} KB)")
        
        except Exception as e:
            print(f"⚠ 警告：保存子图时出错: {str(e)[:100]}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"⚠ 警告：无法显示图形窗口: {str(e)[:50]}")
    
    print("="*80)
    return True


def visualize_multi_point_spectrum(spectrum_data_dict: dict, save_dir: str = None):
    """
    可视化多个点的频谱数据，为每个点生成独立的分析图
    
    Parameters:
    -----------
    spectrum_data_dict : dict
        多点频谱数据字典，键为点索引，值为频谱数据
    save_dir : str, optional
        保存图像的目录路径
    
    Returns:
    --------
    bool
        成功返回True，失败返回False
    """
    if not spectrum_data_dict:
        print("\n⊙ 无频谱数据需要可视化")
        return False
    
    print(f"\n🎨 开始可视化 {len(spectrum_data_dict)} 个点的频谱数据...")
    
    success_count = 0
    for point_idx, spectrum_data in spectrum_data_dict.items():
        try:
            point = spectrum_data['point']
            direction = spectrum_data['direction']
            
            print(f"\n  正在处理点{point_idx+1}: ({point[0]}, {point[1]}) - {direction.upper()}方向")
            
            # 为每个点创建专用文件夹和保存路径
            if save_dir:
                # 创建像素点专用文件夹
                pixel_folder_name = f"spectrum_point_{point[0]}_{point[1]}_{direction}"
                pixel_folder_path = os.path.join(save_dir, pixel_folder_name)
                os.makedirs(pixel_folder_path, exist_ok=True)
                
                # 直接设置保存到像素点文件夹中的路径
                point_save_path = os.path.join(pixel_folder_path, f"spectrum_point_{point[0]}_{point[1]}_{direction}_analysis.png")
            else:
                point_save_path = None
            
            # 调用单点可视化函数
            result = visualize_spectrum(spectrum_data, point_save_path)
            
            if result:
                success_count += 1
                print(f"    ✓ 点{point_idx+1} 频谱分析完成")
            else:
                print(f"    ✗ 点{point_idx+1} 频谱分析失败")
                
        except Exception as e:
            print(f"    ✗ 点{point_idx+1} 处理出错: {str(e)[:100]}")
    
    print(f"\n🎨 多点频谱可视化完成: {success_count}/{len(spectrum_data_dict)} 个点成功")
    return success_count > 0


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


def main():
    """
    主函数
    
    ============================================================================
    📋 配置说明：所有可配置参数都在下方，修改后直接运行即可
    ============================================================================
    """
    
    # ========================================================================
    # 🔧 配置参数区域 - 在这里修改所有参数
    # ========================================================================
    
    # -------------------- 1. 投影仪参数 --------------------
    PROJECTOR_WIDTH = 1920      # 投影仪宽度（像素）
    PROJECTOR_HEIGHT = 1080     # 投影仪高度（像素）
    
    # -------------------- 2. FSA采集参数 --------------------
    NUM_FREQUENCIES = 151      # 采集的频率数量（M值）
                                # 注意：总图像数 = NUM_FREQUENCIES × 4 (四步相移)
    FREQUENCY_STEP = 1.0        # 频率步长（Hz）
                                # 1.0: 标准步长，频率为 0, 1, 2, ..., 150 Hz
    
    # -------------------- 3. 解码方向选择 --------------------
    # 控制解码哪个方向：
    # - "both": 解码两个方向（水平+垂直），获得完整的2D对应关系
    # - "ver" : 仅解码垂直条纹（获取投影仪列坐标/X坐标）
    # - "hor" : 仅解码水平条纹（获取投影仪行坐标/Y坐标）
    DECODE_DIRECTION = "both"   # 可选: "both", "ver", "hor"
    
    # -------------------- 4. 图像路径配置 --------------------
    # 请根据实际情况修改以下路径
    IMAGE_DIR_HOR = r"E:\images\2025_12_27\images_FSM\place1\03_bookshelf_white_cycle1\FSM_horizontal_f0_f150"  # 水平条纹图像文件夹
    IMAGE_DIR_VER = r"E:\images\2025_12_27\images_FSM\place1\03_bookshelf_white_cycle1\FSM_vertical_f0_f150"    # 垂直条纹图像文件夹
    OUTPUT_DIR = r"results2\fsa_decode"  # 输出结果文件夹
    # 提示：如果路径包含中文，确保使用 r"..." 原始字符串
    
    # -------------------- 5. 处理参数 --------------------
    BLOCK_SIZE = 100            # 分块处理大小
    USE_CACHE = True            # 是否使用缓存（加速重复加载）
    VERIFY_SEQUENCE = True      # 是否验证图像序列顺序
    
    # -------------------- 6. 内存模式选择 --------------------
    # 🔥 LOW_MEMORY_MODE: 低内存模式开关
    # 
    # False (标准模式 - 推荐):
    #   - 内存占用: 约 12GB
    #   - 速度: 快（约需 1-3 分钟）
    #   - 适用: 内存 ≥ 16GB 的系统
    #
    # True (低内存模式):
    #   - 内存占用: 约 100MB
    #   - 速度: 较慢（约需 10-30 分钟）
    #   - 适用: 内存 < 12GB 的系统
    #   - 推荐: 如果遇到内存错误，请设为 True
    
    LOW_MEMORY_MODE = False     # 内存不足时设为 True
    
    # -------------------- 7. 频谱分析参数 --------------------
    # 用途：指定一个或多个像素点，解码器会保存这些点的完整DFT频谱
    # 格式：(x, y) 或 [(x1, y1), (x2, y2), ...] 或 None
    # 
    # 示例：
    #   SPECTRUM_POINTS = (1200, 1000)                    # 分析单个像素点
    #   SPECTRUM_POINTS = [(1200, 1000), (800, 600)]      # 分析多个像素点
    #   SPECTRUM_POINTS = None                             # 不进行频谱分析
    #
    # 注意：每个像素点的频谱数据将保存在独立的文件夹中
    SPECTRUM_POINTS = (1200, 1000) # [(1200, 1000), (800, 600), (1500, 800)]  # 设为None则不进行频谱分析
    
    # ========================================================================
    # ✅ 配置参数区域结束 - 下方为程序执行代码，一般不需要修改
    # ========================================================================
    
    # 调用解码函数
    decode_fsa_images(
        projector_width=PROJECTOR_WIDTH,
        projector_height=PROJECTOR_HEIGHT,
        num_frequencies=NUM_FREQUENCIES,
        decode_direction=DECODE_DIRECTION,
        image_dir_hor=IMAGE_DIR_HOR,
        image_dir_ver=IMAGE_DIR_VER,
        output_dir=OUTPUT_DIR,
        block_size=BLOCK_SIZE,
        use_cache=USE_CACHE,
        verify_sequence=VERIFY_SEQUENCE,
        low_memory_mode=LOW_MEMORY_MODE,
        spectrum_points=SPECTRUM_POINTS,
        frequency_step=FREQUENCY_STEP
    )


if __name__ == "__main__":
    main()

