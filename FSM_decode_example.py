"""
FSA解码使用示例

本文件展示如何使用 FSM_decode_core.py 中的FSADecoder类
对实际采集的条纹图像进行解码

支持的解码模式：
---------------
1. 解码两个方向（both）：获得完整的2D像素对应关系
2. 仅解码垂直方向（ver）：获得投影仪列坐标
3. 仅解码水平方向（hor）：获得投影仪行坐标

新增后处理功能（恢复自 decode_fsa.py）：
-----------------------------------------
4. 多峰检测（MultiPeakDetector）：找出频谱中所有候选峰值
5. 极线约束校正（EpipolarCorrector）：利用极线几何修正误匹配

图像序列的组织方式：
-------------------
- 0.png, 1.png, 2.png, 3.png → 频率0的 (0°, 90°, 180°, 270°)
- 4.png, 5.png, 6.png, 7.png → 频率1的 (0°, 90°, 180°, 270°)
- ... 以此类推
"""

import numpy as np
import os
from FSM_decode_core import FSADecoder, MultiPeakDetector, EpipolarCorrector


def save_spectrum_data(spectrum_data, output_dir, direction, spectrum_points):
    """
    保存多点频谱数据到不同的文件夹
    """
    if spectrum_data is None:
        print(f"\n⊙ {direction.upper()}方向: 无频谱数据需要保存")
        return
    
    if isinstance(spectrum_points, tuple) and len(spectrum_points) == 2 and not isinstance(spectrum_points[0], tuple):
        points_list = [spectrum_points]
    else:
        points_list = list(spectrum_points) if spectrum_points else []
    
    print(f"\n✓ 保存{direction.upper()}方向频谱数据:")
    
    for point_idx, point_data in spectrum_data.items():
        point = point_data['point']
        
        point_dir = os.path.join(output_dir, f"spectrum_point_{point[0]}_{point[1]}_{direction}")
        os.makedirs(point_dir, exist_ok=True)
        
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
        频谱分析点坐标
    frequency_step : float
        频率步长（Hz），默认为1.0
        
    Returns:
    --------
    output_dir : str
        输出目录路径
    spectrum_data_list : list
        频谱数据列表
    images_cache : dict
        缓存的图像数据（用于后续后处理），键为 'hor' 和/或 'ver'
    """
    
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
    
    if DECODE_DIRECTION not in ["both", "ver", "hor"]:
        print(f"[ERROR] 错误: DECODE_DIRECTION 参数无效: '{DECODE_DIRECTION}'")
        print("   有效值: 'both', 'ver', 'hor'")
        return None, [], {}
    
    print("="*80)
    print("FSA 解码程序")
    print("="*80)
    print(f"投影仪分辨率: {PROJECTOR_WIDTH} × {PROJECTOR_HEIGHT}")
    print(f"采集频率数: {NUM_FREQUENCIES}")
    print(f"频率步长: {FREQUENCY_STEP} Hz")
    print(f"最高频率: {(NUM_FREQUENCIES-1) * FREQUENCY_STEP} Hz")
    print(f"总图像数: {NUM_FREQUENCIES * 4} (4步相移)")
    print(f"解码方向: {DECODE_DIRECTION.upper()}")
    
    # ★ 显示FFT长度信息（帮助用户理解DFT策略）
    if FREQUENCY_STEP == 1.0:
        fft_h = PROJECTOR_HEIGHT
        fft_w = PROJECTOR_WIDTH
        print(f"★ FFT长度策略: 使用投影仪分辨率（零填充插值）")
        print(f"  水平方向FFT长度: {fft_h} (投影仪高度)")
        print(f"  垂直方向FFT长度: {fft_w} (投影仪宽度)")
        print(f"  → argmax直接输出投影仪坐标，精度: 1像素")
    else:
        print(f"★ FFT长度策略: 使用频率数量M={NUM_FREQUENCIES}")
        print(f"  坐标映射比例(hor): {PROJECTOR_HEIGHT/NUM_FREQUENCIES:.4f}")
        print(f"  坐标映射比例(ver): {PROJECTOR_WIDTH/NUM_FREQUENCIES:.4f}")
    
    if DECODE_DIRECTION == "both":
        print("  → 解码两个方向（水平+垂直），获得完整2D对应关系")
    elif DECODE_DIRECTION == "ver":
        print("  → 仅解码垂直条纹，获得投影仪列坐标（X坐标）")
    elif DECODE_DIRECTION == "hor":
        print("  → 仅解码水平条纹，获得投影仪行坐标（Y坐标）")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建解码器实例
    decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT, FREQUENCY_STEP)
    
    # 缓存图像数据（用于后续多峰检测后处理）
    images_cache = {}
    
    # ==================== 解码水平方向 ====================
    if DECODE_DIRECTION in ["both", "hor"]:
        print("\n" + "="*80)
        print("解码水平条纹（获取投影仪行坐标 / Y坐标）")
        print("="*80)
    
    if DECODE_DIRECTION in ["both", "hor"] and os.path.exists(IMAGE_DIR_HOR):
        try:
            images_hor = decoder.load_images(
                image_dir=IMAGE_DIR_HOR,
                num_frequencies=NUM_FREQUENCIES,
                image_pattern="{:d}.png",
                use_cache=USE_CACHE,
                verify_sequence=VERIFY_SEQUENCE
            )
            
            # 缓存图像用于后续后处理
            images_cache['hor'] = images_hor
            
            disparity_hor, peak_hor, spectrum_hor = decoder.decode_direction(
                images=images_hor,
                direction='hor',
                num_frequencies=NUM_FREQUENCIES,
                block_size=BLOCK_SIZE,
                low_memory=LOW_MEMORY_MODE,
                spectrum_points=SPECTRUM_POINTS
            )
            
            decoder.save_results(disparity_hor, peak_hor, OUTPUT_DIR, 'hor')
            
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
            images_ver = decoder.load_images(
                image_dir=IMAGE_DIR_VER,
                num_frequencies=NUM_FREQUENCIES,
                image_pattern="{:d}.png",
                use_cache=USE_CACHE,
                verify_sequence=VERIFY_SEQUENCE
            )
            
            # 缓存图像用于后续后处理
            images_cache['ver'] = images_ver
            
            disparity_ver, peak_ver, spectrum_ver = decoder.decode_direction(
                images=images_ver,
                direction='ver',
                num_frequencies=NUM_FREQUENCIES,
                block_size=BLOCK_SIZE,
                low_memory=LOW_MEMORY_MODE,
                spectrum_points=SPECTRUM_POINTS
            )
            
            decoder.save_results(disparity_ver, peak_ver, OUTPUT_DIR, 'ver')
            
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
    spectrum_data_list = []
    
    if DECODE_DIRECTION in ["both", "hor"] and 'spectrum_hor' in locals() and spectrum_hor is not None:
        save_spectrum_data(spectrum_hor, OUTPUT_DIR, 'hor', SPECTRUM_POINTS)
        spectrum_data_list.append(spectrum_hor)
        visualize_multi_point_spectrum(spectrum_hor, OUTPUT_DIR)
    
    if DECODE_DIRECTION in ["both", "ver"] and 'spectrum_ver' in locals() and spectrum_ver is not None:
        save_spectrum_data(spectrum_ver, OUTPUT_DIR, 'ver', SPECTRUM_POINTS)
        spectrum_data_list.append(spectrum_ver)
        visualize_multi_point_spectrum(spectrum_ver, OUTPUT_DIR)
    
    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✓ 解码完成！")
    print("="*80)
    print(f"\n结果保存位置: {OUTPUT_DIR}")
    print(f"解码方向: {DECODE_DIRECTION.upper()}")
    print("\n生成的文件:")
    
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
            file_size = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✓ {filename:20s} - {description} ({file_size:.2f} MB)")
        else:
            print(f"  ✗ {filename:20s} - 未生成")
    
    print("\n后续步骤:")
    if DECODE_DIRECTION == "both":
        print("  ✓ 已获得完整的2D像素对应关系（投影仪X和Y坐标）")
        print("  1. 使用峰值图过滤低质量像素点")
        print("  2. [可选] 使用极线约束校正修正误匹配（需要标定参数）")
        print("  3. 结合双目标定参数进行三维重建")
    elif DECODE_DIRECTION == "ver":
        print("  ⊙ 仅获得投影仪列坐标（X坐标）")
        print("  ! 如需完整的2D对应关系，还需要采集并解码水平条纹（Y坐标）")
    elif DECODE_DIRECTION == "hor":
        print("  ⊙ 仅获得投影仪行坐标（Y坐标）")
        print("  ! 如需完整的2D对应关系，还需要采集并解码垂直条纹（X坐标）")
    
    return OUTPUT_DIR, spectrum_data_list, images_cache


def run_epipolar_correction(images_cache, output_dir, projector_width, projector_height,
                            num_frequencies, frequency_step,
                            pixel_coords_to_correct,
                            cam_intrinsic, cam_distortion, cam_rotation, cam_translation,
                            proj_intrinsic, proj_distortion, proj_rotation, proj_translation,
                            peak_limit=0.5):
    """
    执行极线约束后处理校正
    
    恢复自 decode_fsa.py 中的 fix_fsa_disparity() 功能。
    
    此函数对指定的像素点执行以下操作：
    1. 使用 MultiPeakDetector 在频谱中找出所有候选峰值
    2. 使用 EpipolarCorrector 利用极线约束选出最佳匹配
    3. 更新视差图
    
    Parameters:
    -----------
    images_cache : dict
        图像缓存，包含 'hor' 和 'ver' 两个方向的图像数据
    output_dir : str
        解码结果目录
    projector_width, projector_height : int
        投影仪分辨率
    num_frequencies : int
        频率数量
    frequency_step : float
        频率步长
    pixel_coords_to_correct : list of (x, y)
        需要校正的相机像素坐标列表
    cam_intrinsic, cam_distortion, cam_rotation, cam_translation : np.ndarray
        相机标定参数
    proj_intrinsic, proj_distortion, proj_rotation, proj_translation : np.ndarray
        投影仪标定参数
    peak_limit : float
        多峰检测阈值，默认0.5
        
    Returns:
    --------
    results : list of dict
        每个像素的校正结果
    corrected_disp_hor : np.ndarray
        校正后的水平视差图
    corrected_disp_ver : np.ndarray
        校正后的垂直视差图
    """
    print("\n" + "="*80)
    print("极线约束后处理校正")
    print("="*80)
    print(f"待校正像素数: {len(pixel_coords_to_correct)}")
    print(f"多峰检测阈值: {peak_limit}")
    
    # 检查必要数据
    if 'hor' not in images_cache or 'ver' not in images_cache:
        print("❌ 错误: 需要同时有水平和垂直方向的图像数据才能执行极线约束校正")
        return None, None, None
    
    # 加载当前视差图
    disp_hor_path = os.path.join(output_dir, "disp_hor.npy")
    disp_ver_path = os.path.join(output_dir, "disp_ver.npy")
    
    if not os.path.exists(disp_hor_path) or not os.path.exists(disp_ver_path):
        print("❌ 错误: 找不到视差图文件，请先完成解码")
        return None, None, None
    
    raw_disp_hor = np.load(disp_hor_path)
    raw_disp_ver = np.load(disp_ver_path)
    
    # 步骤1: 多峰检测（恢复自 decode_fsa.py 的 get_peaks_position）
    print("\n步骤1: 多峰检测...")
    
    print("  检测水平方向候选峰值...")
    peaks_hor = MultiPeakDetector.get_peaks_for_pixels(
        images=images_cache['hor'],
        pixel_coords=pixel_coords_to_correct,
        direction='hor',
        proj_height=projector_height,
        proj_width=projector_width,
        num_frequencies=num_frequencies,
        frequency_step=frequency_step,
        peak_limit=peak_limit
    )
    
    print("  检测垂直方向候选峰值...")
    peaks_ver = MultiPeakDetector.get_peaks_for_pixels(
        images=images_cache['ver'],
        pixel_coords=pixel_coords_to_correct,
        direction='ver',
        proj_height=projector_height,
        proj_width=projector_width,
        num_frequencies=num_frequencies,
        frequency_step=frequency_step,
        peak_limit=peak_limit
    )
    
    for i, (x, y) in enumerate(pixel_coords_to_correct):
        print(f"  像素({x},{y}): hor候选={peaks_hor[i]}, ver候选={peaks_ver[i]}")
    
    # 步骤2: 极线约束校正（恢复自 decode_fsa.py 的 fix_fsa_disparity）
    print("\n步骤2: 极线约束校正...")
    
    corrector = EpipolarCorrector(
        cam_intrinsic=cam_intrinsic,
        cam_distortion=cam_distortion,
        cam_rotation=cam_rotation,
        cam_translation=cam_translation,
        proj_intrinsic=proj_intrinsic,
        proj_distortion=proj_distortion,
        proj_rotation=proj_rotation,
        proj_translation=proj_translation
    )
    
    results = corrector.correct_disparity(
        pixel_coords=pixel_coords_to_correct,
        peaks_hor=peaks_hor,
        peaks_ver=peaks_ver,
        raw_disp_hor=raw_disp_hor,
        raw_disp_ver=raw_disp_ver
    )
    
    # 步骤3: 应用校正并保存
    print("\n步骤3: 应用校正...")
    
    corrected_disp_hor = raw_disp_hor.copy()
    corrected_disp_ver = raw_disp_ver.copy()
    
    for i, (result, (cx, cy)) in enumerate(zip(results, pixel_coords_to_correct)):
        if result['corrected']:
            old_py = result.get('original_py', corrected_disp_hor[cy, cx])
            old_px = result.get('original_px', corrected_disp_ver[cy, cx])
            new_py = result['py']
            new_px = result['px']
            dist = result['distance']
            
            corrected_disp_hor[cy, cx] = new_py
            corrected_disp_ver[cy, cx] = new_px
            
            print(f"  像素({cx},{cy}): "
                  f"hor: {old_py:.1f} → {new_py:.1f}, "
                  f"ver: {old_px:.1f} → {new_px:.1f}, "
                  f"极线距离: {dist:.2f}")
        else:
            print(f"  像素({cx},{cy}): 未校正（无候选峰值）")
    
    # 保存校正结果
    np.save(os.path.join(output_dir, "disp_hor_corrected.npy"), corrected_disp_hor)
    np.save(os.path.join(output_dir, "disp_ver_corrected.npy"), corrected_disp_ver)
    print(f"\n✓ 校正后的视差图已保存到 {output_dir}")
    
    return results, corrected_disp_hor, corrected_disp_ver


def visualize_spectrum(spectrum_data: dict, save_path: str = None):
    """
    可视化频谱数据，展示FSA解码的完整过程
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\n[ERROR] 频谱可视化错误：无法导入 matplotlib")
        print(f"错误原因：{str(e)[:100]}")
        return None
    
    import matplotlib.font_manager as fm
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_candidates = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'DejaVu Sans', 
                       'Arial Unicode MS', 'WenQuanYi Zen Hei']
    selected_font = 'DejaVu Sans'
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    
    print("\n" + "="*80)
    print("频谱可视化分析")
    print("="*80)
    
    point = spectrum_data['point']
    direction = spectrum_data['direction']
    N = spectrum_data['N']
    M = spectrum_data['num_frequencies']
    frequency_step = spectrum_data.get('frequency_step', 1.0)
    
    print(f"分析点: ({point[0]}, {point[1]})")
    print(f"解码方向: {'水平 (投影仪行坐标)' if direction == 'hor' else '垂直 (投影仪列坐标)'}")
    print(f"频率数量: {M}")
    print(f"频率步长: {frequency_step} Hz")
    print(f"FFT长度(N): {N}")
    print(f"最大峰值位置: {spectrum_data['peak_index']}")
    print(f"最大峰值强度: {spectrum_data['peak_value']:.4f}")
    
    fig = plt.figure(figsize=(20, 10))
    
    freq_axis = np.arange(M) * frequency_step
    
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
    ax4.set_title(f'4. 完整DFT频谱（线性坐标）\nFFT长度={N}, 峰值={spectrum_data["peak_value"]:.4f}')
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
    ax5.set_title('5. 完整DFT频谱（对数坐标）')
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
    
    # 7. 前20个峰值分析
    ax7 = plt.subplot(2, 4, 7)
    top20_indices = np.argsort(spectrum)[-20:][::-1]
    top20_values = spectrum[top20_indices]
    
    ax7.plot(top20_indices, top20_values, 'o-', linewidth=2, markersize=6, color='blue', alpha=0.7)
    ax7.plot(top20_indices[0], top20_values[0], 'o', markersize=10, color='red', 
             label=f'最大峰值 (位置={top20_indices[0]}, 强度={top20_values[0]:.2f})')
    
    for ii in range(min(10, len(top20_indices))):
        y_offset = 0.15 if ii % 2 == 0 else -0.25
        ax7.text(top20_indices[ii], top20_values[ii] + y_offset, f'{top20_values[ii]:.2f}', 
                fontsize=8, ha='center', va='center', 
                color='red' if ii == 0 else 'darkblue', fontweight='bold')
    
    ax7.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标")
    ax7.set_ylabel('归一化幅值')
    ax7.set_title('7. 前20个峰值分析（按强度排序）')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)
    
    # 8. 峰值强度随坐标变化
    ax8 = plt.subplot(2, 4, 8)
    sorted_order = np.argsort(top20_indices)
    sorted_indices = top20_indices[sorted_order]
    sorted_values = top20_values[sorted_order]
    max_peak_pos = np.where(sorted_indices == top20_indices[0])[0][0]
    
    ax8.plot(sorted_indices, sorted_values, 'o-', linewidth=2, markersize=6, color='blue', alpha=0.7)
    ax8.plot(sorted_indices[max_peak_pos], sorted_values[max_peak_pos], 'o', markersize=10, color='red', 
             label=f'最大峰值 (位置={sorted_indices[max_peak_pos]})', zorder=5)
    
    ax8.set_xlabel(f"投影仪{'行' if direction == 'hor' else '列'}坐标")
    ax8.set_ylabel('归一化幅值')
    ax8.set_title('8. 峰值强度随坐标变化（按坐标排序）')
    ax8.legend(loc='best')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'FSA频谱分析 - 像素({point[0]}, {point[1]}) - {direction.upper()}方向\n'
                 f'FFT长度={N}, 峰值位置: {peak_idx}, 峰值强度: {spectrum_data["peak_value"]:.4f}', 
                 fontsize=16, fontweight='bold')
    
    if save_path:
        pixel_save_dir = os.path.dirname(save_path)
        os.makedirs(pixel_save_dir, exist_ok=True)
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=150, bbox_inches=None, facecolor='white', edgecolor='none')
        print(f"\n✓ 频谱图像已保存到: {save_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"⚠ 警告：无法显示图形窗口: {str(e)[:50]}")
    
    print("="*80)
    return True


def visualize_multi_point_spectrum(spectrum_data_dict: dict, save_dir: str = None):
    """
    可视化多个点的频谱数据
    """
    if not spectrum_data_dict:
        print("\n⊙ 无频谱数据需要可视化")
        return False
    
    print(f"\n开始可视化 {len(spectrum_data_dict)} 个点的频谱数据...")
    
    success_count = 0
    for point_idx, spectrum_data in spectrum_data_dict.items():
        try:
            point = spectrum_data['point']
            direction = spectrum_data['direction']
            
            print(f"\n  正在处理点{point_idx+1}: ({point[0]}, {point[1]}) - {direction.upper()}方向")
            
            if save_dir:
                pixel_folder_name = f"spectrum_point_{point[0]}_{point[1]}_{direction}"
                pixel_folder_path = os.path.join(save_dir, pixel_folder_name)
                os.makedirs(pixel_folder_path, exist_ok=True)
                point_save_path = os.path.join(pixel_folder_path, 
                    f"spectrum_point_{point[0]}_{point[1]}_{direction}_analysis.png")
            else:
                point_save_path = None
            
            result = visualize_spectrum(spectrum_data, point_save_path)
            
            if result:
                success_count += 1
                print(f"    ✓ 点{point_idx+1} 频谱分析完成")
            else:
                print(f"    ✗ 点{point_idx+1} 频谱分析失败")
                
        except Exception as e:
            print(f"    ✗ 点{point_idx+1} 处理出错: {str(e)[:100]}")
    
    print(f"\n多点频谱可视化完成: {success_count}/{len(spectrum_data_dict)} 个点成功")
    return success_count > 0


def load_and_inspect_results(result_dir: str):
    """
    加载并检查解码结果
    """
    print("\n" + "="*80)
    print("检查解码结果")
    print("="*80)
    
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
            
            # ★ 新增：检查离散值情况（帮助诊断量化阶梯问题）
            if 'disp' in filename:
                unique_count = len(np.unique(data.astype(np.int32)))
                total_pixels = data.size
                print(f"  唯一整数值数: {unique_count}")
                if unique_count < 100:
                    print(f"  ⚠ 警告：唯一值过少，可能存在量化阶梯效应！")
                    print(f"    建议检查FFT长度设置是否使用了投影仪分辨率")
        else:
            print(f"\n✗ {filename}: 文件不存在")
    
    if 'disp_hor.npy' in results and 'disp_ver.npy' in results:
        print("\n" + "-"*80)
        print("相机-投影仪像素对应关系已建立")
        print("-"*80)
        print("  xp = disparity_ver[yc, xc]  (投影仪列坐标)")
        print("  yp = disparity_hor[yc, xc]  (投影仪行坐标)")
        
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
    # 🔧 配置参数区域
    # ========================================================================
    
    # -------------------- 1. 投影仪参数 --------------------
    PROJECTOR_WIDTH = 1920      # 投影仪宽度（像素）
    PROJECTOR_HEIGHT = 1080     # 投影仪高度（像素）
    
    # -------------------- 2. FSA采集参数 --------------------
    NUM_FREQUENCIES = 351       # 采集的频率数量（M值）
    FREQUENCY_STEP = 1.0        # 频率步长（Hz）
                                # ★ 当FREQUENCY_STEP=1.0时，FFT长度=投影仪分辨率（零填充）
                                #   argmax直接输出投影仪坐标，精度1像素
                                # ★ 当FREQUENCY_STEP!=1.0时，FFT长度=NUM_FREQUENCIES
                                #   argmax需要映射到投影仪坐标
    
    # -------------------- 3. 解码方向选择 --------------------
    DECODE_DIRECTION = "both"   # 可选: "both", "ver", "hor"
    
    # -------------------- 4. 图像路径配置 --------------------
    IMAGE_DIR_HOR = r"E:\images\before_2025_12_27\images_FSM\place1\03_bookshelf_white_cycle1\FSM_horizontal_f0_f150"
    IMAGE_DIR_VER = r"E:\images\before_2025_12_27\images_FSM\place1\03_bookshelf_white_cycle1\FSM_vertical_f0_f150"
    OUTPUT_DIR = r"results\fsa_decode"
    
    # -------------------- 5. 处理参数 --------------------
    BLOCK_SIZE = 100
    USE_CACHE = True
    VERIFY_SEQUENCE = True
    
    # -------------------- 6. 内存模式选择 --------------------
    LOW_MEMORY_MODE = True
    
    # -------------------- 7. 频谱分析参数 --------------------
    SPECTRUM_POINTS = (1200, 1000)
    
    # -------------------- 8. 极线约束校正参数（可选） --------------------
    # 设置为True以启用后处理校正（需要标定参数）
    ENABLE_EPIPOLAR_CORRECTION = False
    
    # 需要校正的像素坐标列表（仅当ENABLE_EPIPOLAR_CORRECTION=True时使用）
    PIXELS_TO_CORRECT = [(900, 495)]  # 对应原始代码中的 ax=900, ay=495
    
    # 多峰检测阈值（恢复自原始代码中的limit=0.5）
    PEAK_LIMIT = 0.5
    
    # 标定参数（需要根据实际标定结果填写）
    # 以下为示例值，实际使用时请替换为真实标定数据
    CAM_INTRINSIC = np.eye(3)       # 相机内参 (3x3)
    CAM_DISTORTION = np.zeros(5)    # 相机畸变系数
    CAM_ROTATION = np.zeros(3)      # 相机旋转向量 (Rodrigues)
    CAM_TRANSLATION = np.zeros(3)   # 相机平移向量
    PROJ_INTRINSIC = np.eye(3)      # 投影仪内参 (3x3)
    PROJ_DISTORTION = np.zeros(5)   # 投影仪畸变系数
    PROJ_ROTATION = np.zeros(3)     # 投影仪旋转向量 (Rodrigues)
    PROJ_TRANSLATION = np.zeros(3)  # 投影仪平移向量
    
    # ========================================================================
    # ✅ 配置参数区域结束
    # ========================================================================
    
    # 执行解码
    output_dir, spectrum_data_list, images_cache = decode_fsa_images(
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
    
    # 执行极线约束校正（如果启用）
    if ENABLE_EPIPOLAR_CORRECTION and output_dir is not None:
        if DECODE_DIRECTION == "both":
            run_epipolar_correction(
                images_cache=images_cache,
                output_dir=OUTPUT_DIR,
                projector_width=PROJECTOR_WIDTH,
                projector_height=PROJECTOR_HEIGHT,
                num_frequencies=NUM_FREQUENCIES,
                frequency_step=FREQUENCY_STEP,
                pixel_coords_to_correct=PIXELS_TO_CORRECT,
                cam_intrinsic=CAM_INTRINSIC,
                cam_distortion=CAM_DISTORTION,
                cam_rotation=CAM_ROTATION,
                cam_translation=CAM_TRANSLATION,
                proj_intrinsic=PROJ_INTRINSIC,
                proj_distortion=PROJ_DISTORTION,
                proj_rotation=PROJ_ROTATION,
                proj_translation=PROJ_TRANSLATION,
                peak_limit=PEAK_LIMIT
            )
        else:
            print("\n⚠ 极线约束校正需要同时解码两个方向（DECODE_DIRECTION='both'）")


if __name__ == "__main__":
    main()
