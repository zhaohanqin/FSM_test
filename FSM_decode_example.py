"""
FSA解码使用示例

本文件展示如何使用 FSM_decode_core.py 中的FSADecoder类对实际采集的条纹图像进行解码。

支持的解码模式：
  "both" → 解码两个方向，获得完整2D像素对应关系
  "ver"  → 仅解码垂直方向，获得投影仪列坐标
  "hor"  → 仅解码水平方向，获得投影仪行坐标

★ FREQUENCY_STEP 参数说明（核心改进）：
-----------------------------------------
  FREQUENCY_STEP = 1.0（默认/标准模式）
    频率序列：0, 1, 2, ..., NUM_FREQUENCIES-1
    FFT长度 = 投影仪分辨率，argmax 直接输出投影仪坐标，精度 1 像素

  FREQUENCY_STEP = 0.5（新增支持）
    频率序列：0, 0.5, 1.0, ..., (NUM_FREQUENCIES-1)*0.5
    示例：最高频率150Hz → NUM_FREQUENCIES=301，总图像数=1204张
    FFT长度 = 2 × 投影仪分辨率，有效bin范围 [0, proj_dim-1]
    argmax 仍直接输出投影仪坐标，无量化阶梯效应
    ⚠ 注意：NUM_FREQUENCIES 须设为 301 以匹配实际采集图像数量

  其他步长（如 0.25）：通用公式 fft_length = proj_dim/step 自动处理

图像序列组织方式（以4步相移为例）：
  0.png~3.png   → 频率0   的 (0°, 90°, 180°, 270°)
  4.png~7.png   → 频率1*s 的 (0°, 90°, 180°, 270°)
  ...（s = FREQUENCY_STEP）
"""

import numpy as np
import os
from FSM_decode_core import FSADecoder, MultiPeakDetector, EpipolarCorrector, CalibrationReader


# =============================================================================
# 频谱数据保存
# =============================================================================

def save_spectrum_data(spectrum_data, output_dir, direction, spectrum_points):
    """保存多点频谱数据"""
    if spectrum_data is None:
        print(f"\n⊙ {direction.upper()}方向: 无频谱数据需要保存")
        return

    print(f"\n✓ 保存{direction.upper()}方向频谱数据:")
    for point_idx, point_data in spectrum_data.items():
        point = point_data['point']
        point_dir = os.path.join(output_dir, f"spectrum_point_{point[0]}_{point[1]}_{direction}")
        os.makedirs(point_dir, exist_ok=True)
        np.save(os.path.join(point_dir, f"spectrum_{direction}.npy"), point_data)
        print(f"  点{point_idx+1} ({point[0]}, {point[1]}): 已保存到 {point_dir}")
    print(f"  共保存了 {len(spectrum_data)} 个点的{direction.upper()}方向频谱数据")


# =============================================================================
# 主解码函数
# =============================================================================

def decode_fsa_images(projector_width, projector_height, num_frequencies, decode_direction,
                      image_dir_hor, image_dir_ver, output_dir, block_size, use_cache,
                      verify_sequence, low_memory_mode, spectrum_points, frequency_step=1.0,
                      multi_peak_scan=False, multi_peak_scan_limit=0.5):
    """
    使用FSADecoder解码实际采集的图像。

    Parameters
    ----------
    projector_width, projector_height : int
    num_frequencies : int
        频率数量 M。FREQUENCY_STEP=0.5 时须设为 301（最高频150Hz）。
    decode_direction : str
        "both" / "ver" / "hor"
    image_dir_hor, image_dir_ver : str
    output_dir : str
    block_size : int
    use_cache, verify_sequence, low_memory_mode : bool
    spectrum_points : tuple(x,y) or list of tuples or None
    frequency_step : float
        频率步长。1.0=标准；0.5=新增支持；其他正数=通用公式自动处理。
    multi_peak_scan : bool
    multi_peak_scan_limit : float

    Returns
    -------
    output_dir, spectrum_data_list, images_cache
    """
    if decode_direction not in ("both", "ver", "hor"):
        print(f"[ERROR] DECODE_DIRECTION 无效: '{decode_direction}'，有效值: 'both','ver','hor'")
        return None, [], {}

    print("="*80)
    print("FSA 解码程序")
    print("="*80)
    print(f"投影仪分辨率: {projector_width} × {projector_height}")
    print(f"采集频率数: {num_frequencies}，频率步长: {frequency_step} Hz")
    print(f"最高频率: {(num_frequencies-1)*frequency_step} Hz，总图像数: {num_frequencies*4}")
    print(f"解码方向: {decode_direction.upper()}")

    # ★ 显示通用公式计算出的FFT参数
    if frequency_step == 1.0:
        fft_h, fft_w = projector_height, projector_width
        print(f"★ FFT长度: {fft_h}(hor) / {fft_w}(ver)，有效bin=全部，argmax直接输出坐标")
    else:
        fft_h = int(round(projector_height / frequency_step))
        fft_w = int(round(projector_width  / frequency_step))
        print(f"★ FFT长度: {fft_h}(hor) / {fft_w}(ver)")
        print(f"  有效bin: 0~{projector_height-1}(hor) / 0~{projector_width-1}(ver)")
        print(f"  argmax 在有效bin范围内直接输出投影仪坐标，无量化阶梯")
        if frequency_step == 0.5:
            old_cs_h = projector_height / num_frequencies
            old_cs_w = projector_width  / num_frequencies
            print(f"  （旧方案量化间隔：hor≈{old_cs_h:.2f}px, ver≈{old_cs_w:.2f}px，已消除）")

    print(f"输出目录: {output_dir}")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    decoder     = FSADecoder(projector_width, projector_height, frequency_step)
    images_cache = {}
    spectrum_data_list = []

    # ==================== 水平方向 ====================
    if decode_direction in ("both", "hor"):
        print("\n" + "="*80)
        print("解码水平条纹（获取投影仪行坐标 / Y坐标）")
        print("="*80)

        if os.path.exists(image_dir_hor):
            try:
                images_hor = decoder.load_images(image_dir_hor, num_frequencies,
                                                  use_cache=use_cache,
                                                  verify_sequence=verify_sequence)
                images_cache['hor'] = images_hor

                disparity_hor, peak_hor, spectrum_hor = decoder.decode_direction(
                    images=images_hor, direction='hor',
                    num_frequencies=num_frequencies, block_size=block_size,
                    low_memory=low_memory_mode, spectrum_points=spectrum_points
                )
                decoder.save_results(disparity_hor, peak_hor, output_dir, 'hor')

                print(f"\n质量评估（水平）: "
                      f"高质量(>1.0)={np.sum(peak_hor>1.0)/peak_hor.size*100:.1f}%  "
                      f"中等(0.5~1.0)={np.sum((peak_hor>0.5)&(peak_hor<=1.0))/peak_hor.size*100:.1f}%  "
                      f"低(<0.5)={np.sum(peak_hor<=0.5)/peak_hor.size*100:.1f}%")

                if spectrum_hor:
                    save_spectrum_data(spectrum_hor, output_dir, 'hor', spectrum_points)
                    spectrum_data_list.append(spectrum_hor)
                    visualize_multi_point_spectrum(spectrum_hor, output_dir)

            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"\n⚠ 图像文件夹不存在: {image_dir_hor}")
    else:
        print("\n⊙ 根据配置，跳过水平方向解码")

    # ==================== 垂直方向 ====================
    if decode_direction in ("both", "ver"):
        print("\n" + "="*80)
        print("解码垂直条纹（获取投影仪列坐标 / X坐标）")
        print("="*80)

        if os.path.exists(image_dir_ver):
            try:
                images_ver = decoder.load_images(image_dir_ver, num_frequencies,
                                                  use_cache=use_cache,
                                                  verify_sequence=verify_sequence)
                images_cache['ver'] = images_ver

                disparity_ver, peak_ver, spectrum_ver = decoder.decode_direction(
                    images=images_ver, direction='ver',
                    num_frequencies=num_frequencies, block_size=block_size,
                    low_memory=low_memory_mode, spectrum_points=spectrum_points
                )
                decoder.save_results(disparity_ver, peak_ver, output_dir, 'ver')

                print(f"\n质量评估（垂直）: "
                      f"高质量(>1.0)={np.sum(peak_ver>1.0)/peak_ver.size*100:.1f}%  "
                      f"中等(0.5~1.0)={np.sum((peak_ver>0.5)&(peak_ver<=1.0))/peak_ver.size*100:.1f}%  "
                      f"低(<0.5)={np.sum(peak_ver<=0.5)/peak_ver.size*100:.1f}%")

                if spectrum_ver:
                    save_spectrum_data(spectrum_ver, output_dir, 'ver', spectrum_points)
                    spectrum_data_list.append(spectrum_ver)
                    visualize_multi_point_spectrum(spectrum_ver, output_dir)

            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"\n⚠ 图像文件夹不存在: {image_dir_ver}")
    else:
        print("\n⊙ 根据配置，跳过垂直方向解码")

    # ==================== 多峰像素扫描（可选） ====================
    if multi_peak_scan and images_cache:
        print("\n" + "="*80)
        print("多峰像素扫描")
        print("="*80)
        all_mp_coords = set()

        for dir_key in ('hor', 'ver'):
            if dir_key not in images_cache:
                continue
            mp_pixels = MultiPeakDetector.scan_image_for_multi_peaks(
                images=images_cache[dir_key], direction=dir_key,
                proj_height=projector_height, proj_width=projector_width,
                num_frequencies=num_frequencies, frequency_step=frequency_step,
                peak_limit=multi_peak_scan_limit
            )
            if mp_pixels:
                print(f"\n  {dir_key.upper()}方向 多峰像素 TOP20:")
                print(f"  {'序':>3}  {'坐标':>14}  {'主峰':>6}  {'主强':>6}  {'次峰':>6}  {'次强':>6}  {'比值':>5}")
                for i, p in enumerate(mp_pixels[:20]):
                    print(f"  {i+1:3d}  ({p['x']:5d},{p['y']:5d})  "
                          f"{p['primary_pos']:6d}  {p['primary_val']:6.3f}  "
                          f"{p['secondary_pos']:6d}  {p['secondary_val']:6.3f}  "
                          f"{p['ratio']:5.3f}")
                for p in mp_pixels:
                    all_mp_coords.add((p['x'], p['y']))

        if all_mp_coords:
            coords_sorted = sorted(all_mp_coords)
            print(f"\n★ 共 {len(coords_sorted)} 个多峰像素，可复制到 PIXELS_TO_CORRECT：")
            show = coords_sorted[:50]
            print("  PIXELS_TO_CORRECT = [" +
                  ", ".join(f"({x},{y})" for x, y in show) +
                  (", ...]" if len(coords_sorted) > 50 else "]"))

    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✓ 解码完成！")
    print("="*80)
    print(f"结果目录: {output_dir}")

    for fname, desc in [("disp_hor.npy","投影仪行坐标"), ("peak_hor.npy","水平置信度"),
                        ("disp_ver.npy","投影仪列坐标"), ("peak_ver.npy","垂直置信度")]:
        fp = os.path.join(output_dir, fname)
        if os.path.exists(fp):
            print(f"  ✓ {fname:22s} {desc}  ({os.path.getsize(fp)/1024/1024:.2f} MB)")

    return output_dir, spectrum_data_list, images_cache


# =============================================================================
# 极线约束校正
# =============================================================================

def run_epipolar_correction(images_cache, output_dir, projector_width, projector_height,
                            num_frequencies, frequency_step, pixel_coords_to_correct,
                            cam_intrinsic, cam_distortion, cam_rotation, cam_translation,
                            proj_intrinsic, proj_distortion, proj_rotation, proj_translation,
                            peak_limit=0.5, enable_spectrum_visualization=True):
    """
    执行极线约束后处理校正（恢复自 decode_fsa.py 的 fix_fsa_disparity()）。
    """
    print("\n" + "="*80)
    print("极线约束后处理校正")
    print("="*80)
    print(f"待校正像素: {len(pixel_coords_to_correct)} 个，多峰阈值: {peak_limit}")

    if 'hor' not in images_cache or 'ver' not in images_cache:
        print("❌ 需要同时有 hor 和 ver 图像才能执行极线约束校正")
        return None, None, None

    disp_hor_path = os.path.join(output_dir, "disp_hor.npy")
    disp_ver_path = os.path.join(output_dir, "disp_ver.npy")
    if not (os.path.exists(disp_hor_path) and os.path.exists(disp_ver_path)):
        print("❌ 找不到视差图文件，请先完成解码")
        return None, None, None

    raw_disp_hor = np.load(disp_hor_path)
    raw_disp_ver = np.load(disp_ver_path)

    print("\n步骤1: 多峰检测...")
    peaks_hor = MultiPeakDetector.get_peaks_for_pixels(
        images_cache['hor'], pixel_coords_to_correct, 'hor',
        projector_height, projector_width, num_frequencies, frequency_step, peak_limit)
    peaks_ver = MultiPeakDetector.get_peaks_for_pixels(
        images_cache['ver'], pixel_coords_to_correct, 'ver',
        projector_height, projector_width, num_frequencies, frequency_step, peak_limit)

    for i, (x, y) in enumerate(pixel_coords_to_correct):
        print(f"  ({x},{y}): hor候选={peaks_hor[i]}, ver候选={peaks_ver[i]}")

    if enable_spectrum_visualization:
        print("\n步骤1.5: 校正像素频谱可视化...")
        ep_sp_dir = os.path.join(output_dir, "epipolar_spectrum")
        os.makedirs(ep_sp_dir, exist_ok=True)
        for dir_key in ('hor', 'ver'):
            if dir_key not in images_cache:
                continue
            sp_dict = MultiPeakDetector.generate_spectrum_data_for_pixels(
                images_cache[dir_key], pixel_coords_to_correct, dir_key,
                projector_height, projector_width, num_frequencies, frequency_step)
            for pt_idx, sp_data in sp_dict.items():
                pt = sp_data['point']
                pt_dir = os.path.join(ep_sp_dir, f"pixel_{pt[0]}_{pt[1]}")
                os.makedirs(pt_dir, exist_ok=True)
                np.save(os.path.join(pt_dir, f"spectrum_{dir_key}.npy"), sp_data)
                save_path = os.path.join(pt_dir, f"spectrum_{pt[0]}_{pt[1]}_{dir_key}.png")
                try:
                    visualize_spectrum(sp_data, save_path)
                except Exception as e:
                    print(f"    ⚠ 可视化失败: {str(e)[:80]}")
        print(f"  ✓ 频谱分析图已保存到: {ep_sp_dir}")

    print("\n步骤2: 极线约束校正...")
    corrector = EpipolarCorrector(
        cam_intrinsic, cam_distortion, cam_rotation, cam_translation,
        proj_intrinsic, proj_distortion, proj_rotation, proj_translation)
    results = corrector.correct_disparity(
        pixel_coords_to_correct, peaks_hor, peaks_ver, raw_disp_hor, raw_disp_ver)

    print("\n步骤3: 应用校正...")
    corr_hor = raw_disp_hor.copy()
    corr_ver = raw_disp_ver.copy()
    for result, (cx, cy) in zip(results, pixel_coords_to_correct):
        if result['corrected']:
            old_py = result.get('original_py', corr_hor[cy, cx])
            old_px = result.get('original_px', corr_ver[cy, cx])
            corr_hor[cy, cx] = result['py']
            corr_ver[cy, cx] = result['px']
            print(f"  ({cx},{cy}): hor {old_py:.1f}→{result['py']:.1f}, "
                  f"ver {old_px:.1f}→{result['px']:.1f}, 极线距={result['distance']:.2f}")
        else:
            print(f"  ({cx},{cy}): 未校正（无候选峰值）")

    np.save(os.path.join(output_dir, "disp_hor_corrected.npy"), corr_hor)
    np.save(os.path.join(output_dir, "disp_ver_corrected.npy"), corr_ver)
    print(f"\n✓ 校正后视差图已保存到 {output_dir}")
    return results, corr_hor, corr_ver


# =============================================================================
# 频谱可视化
# =============================================================================

def visualize_spectrum(spectrum_data: dict, save_path: str = None):
    """可视化频谱数据，输出 2×4 共8张子图的总图与独立子图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
    except ImportError as e:
        print(f"[ERROR] 无法导入 matplotlib: {e}")
        return None

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in ['SimHei', 'Microsoft YaHei', 'STHeiti', 'DejaVu Sans']:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9

    point          = spectrum_data['point']
    direction      = spectrum_data['direction']
    N              = spectrum_data['N']
    valid_length   = spectrum_data.get('valid_length', N)   # ★ 兼容旧格式
    M              = spectrum_data['num_frequencies']
    frequency_step = spectrum_data.get('frequency_step', 1.0)
    spectrum       = spectrum_data['spectrum']
    frequencies    = spectrum_data['frequencies']
    peak_idx       = spectrum_data['peak_index']
    dir_label      = '行' if direction == 'hor' else '列'
    freq_axis      = np.arange(M) * frequency_step
    D_mag          = np.abs(spectrum_data['Dc'] + 1j * spectrum_data['Ds'])
    sp_log         = np.log10(spectrum + 1e-10)
    zoom_s         = max(0, peak_idx - 100)
    zoom_e         = min(N, peak_idx + 100)

    # 前20个峰值（仅在有效范围内排序）
    top20_idx = np.argsort(spectrum[:valid_length])[-20:][::-1]
    top20_val = spectrum[top20_idx]
    sord      = np.argsort(top20_idx)
    s_idx     = top20_idx[sord];  s_val = top20_val[sord]
    s_max_pos = np.where(s_idx == top20_idx[0])[0][0]

    def sp1(ax):
        for I, lbl, mk in [(spectrum_data['I0'],'I0(0°)','o-'),
                            (spectrum_data['I1'],'I1(90°)','s-'),
                            (spectrum_data['I2'],'I2(180°)','^-'),
                            (spectrum_data['I3'],'I3(270°)','d-')]:
            ax.plot(freq_axis, I, mk, label=lbl, lw=1.5, ms=3)
        ax.set(xlabel='频率(Hz)', ylabel='强度',
               title=f'1. 原始强度序列\n(M={M}，步长={frequency_step}Hz)')
        ax.legend(); ax.grid(alpha=0.3)

    def sp2(ax):
        ax.plot(freq_axis, spectrum_data['Dc'], 'o-', label='Dc=I0-I2', lw=1.5, ms=3)
        ax.plot(freq_axis, spectrum_data['Ds'], 's-', label='Ds=I3-I1', lw=1.5, ms=3)
        ax.set(xlabel='频率(Hz)', ylabel='差分值', title='2. 差分信号')
        ax.legend(); ax.grid(alpha=0.3)

    def sp3(ax):
        ax.plot(freq_axis, D_mag, 'o-', color='purple', lw=1.5, ms=3)
        ax.set(xlabel='频率(Hz)', ylabel='|D|', title='3. 复数信号幅值\nD=Dc+j·Ds')
        ax.grid(alpha=0.3)

    def sp4(ax):
        ax.plot(frequencies[:valid_length], spectrum[:valid_length], '-', lw=1)
        ax.axvline(peak_idx, color='red', ls='--', lw=2, label=f'峰值={peak_idx}')
        if valid_length < N:
            ax.axvline(valid_length-1, color='gray', ls=':', lw=1,
                       label=f'有效边界={valid_length-1}')
        ax.set(xlabel=f'投影仪{dir_label}坐标', ylabel='归一化幅值',
               title=f'4. 有效DFT频谱(线性)\n有效bin=0~{valid_length-1}')
        ax.legend(); ax.grid(alpha=0.3)

    def sp5(ax):
        ax.plot(frequencies[:valid_length], sp_log[:valid_length], '-', color='green', lw=1)
        ax.axvline(peak_idx, color='red', ls='--', lw=2, label=f'峰值={peak_idx}')
        ax.set(xlabel=f'投影仪{dir_label}坐标', ylabel='log₁₀(幅值)', title='5. 有效DFT频谱(对数)')
        ax.legend(); ax.grid(alpha=0.3)

    def sp6(ax):
        ax.plot(frequencies[zoom_s:zoom_e], spectrum[zoom_s:zoom_e], 'o-', lw=2, ms=4, color='purple')
        ax.axvline(peak_idx, color='red', ls='--', lw=2, label=f'峰值={peak_idx}')
        ax.set(xlabel=f'投影仪{dir_label}坐标', ylabel='归一化幅值',
               title=f'6. 峰值局部放大\n[{zoom_s},{zoom_e}]')
        ax.legend(); ax.grid(alpha=0.3)

    def sp7(ax):
        ax.plot(top20_idx, top20_val, 'o-', lw=2, ms=6, color='blue', alpha=0.7)
        ax.plot(top20_idx[0], top20_val[0], 'o', ms=10, color='red',
                label=f'最大峰值(位置={top20_idx[0]},强度={top20_val[0]:.2f})')
        ax.set(xlabel=f'投影仪{dir_label}坐标', ylabel='归一化幅值',
               title='7. 前20峰值（有效范围，按强度降序）')
        ax.legend(); ax.grid(alpha=0.3)

    def sp8(ax):
        ax.plot(s_idx, s_val, 'o-', lw=2, ms=6, color='blue', alpha=0.7)
        ax.plot(s_idx[s_max_pos], s_val[s_max_pos], 'o', ms=10, color='red',
                label=f'最大峰值(位置={s_idx[s_max_pos]})', zorder=5)
        ax.set(xlabel=f'投影仪{dir_label}坐标', ylabel='归一化幅值',
               title='8. 峰值强度随坐标变化（按坐标排序）')
        ax.legend(); ax.grid(alpha=0.3)

    drawers = [sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8]
    names   = ["1_raw_intensity", "2_diff_signal", "3_complex_mag",
               "4_spectrum_linear", "5_spectrum_log", "6_peak_zoom",
               "7_top20_peaks", "8_peaks_by_coord"]

    fig = plt.figure(figsize=(20, 10))
    for i, fn in enumerate(drawers):
        fn(plt.subplot(2, 4, i+1))
    plt.suptitle(f'FSA频谱分析 - 像素({point[0]},{point[1]}) - {direction.upper()}\n'
                 f'FFT={N}, 有效bin={valid_length}, 峰值位置={peak_idx}, '
                 f'强度={spectrum_data["peak_value"]:.4f}',
                 fontsize=14, fontweight='bold')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 总图已保存: {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

    if save_path:
        sub_dir  = os.path.join(os.path.dirname(save_path), "subplots")
        os.makedirs(sub_dir, exist_ok=True)
        base     = os.path.splitext(os.path.basename(save_path))[0]
        title_px = f'像素({point[0]},{point[1]}) - {direction.upper()}'
        for fn, nm in zip(drawers, names):
            fig_s = plt.figure(figsize=(8, 6))
            fn(fig_s.add_subplot(1, 1, 1))
            fig_s.suptitle(title_px, fontsize=11, fontweight='bold')
            fig_s.tight_layout(rect=[0, 0, 1, 0.94])
            fig_s.savefig(os.path.join(sub_dir, f"{base}_{nm}.png"),
                          dpi=150, facecolor='white')
            plt.close(fig_s)
        print(f"✓ 8张独立子图已保存: {sub_dir}")

    return True


def visualize_multi_point_spectrum(spectrum_data_dict: dict, save_dir: str = None):
    """批量可视化多个点的频谱"""
    if not spectrum_data_dict:
        return False
    print(f"\n开始可视化 {len(spectrum_data_dict)} 个点的频谱...")
    ok = 0
    for pt_idx, sp_data in spectrum_data_dict.items():
        pt  = sp_data['point'];  dr = sp_data['direction']
        save_path = (os.path.join(save_dir,
                     f"spectrum_point_{pt[0]}_{pt[1]}_{dr}",
                     f"spectrum_{pt[0]}_{pt[1]}_{dr}_analysis.png")
                     if save_dir else None)
        try:
            if visualize_spectrum(sp_data, save_path):
                ok += 1
        except Exception as e:
            print(f"  ✗ 点{pt_idx+1} 出错: {str(e)[:80]}")
    print(f"频谱可视化完成: {ok}/{len(spectrum_data_dict)} 个点成功")
    return ok > 0


def load_and_inspect_results(result_dir: str):
    """加载并检查解码结果"""
    print("\n" + "="*80)
    print("检查解码结果")
    print("="*80)
    results = {}
    for fname in ('disp_hor.npy', 'peak_hor.npy', 'disp_ver.npy', 'peak_ver.npy'):
        fp = os.path.join(result_dir, fname)
        if os.path.exists(fp):
            d = np.load(fp)
            results[fname] = d
            print(f"\n✓ {fname}: shape={d.shape}, range=[{d.min():.2f},{d.max():.2f}], mean={d.mean():.2f}")
            if 'disp' in fname:
                uniq = len(np.unique(d.astype(np.int32)))
                if uniq < 100:
                    print(f"  ⚠ 唯一整数值仅 {uniq} 个，可能存在量化阶梯")
        else:
            print(f"\n✗ {fname}: 文件不存在")

    if 'disp_hor.npy' in results and 'disp_ver.npy' in results:
        yc, xc = 500, 500
        if yc < results['disp_hor.npy'].shape[0] and xc < results['disp_hor.npy'].shape[1]:
            yp = results['disp_hor.npy'][yc, xc]
            xp = results['disp_ver.npy'][yc, xc]
            print(f"\n示例: 相机像素({xc},{yc}) → 投影仪像素({xp:.2f},{yp:.2f})")


# =============================================================================
# main()
# =============================================================================

def main():
    """
    主函数 — 所有可配置参数均在下方配置区域内修改后直接运行

    ============================================================================
    配置说明
    ============================================================================
    """

    # =========================================================================
    # 🔧 配置参数区域（仅需修改此区域）
    # =========================================================================

    # ---------- 1. 投影仪参数 ----------
    PROJECTOR_WIDTH  = 1920
    PROJECTOR_HEIGHT = 1080

    # ---------- 2. FSA采集参数 ----------
    NUM_FREQUENCIES = 301   # 频率数量 M
    FREQUENCY_STEP  = 0.5   # 频率步长（Hz）
    #
    # ★★★ FREQUENCY_STEP 说明 ★★★
    #
    #   FREQUENCY_STEP = 1.0（标准模式）
    #     频率: 0,1,...,150，总图像: 151×4=604张
    #     FFT长度 = 投影仪分辨率，argmax 直接输出坐标，精度1像素
    #
    #   FREQUENCY_STEP = 0.5（半步长模式，新增支持）
    #     频率: 0,0.5,1.0,...,150，→ NUM_FREQUENCIES = 301，总图像: 301×4=1204张
    #     FFT长度 = 2×投影仪分辨率，有效bin=[0,proj_dim-1]
    #     argmax 仍直接输出坐标，无量化阶梯效应
    #     ⚠ 记得同步修改 NUM_FREQUENCIES = 301
    #
    #   示例（切换到半步长模式）：
    #     NUM_FREQUENCIES = 301
    #     FREQUENCY_STEP  = 0.5

    # ---------- 3. 解码方向 ----------
    DECODE_DIRECTION = "both"   # "both" / "ver" / "hor"

    # ---------- 4. 路径配置 ----------
    IMAGE_DIR_HOR = r"H:\images\before_2025_12_27\images_FSM\place1\04_bookshelf_white_cycle0_5\FSM_horizontal_f0_f150"
    IMAGE_DIR_VER = r"H:\images\before_2025_12_27\images_FSM\place1\04_bookshelf_white_cycle0_5\FSM_vertical_f0_f150"
    OUTPUT_DIR    = r"results\fsa_decode"

    # ---------- 5. 处理参数 ----------
    BLOCK_SIZE      = 100
    USE_CACHE       = True
    VERIFY_SEQUENCE = True

    # ---------- 6. 内存模式 ----------
    LOW_MEMORY_MODE = True
    # ★ FREQUENCY_STEP=0.5 时内存占用约为标准模式的2倍
    #   内存不足时可减小 BLOCK_SIZE 或设 LOW_MEMORY_MODE=True

    # ---------- 7. 频谱分析点 ----------
    SPECTRUM_POINTS = (1200, 1000)   # 单点：(x,y)；多点：[(x1,y1),(x2,y2),...]

    # ---------- 7.5 多峰像素扫描（可选） ----------
    ENABLE_MULTI_PEAK_SCAN = False
    MULTI_PEAK_SCAN_LIMIT  = 0.5

    # ---------- 8. 极线约束校正（可选） ----------
    ENABLE_EPIPOLAR_CORRECTION  = False
    CALIBRATION_FILE            = r"标定结果.txt"
    PIXELS_TO_CORRECT           = [(900, 495)]
    PEAK_LIMIT                  = 0.5
    ENABLE_EPIPOLAR_SPECTRUM_VIZ = True

    # =========================================================================
    # ✅ 配置参数区域结束
    # =========================================================================

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
        frequency_step=FREQUENCY_STEP,
        multi_peak_scan=ENABLE_MULTI_PEAK_SCAN,
        multi_peak_scan_limit=MULTI_PEAK_SCAN_LIMIT
    )

    if ENABLE_EPIPOLAR_CORRECTION and output_dir is not None:
        if DECODE_DIRECTION != "both":
            print("\n⚠ 极线约束校正需要 DECODE_DIRECTION='both'")
        else:
            try:
                calib_data      = CalibrationReader.read_calibration_file(CALIBRATION_FILE)
                epipolar_params = CalibrationReader.extract_epipolar_params(calib_data)
                run_epipolar_correction(
                    images_cache=images_cache,
                    output_dir=OUTPUT_DIR,
                    projector_width=PROJECTOR_WIDTH,
                    projector_height=PROJECTOR_HEIGHT,
                    num_frequencies=NUM_FREQUENCIES,
                    frequency_step=FREQUENCY_STEP,
                    pixel_coords_to_correct=PIXELS_TO_CORRECT,
                    cam_intrinsic=epipolar_params['cam_intrinsic'],
                    cam_distortion=epipolar_params['cam_distortion'],
                    cam_rotation=epipolar_params['cam_rotation'],
                    cam_translation=epipolar_params['cam_translation'],
                    proj_intrinsic=epipolar_params['proj_intrinsic'],
                    proj_distortion=epipolar_params['proj_distortion'],
                    proj_rotation=epipolar_params['proj_rotation'],
                    proj_translation=epipolar_params['proj_translation'],
                    peak_limit=PEAK_LIMIT,
                    enable_spectrum_visualization=ENABLE_EPIPOLAR_SPECTRUM_VIZ
                )
            except FileNotFoundError:
                print(f"\n❌ 找不到标定文件: {CALIBRATION_FILE}")
            except Exception as e:
                print(f"\n❌ 极线约束校正出错: {e}")
                import traceback; traceback.print_exc()

    if output_dir is not None:
        load_and_inspect_results(output_dir)


if __name__ == "__main__":
    main()
