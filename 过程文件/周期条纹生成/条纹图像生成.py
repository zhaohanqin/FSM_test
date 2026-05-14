# -*- coding: utf-8 -*-
# Generate N-step phase-shifted sine BMPs (支持N步相移):
#   - horizontal: 1080×1
#   - vertical: 1×1920
#   - full images: 1920×1080 (for both directions)
# Phase order: 0°, 360°/N, 720°/N, ..., 360°×(N-1)/N
# Examples:
#   - N=3: 0°, 120°, 240°
#   - N=4: 0°, 90°, 180°, 270°
#   - N=5: 0°, 72°, 144°, 216°, 288°
# Requires: Pillow (PIL) and NumPy
#   pip install pillow numpy

import os
import math
import numpy as np
from PIL import Image

def clamp_u8(arr: np.ndarray) -> np.ndarray:
    """Clamp to [0,255] and cast to uint8."""
    return np.clip(arr, 0, 255).astype(np.uint8)

def generate_sine_1d(length: int, cycles: float, phase_deg: float, 
                     amplitude: float = 127.5, offset: float = 127.5) -> np.ndarray:
    """
    Generate a 1D sine wave mapped to [0,255].
    
    Parameters:
        length: number of pixels (for horizontal stripes use HEIGHT, for vertical use WIDTH)
        cycles: number of full sine periods along the 1D array
        phase_deg: phase in degrees (e.g., 0, 90, 180, 270)
        amplitude: 振幅，控制调制度 (默认127.5表示最大调制度)
        offset: 偏移量/平均亮度 (默认127.5)
    
    调制度计算:
        Imin = offset - amplitude
        Imax = offset + amplitude
        调制度 = (Imax - Imin) / (Imax + Imin) = amplitude / offset
        
    示例:
        - amplitude=127.5, offset=127.5 → 调制度=1.0 (最大, 范围0-255)
        - amplitude=63.75, offset=127.5 → 调制度=0.5 (50%, 范围64-191)
        - amplitude=95.625, offset=127.5 → 调制度=0.75 (75%, 范围32-223)
    """
    two_pi = math.tau  # 2*pi
    phase = math.radians(phase_deg)
    x = np.arange(length, dtype=np.float64)
    angle = two_pi * (x / float(length)) * cycles + phase
    vals = np.rint(offset + amplitude * np.cos(angle))  # 调制度可控的余弦波(与test.py一致)
    return clamp_u8(vals)

def save_fringe_patterns(height: int, width: int, cycles: float, output_dir_1d: str, output_dir_full: str,
                        phase_shift_step: int = 4, amplitude: float = 127.5, offset: float = 127.5,
                        direction: str = 'both', save_1d_data: bool = True, 
                        save_full_image: bool = True, save_full_4phase: bool = True):
    """
    Generate N-step phase-shifted fringe patterns:
    - horizontal stripes (1080×1) - 1D data
    - vertical stripes (1×1920) - 1D data
    - full 1920×1080 patterns for both directions
    
    Parameters:
        height: 图像高度
        width: 图像宽度
        cycles: 条纹周期数
        output_dir_1d: 一维数据保存目录
        output_dir_full: 完整图像保存目录
        phase_shift_step: 相移步数N（默认4）
        amplitude: 振幅，控制调制度 (默认127.5)
        offset: 偏移量/平均亮度 (默认127.5)
        direction: 条纹方向 ('vertical', 'horizontal', 'both')
        save_1d_data: 是否保存一维数据（默认True）
        save_full_image: 是否保存完整图像（默认True）
        save_full_4phase: 完整图像是否生成4个相位（默认True，False则只生成0°）
    """
    # ✅ 动态生成N步相位
    # N步相移：phase_i = 360° × i / N，其中 i = 0, 1, ..., N-1
    N = phase_shift_step
    phases = [360.0 * i / N for i in range(N)]
    
    # 如果选择both，则递归调用生成两个方向
    if direction == 'both':
        save_fringe_patterns(height, width, cycles, output_dir_1d, output_dir_full, phase_shift_step, 
                           amplitude, offset, 'vertical', save_1d_data, save_full_image, save_full_4phase)
        save_fringe_patterns(height, width, cycles, output_dir_1d, output_dir_full, phase_shift_step, 
                           amplitude, offset, 'horizontal', save_1d_data, save_full_image, save_full_4phase)
        return
    
    if direction == 'horizontal':
        # --- Horizontal stripes (1D data: height×1) ---
        if save_1d_data:
            os.makedirs(output_dir_1d, exist_ok=True)
            for i, ph in enumerate(phases, start=1):
                col = generate_sine_1d(height, cycles, ph, amplitude, offset)
                img = Image.fromarray(col.reshape((height, 1)), "L")
                fname = f"sine_horizontal_1x{height}_cycles{int(cycles)}_step{i}_phase{ph:.1f}.bmp"
                img.save(os.path.join(output_dir_1d, fname), format="BMP")
                print(f"✅ Saved horizontal strip: {fname}")

        # --- Full-size horizontal patterns ---
        if save_full_image:
            os.makedirs(output_dir_full, exist_ok=True)
            if save_full_4phase:
                for i, ph in enumerate(phases, start=1):
                    col = generate_sine_1d(height, cycles, ph, amplitude, offset)
                    full_h = np.tile(col.reshape((height, 1)), (1, width))
                    img_h = Image.fromarray(full_h, "L")
                    fname_h = f"full_horizontal_{width}x{height}_cycles{int(cycles)}_step{i}_phase{ph:.1f}.bmp"
                    img_h.save(os.path.join(output_dir_full, fname_h), format="BMP")
                    print(f"✅ Saved full horizontal: {fname_h}")
            else:
                # 只生成0°相位
                col = generate_sine_1d(height, cycles, 0, amplitude, offset)
                full_h = np.tile(col.reshape((height, 1)), (1, width))
                img_h = Image.fromarray(full_h, "L")
                fname_h = f"full_horizontal_{width}x{height}_cycles{int(cycles)}_step1_phase0.0.bmp"
                img_h.save(os.path.join(output_dir_full, fname_h), format="BMP")
                print(f"✅ Saved full horizontal: {fname_h}")

    elif direction == 'vertical':
        # --- Vertical stripes (1D data: 1×width) ---
        if save_1d_data:
            os.makedirs(output_dir_1d, exist_ok=True)
            for i, ph in enumerate(phases, start=1):
                row = generate_sine_1d(width, cycles, ph, amplitude, offset)
                img = Image.fromarray(row.reshape((1, width)), "L")
                fname = f"sine_vertical_1x{width}_cycles{int(cycles)}_step{i}_phase{ph:.1f}.bmp"
                img.save(os.path.join(output_dir_1d, fname), format="BMP")
                print(f"✅ Saved vertical strip: {fname}")

        # --- Full-size vertical patterns ---
        if save_full_image:
            os.makedirs(output_dir_full, exist_ok=True)
            if save_full_4phase:
                for i, ph in enumerate(phases, start=1):
                    row = generate_sine_1d(width, cycles, ph, amplitude, offset)
                    full_v = np.tile(row, (height, 1))
                    img_v = Image.fromarray(full_v, "L")
                    fname_v = f"full_vertical_{width}x{height}_cycles{int(cycles)}_step{i}_phase{ph:.1f}.bmp"
                    img_v.save(os.path.join(output_dir_full, fname_v), format="BMP")
                    print(f"✅ Saved full vertical: {fname_v}")
            else:
                # 只生成0°相位
                row = generate_sine_1d(width, cycles, 0, amplitude, offset)
                full_v = np.tile(row, (height, 1))
                img_v = Image.fromarray(full_v, "L")
                fname_v = f"full_vertical_{width}x{height}_cycles{int(cycles)}_step1_phase0.0.bmp"
                img_v.save(os.path.join(output_dir_full, fname_v), format="BMP")
                print(f"✅ Saved full vertical: {fname_v}")
    
    else:
        raise ValueError(f"不支持的方向: {direction}，请选择 'vertical', 'horizontal' 或 'both'")

if __name__ == "__main__":
    # ========================================
    # 参数配置
    # ========================================
    
    # ========================================
    # 参数模式选择
    # ========================================
    USE_FREQUENCY_MODE = True  # True: 使用频率参数模式, False: 使用周期参数模式
    
    # ========================================
    # 周期参数（周期模式）
    # ========================================
    cycles_min = 0        # 最小周期数（DC分量，0个周期）
    cycles_max = 49       # 最大周期数（图像中最多有多少个完整周期）
    cycles_step = 1       # 周期增量（每次增加多少个周期）
    
    # 💡 周期数说明：
    # - cycles = 0: DC分量（均匀灰度）
    # - cycles = 1: 图像中有1个完整周期
    # - cycles = 10: 图像中有10个完整周期（条纹较稀疏）
    # - cycles = 100: 图像中有100个完整周期（条纹较密集）
    # - cycles = 200: 图像中有200个完整周期（条纹非常密集）
    
    # ========================================
    # 频率参数（频率模式）
    # ========================================
    f_min = 0.00       # 最小空间频率 (cycles/pixel)
    f_max = 0.20       # 最大空间频率 (cycles/pixel)
    delta_f = 0.00026  # 频率步长
    
    # 投影仪分辨率参数
    width = 1920       # 投影仪列数（水平分辨率）
    height = 1080      # 投影仪行数（垂直分辨率）
    
    # 条纹方向选择
    direction = 'both'  # 可选: 'vertical', 'horizontal' 或 'both'（同时生成两个方向）
    
    # ✅ 相移步数N（默认4，可改为3、5等）
    PHASE_SHIFT_STEP = 3 # N步相移
    # N=3: 0°, 120°, 240°
    # N=4: 0°, 90°, 180°, 270°
    # N=5: 0°, 72°, 144°, 216°, 288°
    
    # ============ 调制度设置 ============
    # 默认: amplitude=127.5, offset=127.5 → 调制度=1.0 (最大, 范围0-255)
    # 示例: amplitude=63.75, offset=127.5 → 调制度=0.5 (50%, 范围64-191)
    # 示例: amplitude=95.625, offset=127.5 → 调制度=0.75 (75%, 范围32-223)
    AMPLITUDE = 127.5  # 振幅 (修改此值来调整调制度)
    OFFSET = 127.5     # 平均亮度
    # ==================================
    
    # ========================================
    # 输出控制 - 您可以选择生成哪些文件
    # ========================================
    save_1d_data = True          # 是否保存一维数据（总是生成N个相位）
    save_full_image = True       # 是否保存完整图像（可直接投影）
    save_full_4phase = True      # 完整图像是否生成N个相位（False则只生成0°）
    
    # 💡 使用建议：
    # - 只需要投影（单相位）：save_1d_data=False, save_full_image=True, save_full_4phase=False
    # - 只需要投影（N相位）：save_1d_data=False, save_full_image=True, save_full_4phase=True
    # - 只需要数值分析：save_1d_data=True, save_full_image=False
    # - 完整数据集：save_1d_data=True, save_full_image=True, save_full_4phase=True
    
    # 输出目录
    # 如果是both模式，目录名不包含方向；否则包含方向
    dir_suffix = direction if direction != 'both' else 'both'
    output_dir_1d_base = rf"E:/fringe_patterns_full_sine_{dir_suffix}/1d_data"
    output_dir_full_base = rf"E:/fringe_patterns_full_sine_{dir_suffix}/full_images"
    
    # ========================================
    # 生成条纹
    # ========================================
    
    # 根据模式计算周期序列
    if USE_FREQUENCY_MODE:
        # 频率模式：计算频率序列，然后转换为周期数
        frequencies = np.arange(f_min, f_max + delta_f, delta_f)
        total_freqs = len(frequencies)
        
        # 根据方向确定图像长度（用于频率到周期的转换）
        if direction == 'vertical':
            image_length = width
        elif direction == 'horizontal':
            image_length = height
        else:  # both
            image_length = width  # 使用width作为参考（垂直方向）
        
        # 将频率转换为周期数：cycles = frequency * image_length
        cycles_list = frequencies * image_length
        total_cycles = len(cycles_list)
        
        # 计算周期范围（用于显示）
        cycles_min_calc = f_min * image_length
        cycles_max_calc = f_max * image_length
    else:
        # 周期模式：直接使用周期参数
        cycles_list = np.arange(cycles_min, cycles_max + cycles_step, cycles_step)
        total_cycles = len(cycles_list)
        cycles_min_calc = cycles_min
        cycles_max_calc = cycles_max
    
    print("=" * 80)
    print(f"📊 条纹图案生成器 - {PHASE_SHIFT_STEP}步相移")
    print("=" * 80)
    print(f"📐 条纹方向:  {direction.upper()}")
    print(f"📏 投影分辨率: {width} x {height}")
    if direction == 'vertical':
        print(f"📍 图像宽度:  {width} 像素")
    elif direction == 'horizontal':
        print(f"📍 图像高度:  {height} 像素")
    else:  # both
        print(f"📍 图像宽度 (垂直条纹):  {width} 像素")
        print(f"📍 图像高度 (水平条纹):  {height} 像素")
    
    if USE_FREQUENCY_MODE:
        print(f"🔢 频率范围:  {f_min:.6f} ~ {f_max:.6f} cycles/pixel")
        print(f"📈 频率步长:  {delta_f:.6f}")
        print(f"🎬 频率点数:  {total_freqs} 个")
        if direction == 'vertical':
            print(f"💡 对应周期范围:  {cycles_min_calc:.2f} ~ {cycles_max_calc:.2f} 个周期 (基于宽度 {width})")
        elif direction == 'horizontal':
            print(f"💡 对应周期范围:  {cycles_min_calc:.2f} ~ {cycles_max_calc:.2f} 个周期 (基于高度 {height})")
        else:  # both
            print(f"💡 对应周期范围 (垂直):  {f_min * width:.2f} ~ {f_max * width:.2f} 个周期 (基于宽度 {width})")
            print(f"💡 对应周期范围 (水平):  {f_min * height:.2f} ~ {f_max * height:.2f} 个周期 (基于高度 {height})")
    else:
        print(f"🔢 周期范围:  {cycles_min} ~ {cycles_max} 个周期")
        print(f"📈 周期步长:  {cycles_step}")
        print(f"🎬 周期点数:  {total_cycles} 个")
    print(f"⚙️  调制度参数:  amplitude={AMPLITUDE}, offset={OFFSET}")
    print("=" * 80)
    print(f"💾 输出选项:")
    
    # 计算文件数量倍数（both模式下文件数翻倍）
    direction_multiplier = 2 if direction == 'both' else 1
    
    if save_1d_data:
        if direction == 'vertical':
            print(f"   ✅ 一维数据 (1 x {width}): 每周期数{PHASE_SHIFT_STEP}个相位 → {output_dir_1d_base}")
        elif direction == 'horizontal':
            print(f"   ✅ 一维数据 ({height} x 1): 每周期数{PHASE_SHIFT_STEP}个相位 → {output_dir_1d_base}")
        else:  # both
            print(f"   ✅ 一维数据 (垂直+水平): 每周期数{PHASE_SHIFT_STEP}个相位×2方向 → {output_dir_1d_base}")
    else:
        print(f"   ❌ 一维数据: 不保存")
    if save_full_image:
        if save_full_4phase:
            print(f"   ✅ 完整图像 ({width} x {height}): 每周期数{PHASE_SHIFT_STEP}个相位{' × 2方向' if direction == 'both' else ''} → {output_dir_full_base}")
        else:
            print(f"   ✅ 完整图像 ({width} x {height}): 每周期数1个相位(0°){' × 2方向' if direction == 'both' else ''} → {output_dir_full_base}")
    else:
        print(f"   ❌ 完整图像: 不保存")
    print("=" * 80)
    
    if USE_FREQUENCY_MODE:
        print(f"\n开始生成 {total_freqs} 个频率的{PHASE_SHIFT_STEP}步相移条纹图案...\n")
    else:
        print(f"\n开始生成 {total_cycles} 个周期的{PHASE_SHIFT_STEP}步相移条纹图案...\n")
    
    for idx, cycles in enumerate(cycles_list, 1):
        CYCLES = float(cycles)
        
        # 根据模式确定文件夹命名
        if USE_FREQUENCY_MODE:
            # 频率模式：使用频率值命名文件夹（类似周期版的f×ck值）
            # 统一使用width作为参考，确保both模式下垂直和水平条纹保存在同一文件夹
            f_ck_value = frequencies[idx-1] * width
            folder_name = f"f{f_ck_value:.1f}".replace('.', '_')
            if f_ck_value == int(f_ck_value):
                folder_name = f"f{int(f_ck_value)}"
        else:
            # 周期模式：使用周期数命名文件夹
            folder_name = f"{int(cycles)}"
        
        output_dir_1d = rf"{output_dir_1d_base}/{PHASE_SHIFT_STEP}step/{folder_name}"
        output_dir_full = rf"{output_dir_full_base}/{PHASE_SHIFT_STEP}step/{folder_name}"
        
        # 在频率模式下，both方向需要为每个方向分别计算周期数
        if USE_FREQUENCY_MODE and direction == 'both':
            freq_val = frequencies[idx-1]
            # 垂直方向：cycles = frequency * width
            cycles_vert = freq_val * width
            # 水平方向：cycles = frequency * height
            cycles_hor = freq_val * height
            print(f"[{idx:3d}/{total_cycles}] 生成频率 {freq_val:.6f} cycles/pixel")
            print(f"   垂直方向周期数: {cycles_vert:.2f}, 水平方向周期数: {cycles_hor:.2f}")
        elif USE_FREQUENCY_MODE:
            freq_val = frequencies[idx-1]
            print(f"[{idx:3d}/{total_cycles}] 生成频率 {freq_val:.6f} cycles/pixel (周期数 {CYCLES:.2f}) 的条纹图案...")
        else:
            print(f"[{idx:3d}/{total_cycles}] 生成周期数 {CYCLES} 的条纹图案...")
        
        # 在频率模式下，both方向需要为每个方向分别计算周期数
        if USE_FREQUENCY_MODE and direction == 'both':
            
            # 先处理垂直方向
            save_fringe_patterns(height, width, cycles_vert, output_dir_1d, output_dir_full,
                               phase_shift_step=PHASE_SHIFT_STEP,
                               amplitude=AMPLITUDE, offset=OFFSET,
                               direction='vertical',
                               save_1d_data=save_1d_data,
                               save_full_image=save_full_image,
                               save_full_4phase=save_full_4phase)
            # 再处理水平方向
            save_fringe_patterns(height, width, cycles_hor, output_dir_1d, output_dir_full,
                               phase_shift_step=PHASE_SHIFT_STEP,
                               amplitude=AMPLITUDE, offset=OFFSET,
                               direction='horizontal',
                               save_1d_data=save_1d_data,
                               save_full_image=save_full_image,
                               save_full_4phase=save_full_4phase)
        else:
            # 周期模式或单一方向：直接使用计算的周期数
            save_fringe_patterns(height, width, CYCLES, output_dir_1d, output_dir_full,
                               phase_shift_step=PHASE_SHIFT_STEP,
                               amplitude=AMPLITUDE, offset=OFFSET,
                               direction=direction,
                               save_1d_data=save_1d_data,
                               save_full_image=save_full_image,
                               save_full_4phase=save_full_4phase)
        
        if save_1d_data:
            if USE_FREQUENCY_MODE:
                print(f"✅ 频率 {frequencies[idx-1]:.6f} 的一维数据已保存到: {os.path.abspath(output_dir_1d)}")
            else:
                print(f"✅ 周期数 {CYCLES} 的一维数据已保存到: {os.path.abspath(output_dir_1d)}")
        if save_full_image:
            if USE_FREQUENCY_MODE:
                print(f"✅ 频率 {frequencies[idx-1]:.6f} 的完整图像已保存到: {os.path.abspath(output_dir_full)}")
            else:
                print(f"✅ 周期数 {CYCLES} 的完整图像已保存到: {os.path.abspath(output_dir_full)}")
        print()
    
    print("=" * 80)
    print("✅ 所有条纹图案生成完成！")
    print("=" * 80)
    print(f"📊 生成统计:")
    print(f"   - 周期点数:   {total_cycles}")
    print(f"   - 条纹方向:   {direction}")
    if save_1d_data:
        if direction == 'both':
            print(f"   - 一维图像:   {total_cycles} 文件夹 × {PHASE_SHIFT_STEP}相位 × 2方向 = {total_cycles * PHASE_SHIFT_STEP * 2} 张 → {output_dir_1d_base}")
        else:
            print(f"   - 一维图像:   {total_cycles} 文件夹 × {PHASE_SHIFT_STEP}相位 = {total_cycles * PHASE_SHIFT_STEP} 张 → {output_dir_1d_base}")
    else:
        print(f"   - 一维图像:   未生成（已禁用）")
    if save_full_image:
        if save_full_4phase:
            if direction == 'both':
                print(f"   - 完整图像:   {total_cycles} 文件夹 × {PHASE_SHIFT_STEP}相位 × 2方向 = {total_cycles * PHASE_SHIFT_STEP * 2} 张 → {output_dir_full_base}")
            else:
                print(f"   - 完整图像:   {total_cycles} 文件夹 × {PHASE_SHIFT_STEP}相位 = {total_cycles * PHASE_SHIFT_STEP} 张 → {output_dir_full_base}")
        else:
            if direction == 'both':
                print(f"   - 完整图像:   {total_cycles} 文件夹 × 1相位 × 2方向 = {total_cycles * 2} 张 → {output_dir_full_base}")
            else:
                print(f"   - 完整图像:   {total_cycles} 文件夹 × 1相位 = {total_cycles} 张 → {output_dir_full_base}")
    else:
        print(f"   - 完整图像:   未生成（已禁用）")
    print("=" * 80)
