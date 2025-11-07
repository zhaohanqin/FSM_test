"""
结构光条纹生成程序 - 空间频率版（四步相移）
基于论文: Multi-Frequency Nonlinear Methods for 3D Shape Measurement

================================================================================
核心概念解释
================================================================================

1. 【空间频率 (Spatial Frequency)】
   - 定义：单位像素内包含的完整周期数
   - 符号：f
   - 单位：cycles/pixel（周期/像素）
   - 公式：f = N / L
     其中 N 是图像中的总周期数，L 是图像的像素长度
   - 示例：f = 0.075 cycles/pixel 表示每个像素包含0.075个周期
   
   【物理意义】
   - 空间频率越大 → 条纹越密集 → 周期越短
   - 空间频率越小 → 条纹越稀疏 → 周期越长
   - f = 0 时为 DC 分量（均匀灰度，无条纹）

2. 【周期数 (Number of Cycles)】
   - 定义：整个图像中包含的完整正弦波周期数
   - 符号：N
   - 单位：cycles（周期）
   - 公式：N = f × L
     其中 f 是空间频率，L 是图像的像素长度
   - 示例：N = 144 表示图像中有 144 个完整的明暗交替周期
   
   【物理意义】
   - 周期数 = 0：DC 分量，图像为均匀灰度
   - 周期数 = 1：图像中只有 1 个完整的正弦波
   - 周期数 = 144：图像中有 144 个完整的明暗条纹周期

3. 【周期/单周期像素数 (Period in Pixels)】
   - 定义：一个完整正弦波周期所占用的像素数
   - 符号：T
   - 单位：pixels/cycle（像素/周期）
   - 公式：T = L / N = 1 / f
     其中 L 是图像像素长度，N 是周期数，f 是空间频率
   - 示例：T = 13.3 px 表示每个完整周期占用 13.3 个像素
   
   【物理意义】
   - 周期越大 → 条纹越宽 → 频率越低
   - 周期越小 → 条纹越窄 → 频率越高
   - 一个周期包含一个完整的明暗变化（从最暗→最亮→最暗）

================================================================================
参数关系示例（垂直条纹，图像宽度 W = 1920 像素）
================================================================================

示例 1：低频条纹（稀疏）
  - 周期数 N = 10
  - 空间频率 f = N / W = 10 / 1920 ≈ 0.0052 cycles/pixel
  - 单周期像素数 T = W / N = 1920 / 10 = 192 px
  → 图像中有 10 个很宽的条纹，每个条纹占 192 像素

示例 2：中频条纹
  - 周期数 N = 144（您的例子）
  - 空间频率 f = N / W = 144 / 1920 = 0.075 cycles/pixel
  - 单周期像素数 T = W / N = 1920 / 144 ≈ 13.3 px
  → 图像中有 144 个条纹，每个条纹占约 13.3 像素

示例 3：高频条纹（密集）
  - 周期数 N = 500
  - 空间频率 f = N / W = 500 / 1920 ≈ 0.2604 cycles/pixel
  - 单周期像素数 T = W / N = 1920 / 500 = 3.84 px
  → 图像中有 500 个很窄的条纹，每个条纹只占约 3.84 像素

================================================================================
正弦条纹公式
================================================================================

I(x) = P_A + P_B × sin(2π × f × x + φ)

其中：
  - I(x): 位置 x 处的像素灰度值
  - P_A: 直流分量（背景亮度，通常为 127.5）
  - P_B: 调制幅度（对比度，通常为 127.5）
  - f: 空间频率（cycles/pixel）
  - x: 像素位置坐标（0, 1, 2, ..., W-1）
  - φ: 相位偏移（四步相移：0°, 90°, 180°, 270°）

================================================================================
"""

import os
import math
import numpy as np
from PIL import Image

def clamp_u8(arr: np.ndarray) -> np.ndarray:
    """将数组限制在[0,255]范围并转换为uint8类型"""
    return np.clip(arr, 0, 255).astype(np.uint8)

def generate_sine_1d(length: int, frequency: float, phase_deg: float = 0.0,
                     P_A: float = 127.5, P_B: float = 127.5) -> np.ndarray:
    """
    根据标准四步相移公式生成一维正弦波: I(c) = P_A + P_B * sin(2*pi*f*c + phase)
    
    Parameters:
        length: 像素数量（投影仪分辨率，如1920）
        frequency: 空间频率 f (单位: cycles/pixel)，频率越大条纹越密
        phase_deg: 相位（度），用于四步相移（0, 90, 180, 270）
        P_A: 直流分量（背景亮度，默认127.5）
        P_B: 调制幅度（对比度，默认127.5）
    
    Returns:
        一维数组，shape=(length,)
    
    公式: I(c_x) = P_A + P_B * sin(2*pi*f*c_x + phase)
    其中 c_x 是像素坐标位置 [0, length-1]，f 是空间频率
    参考: generate_fringe_patterns.py 的标准实现
    """
    two_pi = math.tau  # 2*pi
    phase = math.radians(phase_deg)
    c_x = np.arange(length, dtype=np.float64)  # c_x: 像素坐标位置
    angle = two_pi * frequency * c_x + phase   # 2*pi*f*c_x + phase
    vals = P_A + P_B * np.sin(angle)           # P_A + P_B * sin(...) - 使用sin而非cos！
    return clamp_u8(vals)

def generate_fringe_for_frequency(frequency: float, height: int, width: int, 
                                  direction: str, output_dir_1d: str, 
                                  output_dir_full: str,
                                  save_1d: bool = True,
                                  save_full: bool = True,
                                  save_full_4phase: bool = False,
                                  P_A: float = 127.5, P_B: float = 127.5):
    """
    根据给定的空间频率生成正弦条纹图案并保存
    
    Parameters:
        frequency: 空间频率 f (单位: cycles/pixel)，频率越大条纹越密集
        height: 图像高度
        width: 图像宽度
        direction: 'vertical', 'horizontal' 或 'both'，条纹方向
        output_dir_1d: 一维数据保存目录
        output_dir_full: 完整图像保存目录
        save_1d: 是否保存一维数据（默认True，总是生成4个相位）
        save_full: 是否保存完整图像（默认True）
        save_full_4phase: 完整图像是否生成4个相位（False则只生成0°）
        P_A: 直流分量（背景亮度，默认127.5）
        P_B: 调制幅度（对比度，默认127.5）
    """
    # 四步相移：标准顺序 0°, 90°, 180°, 270°
    # 参考公式: phase_shift = i * 2π/N，其中 N=4
    phases = [0, 90, 180, 270]  # 相位角度（度）
    
    # 如果选择both，则递归调用生成两个方向
    if direction == 'both':
        generate_fringe_for_frequency(frequency, height, width, 'vertical', 
                                     output_dir_1d, output_dir_full,
                                     save_1d, save_full, save_full_4phase, P_A, P_B)
        generate_fringe_for_frequency(frequency, height, width, 'horizontal', 
                                     output_dir_1d, output_dir_full,
                                     save_1d, save_full, save_full_4phase, P_A, P_B)
        return
    
    # 统一的文件夹命名（基于垂直方向的 f×ck 值）
    # 这样可以确保在 both 模式下，垂直和水平条纹保存在同一个文件夹
    f_ck_value = frequency * width  # 统一使用 width 作为参考
    folder_name = f"f{f_ck_value:.1f}".replace('.', '_')
    if f_ck_value == int(f_ck_value):
        folder_name = f"f{int(f_ck_value)}"
    
    if direction == 'vertical':
        # 垂直条纹：沿 X 轴变化，ck = width
        ck = width
        
        # 文件名前缀（用于both模式）
        file_prefix = "vertical_"
        
        # 创建子文件夹
        if save_1d:
            dir_1d = os.path.join(output_dir_1d, folder_name)
            os.makedirs(dir_1d, exist_ok=True)
        if save_full:
            dir_full = os.path.join(output_dir_full, folder_name)
            os.makedirs(dir_full, exist_ok=True)
        
        # 保存一维数据（1 x width）- 总是生成4个相位
        if save_1d:
            for i, phase_deg in enumerate(phases):
                # 按标准四步相移顺序生成：0°, 90°, 180°, 270°
                pattern_1d = generate_sine_1d(width, frequency, phase_deg, P_A, P_B)
                img_1d = Image.fromarray(pattern_1d.reshape((1, width)), "L")
                fname_1d = f"{file_prefix}{folder_name}_phase{phase_deg}.bmp"
                img_1d.save(os.path.join(dir_1d, fname_1d), format="BMP")
        
        # 保存完整图像（height x width）
        if save_full:
            if save_full_4phase:
                # 生成4个相位，按标准顺序：0°, 90°, 180°, 270°
                for i, phase_deg in enumerate(phases):
                    pattern_1d = generate_sine_1d(width, frequency, phase_deg, P_A, P_B)
                    full_pattern = np.tile(pattern_1d, (height, 1))
                    img_full = Image.fromarray(full_pattern, "L")
                    fname_full = f"{file_prefix}{folder_name}_phase{phase_deg}.bmp"
                    img_full.save(os.path.join(dir_full, fname_full), format="BMP")
            else:
                # 只生成0°相位
                pattern_1d = generate_sine_1d(width, frequency, 0, P_A, P_B)
                full_pattern = np.tile(pattern_1d, (height, 1))
                img_full = Image.fromarray(full_pattern, "L")
                fname_full = f"{file_prefix}{folder_name}_phase0.bmp"
                img_full.save(os.path.join(dir_full, fname_full), format="BMP")
        
        # 打印进度信息
        period_str = f"{1/frequency:.1f}" if frequency > 0 else "∞"
        save_info = []
        if save_1d:
            save_info.append("1D×4")
        if save_full:
            if save_full_4phase:
                save_info.append("Full×4")
            else:
                save_info.append("Full×1")
        print(f"✅ {folder_name:8s} [V] f={frequency:.6f} (f×ck={ck * frequency:.1f}) | 周期={period_str}px | {'+'.join(save_info) if save_info else '无'}")
        
    elif direction == 'horizontal':
        # 水平条纹：沿 Y 轴变化，ck = height
        ck = height
        
        # 文件名前缀（用于both模式）
        file_prefix = "horizontal_"
        
        # 创建子文件夹
        if save_1d:
            dir_1d = os.path.join(output_dir_1d, folder_name)
            os.makedirs(dir_1d, exist_ok=True)
        if save_full:
            dir_full = os.path.join(output_dir_full, folder_name)
            os.makedirs(dir_full, exist_ok=True)
        
        # 保存一维数据（height x 1）- 总是生成4个相位
        if save_1d:
            for i, phase_deg in enumerate(phases):
                # 按标准四步相移顺序生成：0°, 90°, 180°, 270°
                pattern_1d = generate_sine_1d(height, frequency, phase_deg, P_A, P_B)
                img_1d = Image.fromarray(pattern_1d.reshape((height, 1)), "L")
                fname_1d = f"{file_prefix}{folder_name}_phase{phase_deg}.bmp"
                img_1d.save(os.path.join(dir_1d, fname_1d), format="BMP")
        
        # 保存完整图像（height x width）
        if save_full:
            if save_full_4phase:
                # 生成4个相位，按标准顺序：0°, 90°, 180°, 270°
                for i, phase_deg in enumerate(phases):
                    pattern_1d = generate_sine_1d(height, frequency, phase_deg, P_A, P_B)
                    full_pattern = np.tile(pattern_1d.reshape((height, 1)), (1, width))
                    img_full = Image.fromarray(full_pattern, "L")
                    fname_full = f"{file_prefix}{folder_name}_phase{phase_deg}.bmp"
                    img_full.save(os.path.join(dir_full, fname_full), format="BMP")
            else:
                # 只生成0°相位
                pattern_1d = generate_sine_1d(height, frequency, 0, P_A, P_B)
                full_pattern = np.tile(pattern_1d.reshape((height, 1)), (1, width))
                img_full = Image.fromarray(full_pattern, "L")
                fname_full = f"{file_prefix}{folder_name}_phase0.bmp"
                img_full.save(os.path.join(dir_full, fname_full), format="BMP")
        
        # 打印进度信息
        period_str = f"{1/frequency:.1f}" if frequency > 0 else "∞"
        save_info = []
        if save_1d:
            save_info.append("1D×4")
        if save_full:
            if save_full_4phase:
                save_info.append("Full×4")
            else:
                save_info.append("Full×1")
        print(f"✅ {folder_name:8s} [H] f={frequency:.6f} (f×ck={ck * frequency:.1f}) | 周期={period_str}px | {'+'.join(save_info) if save_info else '无'}")
    
    else:
        raise ValueError(f"不支持的方向: {direction}，请选择 'vertical', 'horizontal' 或 'both'")


if __name__ == "__main__":
    # ========================================
    # 参数配置
    # ========================================
    
    # 频率参数
    f_min = 0.00       # 最小空间频率 (cycles/pixel)
    f_max = 0.20      # 最大空间频率 (cycles/pixel)
    delta_f = 0.00026  # 频率步长
    
    # 投影仪分辨率参数
    width = 1920       # 投影仪列数（水平分辨率）
    height = 1080      # 投影仪行数（垂直分辨率）
    
    # 条纹方向选择
    direction = 'both'  # 可选: 'vertical', 'horizontal' 或 'both'（同时生成两个方向）
    
    # 调制参数
    P_A = 127.5        # 直流分量（背景亮度，0-255）
    P_B = 127.5        # 调制幅度（对比度，0-255）
    
    # ========================================
    # 输出控制 - 您可以选择生成哪些文件
    # ========================================
    save_1d_data = True          # 是否保存一维数据（总是生成4个相位）
    save_full_image = True        # 是否保存完整图像（可直接投影）
    save_full_4phase = True     # 完整图像是否生成4个相位（False则只生成0°）
    
    # 💡 使用建议：
    # - 只需要投影（单相位）：save_1d_data=False, save_full_image=True, save_full_4phase=False
    # - 只需要投影（四相位）：save_1d_data=False, save_full_image=True, save_full_4phase=True
    # - 只需要数值分析：save_1d_data=True, save_full_image=False
    # - 完整数据集：save_1d_data=True, save_full_image=True, save_full_4phase=True
    
    # 输出目录
    # 如果是both模式，目录名不包含方向；否则包含方向
    dir_suffix = direction if direction != 'both' else 'both'
    output_dir_1d = fr"E:\1_images_{dir_suffix}/1d_data"
    output_dir_full = fr"E:\1_images_{dir_suffix}/full_images"
    
    # ========================================
    # 生成条纹
    # ========================================
    
    # 计算频率序列
    frequencies = np.arange(f_min, f_max + delta_f, delta_f)
    total_freqs = len(frequencies)
    
    print("=" * 70)
    print("📊 结构光条纹生成程序（四步相移版）")
    print("   基于论文: Multi-Frequency Nonlinear Methods")
    print("=" * 70)
    print(f"📐 条纹方向:  {direction.upper()}")
    print(f"📏 投影分辨率: {width} x {height}")
    if direction == 'vertical':
        print(f"📍 投影列数 ck: {width}")
    elif direction == 'horizontal':
        print(f"📍 投影行数 ck: {height}")
    else:  # both
        print(f"📍 投影列数 ck (垂直条纹): {width}")
        print(f"📍 投影行数 ck (水平条纹): {height}")
    print(f"🔢 频率范围:  {f_min:.6f} ~ {f_max:.6f} cycles/pixel")
    print(f"📈 频率步长:  {delta_f:.6f}")
    print(f"🎬 频率点数:  {total_freqs} 个")
    print(f"⚙️  调制参数:  P_A={P_A}, P_B={P_B}")
    print("=" * 70)
    print(f"💾 输出选项:")
    
    # 计算文件数量倍数（both模式下文件数翻倍）
    direction_multiplier = 2 if direction == 'both' else 1
    
    if save_1d_data:
        if direction == 'vertical':
            print(f"   ✅ 一维数据 (1 x {width}): 每频率4个相位 → {output_dir_1d}")
            print(f"      总文件数: {total_freqs} × 4 = {total_freqs * 4} 张")
        elif direction == 'horizontal':
            print(f"   ✅ 一维数据 ({height} x 1): 每频率4个相位 → {output_dir_1d}")
            print(f"      总文件数: {total_freqs} × 4 = {total_freqs * 4} 张")
        else:  # both
            print(f"   ✅ 一维数据 (垂直+水平): 每频率4个相位×2方向 → {output_dir_1d}")
            print(f"      总文件数: {total_freqs} × 4 × 2 = {total_freqs * 4 * 2} 张")
        print(f"      文件结构: f0/{direction if direction != 'both' else 'vertical_'}f0_phase0.bmp, ...")
    else:
        print(f"   ❌ 一维数据: 不保存")
    if save_full_image:
        if save_full_4phase:
            print(f"   ✅ 完整图像 ({width} x {height}): 每频率4个相位{' × 2方向' if direction == 'both' else ''} → {output_dir_full}")
            print(f"      总文件数: {total_freqs} × 4 × {direction_multiplier} = {total_freqs * 4 * direction_multiplier} 张")
            print(f"      文件结构: f0/{direction if direction != 'both' else 'vertical_'}f0_phase0.bmp, ...")
        else:
            print(f"   ✅ 完整图像 ({width} x {height}): 每频率1个相位(0°){' × 2方向' if direction == 'both' else ''} → {output_dir_full}")
            print(f"      总文件数: {total_freqs} × {direction_multiplier} = {total_freqs * direction_multiplier} 张")
            print(f"      文件结构: f0/{direction if direction != 'both' else 'vertical_'}f0_phase0.bmp, ...")
    else:
        print(f"   ❌ 完整图像: 不保存")
    print("=" * 70)
    
    # 显示周期范围
    if f_max > 0:
        period_max = 1 / f_min if f_min > 0 else float('inf')
        period_min = 1 / f_max
        print(f"\n💡 条纹周期范围: {period_min:.1f} ~ {period_max} 像素/周期")
        print(f"   (周期随频率增加而减小，条纹变密集)\n")
    
    print(f"开始生成 {total_freqs} 个频率的条纹图案...\n")

# === 生成对应频率的条纹图像 ===
    for idx, f in enumerate(frequencies, 1):
        print(f"[{idx}/{total_freqs}] ", end='')
        generate_fringe_for_frequency(
            frequency=f, 
            height=height, 
            width=width,
            direction=direction,
            output_dir_1d=output_dir_1d,
            output_dir_full=output_dir_full,
            save_1d=save_1d_data,
            save_full=save_full_image,
            save_full_4phase=save_full_4phase,
            P_A=P_A,
            P_B=P_B
        )
    
    print("\n" + "=" * 70)
    print("✅ 所有条纹图案生成完成！")
    print("=" * 70)
    print(f"📊 生成统计:")
    print(f"   - 频率点数:   {total_freqs}")
    print(f"   - 条纹方向:   {direction}")
    if save_1d_data:
        if direction == 'both':
            print(f"   - 一维图像:   {total_freqs} × 4相位 × 2方向 = {total_freqs * 4 * 2} 张 → {output_dir_1d}")
        else:
            print(f"   - 一维图像:   {total_freqs} × 4相位 = {total_freqs * 4} 张 → {output_dir_1d}")
    else:
        print(f"   - 一维图像:   未生成（已禁用）")
    if save_full_image:
        if save_full_4phase:
            if direction == 'both':
                print(f"   - 完整图像:   {total_freqs} × 4相位 × 2方向 = {total_freqs * 4 * 2} 张 → {output_dir_full}")
            else:
                print(f"   - 完整图像:   {total_freqs} × 4相位 = {total_freqs * 4} 张 → {output_dir_full}")
        else:
            if direction == 'both':
                print(f"   - 完整图像:   {total_freqs} × 1相位 × 2方向 = {total_freqs * 2} 张 → {output_dir_full}")
            else:
                print(f"   - 完整图像:   {total_freqs} × 1相位 = {total_freqs} 张 → {output_dir_full}")
    else:
        print(f"   - 完整图像:   未生成（已禁用）")
    print("=" * 70)
    
    print("\n💡 使用说明:")
    if direction == 'both':
        print("   📂 文件组织结构（Both模式）:")
        print("      - 每个频率一个文件夹（f0, f0_5, f1, ...）")
        print("      - 文件夹内包含垂直和水平两个方向的条纹")
        print("      - 文件命名: vertical_f{频率标识}_phase{相位}.bmp, horizontal_f{频率标识}_phase{相位}.bmp")
    if save_1d_data:
        print("   ✅ 一维数据（4个相位：0°, 90°, 180°, 270°）")
        if direction == 'vertical':
            print(f"      - 尺寸: 1 x {width} 像素")
        elif direction == 'horizontal':
            print(f"      - 尺寸: {height} x 1 像素")
        else:  # both
            print(f"      - 垂直条纹: 1 x {width} 像素")
            print(f"      - 水平条纹: {height} x 1 像素")
    if save_full_image:
        if save_full_4phase:
            print("   ✅ 完整图像（4个相位：0°, 90°, 180°, 270°）")
        else:
            print("   ✅ 完整图像（仅0°相位）")
        print(f"      - 尺寸: {width} x {height} 像素")
    print("\n   📝 文件命名规则:")
    print(f"      - 文件夹: f{{f×width:.1f}} （例如：f0, f1_9, f10）")
    if direction == 'both':
        print(f"        注意：Both模式下，文件夹名基于垂直方向（f×width），两个方向的条纹保存在同一文件夹")
        print(f"      - 文件: {{direction}}_f{{频率标识}}_phase{{相位}}.bmp")
        print(f"        · 垂直条纹: vertical_f0_phase0.bmp, vertical_f144_phase90.bmp, ...")
        print(f"        · 水平条纹: horizontal_f0_phase0.bmp, horizontal_f144_phase90.bmp, ...")
    else:
        print(f"      - 文件: {direction}_f{{频率标识}}_phase{{相位}}.bmp （例如：{direction}_f144_phase0.bmp, {direction}_f144_phase90.bmp）")
    print(f"\n   🔧 修改输出选项: save_1d_data, save_full_image, save_full_4phase")
    print(f"   🔧 更改方向: 修改 direction 参数 ('vertical', 'horizontal', 'both')")
    print("=" * 70)
