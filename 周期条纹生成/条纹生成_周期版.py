"""
结构光条纹生成程序 - 周期数版（四步相移）
基于论文: Multi-Frequency Nonlinear Methods for 3D Shape Measurement

================================================================================
核心概念解释
================================================================================

1. 【周期数 (Number of Cycles)】
   - 定义：整个图像中包含的完整正弦波周期数
   - 符号：N
   - 单位：cycles（周期）
   - 输入方式：本程序直接指定周期数
   - 示例：N = 144 表示图像中有 144 个完整的明暗交替周期
   
   【物理意义】
   - 周期数 = 0：DC 分量，图像为均匀灰度（无条纹）
   - 周期数 = 1：图像中只有 1 个完整的正弦波周期
   - 周期数 = 144：图像中有 144 个完整的明暗条纹周期
   - 周期数越大 → 条纹越密集 → 单周期像素数越小

2. 【周期/单周期像素数 (Period in Pixels)】
   - 定义：一个完整正弦波周期所占用的像素数
   - 符号：T
   - 单位：pixels/cycle（像素/周期）
   - 公式：T = L / N
     其中 L 是图像像素长度（如1920），N 是周期数
   - 示例：T = 13.3 px 表示每个完整周期占用 13.3 个像素
   
   【物理意义】
   - 周期越大 → 条纹越宽 → 频率越低 → 条纹稀疏
   - 周期越小 → 条纹越窄 → 频率越高 → 条纹密集
   - 一个周期包含一个完整的明暗变化（从最暗→最亮→最暗）

3. 【空间频率 (Spatial Frequency)】
   - 定义：单位像素内包含的完整周期数
   - 符号：f
   - 单位：cycles/pixel（周期/像素）
   - 公式：f = N / L = 1 / T
     其中 N 是周期数，L 是图像像素长度，T 是单周期像素数
   - 示例：f = 0.075 cycles/pixel 表示每个像素包含 0.075 个周期
   
   【物理意义】
   - 空间频率越大 → 条纹越密集 → 周期越短
   - 空间频率越小 → 条纹越稀疏 → 周期越长
   - f = 0 时为 DC 分量（均匀灰度，无条纹）

================================================================================
三者关系
================================================================================

对于垂直条纹（图像宽度 W = 1920 像素）：

  周期数 N  ←→  单周期像素数 T  ←→  空间频率 f
  
  关系公式：
    N = W / T = f × W
    T = W / N = 1 / f
    f = N / W = 1 / T

  【相互转换】
  - 知道周期数 N，可求：T = W / N，f = N / W
  - 知道单周期像素数 T，可求：N = W / T，f = 1 / T
  - 知道空间频率 f，可求：N = f × W，T = 1 / f

================================================================================
参数关系示例（垂直条纹，图像宽度 W = 1920 像素）
================================================================================

示例 1：低频条纹（稀疏）
  - 周期数 N = 10
  - 单周期像素数 T = W / N = 1920 / 10 = 192 px
  - 空间频率 f = N / W = 10 / 1920 ≈ 0.0052 cycles/pixel
  → 图像中有 10 个很宽的条纹，每个条纹占 192 像素

示例 2：中频条纹（您的例子）
  - 周期数 N = 144
  - 单周期像素数 T = W / N = 1920 / 144 ≈ 13.3 px
  - 空间频率 f = N / W = 144 / 1920 = 0.075 cycles/pixel
  → 图像中有 144 个条纹，每个条纹占约 13.3 像素

示例 3：高频条纹（密集）
  - 周期数 N = 500
  - 单周期像素数 T = W / N = 1920 / 500 = 3.84 px
  - 空间频率 f = N / W = 500 / 1920 ≈ 0.2604 cycles/pixel
  → 图像中有 500 个很窄的条纹，每个条纹只占约 3.84 像素

================================================================================
正弦条纹公式
================================================================================

I(x) = P_A + P_B × sin(2π × f × x + φ)
     = P_A + P_B × sin(2π × (N/W) × x + φ)

其中：
  - I(x): 位置 x 处的像素灰度值
  - P_A: 直流分量（背景亮度，通常为 127.5）
  - P_B: 调制幅度（对比度，通常为 127.5）
  - N: 周期数（本程序的输入参数）
  - W: 图像宽度（像素）
  - f = N / W: 空间频率（由周期数计算得出）
  - x: 像素位置坐标（0, 1, 2, ..., W-1）
  - φ: 相位偏移（四步相移：0°, 90°, 180°, 270°）

================================================================================
程序特点
================================================================================

本程序采用【周期数】作为输入参数，相比频率输入更直观：
  - 优点 1：直接指定条纹数量，便于理解（如"我要100个条纹"）
  - 优点 2：周期数为整数时，条纹边界对齐（无截断）
  - 优点 3：适合需要特定条纹数量的应用场景

对比：
  - 本程序：指定周期数 → 自动计算频率
  - 频率版：指定频率 → 自动计算周期数

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
        frequency: 空间频率 f (单位: cycles/pixel)
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

def generate_fringe_for_cycles(cycles: float, height: int, width: int, 
                                direction: str, output_dir_1d: str, 
                                output_dir_full: str,
                                save_1d: bool = True,
                                save_full: bool = True,
                                save_full_4phase: bool = False,
                                P_A: float = 127.5, P_B: float = 127.5):
    """
    根据给定的周期数生成正弦条纹图案并保存
    
    Parameters:
        cycles: 图像中的周期数（整个图像包含多少个完整周期）
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
        generate_fringe_for_cycles(cycles, height, width, 'vertical', 
                                   output_dir_1d, output_dir_full,
                                   save_1d, save_full, save_full_4phase, P_A, P_B)
        generate_fringe_for_cycles(cycles, height, width, 'horizontal', 
                                   output_dir_1d, output_dir_full,
                                   save_1d, save_full, save_full_4phase, P_A, P_B)
        return
    
    # 文件夹命名：f0, f0.5, f1, f1.5 等
    folder_name = f"f{cycles:.1f}".replace('.', '_')  # f0_5 而不是 f0.5（避免文件系统问题）
    # 如果是整数，简化命名
    if cycles == int(cycles):
        folder_name = f"f{int(cycles)}"
    
    if direction == 'vertical':
        # 垂直条纹：沿 X 轴变化
        # 周期数 = cycles，图像宽度 = width
        # 空间频率 = cycles / width
        frequency = cycles / width if width > 0 else 0
        image_length = width
        
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
        
        # 计算实际周期（像素数）
        period_px = width / cycles if cycles > 0 else float('inf')
        
    elif direction == 'horizontal':
        # 水平条纹：沿 Y 轴变化
        # 周期数 = cycles，图像高度 = height
        # 空间频率 = cycles / height
        frequency = cycles / height if height > 0 else 0
        image_length = height
        
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
        
        # 计算实际周期（像素数）
        period_px = height / cycles if cycles > 0 else float('inf')
    
    else:
        raise ValueError(f"不支持的方向: {direction}，请选择 'vertical', 'horizontal' 或 'both'")
    
    # 打印进度信息
    period_str = f"{period_px:.1f}" if period_px != float('inf') else "∞"
    save_info = []
    if save_1d:
        save_info.append("1D×4")
    if save_full:
        if save_full_4phase:
            save_info.append("Full×4")
        else:
            save_info.append("Full×1")
    
    print(f"✅ {folder_name:8s} [{direction[0].upper()}] 周期数={cycles:6.1f} | 周期={period_str:8s}px | 空间频率={frequency:.6f} | 保存: {'+'.join(save_info) if save_info else '无'}")


if __name__ == "__main__":
    # ========================================
    # 参数配置
    # ========================================
    
    # 周期参数（新的定义方式）
    cycles_min = 0        # 最小周期数（DC分量，0个周期）
    cycles_max = 150      # 最大周期数（图像中最多有多少个完整周期）
    cycles_step = 0.5     # 周期增量（每次增加多少个周期）
    
    # 💡 周期数说明：
    # - cycles = 0: DC分量（均匀灰度）
    # - cycles = 1: 图像中有1个完整周期
    # - cycles = 10: 图像中有10个完整周期（条纹较稀疏）
    # - cycles = 100: 图像中有100个完整周期（条纹较密集）
    # - cycles = 200: 图像中有200个完整周期（条纹非常密集）
    
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
    save_full_image = False       # 是否保存完整图像（可直接投影）
    save_full_4phase = False     # 完整图像是否生成4个相位（False则只生成0°）
    
    # 💡 使用建议：
    # - 只需要投影（单相位）：save_1d_data=False, save_full_image=True, save_full_4phase=False
    # - 只需要投影（四相位）：save_1d_data=False, save_full_image=True, save_full_4phase=True
    # - 只需要数值分析：save_1d_data=True, save_full_image=False
    # - 完整数据集：save_1d_data=True, save_full_image=True, save_full_4phase=True
    
    # 输出目录
    # 如果是both模式，目录名不包含方向；否则包含方向
    dir_suffix = direction if direction != 'both' else 'both'
    output_dir_1d = fr"F:\{cycles_step}cycles_images_{dir_suffix}/1d_data"
    output_dir_full = fr"F:\{cycles_step}cycles_images_{dir_suffix}/full_images"
    
    # ========================================
    # 生成条纹
    # ========================================
    
    # 计算周期序列
    cycles_list = np.arange(cycles_min, cycles_max + cycles_step, cycles_step)
    total_cycles = len(cycles_list)
    
    print("=" * 80)
    print("📊 结构光条纹生成程序（周期数版 - 四步相移）")
    print("   基于论文: Multi-Frequency Nonlinear Methods")
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
    print(f"🔢 周期范围:  {cycles_min} ~ {cycles_max} 个周期")
    print(f"📈 周期步长:  {cycles_step}")
    print(f"🎬 周期点数:  {total_cycles} 个")
    print(f"⚙️  调制参数:  P_A={P_A}, P_B={P_B}")
    print("=" * 80)
    print(f"💾 输出选项:")
    
    # 计算文件数量倍数（both模式下文件数翻倍）
    direction_multiplier = 2 if direction == 'both' else 1
    
    if save_1d_data:
        if direction == 'vertical':
            print(f"   ✅ 一维数据 (1 x {width}): 每周期数4个相位 → {output_dir_1d}")
            print(f"      总文件数: {total_cycles} × 4 = {total_cycles * 4} 张")
        elif direction == 'horizontal':
            print(f"   ✅ 一维数据 ({height} x 1): 每周期数4个相位 → {output_dir_1d}")
            print(f"      总文件数: {total_cycles} × 4 = {total_cycles * 4} 张")
        else:  # both
            print(f"   ✅ 一维数据 (垂直+水平): 每周期数4个相位×2方向 → {output_dir_1d}")
            print(f"      总文件数: {total_cycles} × 4 × 2 = {total_cycles * 4 * 2} 张")
        print(f"      文件结构: f0/{direction if direction != 'both' else 'vertical_'}f0_phase0.bmp, ...")
    else:
        print(f"   ❌ 一维数据: 不保存")
    if save_full_image:
        if save_full_4phase:
            print(f"   ✅ 完整图像 ({width} x {height}): 每周期数4个相位{' × 2方向' if direction == 'both' else ''} → {output_dir_full}")
            print(f"      总文件数: {total_cycles} × 4 × {direction_multiplier} = {total_cycles * 4 * direction_multiplier} 张")
            print(f"      文件结构: f0/{direction if direction != 'both' else 'vertical_'}f0_phase0.bmp, ...")
        else:
            print(f"   ✅ 完整图像 ({width} x {height}): 每周期数1个相位(0°){' × 2方向' if direction == 'both' else ''} → {output_dir_full}")
            print(f"      总文件数: {total_cycles} × {direction_multiplier} = {total_cycles * direction_multiplier} 张")
            print(f"      文件结构: f0/{direction if direction != 'both' else 'vertical_'}f0_phase0.bmp, ...")
    else:
        print(f"   ❌ 完整图像: 不保存")
    print("=" * 80)
    
    # 显示周期与空间频率的对应关系
    if cycles_max > 0:
        if direction == 'vertical':
            freq_min = cycles_min / width if cycles_min > 0 else 0
            freq_max = cycles_max / width
            period_min = width / cycles_max
            period_max = width / cycles_min if cycles_min > 0 else float('inf')
            print(f"\n💡 参数对应关系（垂直条纹）:")
            print(f"   - 周期数范围: {cycles_min} ~ {cycles_max} 个完整周期")
            print(f"   - 空间频率范围: {freq_min:.6f} ~ {freq_max:.6f} cycles/pixel")
            print(f"   - 单周期像素数: {period_min:.1f} ~ {period_max} 像素/周期")
            print(f"   (周期数越大 → 条纹越密集 → 单周期像素数越小)\n")
        elif direction == 'horizontal':
            freq_min = cycles_min / height if cycles_min > 0 else 0
            freq_max = cycles_max / height
            period_min = height / cycles_max
            period_max = height / cycles_min if cycles_min > 0 else float('inf')
            print(f"\n💡 参数对应关系（水平条纹）:")
            print(f"   - 周期数范围: {cycles_min} ~ {cycles_max} 个完整周期")
            print(f"   - 空间频率范围: {freq_min:.6f} ~ {freq_max:.6f} cycles/pixel")
            print(f"   - 单周期像素数: {period_min:.1f} ~ {period_max} 像素/周期")
            print(f"   (周期数越大 → 条纹越密集 → 单周期像素数越小)\n")
        else:  # both
            # 垂直条纹
            freq_min_v = cycles_min / width if cycles_min > 0 else 0
            freq_max_v = cycles_max / width
            period_min_v = width / cycles_max
            period_max_v = width / cycles_min if cycles_min > 0 else float('inf')
            # 水平条纹
            freq_min_h = cycles_min / height if cycles_min > 0 else 0
            freq_max_h = cycles_max / height
            period_min_h = height / cycles_max
            period_max_h = height / cycles_min if cycles_min > 0 else float('inf')
            
            print(f"\n💡 参数对应关系:")
            print(f"   【垂直条纹】")
            print(f"   - 周期数范围: {cycles_min} ~ {cycles_max} 个完整周期")
            print(f"   - 空间频率范围: {freq_min_v:.6f} ~ {freq_max_v:.6f} cycles/pixel")
            print(f"   - 单周期像素数: {period_min_v:.1f} ~ {period_max_v} 像素/周期")
            print(f"   【水平条纹】")
            print(f"   - 周期数范围: {cycles_min} ~ {cycles_max} 个完整周期")
            print(f"   - 空间频率范围: {freq_min_h:.6f} ~ {freq_max_h:.6f} cycles/pixel")
            print(f"   - 单周期像素数: {period_min_h:.1f} ~ {period_max_h} 像素/周期")
            print(f"   (周期数越大 → 条纹越密集 → 单周期像素数越小)\n")
    
    print(f"开始生成 {total_cycles} 个周期的条纹图案...\n")
    print(f"{'文件夹':8s} | {'周期数':8s} | {'周期(px)':10s} | 空间频率 | 保存信息")
    print("-" * 80)
    
    # === 生成对应周期的条纹图像 ===
    for idx, c in enumerate(cycles_list, 1):
        print(f"[{idx:3d}/{total_cycles}] ", end='')
        generate_fringe_for_cycles(
            cycles=c, 
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
    
    print("\n" + "=" * 80)
    print("✅ 所有条纹图案生成完成！")
    print("=" * 80)
    print(f"📊 生成统计:")
    print(f"   - 周期点数:   {total_cycles}")
    print(f"   - 条纹方向:   {direction}")
    if save_1d_data:
        if direction == 'both':
            print(f"   - 一维图像:   {total_cycles} 文件夹 × 4相位 × 2方向 = {total_cycles * 4 * 2} 张 → {output_dir_1d}")
        else:
            print(f"   - 一维图像:   {total_cycles} 文件夹 × 4相位 = {total_cycles * 4} 张 → {output_dir_1d}")
    else:
        print(f"   - 一维图像:   未生成（已禁用）")
    if save_full_image:
        if save_full_4phase:
            if direction == 'both':
                print(f"   - 完整图像:   {total_cycles} 文件夹 × 4相位 × 2方向 = {total_cycles * 4 * 2} 张 → {output_dir_full}")
            else:
                print(f"   - 完整图像:   {total_cycles} 文件夹 × 4相位 = {total_cycles * 4} 张 → {output_dir_full}")
        else:
            if direction == 'both':
                print(f"   - 完整图像:   {total_cycles} 文件夹 × 1相位 × 2方向 = {total_cycles * 2} 张 → {output_dir_full}")
            else:
                print(f"   - 完整图像:   {total_cycles} 文件夹 × 1相位 = {total_cycles} 张 → {output_dir_full}")
    else:
        print(f"   - 完整图像:   未生成（已禁用）")
    print("=" * 80)
    
    print("\n💡 使用说明:")
    if direction == 'both':
        print("   📂 文件组织结构（Both模式）:")
        print("      - 每个周期数对应一个文件夹（f0, f0_5, f1, f1_5, ...）")
        print("      - 文件夹内包含垂直和水平两个方向的条纹")
        print("      - 文件命名: vertical_f{周期数}_phase{相位}.bmp, horizontal_f{周期数}_phase{相位}.bmp")
    else:
        print("   📂 文件组织结构:")
        print("      - 每个周期数对应一个文件夹（f0, f0_5, f1, f1_5, ...）")
        print(f"      - 每个文件夹内包含4个相位文件（{direction}_f{{周期数}}_phase0.bmp, {direction}_f{{周期数}}_phase90.bmp, {direction}_f{{周期数}}_phase180.bmp, {direction}_f{{周期数}}_phase270.bmp）")
        print("      - 方便批量导入和处理")
    if save_1d_data:
        print("\n   ✅ 一维数据（4个相位：0°, 90°, 180°, 270°）")
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
    print("      - 文件夹: f{周期数} （例如：f0, f0_5, f1, f10）")
    if direction == 'both':
        print(f"      - 文件: {{direction}}_f{{周期数}}_phase{{相位}}.bmp")
        print(f"        · 垂直条纹: vertical_f0_phase0.bmp, vertical_f144_phase90.bmp, ...")
        print(f"        · 水平条纹: horizontal_f0_phase0.bmp, horizontal_f144_phase90.bmp, ...")
    else:
        print(f"      - 文件: {direction}_f{{周期数}}_phase{{相位}}.bmp （例如：{direction}_f144_phase0.bmp, {direction}_f144_phase90.bmp）")
    print(f"\n   🔧 修改输出选项: save_1d_data, save_full_image, save_full_4phase")
    print(f"   🔧 更改周期范围: cycles_min, cycles_max, cycles_step")
    print(f"   🔧 更改方向: direction 参数 ('vertical', 'horizontal', 'both')")
    print("=" * 80)

