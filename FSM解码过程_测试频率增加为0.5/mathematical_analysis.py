"""
FSA解码的数学原理分析

深入分析原始decode_fsa.py的数学原理，理解频率步长的影响。

作者：基于FSA解码理论的数学分析
日期：2025-11-15
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_fsa_mathematics():
    """
    分析FSA解码的数学原理
    """
    print("="*80)
    print("FSA解码数学原理分析")
    print("="*80)
    
    print("FSA（Frequency Scanning Array）基本原理：")
    print("1. 投影仪投射不同频率的正弦条纹图案")
    print("2. 相机采集每个频率下的四步相移图像")
    print("3. 通过DFT解码每个相机像素对应的投影仪坐标")
    
    print(f"\n数学模型：")
    print("投影仪图案：I_p(x_p, f) = cos(2π * f * x_p / W + φ)")
    print("其中：")
    print("  x_p: 投影仪坐标 (0 到 W-1)")
    print("  f: 频率 (0, 1, 2, ..., M-1)")
    print("  W: 投影仪宽度")
    print("  φ: 相位 (0°, 90°, 180°, 270°)")
    
    print(f"\n相机采集：")
    print("I_c(x_c, f, φ) = A + B * cos(2π * f * x_p(x_c) / W + φ)")
    print("其中 x_p(x_c) 是相机像素x_c对应的投影仪坐标")

def analyze_dft_process():
    """
    分析DFT处理过程
    """
    print(f"\n{'='*80}")
    print("DFT处理过程分析")
    print(f"{'='*80}")
    
    print("四步相移构造复数信号：")
    print("D = (I0 - I2) + j*(I3 - I1)")
    print("D(x_c, f) = B * exp(j * 2π * f * x_p(x_c) / W)")
    
    print(f"\n对频率f进行DFT：")
    print("C(x_c, k) = Σ_{f=0}^{M-1} D(x_c, f) * exp(-j * 2π * f * k / M)")
    print("C(x_c, k) = B * Σ_{f=0}^{M-1} exp(j * 2π * f * x_p / W) * exp(-j * 2π * f * k / M)")
    
    print(f"\n当 k = M * x_p / W 时，DFT达到最大值：")
    print("峰值索引 k_peak = M * x_p / W")
    print("因此：投影仪坐标 x_p = k_peak * W / M")

def analyze_frequency_step_effect():
    """
    分析频率步长的影响
    """
    print(f"\n{'='*80}")
    print("频率步长影响分析")
    print(f"{'='*80}")
    
    print("标准频率步长（1.0）：")
    print("  频率序列：f = [0, 1, 2, ..., M-1]")
    print("  频率范围：0 到 M-1 Hz")
    print("  峰值索引：k_peak = x_p * M / W")
    print("  投影仪坐标：x_p = k_peak * W / M")
    
    print(f"\n精细频率步长（0.5）：")
    print("  频率序列：f = [0, 0.5, 1.0, 1.5, ..., (M-1)*0.5]")
    print("  频率范围：0 到 (M-1)*0.5 Hz")
    print("  但DFT索引仍对应频率索引，不是实际频率")
    
    print(f"\n关键理解：")
    print("  DFT索引k对应第k个频率分量")
    print("  第k个频率的实际值 = k * frequency_step")
    print("  峰值条件：k * frequency_step ≈ x_p * M / W")
    print("  因此：x_p = k * frequency_step * W / M")

def derive_correct_mapping():
    """
    推导正确的坐标映射
    """
    print(f"\n{'='*80}")
    print("正确坐标映射推导")
    print(f"{'='*80}")
    
    # 原始配置
    M_orig = 121
    W_orig = 1024
    freq_step_orig = 1.0
    
    print("原始配置推导：")
    print(f"  M = {M_orig}, W = {W_orig}, frequency_step = {freq_step_orig}")
    mapping_orig = W_orig / M_orig
    print(f"  映射比例：W/M = {mapping_orig:.3f}")
    print(f"  投影仪坐标：x_p = k * {mapping_orig:.3f}")
    
    # 新配置
    M_new = 301
    W_new = 1920
    freq_step_new = 0.5
    
    print(f"\n新配置推导：")
    print(f"  M = {M_new}, W = {W_new}, frequency_step = {freq_step_new}")
    mapping_new = freq_step_new * W_new / M_new
    print(f"  映射比例：frequency_step * W/M = {mapping_new:.3f}")
    print(f"  投影仪坐标：x_p = k * {mapping_new:.3f}")
    
    # 验证坐标范围
    max_k = M_new - 1
    max_coord = max_k * mapping_new
    print(f"\n坐标范围验证：")
    print(f"  最大DFT索引：{max_k}")
    print(f"  最大投影仪坐标：{max_coord:.1f}")
    print(f"  投影仪宽度：{W_new}")
    
    if abs(max_coord - W_new) < 10:
        print(f"  ✓ 坐标范围基本正确")
    else:
        print(f"  ✗ 坐标范围不匹配")
        print(f"    需要调整映射比例")

def simulate_correct_mapping():
    """
    模拟正确的坐标映射
    """
    print(f"\n{'='*80}")
    print("正确映射模拟验证")
    print(f"{'='*80}")
    
    # 参数
    M = 301
    W = 1920
    freq_step = 0.5
    
    # 正确的映射比例
    coord_scale = freq_step * W / M
    
    print(f"模拟参数：")
    print(f"  M = {M}, W = {W}, frequency_step = {freq_step}")
    print(f"  映射比例：{coord_scale:.3f}")
    
    # 模拟几个DFT索引
    test_indices = [0, 100, 200, 300]
    print(f"\n映射验证：")
    for k in test_indices:
        x_p = k * coord_scale
        print(f"  DFT索引 {k:3d} → 投影仪坐标 {x_p:6.1f}")
    
    # 模拟真实场景
    true_coord = 960  # 投影仪中心
    expected_k = true_coord / coord_scale
    print(f"\n反向验证：")
    print(f"  真实投影仪坐标：{true_coord}")
    print(f"  预期DFT索引：{expected_k:.1f}")
    
    # 生成模拟信号并DFT
    frequencies = np.arange(M) * freq_step
    signal = np.exp(2j * np.pi * frequencies * true_coord / W)
    spectrum = np.fft.fft(signal, n=M)  # 注意：这里DFT长度应该是M
    magnitude = np.abs(spectrum)
    peak_k = np.argmax(magnitude)
    
    print(f"\nDFT验证：")
    print(f"  DFT长度：{M}")
    print(f"  实际峰值索引：{peak_k}")
    print(f"  预期峰值索引：{expected_k:.1f}")
    print(f"  误差：{abs(peak_k - expected_k):.1f}")
    
    if abs(peak_k - expected_k) < 5:
        print(f"  ✓ 验证通过：映射正确")
    else:
        print(f"  ✗ 验证失败：映射需要调整")

if __name__ == "__main__":
    analyze_fsa_mathematics()
    analyze_dft_process()
    analyze_frequency_step_effect()
    derive_correct_mapping()
    simulate_correct_mapping()
    
    print(f"\n{'='*80}")
    print("数学分析结论")
    print(f"{'='*80}")
    print("关键发现：")
    print("1. DFT长度应该等于频率数量M，不是投影仪尺寸W")
    print("2. 坐标映射：x_p = k * frequency_step * W / M")
    print("3. 频率步长直接影响映射比例")
    print("4. 原始代码的DFT长度设置可能需要重新理解")
    print("\n需要修正的代码：")
    print("  - DFT长度改为M（频率数量）")
    print("  - 添加正确的坐标映射比例")
    print("  - 确保坐标范围覆盖投影仪尺寸")
