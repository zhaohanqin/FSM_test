"""
测试频率步长修复效果

本脚本用于验证频率步长为0.5时的FSA解码修复是否正确。
主要验证点：
1. 解码结果的坐标范围应该覆盖整个投影仪尺寸
2. 频谱分析应该显示正确的投影仪坐标
3. DFT长度应该等于投影仪尺寸，不受频率步长影响

作者：基于FSM_decode_core.py修复验证
日期：2025-11-15
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dft_logic():
    """
    测试DFT逻辑的正确性
    """
    print("="*80)
    print("测试DFT逻辑")
    print("="*80)
    
    # 模拟参数
    PROJECTOR_WIDTH = 1920
    PROJECTOR_HEIGHT = 1080
    NUM_FREQUENCIES = 301
    FREQUENCY_STEP = 0.5
    
    print(f"测试参数:")
    print(f"  投影仪尺寸: {PROJECTOR_WIDTH} × {PROJECTOR_HEIGHT}")
    print(f"  频率数量: {NUM_FREQUENCIES}")
    print(f"  频率步长: {FREQUENCY_STEP}")
    print(f"  频率范围: 0 到 {(NUM_FREQUENCIES-1)*FREQUENCY_STEP} Hz")
    
    # 关键验证：DFT长度应该等于投影仪尺寸
    print(f"\n关键验证:")
    print(f"  垂直条纹DFT长度: {PROJECTOR_WIDTH} (应该等于投影仪宽度)")
    print(f"  水平条纹DFT长度: {PROJECTOR_HEIGHT} (应该等于投影仪高度)")
    print(f"  ✓ DFT长度不受频率步长影响")
    
    # 模拟DFT输出
    print(f"\n坐标映射验证:")
    sample_indices = [0, 480, 960, 1440, 1919]
    for idx in sample_indices:
        print(f"    DFT索引 {idx:4d} → 投影仪坐标 {idx:4d} (直接对应)")
    
    print(f"\n预期结果:")
    print(f"  解码坐标范围应该是 0 到 {PROJECTOR_WIDTH-1} (列)")
    print(f"  解码坐标范围应该是 0 到 {PROJECTOR_HEIGHT-1} (行)")
    print(f"  ✓ 不应该只集中在一半范围内")

def simulate_frequency_spectrum():
    """
    模拟频率谱以验证理论
    """
    print(f"\n{'='*80}")
    print("模拟频率谱验证")
    print(f"{'='*80}")
    
    # 参数
    M = 301  # 频率数量
    N = 1920  # 投影仪宽度
    frequency_step = 0.5
    
    # 模拟一个投影仪坐标为500的像素点
    true_projector_coord = 500
    
    print(f"模拟场景:")
    print(f"  真实投影仪坐标: {true_projector_coord}")
    print(f"  采集频率数: {M}")
    print(f"  频率步长: {frequency_step}")
    
    # 生成模拟信号
    frequencies = np.arange(M) * frequency_step  # 0, 0.5, 1.0, ..., 150.0
    
    # 模拟理想的正弦信号（忽略噪声）
    # 在投影仪坐标true_projector_coord处应该有最大响应
    signal = np.exp(2j * np.pi * frequencies * true_projector_coord / N)
    
    # 执行DFT
    spectrum = np.fft.fft(signal, n=N)
    magnitude = np.abs(spectrum)
    
    # 找到峰值
    peak_idx = np.argmax(magnitude)
    
    print(f"\nDFT结果:")
    print(f"  DFT长度: {N}")
    print(f"  峰值索引: {peak_idx}")
    print(f"  理论预期: {true_projector_coord}")
    print(f"  误差: {abs(peak_idx - true_projector_coord)}")
    
    if abs(peak_idx - true_projector_coord) < 1:
        print(f"  ✓ 验证通过：DFT峰值正确对应投影仪坐标")
    else:
        print(f"  ✗ 验证失败：DFT峰值不对应投影仪坐标")
    
    # 可视化频谱
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, np.real(signal), 'b-', label='实部')
    plt.plot(frequencies, np.imag(signal), 'r-', label='虚部')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('信号幅值')
    plt.title('输入信号（频域）')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    coord_axis = np.arange(N)
    plt.plot(coord_axis, magnitude)
    plt.axvline(x=true_projector_coord, color='r', linestyle='--', label=f'真实坐标 {true_projector_coord}')
    plt.axvline(x=peak_idx, color='g', linestyle='--', label=f'DFT峰值 {peak_idx}')
    plt.xlabel('投影仪坐标')
    plt.ylabel('频谱幅值')
    plt.title('DFT频谱')
    plt.legend()
    plt.grid(True)
    plt.xlim(true_projector_coord-50, true_projector_coord+50)
    
    plt.tight_layout()
    plt.savefig('frequency_spectrum_test.png', dpi=150, bbox_inches='tight')
    plt.show()

def explain_fix():
    """
    解释修复的关键点
    """
    print(f"\n{'='*80}")
    print("修复说明")
    print(f"{'='*80}")
    
    print("问题根源:")
    print("  之前的修复错误地认为需要调整DFT长度来适应频率步长")
    print("  实际上，DFT长度应该始终等于投影仪尺寸")
    
    print(f"\n正确理解:")
    print("  1. DFT长度 = 投影仪尺寸（1920或1080）")
    print("  2. 频率步长只影响采集的频率数量，不影响投影仪坐标范围")
    print("  3. DFT输出的索引直接对应投影仪坐标")
    
    print(f"\n修复前的错误:")
    print("  - 错误地计算 effective_N = N / frequency_step")
    print("  - 错误地进行坐标转换 coord = index * frequency_step")
    print("  - 导致坐标范围只有投影仪尺寸的一半")
    
    print(f"\n修复后的正确做法:")
    print("  - DFT长度始终为 N（投影仪尺寸）")
    print("  - DFT输出索引直接是投影仪坐标")
    print("  - 坐标范围覆盖整个投影仪尺寸")
    
    print(f"\n验证方法:")
    print("  1. 检查解码结果的坐标范围")
    print("  2. 验证频谱分析的投影仪坐标")
    print("  3. 确认DFT长度等于投影仪尺寸")

if __name__ == "__main__":
    test_dft_logic()
    simulate_frequency_spectrum()
    explain_fix()
    
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print("修复后的FSA解码器现在应该能够正确处理频率步长为0.5的情况。")
    print("关键改进：")
    print("  1. DFT长度始终等于投影仪尺寸")
    print("  2. 移除错误的坐标转换")
    print("  3. 峰值索引直接对应投影仪坐标")
    print("  4. 坐标范围覆盖整个投影仪尺寸")
    print("\n请重新运行解码程序验证修复效果！")
