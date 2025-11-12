"""
分析单方向重建中视差图误差对高度信息的影响
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载双方向（真实）和单方向（计算）的视差图
print("加载视差图...")
disp_hor_real = np.load("results/fsa_decode/disp_hor.npy")
disp_ver_real = np.load("results/fsa_decode/disp_ver.npy")

# 加载单方向重建时计算的视差图
# 需要先运行一次单方向重建并保存计算结果
print(f"真实水平视差范围: [{disp_hor_real.min()}, {disp_hor_real.max()}]")
print(f"真实垂直视差范围: [{disp_ver_real.min()}, {disp_ver_real.max()}]")

# 分析视差图的分布
valid_mask = (disp_hor_real > 0) & (disp_ver_real > 0)
print(f"\n有效点数: {valid_mask.sum():,}")

# 计算视差的变化范围（反映物体的深度信息）
hor_range = disp_hor_real[valid_mask].max() - disp_hor_real[valid_mask].min()
ver_range = disp_ver_real[valid_mask].max() - disp_ver_real[valid_mask].min()

print(f"\n视差变化范围:")
print(f"  水平方向: {hor_range:.1f} 像素")
print(f"  垂直方向: {ver_range:.1f} 像素")

# 分析视差梯度（反映表面起伏）
from scipy.ndimage import sobel

grad_hor_x = sobel(disp_hor_real, axis=1)
grad_hor_y = sobel(disp_hor_real, axis=0)
grad_ver_x = sobel(disp_ver_real, axis=1)
grad_ver_y = sobel(disp_ver_real, axis=0)

print(f"\n视差梯度统计（反映表面起伏）:")
print(f"  水平视差梯度: 均值={np.abs(grad_hor_x[valid_mask]).mean():.3f}, 最大={np.abs(grad_hor_x[valid_mask]).max():.3f}")
print(f"  垂直视差梯度: 均值={np.abs(grad_ver_x[valid_mask]).mean():.3f}, 最大={np.abs(grad_ver_x[valid_mask]).max():.3f}")

# 可视化视差分布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 水平视差图
im1 = axes[0, 0].imshow(disp_hor_real, cmap='jet', vmin=0, vmax=disp_hor_real.max())
axes[0, 0].set_title(f'水平视差图 (范围: {hor_range:.1f})')
axes[0, 0].set_xlabel('X (像素)')
axes[0, 0].set_ylabel('Y (像素)')
plt.colorbar(im1, ax=axes[0, 0])

# 垂直视差图
im2 = axes[0, 1].imshow(disp_ver_real, cmap='jet', vmin=0, vmax=disp_ver_real.max())
axes[0, 1].set_title(f'垂直视差图 (范围: {ver_range:.1f})')
axes[0, 1].set_xlabel('X (像素)')
axes[0, 1].set_ylabel('Y (像素)')
plt.colorbar(im2, ax=axes[0, 1])

# 水平视差直方图
axes[1, 0].hist(disp_hor_real[valid_mask], bins=100, alpha=0.7, color='blue')
axes[1, 0].set_title('水平视差分布')
axes[1, 0].set_xlabel('视差值')
axes[1, 0].set_ylabel('频数')
axes[1, 0].grid(True, alpha=0.3)

# 垂直视差直方图
axes[1, 1].hist(disp_ver_real[valid_mask], bins=100, alpha=0.7, color='red')
axes[1, 1].set_title('垂直视差分布')
axes[1, 1].set_xlabel('视差值')
axes[1, 1].set_ylabel('频数')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('disparity_analysis.png', dpi=150)
print(f"\n✓ 视差分析图已保存: disparity_analysis.png")

# 关键发现
print("\n" + "="*60)
print("关键发现:")
print("="*60)

if ver_range > hor_range * 1.5:
    print(f"⚠ 垂直视差变化范围 ({ver_range:.1f}) 远大于水平视差 ({hor_range:.1f})")
    print("  → 垂直方向包含更多深度信息")
    print("  → 如果只用水平视差计算垂直视差，会丢失大量高度信息")
elif hor_range > ver_range * 1.5:
    print(f"⚠ 水平视差变化范围 ({hor_range:.1f}) 远大于垂直视差 ({ver_range:.1f})")
    print("  → 水平方向包含更多深度信息")
    print("  → 如果只用垂直视差计算水平视差，会丢失大量高度信息")
else:
    print(f"✓ 两个方向的视差变化范围相近")
    print(f"  水平: {hor_range:.1f}, 垂直: {ver_range:.1f}")

print("\n理论分析:")
print("-" * 60)
print("基础矩阵约束: x_cam^T · F · x_proj = 0")
print("这是一个线性约束，给定一个方向的视差，另一方向有无穷多个解！")
print("我们只能找到满足约束的一个近似解，但不一定是真实解。")
print("\n当物体有显著的高度变化时:")
print("  1. 真实的两个方向视差都会有较大变化")
print("  2. 但基础矩阵计算出的视差可能变化很小")
print("  3. 导致重建的点云被'压扁'，丢失高度信息")
