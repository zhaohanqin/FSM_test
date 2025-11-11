"""
FSA解码结果可视化

本文件提供了多种可视化方法来查看FSA解码的结果：
1. 视差图（伪彩色显示）
2. 峰值强度图（质量图）
3. 统计直方图
4. 特定点的频谱分析
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class FSAVisualizer:
    """FSA解码结果可视化工具"""
    
    def __init__(self, result_dir: str, peak_threshold: float = 0.5):
        """
        初始化可视化工具
        
        Parameters:
        -----------
        result_dir : str
            解码结果文件夹路径
        peak_threshold : float, optional
            峰值阈值，用于过滤低质量像素。默认值为0.5
            - 推荐值：0.5（中等质量以上）
            - 高质量：1.0
            - 包容性强：0.3
        """
        self.result_dir = result_dir
        self.peak_threshold = peak_threshold  # 统一的阈值参数
        self.disparity_hor = None
        self.disparity_ver = None
        self.peak_hor = None
        self.peak_ver = None
        
        print(f"初始化可视化工具，峰值阈值 = {peak_threshold}")
        
        # 加载数据
        self.load_results()
    
    def load_results(self):
        """加载解码结果"""
        print("加载解码结果...")
        
        # 尝试加载所有结果文件
        files = {
            'disp_hor.npy': 'disparity_hor',
            'disp_ver.npy': 'disparity_ver',
            'peak_hor.npy': 'peak_hor',
            'peak_ver.npy': 'peak_ver'
        }
        
        for filename, attr_name in files.items():
            filepath = os.path.join(self.result_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                setattr(self, attr_name, data)
                print(f"  ✓ 加载 {filename}: 形状 {data.shape}")
            else:
                print(f"  ✗ 未找到 {filename}")
    
    def visualize_disparity(self, direction='ver', peak_threshold=None, save_path=None):
        """
        可视化视差图
        
        Parameters:
        -----------
        direction : str
            'hor' 或 'ver'
        peak_threshold : float, optional
            峰值阈值，低于此值的像素将被标记为无效
            如果为None，则使用初始化时设定的统一阈值
        save_path : str, optional
            保存图像的路径
        """
        # 如果没有指定阈值，使用统一阈值
        if peak_threshold is None:
            peak_threshold = self.peak_threshold
        
        if direction == 'hor':
            disparity = self.disparity_hor
            peak = self.peak_hor
            title = '水平视差图（投影仪行坐标）'
        else:
            disparity = self.disparity_ver
            peak = self.peak_ver
            title = '垂直视差图（投影仪列坐标）'
        
        if disparity is None or peak is None:
            print(f"警告: {direction} 方向的数据未加载")
            return
        
        # 创建掩码
        valid_mask = peak >= peak_threshold
        
        # 创建显示图像
        disp_masked = disparity.copy()
        disp_masked[~valid_mask] = np.nan  # 无效区域设为NaN
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：视差图
        im1 = axes[0].imshow(disp_masked, cmap='jet', interpolation='nearest')
        axes[0].set_title(f'{title}\n(峰值阈值 ≥ {peak_threshold})', fontsize=12)
        axes[0].set_xlabel('X (像素)')
        axes[0].set_ylabel('Y (像素)')
        plt.colorbar(im1, ax=axes[0], label='投影仪坐标')
        
        # 右图：峰值强度图
        im2 = axes[1].imshow(peak, cmap='hot', interpolation='nearest', vmin=0, vmax=2)
        axes[1].set_title(f'峰值强度图（质量）\n有效像素: {np.sum(valid_mask)}/{valid_mask.size} '
                         f'({np.sum(valid_mask)/valid_mask.size*100:.1f}%)', fontsize=12)
        axes[1].set_xlabel('X (像素)')
        axes[1].set_ylabel('Y (像素)')
        plt.colorbar(im2, ax=axes[1], label='峰值强度')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    def visualize_statistics(self, save_path=None):
        """
        可视化统计信息
        
        Parameters:
        -----------
        save_path : str, optional
            保存图像的路径
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 定义质量分级的公共变量（两个方向都会使用）
        quality_bins = [0, 0.5, 1.0, 1.5, 2.0, np.inf]
        quality_labels = ['很差\n(<0.5)', '较差\n(0.5-1.0)', '中等\n(1.0-1.5)', '良好\n(1.5-2.0)', '优秀\n(>2.0)']
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        # 水平方向
        if self.disparity_hor is not None and self.peak_hor is not None:
            # 视差分布
            ax1 = plt.subplot(2, 3, 1)
            disp_flat = self.disparity_hor.flatten()
            ax1.hist(disp_flat, bins=100, color='blue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('投影仪行坐标')
            ax1.set_ylabel('像素数量')
            ax1.set_title(f'水平视差分布\n均值={disp_flat.mean():.1f}, 标准差={disp_flat.std():.1f}')
            ax1.grid(True, alpha=0.3)
            
            # 峰值分布
            ax2 = plt.subplot(2, 3, 2)
            peak_flat = self.peak_hor.flatten()
            ax2.hist(peak_flat, bins=100, color='red', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('峰值强度')
            ax2.set_ylabel('像素数量')
            ax2.set_title(f'水平峰值分布\n均值={peak_flat.mean():.3f}, 中值={np.median(peak_flat):.3f}')
            ax2.axvline(x=0.5, color='green', linestyle='--', label='建议阈值=0.5')
            ax2.axvline(x=1.0, color='orange', linestyle='--', label='高质量阈值=1.0')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 质量分布图
            ax3 = plt.subplot(2, 3, 3)
            quality_counts = [np.sum((peak_flat >= quality_bins[i]) & (peak_flat < quality_bins[i+1])) 
                             for i in range(len(quality_bins)-1)]
            bars = ax3.bar(quality_labels, quality_counts, color=colors, edgecolor='black')
            ax3.set_ylabel('像素数量')
            ax3.set_title('水平方向质量分级')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上标注数量
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({height/len(peak_flat)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)
        
        # 垂直方向
        if self.disparity_ver is not None and self.peak_ver is not None:
            # 视差分布
            ax4 = plt.subplot(2, 3, 4)
            disp_flat = self.disparity_ver.flatten()
            ax4.hist(disp_flat, bins=100, color='blue', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('投影仪列坐标')
            ax4.set_ylabel('像素数量')
            ax4.set_title(f'垂直视差分布\n均值={disp_flat.mean():.1f}, 标准差={disp_flat.std():.1f}')
            ax4.grid(True, alpha=0.3)
            
            # 峰值分布
            ax5 = plt.subplot(2, 3, 5)
            peak_flat = self.peak_ver.flatten()
            ax5.hist(peak_flat, bins=100, color='red', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('峰值强度')
            ax5.set_ylabel('像素数量')
            ax5.set_title(f'垂直峰值分布\n均值={peak_flat.mean():.3f}, 中值={np.median(peak_flat):.3f}')
            ax5.axvline(x=0.5, color='green', linestyle='--', label='建议阈值=0.5')
            ax5.axvline(x=1.0, color='orange', linestyle='--', label='高质量阈值=1.0')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 质量分布图
            ax6 = plt.subplot(2, 3, 6)
            quality_counts = [np.sum((peak_flat >= quality_bins[i]) & (peak_flat < quality_bins[i+1])) 
                             for i in range(len(quality_bins)-1)]
            bars = ax6.bar(quality_labels, quality_counts, color=colors, edgecolor='black')
            ax6.set_ylabel('像素数量')
            ax6.set_title('垂直方向质量分级')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上标注数量
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({height/len(peak_flat)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('FSA解码质量统计分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    def visualize_correspondence_map(self, peak_threshold=None, save_path=None):
        """
        可视化完整的相机-投影仪对应关系
        
        Parameters:
        -----------
        peak_threshold : float, optional
            峰值阈值，如果为None，则使用初始化时设定的统一阈值
        save_path : str, optional
            保存路径
        """
        # 如果没有指定阈值，使用统一阈值
        if peak_threshold is None:
            peak_threshold = self.peak_threshold
        
        if self.disparity_hor is None or self.disparity_ver is None:
            print("警告: 需要两个方向的视差数据")
            return
        
        # 计算综合质量
        peak_min = np.minimum(self.peak_hor, self.peak_ver)
        valid_mask = peak_min >= peak_threshold
        
        # 创建RGB图像显示对应关系
        # R通道：投影仪X坐标（归一化）
        # G通道：投影仪Y坐标（归一化）
        # B通道：质量（归一化）
        
        proj_x_norm = self.disparity_ver / self.disparity_ver.max()
        proj_y_norm = self.disparity_hor / self.disparity_hor.max()
        quality_norm = np.clip(peak_min / 2.0, 0, 1)  # 假设峰值最大为2
        
        correspondence_map = np.stack([proj_x_norm, proj_y_norm, quality_norm], axis=2)
        correspondence_map[~valid_mask] = 0  # 无效区域设为黑色
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 左图：对应关系彩色图
        axes[0].imshow(correspondence_map)
        axes[0].set_title(f'相机-投影仪对应关系图\n(R=投影仪X, G=投影仪Y, B=质量)', fontsize=11)
        axes[0].set_xlabel('相机X坐标')
        axes[0].set_ylabel('相机Y坐标')
        
        # 中图：投影仪X坐标
        im1 = axes[1].imshow(np.where(valid_mask, self.disparity_ver, np.nan), 
                             cmap='jet', interpolation='nearest')
        axes[1].set_title('投影仪X坐标（列）', fontsize=11)
        axes[1].set_xlabel('相机X坐标')
        axes[1].set_ylabel('相机Y坐标')
        plt.colorbar(im1, ax=axes[1])
        
        # 右图：投影仪Y坐标
        im2 = axes[2].imshow(np.where(valid_mask, self.disparity_hor, np.nan), 
                             cmap='jet', interpolation='nearest')
        axes[2].set_title('投影仪Y坐标（行）', fontsize=11)
        axes[2].set_xlabel('相机X坐标')
        axes[2].set_ylabel('相机Y坐标')
        plt.colorbar(im2, ax=axes[2])
        
        plt.suptitle(f'FSA解码完整结果 (有效像素: {np.sum(valid_mask)}/{valid_mask.size}, '
                    f'{np.sum(valid_mask)/valid_mask.size*100:.1f}%)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    def plot_spectrum_at_point(self, x, y, direction='ver', 
                               image_dir=None, num_frequencies=121):
        """
        绘制特定点的频谱图（需要原始图像）
        
        Parameters:
        -----------
        x, y : int
            相机像素坐标
        direction : str
            'hor' 或 'ver'
        image_dir : str
            原始图像文件夹路径
        num_frequencies : int
            频率数量
        """
        if image_dir is None:
            print("需要提供原始图像路径")
            return
        
        print(f"加载点 ({x}, {y}) 的图像序列...")
        
        # 加载图像
        images = []
        for i in range(num_frequencies * 4):
            img_path = os.path.join(image_dir, f"{i}.png")
            try:
                import cv2
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            except:
                from PIL import Image
                img = np.array(Image.open(img_path).convert('L'))
            images.append(img)
        
        images = np.array(images, dtype=np.float32)
        
        # 提取四步相移
        I0 = images[0::4, y, x]
        I1 = images[1::4, y, x]
        I2 = images[2::4, y, x]
        I3 = images[3::4, y, x]
        
        # 构造复数信号
        Dc = I0 - I2
        Ds = I3 - I1
        D = Dc + 1j * Ds
        
        # DFT
        if direction == 'hor':
            N = 768  # 投影仪高度
        else:
            N = 1024  # 投影仪宽度
        
        C = np.fft.fft(D, n=N)
        C_magnitude = np.abs(C) / N
        
        # 获取解码结果
        if direction == 'hor':
            decoded_coord = self.disparity_hor[y, x] if self.disparity_hor is not None else None
            peak_value = self.peak_hor[y, x] if self.peak_hor is not None else None
        else:
            decoded_coord = self.disparity_ver[y, x] if self.disparity_ver is not None else None
            peak_value = self.peak_ver[y, x] if self.peak_ver is not None else None
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 左上：原始强度序列
        axes[0, 0].plot(I0, 'o-', label='I0 (0°)', linewidth=1.5)
        axes[0, 0].plot(I1, 's-', label='I1 (90°)', linewidth=1.5)
        axes[0, 0].plot(I2, '^-', label='I2 (180°)', linewidth=1.5)
        axes[0, 0].plot(I3, 'd-', label='I3 (270°)', linewidth=1.5)
        axes[0, 0].set_xlabel('频率索引 k')
        axes[0, 0].set_ylabel('强度值')
        axes[0, 0].set_title(f'点 ({x}, {y}) 的强度序列')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 右上：差分信号
        axes[0, 1].plot(Dc, 'o-', label='Dc = I0 - I2', linewidth=1.5)
        axes[0, 1].plot(Ds, 's-', label='Ds = I3 - I1', linewidth=1.5)
        axes[0, 1].set_xlabel('频率索引 k')
        axes[0, 1].set_ylabel('差分值')
        axes[0, 1].set_title('差分信号（去除直流分量）')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 左下：完整频谱
        axes[1, 0].plot(C_magnitude, '-', linewidth=1.5)
        if decoded_coord is not None:
            axes[1, 0].axvline(x=decoded_coord, color='red', linestyle='--', 
                              label=f'解码位置={decoded_coord:.1f}')
        axes[1, 0].set_xlabel('投影仪坐标')
        axes[1, 0].set_ylabel('幅值')
        axes[1, 0].set_title(f'完整频谱 (峰值={peak_value:.4f})' if peak_value else '完整频谱')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 右下：局部放大
        if decoded_coord is not None:
            center = int(decoded_coord)
            half_width = 50
            start = max(0, center - half_width)
            end = min(N, center + half_width)
            
            axes[1, 1].plot(range(start, end), C_magnitude[start:end], 'o-', linewidth=2)
            axes[1, 1].axvline(x=decoded_coord, color='red', linestyle='--', 
                              label=f'峰值位置={decoded_coord:.2f}')
            axes[1, 1].set_xlabel('投影仪坐标')
            axes[1, 1].set_ylabel('幅值')
            axes[1, 1].set_title('频谱局部放大')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


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
    
    # -------------------- 1. 路径配置 --------------------
    BASE_DIR = r"D:\在校学习\fsa"                                    # 基础目录
    RESULT_DIR = os.path.join(BASE_DIR, "results", "fsa_decode")   # 解码结果目录
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", "visualizations") # 可视化输出目录
    
    # -------------------- 2. 峰值阈值设置 --------------------
    # 🔥 统一峰值阈值 - 修改这一个参数，就可以改变所有可视化的阈值！
    PEAK_THRESHOLD = 0.5  # 可选值：
                          # - 0.3: 包容性强，显示更多像素（可能包含噪声）
                          # - 0.5: 中等质量以上（推荐）
                          # - 0.8: 只显示高质量像素
                          # - 1.0: 只显示最高质量像素
    
    # ========================================================================
    # ✅ 配置参数区域结束 - 下方为程序执行代码，一般不需要修改
    # ========================================================================
    
    print("="*80)
    print("FSA解码结果可视化工具")
    print("="*80)
    print(f"\n📋 配置参数:")
    print(f"  解码结果目录: {RESULT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  峰值阈值: {PEAK_THRESHOLD}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建可视化工具（传入统一阈值）
    visualizer = FSAVisualizer(RESULT_DIR, peak_threshold=PEAK_THRESHOLD)
    
    # 1. 可视化视差图（自动使用统一阈值）
    print("\n生成视差图...")
    visualizer.visualize_disparity(
        direction='ver',
        save_path=os.path.join(OUTPUT_DIR, 'disparity_ver.png')
    )
    
    visualizer.visualize_disparity(
        direction='hor',
        save_path=os.path.join(OUTPUT_DIR, 'disparity_hor.png')
    )
    
    # 2. 统计分析
    print("\n生成统计分析...")
    visualizer.visualize_statistics(
        save_path=os.path.join(OUTPUT_DIR, 'statistics.png')
    )
    
    # 3. 完整对应关系（自动使用统一阈值）
    print("\n生成对应关系图...")
    visualizer.visualize_correspondence_map(
        save_path=os.path.join(OUTPUT_DIR, 'correspondence_map.png')
    )
    
    print("\n" + "="*80)
    print("可视化完成！")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
