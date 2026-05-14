# FSA (频移法) 三维重建完整指南

> 本文档整合了快速开始指南、频移法原理说明和生成结果详解，帮助你全面理解和使用FSA解码系统。

---

## 📑 目录

1. [快速开始](#一快速开始)
2. [频移法原理](#二频移法原理)
3. [生成结果说明](#三生成结果说明)
4. [常见问题](#四常见问题)

---

# 一、快速开始

## ⚡ 5分钟快速上手

### 第1步：准备环境

```bash
# 安装必要的Python包
pip install numpy matplotlib

# 图像读取（二选一）
pip install opencv-python
# 或
pip install Pillow
```

### 第2步：准备数据

确保你的图像按以下结构组织：

```bash
FSM解码过程/
├── data/
│   ├── hor/          # 水平条纹图像
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...       # 共 M×4 张（M为频率数）
│   └── ver/          # 垂直条纹图像
│       ├── 0.png
│       ├── 1.png
│       └── ...       # 共 M×4 张
```

### 第3步：修改配置并运行

打开 `FSM_decode_example.py`，在主函数开头找到配置参数区域：

```python
def main():
    # ========================================================================
    # 🔧 配置参数区域 - 在这里修改所有参数
    # ========================================================================
    
    # -------------------- 1. 投影仪参数 --------------------
    PROJECTOR_WIDTH = 1920      # 投影仪宽度（像素）
    PROJECTOR_HEIGHT = 1080     # 投影仪高度（像素）
    
    # -------------------- 2. FSA采集参数 --------------------
    NUM_FREQUENCIES = 151       # 采集的频率数量（M值）
    
    # -------------------- 3. 解码方向选择 --------------------
    DECODE_DIRECTION = "both"   # 可选: "both", "ver", "hor"
    
    # -------------------- 4. 图像路径配置 --------------------
    IMAGE_DIR_HOR = r"F:\FSM_horizontal_f0_f150_images"
    IMAGE_DIR_VER = r"F:\FSM_vertical_f0_f150_images"
    OUTPUT_DIR = r"D:\在校学习\fsa\results\fsa_decode"
    
    # ... 其他参数
```

然后运行：

```bash
python FSM_decode_example.py
```

### 第4步：查看结果

```bash
# 可视化结果
python FSM_visualize.py
```

---

## 📖 核心概念速查

### FSA是什么？

**FSA (Frequency Shift Analysis)** = 频移法

- 投影一系列**频率递增**的正弦条纹
- 每个频率投影**4张相移图**（0°, 90°, 180°, 270°）
- 通过**DFT（傅里叶变换）**解码出投影仪坐标

### 核心公式

```
对于相机像素 (xc, yc)：

1. 采集 M 个频率的图像：I0, I1, I2, I3 (共 M×4 张)

2. 构造复数信号：
   D[k] = (I0[k] - I2[k]) + j*(I3[k] - I1[k])

3. DFT变换：
   C = FFT(D)

4. 峰值检测：
   投影仪坐标 = argmax(|C|)
```

### 关键参数说明

| 参数 | 说明 | 典型值 | 如何确定 |
|------|------|--------|----------|
| `PROJECTOR_WIDTH` | 投影仪宽度 | 1920 | 查看投影仪分辨率 |
| `PROJECTOR_HEIGHT` | 投影仪高度 | 1080 | 查看投影仪分辨率 |
| `NUM_FREQUENCIES` | 采集频率数 | 121-151 | 通常为投影仪分辨率的1/8-1/12 |
| `PEAK_LIMIT` | 峰值阈值 | 0.5-0.8 | 根据质量要求调整 |

---

## 💡 常见使用场景

### 场景1：作为独立工具使用

```bash
# 1. 修改 FSM_decode_example.py 中的配置参数
# 2. 运行解码
python FSM_decode_example.py

# 3. 查看结果
python FSM_visualize.py
```

### 场景2：集成到你的Python项目

```python
from FSM_decode_core import FSADecoder

# 创建解码器
decoder = FSADecoder(1920, 1080)

# 加载并解码
images = decoder.load_images("./data/ver", 151)
disp, peak, _ = decoder.decode_direction(images, 'ver', 151)

# 使用结果
valid_mask = peak > 0.5
print(f"有效像素: {valid_mask.sum()}")
```

---

## 🔍 输出结果说明

### 生成的文件

```bash
results/fsa_decode/
├── disp_hor.npy      # 投影仪行坐标（Y坐标）
├── peak_hor.npy      # 水平质量图
├── disp_ver.npy      # 投影仪列坐标（X坐标）
├── peak_ver.npy      # 垂直质量图
├── spectrum_hor.npy  # 水平方向频谱数据（调试用）
└── spectrum_ver.npy  # 垂直方向频谱数据（调试用）
```

### 质量评估标准

| 峰值强度 | 质量等级 | 建议 |
|----------|----------|------|
| > 1.5 | 优秀 ⭐⭐⭐⭐⭐ | 可直接使用 |
| 1.0 - 1.5 | 良好 ⭐⭐⭐⭐ | 可使用 |
| 0.5 - 1.0 | 中等 ⭐⭐⭐ | 谨慎使用 |
| < 0.5 | 较差 ⭐ | 建议过滤 |

---

## ⚠️ 常见问题快速解决

### 问题1：FileNotFoundError

**解决方法**：
1. 检查路径是否正确
2. 确认图像文件是否存在
3. 修改配置参数区域的路径

### 问题2：内存不足

**解决方法**：
```python
# 在配置参数区域设置
LOW_MEMORY_MODE = True  # 从False改为True
```

### 问题3：解码结果噪点多

**解决方法**：
```python
# 提高质量阈值
PEAK_LIMIT = 1.0  # 从0.5改为1.0
```

---

# 二、频移法原理

## 基本原理

### 1. 条纹投影的基本概念

频移法通过投影空间变化的正弦条纹：

$$I_{\text{proj}}(x, \theta_k, \phi) = A \sin(2\pi \nu_k x + \phi)$$

其中：
- $x$：投影仪列坐标
- $\nu_k$：第 $k$ 个频率
- $\phi$：相位偏移（0°, 90°, 180°, 270°）
- $A$：条纹强度幅值
- $k = 0, 1, \ldots, M-1$（$M$ 为总频率数）

### 2. 四步相移法

为了消除环境光和提高信号质量，采用四步相移法：
- 相位 0°: $I_0 = B + A\sin(\theta)$
- 相位 90°: $I_1 = B + A\sin(\theta + \pi/2)$
- 相位 180°: $I_2 = B + A\sin(\theta + \pi)$
- 相位 270°: $I_3 = B + A\sin(\theta + 3\pi/2)$

### 3. 信号恢复

通过差分操作消除环境光 $B$：

$$D_c = I_0 - I_2 = 2A\sin(\theta)$$
$$D_s = I_3 - I_1 = -2A\cos(\theta)$$

构造复数信号：

$$D[k] = D_c[k] + j D_s[k] = 2A e^{-j\theta_k}$$

---

## 核心算法

### 算法流程图

```
输入：M×4 张图像序列
  ↓
[步骤1] 图像分组
  I0, I1, I2, I3：形状 (M, height, width)
  ↓
[步骤2] 差分处理
  Dc = I0 - I2  (消除环境光)
  Ds = I3 - I1
  ↓
[步骤3] 复数信号构造
  D[k] = Dc[k] + j·Ds[k]
  ↓
[步骤4] DFT 变换
  C = FFT(D, n=N)
  ↓
[步骤5] 峰值检测
  disparity = argmax(|C|)
  peak = max(|C|) / N
  ↓
输出：disparity（投影仪坐标）, peak（置信度）
```

### Python伪代码

```python
def decode_fsa(images, num_frequencies, dft_length):
    # 步骤1：提取相移序列
    I0 = images[0::4]  # 第0, 4, 8, ... 张图像
    I1 = images[1::4]
    I2 = images[2::4]
    I3 = images[3::4]
    
    # 步骤2：差分处理
    Dc = I0 - I2
    Ds = I3 - I1
    
    # 步骤3：复数信号
    D = Dc + 1j * Ds
    
    # 步骤4：DFT 变换
    C = np.fft.fft(D, n=dft_length, axis=0)
    
    # 步骤5：峰值检测
    magnitude = np.abs(C)
    disparity = np.argmax(magnitude, axis=0)
    peak = np.max(magnitude, axis=0) / dft_length
    
    return disparity, peak
```

---

## 实现细节

### 1. 低内存模式

**原理**：逐行处理而非一次性加载整个图像

**优点**：
- 内存占用从 12GB 降低到 100MB
- 适合内存受限的环境

**缺点**：
- 速度较慢（约 10-30 分钟 vs 1-3 分钟）

### 2. 子像素精度优化

采用质心法（Centroid Method）达到亚像素精度：

```python
# 找到峰值附近的三个点
peak_idx = np.argmax(C_magnitude)
idx_left = (peak_idx - 1) % N
idx_right = (peak_idx + 1) % N

# 计算加权平均
offset = (val_right - val_left) / (val_left + val_center + val_right)
peak_coord = peak_idx + offset * 0.5
```

**优点**：将精度提高到 0.1-0.5 像素

---

# 三、生成结果说明

## 解码结果文件（.npy文件）

### 1. disp_ver.npy - 垂直视差图

**含义**：相机像素到投影仪列坐标（X坐标）的映射

**数组形状**：(height, width) - 相机图像尺寸

**示例**：
```python
import numpy as np

disp_ver = np.load('disp_ver.npy')
camera_x, camera_y = 1200, 1000
projector_x = disp_ver[camera_y, camera_x]

print(f"相机({camera_x},{camera_y}) → 投影仪X={projector_x:.2f}")
```

### 2. peak_ver.npy - 垂直峰值强度图

**含义**：解码质量评估指标

**数值范围与含义**：

| 峰值强度范围 | 质量等级 | 含义 |
|------------|---------|------|
| > 1.0 | 优秀 | 信号强、峰值尖锐 |
| 0.5 - 1.0 | 中等 | 信号较弱、有干扰 |
| < 0.5 | 差 | 信号弱、不可靠 |

**示例**：
```python
peak_ver = np.load('peak_ver.npy')
quality = peak_ver[camera_y, camera_x]

if quality > 1.0:
    print("✅ 高质量，可靠")
elif quality > 0.5:
    print("⚠️ 中等质量")
else:
    print("❌ 低质量，建议丢弃")
```

### 3. disp_hor.npy 和 peak_hor.npy

与垂直方向类似，但存储的是投影仪行坐标（Y坐标）和水平方向的质量。

**完整2D对应关系**：
```python
disp_hor = np.load('disp_hor.npy')
disp_ver = np.load('disp_ver.npy')

projector_x = disp_ver[camera_y, camera_x]
projector_y = disp_hor[camera_y, camera_x]

print(f"相机({camera_x}, {camera_y})")
print(f"  → 投影仪({projector_x:.2f}, {projector_y:.2f})")
```

---

## 频谱分析图像

当设置 `SPECTRUM_POINT = (x, y)` 后，会生成详细的频谱分析图像。

### 关键图像说明

#### 1. 原始强度序列
- 显示采集的原始数据（I0, I1, I2, I3）
- 用于检查采集质量

#### 2. 完整频谱（线性坐标）⭐
- X轴：投影仪坐标（0-1919）
- Y轴：归一化幅值
- 红色虚线：最大峰值位置

**如何解读**：
```
理想情况（高质量）：
幅值
  |        █      ← 只有一个尖锐的主峰
 5|        █
  |       ███
  |_____███████_____ 投影仪坐标

存在歧义（低质量）：
幅值
  |   █       █  ← 两个相近的峰值
 5|   █       █
  |_███_______███___ 投影仪坐标
```

#### 3. 峰值分析图 ⭐⭐⭐

**最重要的一张图！**

显示前20个最强峰值的位置和强度：
- X轴：具体的投影仪坐标
- Y轴：峰值强度
- 红色大圆点：最大峰值

**快速判断标准**：
1. 峰值是否集中？✅ 集中 → 好
2. 最大峰值是否 > 1.0？✅ 是 → 好
3. 主峰是否远大于次峰？✅ 是 → 好

---

## 可视化结果

### statistics.png - 统计分析

包含6个子图，提供完整的质量统计：

**质量分级（最重要）**：
- 🟢 优秀（>2.0）：信号极好
- 🟢 良好（1.5-2.0）：信号好
- 🟡 中等（1.0-1.5）：信号可用
- 🟠 较差（0.5-1.0）：信号弱
- 🔴 很差（<0.5）：背景/无效

**评判标准**：
- ✅ 优秀像素 > 40% → 质量很好
- ⚠️ 优秀像素 20-40% → 质量中等
- ❌ 优秀像素 < 20% → 质量较差

---

# 四、常见问题

## Q1：为什么峰值强度图有很多黑色区域？

**A**：黑色区域代表低质量或无效区域，主要原因：
1. 背景区域（最常见）
2. 阴影区域
3. 高光区域
4. 物体边缘

**这是正常的！** 只要有效区域的峰值高即可。

## Q2：如何判断解码是否成功？

**A**：看这3个指标：
1. 优秀像素占比 > 40% ✅
2. 有效像素占比 > 45% ✅
3. 峰值强度图物体区域有大量亮色 ✅

## Q3：内存不足怎么办？

**A**：在配置参数区域设置：
```python
LOW_MEMORY_MODE = True
```

## Q4：如何提高解码质量？

**A**：
1. 调整相机曝光，避免过曝/欠曝
2. 增加采集频率数量
3. 提高投影仪亮度
4. 确保表面平整、反射良好

---

## 📚 相关文件

- `FSM_decode_core.py` - 核心解码算法
- `FSM_decode_example.py` - 使用示例（配置参数在主函数开头）
- `FSM_visualize.py` - 可视化工具
- `FSM_reconstruct_core.py` - 三维重建核心
- `FSM_reconstruct_example.py` - 三维重建示例

---

**文档版本**: 2.0  
**最后更新**: 2025-11-09  
**整合自**: 快速开始指南.md + 频移法说明.md + 生成结果说明.md
