# FSA解码代码修改说明

---

## 一、问题概括

### 1.1 DFT长度选择不合理为什么导致离散值

FSA解码的核心是对每个像素的M个复数样本执行FFT，然后用`argmax`找峰值位置作为投影仪坐标。FFT长度直接决定了输出频域有多少个"格子"（bin），`argmax`只能返回这些格子的整数索引。

**原始代码**（`frequency_step=1.0`时）：FFT长度 = 投影仪分辨率N（如1920），M个样本被零填充到N点。输出有1920个bin，每个bin对应1个投影仪像素，`argmax`直接给出投影仪坐标，精度为1像素。

**之前的改进版**：FFT长度 = M（如340），输出只有340个bin。`argmax`范围是[0, 339]，乘以`N/M ≈ 5.65`后，输出只能取`0, 5.65, 11.29, 16.94, ...`这些离散值。视差图中每个像素的值都被量化到5.65的整数倍。

### 1.2 为什么离散值导致量化阶梯效应

三维重建中，投影仪坐标（视差值）通过三角化转换为深度。当视差值只能取等间距的离散值时，深度也只能取有限个离散值，在三维点云中表现为一层层平行的"台阶"。间距越大（N/M越大），阶梯越明显。

以M=340、N=1920为例，视差量化步长为5.65像素，对应的深度量化步长取决于基线和焦距，但通常会产生肉眼可见的阶梯。

### 1.3 为什么需要补回 fix_fsa_disparity 和 find_peaks

FSA频谱中，一个相机像素可能在频域存在多个峰值（由反射、遮挡、多路径干扰等引起）。`argmax`只选最大峰，但最大峰不一定是几何正确的匹配。原始代码通过以下机制处理多义性：

- `find_peaks`：找出频谱中**所有**显著峰值（不只是最大的），为每个像素提供多个候选匹配
- `fix_fsa_disparity`：利用极线几何约束（对应点必须在极线上），从多个候选中选出与极线最近的点作为正确匹配

这两个功能对处理复杂场景至关重要，不是"没必要的操作"。

---

## 二、修改思路说明

### 2.1 DFT长度修复

**核心改法**：新增 `_get_fft_params()` 方法，集中管理FFT长度选择逻辑：

```python
def _get_fft_params(self, direction, num_frequencies):
    proj_dim = self.proj_height if direction == "hor" else self.proj_width
    if self.frequency_step == 1.0:
        fft_length = proj_dim    # 投影仪分辨率
        coord_scale = 1.0         # argmax直接是投影仪坐标
    else:
        fft_length = num_frequencies  # M
        coord_scale = proj_dim / num_frequencies
    return fft_length, coord_scale, proj_dim
```

然后 `decode_direction()` 和 `_decode_direction_low_memory()` 都调用此方法获取参数。

### 2.2 后处理功能接入

- `MultiPeakDetector`：独立类，提供 `find_peaks()` 和 `get_peaks_for_pixels()` 两个静态方法
- `EpipolarCorrector`：独立类，构造函数接受标定参数，提供 `get_epipolar_line()` 和 `correct_disparity()` 方法
- `FSM_decode_example.py` 中新增 `run_epipolar_correction()` 函数，整合多峰检测和极线校正的完整流程

### 2.3 修改项与对应问题

| 修改项 | 解决的问题 |
|--------|-----------|
| `_get_fft_params()` 集中FFT参数计算 | DFT长度选择错误 |
| `decode_direction()` 使用 `fft_length` | 视差离散化 / 量化阶梯 |
| `_decode_direction_low_memory()` 同步修复 | 低内存模式的同一问题 |
| 峰值归一化使用 `fft_length` | 峰值强度与原始代码一致 |
| 新增 `MultiPeakDetector` | 恢复多峰检测能力 |
| 新增 `EpipolarCorrector` | 恢复极线约束校正 |
| 新增 `run_epipolar_correction()` | 提供完整的后处理流程 |

---

## 四、逐项修改内容说明

### 4.1 FSM_decode_core.py 的修改

#### 修改点1：新增 `_get_fft_params()` 方法

- **具体改动**：在 `FSADecoder` 类中新增方法，恢复 `decode_fsa.py` 第75行（`N = H if dir == "hor" else W`）和第152-157行的FFT长度分支逻辑
- **作用**：集中管理FFT参数，确保标准模式和低内存模式使用一致的参数
- **预期效果**：消除离散值问题的根源

#### 修改点2：`decode_direction()` 中的FFT调用

- **具体改动**：
  - 第800-801行：`N = num_frequencies` → 调用 `_get_fft_params()` 获取 `fft_length`
  - 第940行：`n=N` → `n=fft_length`
  - 第946行：直接赋值 `freq_index` 或乘以 `coord_scale`（恢复原始代码第170-177行的分支）
  - 第949行：`/ N` → `/ fft_length`
- **作用**：使FFT零填充到投影仪分辨率，argmax直接输出投影仪坐标
- **预期效果**：视差值从 N/M 间隔的离散值变为连续的整数值（1像素精度）

#### 修改点3：`_decode_direction_low_memory()` 参数修改

- **具体改动**：方法签名从接受 `N`（含义模糊）改为接受 `fft_length` 和 `coord_scale`（含义明确）；内部所有 `% N` 改为 `% fft_length`；归一化改用 `fft_length`
- **作用**：低内存模式与标准模式保持一致
- **预期效果**：低内存模式同样获得1像素精度

#### 修改点4：新增 `MultiPeakDetector` 类

- **具体改动**：恢复 `decode_fsa.py` 的 `find_peaks()` (第377-393行) 和 `get_peaks_position()` (第395-433行)
- **作用**：在频谱中找出所有候选峰值
- **改进**：使用 `(position, value)` 元组存储峰值，修复原始代码中浮点数索引访问数组的潜在bug

#### 修改点5：新增 `EpipolarCorrector` 类

- **具体改动**：恢复 `decode_fsa.py` 的 `get_epipolar_line()` (第525-596行)、`fix_fsa_disparity()` (第437-512行) 和 `draw_epipolar()` (第516-523行)
- **作用**：利用极线几何约束从多个候选峰值中选出正确匹配
- **改进**：将原始代码中通过文件路径加载标定数据改为直接传入参数，消除对 pyaoi 的依赖；将原始的诊断工具改造为自动校正函数

### 4.2 FSM_decode_example.py 的修改

#### 修改点6：`decode_fsa_images()` 返回图像缓存

- **具体改动**：新增 `images_cache` 返回值，缓存加载的图像数据
- **作用**：后续多峰检测需要原始图像来重新计算每个像素的FFT频谱
- **预期效果**：避免后处理时重复加载图像

#### 修改点7：新增 `run_epipolar_correction()` 函数

- **具体改动**：整合 MultiPeakDetector 和 EpipolarCorrector 的完整后处理流程
- **作用**：提供一键式极线约束校正
- **预期效果**：对指定像素执行多峰检测→极线约束→选择最佳候选→更新视差

#### 修改点8：main() 中新增极线校正配置区

- **具体改动**：新增 `ENABLE_EPIPOLAR_CORRECTION`、`PIXELS_TO_CORRECT`、标定参数等配置项
- **作用**：用户可以方便地开启/关闭极线校正后处理

#### 修改点9：FFT策略信息显示

- **具体改动**：在 `decode_fsa_images()` 中打印FFT长度策略信息
- **作用**：帮助用户理解当前使用的DFT策略，便于诊断问题

---

## 五、DFT长度相关改动详细说明

### 5.1 原始代码中DFT长度选择逻辑的意义

原始代码 `decode_fsa.py` 第75行和第152-157行：

```python
N = H if dir == "hor" else W      # N = 投影仪分辨率

if frequency_step == 1.0:
    fft_length = N                  # 使用投影仪分辨率
else:
    fft_length = M                  # 使用频率数量
```

**数学意义**：

FSA的复数信号 D[k] (k=0,...,M-1) 在 `frequency_step=1.0` 时满足：

    D[k] ∝ exp(j · 2π · k · x_p / N)

其中 x_p 是投影仪坐标，N是投影仪分辨率。

对 D 执行N点FFT：

    C[n] = Σ D[k] · exp(-j · 2π · n · k / N)

当 n = x_p 时，相位项全部抵消，频谱出现最大峰值。因此 `argmax(|C|)` 直接返回 x_p。

零填充（M→N）的作用是将原本只有M个采样点的频谱"插值"到N个点，使每个投影仪像素都有一个对应的频率bin。这不是增加信息量，而是利用已有信息做更精细的定位。

### 5.2 改进版为什么出问题

改进版代码（修改前）第800-801行：

```python
N = num_frequencies                              # N = M = 340
coord_scale = self.proj_width / num_frequencies  # = 1920/340 ≈ 5.647
```

这导致：
1. FFT只有340个bin，频域采样间隔为 1920/340 ≈ 5.65 像素
2. `argmax` 返回 [0, 339] 的整数
3. 乘以 5.647 后，输出只能取 0, 5.65, 11.29, 16.94, ... 等离散值
4. 相邻两个可能的输出值之间间隔 5.65 像素——这就是"量化步长"

### 5.3 具体修复方式

在 `_get_fft_params()` 中恢复原始逻辑：

```python
if self.frequency_step == 1.0:
    fft_length = proj_dim    # 投影仪分辨率（如1920）
    coord_scale = 1.0         # 不需要映射
else:
    fft_length = num_frequencies
    coord_scale = proj_dim / num_frequencies
```

在 `decode_direction()` 的FFT块中：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)    # fft_length=1920
C_magnitude = np.abs(C)
freq_index = np.argmax(C_magnitude, axis=0)       # 范围 [0, 1919]
disparity[ty:ey, tx:ex] = freq_index              # 直接是投影仪坐标
peak[ty:ey, tx:ex] = np.max(C_magnitude, axis=0) / fft_length
```

### 5.4 为什么能缓解量化阶梯

修复前：视差值 ∈ {0, 5.65, 11.29, 16.94, ...}，步长5.65
修复后：视差值 ∈ {0, 1, 2, 3, 4, 5, ...}，步长1

在三维重建中，深度与视差近似线性关系。步长从5.65降为1，量化阶梯的高度降低到约原来的1/5.65。对于M=121的情况（步长从15.87降为1），改善幅度更大。

### 5.5 验证数据

使用模拟FSA信号（M=340, N=1920）的验证结果：

| 真实位置 | 修复后输出 | 误差 | 修复前输出 | 误差 |
|---------|-----------|------|-----------|------|
| 100     | 100       | 0    | 101.65    | 1.65 |
| 500     | 500       | 0    | 502.59    | 2.59 |
| 957     | 957       | 0    | 954.35    | 2.65 |
| 1500    | 1500      | 0    | 1502.12   | 2.12 |

修复后误差为0（理想信号下），修复前最大误差约2.65像素。
