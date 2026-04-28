# FSM / FSA 频移法原理与实现说明

本文说明本仓库当前主线中的 FSM / FSA（Frequency Shift Analysis，频移法）解码方法。说明以根目录下的 `FSM_decode_core.py` 和 `FSM_decode_example.py` 为准，`initial/decode_fsa.py` 只作为历史过程式实现和算法演进参考。

## 1. 方法目标

结构光三维重建需要知道每个相机像素对应投影仪上的哪个像素。相机拍到的是物体表面反射回来的条纹图像，而不是投影仪坐标本身。FSM 的作用就是从一组频率递增、带相移的条纹图像中，为每个相机像素恢复投影仪坐标：

```text
相机像素 (xc, yc) -> 投影仪像素 (xp, yp)
```

在当前主线代码中，这个对应关系被拆成两个方向解码：

- 水平条纹 `hor`：条纹沿水平方向延伸，灰度沿投影仪纵向变化，用于解码投影仪行坐标 `yp` / Y 坐标。
- 垂直条纹 `ver`：条纹沿垂直方向延伸，灰度沿投影仪横向变化，用于解码投影仪列坐标 `xp` / X 坐标。

解码完成后，主线输出：

- `disp_hor.npy`：每个相机像素对应的投影仪行坐标 `yp`。
- `disp_ver.npy`：每个相机像素对应的投影仪列坐标 `xp`。
- `peak_hor.npy`：水平条纹方向的频谱峰值强度，用作解码质量或置信度。
- `peak_ver.npy`：垂直条纹方向的频谱峰值强度，用作解码质量或置信度。

## 2. 为什么要投影一系列条纹图像

单张条纹图像只能告诉相机像素处的灰度值，不能唯一确定它来自投影仪的哪个位置。因为同一张正弦条纹中会有很多位置灰度相同，亮度还会受到环境光、物体颜色、表面反射率、曝光和噪声影响。

FSM 通过两层序列解决这个问题：

1. 对每个频率投影多张相移图像。
2. 投影多个不同频率的条纹图像。

### 2.1 四步相移用于构造复信号

当前解码代码默认每个频率有 4 张相移图像：

```text
0.png, 1.png, 2.png, 3.png     -> 频率 0 的 0°, 90°, 180°, 270°
4.png, 5.png, 6.png, 7.png     -> 频率 1 的 0°, 90°, 180°, 270°
...
```

对同一个相机像素，在同一个频率下可以得到四个灰度值：

```text
I0 = 0° 相移图像中的灰度
I1 = 90° 相移图像中的灰度
I2 = 180° 相移图像中的灰度
I3 = 270° 相移图像中的灰度
```

然后构造：

```text
Dc = I0 - I2
Ds = I3 - I1
D  = Dc + j * Ds
```

这样做有两个作用：

- `I0 - I2` 和 `I3 - I1` 会抵消大量直流亮度项，包括环境光、背景灰度和物体局部反射率造成的共同亮度偏移。
- `Dc` 和 `Ds` 分别提供近似余弦分量和正弦分量，可以组合成复数信号 `D`，让后续 FFT 直接处理相位随频率变化的规律。

### 2.2 多个频率用于形成“频率轴信号”

FSM 的核心不是只看某一个频率，而是看同一个相机像素在一系列投影频率下的响应。对一个固定相机像素 `(xc, yc)`，程序会沿频率轴取出：

```text
D[0], D[1], D[2], ..., D[M-1]
```

这里 `M` 是采集的频率数量。这个长度为 `M` 的复数序列就是该相机像素的频率响应信号。

如果某个相机像素真实对应投影仪坐标 `u`，那么随着投影频率递增，该点接收到的复数相位会按照与 `u` 有关的速率旋转。换句话说，投影仪坐标被编码成了频率轴上的相位变化率。对这个频率轴信号做 DFT / FFT 后，能量会集中在与投影仪坐标对应的位置，因此频谱峰值位置就可以用来恢复投影仪坐标。

### 2.3 一个更具体的数学视角

旧版 `频移法说明.md` 中有一个有价值的角度：可以把投影条纹写成随投影仪坐标变化的正弦信号。以某一个方向为例，投影仪坐标记为 `u`，第 `k` 个投影频率记为 `f_k`，相移记为 `phi`，理想投影强度可以写成：

```text
I_proj(u, k, phi) = A * sin(2*pi*f_k*u + phi)
```

相机实际看到的不是这个理想强度，而是经过物体反射、环境光和相机响应后的结果。对某个相机像素 `(xc, yc)`，如果它对应投影仪坐标 `u_p`，可以粗略写成：

```text
I_cam(xc, yc, k, phi) = B(xc, yc) + R(xc, yc) * A * sin(2*pi*f_k*u_p + phi)
```

这里：

- `B(xc, yc)` 表示环境光、暗电流、背景亮度等近似直流项。
- `R(xc, yc)` 表示物体表面反射率和局部亮度缩放。
- `u_p` 是希望解码出来的投影仪坐标。

四步相移的差分操作会显著削弱 `B`，并把主要信息转化成与 `2*pi*f_k*u_p` 相关的复数相位。对于当前代码使用的构造方式：

```text
Dc = I0 - I2
Ds = I3 - I1
D  = Dc + j*Ds
```

可以把 `D[k]` 理解为一个随频率 `k` 旋转的复数。不同投影仪坐标 `u_p` 会导致不同的旋转速度；DFT / FFT 的作用就是在频域中寻找这个旋转速度最匹配的位置。

需要注意：正弦还是余弦、相移正方向、`Ds = I3-I1` 还是 `I1-I3` 会影响复数相位的符号，但当前解码最终使用的是幅值谱 `abs(C)` 的峰值位置，因此核心匹配机制不依赖文档中选用的正负号约定。

## 3. 每个相机像素具体做了什么

当前主线在 `FSM_decode_core.py` 的 `FSADecoder.decode_direction()` 中执行解码。对一个方向来说，输入图像数组形状为：

```text
(M * 4, height, width)
```

其中 `height` 和 `width` 是相机图像尺寸。解码过程可以理解为对相机图像中的每个像素独立执行以下步骤。

### 3.1 提取四个相移序列

代码用切片把图像按相位重新分组：

```python
I0 = images[0::4].astype(np.float32)
I1 = images[1::4].astype(np.float32)
I2 = images[2::4].astype(np.float32)
I3 = images[3::4].astype(np.float32)
```

此时 `I0/I1/I2/I3` 的形状都是：

```text
(M, height, width)
```

对任意相机像素 `(x, y)`，`I0[:, y, x]` 就是该像素在全部频率下的 0° 相移灰度序列。

### 3.2 构造频率轴复信号

代码构造：

```python
Dc = I0 - I2
Ds = I3 - I1
D = Dc + 1j * Ds
```

`D` 的形状仍然是：

```text
(M, height, width)
```

但每个像素位置上的数据已经从实数灰度序列变成复数频率响应序列。对单个像素来说：

```text
D[:, y, x] = 该相机像素沿频率轴的一维复信号
```

### 3.3 沿频率维度做 FFT

主线代码按图像块处理，避免一次性为整幅图分配过大的频谱数组。对每个块：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)
```

这里 `axis=0` 表示 FFT 沿频率维度执行，而不是沿图像的行或列执行。也就是说，程序不是对图像空间做二维傅里叶变换，而是对每个相机像素自己的频率响应序列做一维傅里叶变换。

对单个像素来说，可以写成：

```text
C[k] = sum D[m] * exp(-j * 2π * k * m / fft_length)
```

当 `k` 与该相机像素对应的投影仪坐标一致或最接近时，各频率样本的相位会对齐，幅值 `|C[k]|` 最大。

### 3.4 取频谱峰值作为投影仪坐标

主线代码使用：

```python
freq_index = np.argmax(C_magnitude, axis=0)
peak = np.max(C_magnitude, axis=0) / fft_length
```

含义是：

- `freq_index`：每个相机像素频谱最大峰值的位置。
- `peak`：最大峰值强度，经过 `fft_length` 归一化，用作质量评价。

当坐标映射比例 `coord_scale == 1.0` 时：

```python
disparity = freq_index
```

否则：

```python
disparity = freq_index * coord_scale
```

这样就得到了当前方向上的投影仪坐标图。

### 3.5 与 `FSM_decode_core.py` 的代码对应关系

下面这段是主线解码入口中最核心的流程。它把整组图像拆成四个相移序列，再构造复信号：

```python
I0 = images[0::4].astype(np.float32)
I1 = images[1::4].astype(np.float32)
I2 = images[2::4].astype(np.float32)
I3 = images[3::4].astype(np.float32)

Dc = I0 - I2
Ds = I3 - I1
D = Dc + 1j * Ds
```

这段代码对应的是“对所有相机像素同时做同一件事”。例如 `I0[:, y, x]` 表示相机像素 `(x, y)` 在所有频率下的 0° 相移响应；`D[:, y, x]` 表示该像素最终用于 FFT 的一维复数序列。

从数据形状上看：

```text
images: (M*4, Hc, Wc)
I0/I1/I2/I3: (M, Hc, Wc)
Dc/Ds/D: (M, Hc, Wc)
```

其中 `Hc` 和 `Wc` 是相机图像高度和宽度。注意这里的第一维始终是频率维度，所以后续 FFT 必须沿 `axis=0` 执行。

主线为了控制内存，不是一次性对整幅图像做完整频谱保存，而是按块处理：

```python
for ty in range(0, D.shape[1], block_size):
    for tx in range(0, D.shape[2], block_size):
        ey = min(ty + block_size, D.shape[1])
        ex = min(tx + block_size, D.shape[2])

        D_block = D[:, ty:ey, tx:ex]
        C = np.fft.fft(D_block, n=fft_length, axis=0)
        C_magnitude = np.abs(C)
```

`D_block` 的形状是：

```text
(M, block_h, block_w)
```

对这个块做 `np.fft.fft(..., axis=0)` 后，`C` 的形状变成：

```text
(fft_length, block_h, block_w)
```

也就是说，块中的每个相机像素都有一条长度为 `fft_length` 的频谱。程序接着对每个像素在这条频谱上找最大值：

```python
freq_index = np.argmax(C_magnitude, axis=0)

if coord_scale == 1.0:
    disparity[ty:ey, tx:ex] = freq_index
else:
    disparity[ty:ey, tx:ex] = freq_index * coord_scale

peak[ty:ey, tx:ex] = np.max(C_magnitude, axis=0) / fft_length
```

这里有三个输出含义：

- `freq_index` 是频谱峰值所在的 bin。
- `disparity` 是把峰值 bin 解释或映射后的投影仪坐标。
- `peak` 是最大峰值强度，数值越大通常表示该像素的频率响应越明确。

### 3.6 单个像素的等价伪代码

虽然主线代码使用 NumPy 向量化和分块处理，但对一个相机像素 `(x, y)` 来说，可以理解成下面的伪代码：

```python
signal = []

for f in range(M):
    I0 = image[4*f + 0][y, x]
    I1 = image[4*f + 1][y, x]
    I2 = image[4*f + 2][y, x]
    I3 = image[4*f + 3][y, x]

    Dc = I0 - I2
    Ds = I3 - I1
    signal.append(Dc + 1j * Ds)

C = fft(signal, n=fft_length)
coord = argmax(abs(C))
quality = max(abs(C)) / fft_length
```

这段伪代码说明了一个关键点：FSM 解码不是在一张图里找条纹位置，而是对“同一个相机像素随投影频率变化形成的复信号”做频谱分析。

### 3.7 低内存模式的实现价值

旧版 `频移法说明.md` 专门提到过低内存模式，这一点对当前主线仍然有价值。标准模式会把 `D_block` 做成三维块，再对块内所有像素同时 FFT，速度较快；但如果图像尺寸大、`fft_length` 大、频谱分析点多，内存压力会明显增加。

当前 `FSM_decode_core.py` 中保留了 `_decode_direction_low_memory()`，它的思路是逐行处理：

```python
for row in range(height):
    I0_row = images[0::4, row, :].astype(np.float32)
    I1_row = images[1::4, row, :].astype(np.float32)
    I2_row = images[2::4, row, :].astype(np.float32)
    I3_row = images[3::4, row, :].astype(np.float32)

    Dc_row = I0_row - I2_row
    Ds_row = I3_row - I1_row
    D_row = Dc_row + 1j * Ds_row

    for col in range(width):
        signal = D_row[:, col]
        C = np.fft.fft(signal, n=fft_length)
        C_magnitude = np.abs(C)
```

这个模式每次只保留一行的频率响应，输出仍然是完整的 `disparity` 和 `peak`。它的取舍很明确：

- 优点：显著降低中间频谱数组的内存峰值。
- 缺点：逐像素循环更多，通常比标准分块模式慢。

因此，当机器内存足够时，优先使用标准模式；当标准模式内存不足或需要在较小内存环境中跑通流程时，再打开 `LOW_MEMORY_MODE = True`。

### 3.8 图像缓存机制

旧版说明中还提到图像缓存，这在当前主线里由 `ImageLoader.load_images()` 实现。默认缓存路径是输入图像目录下的：

```text
images_cache.npy
```

核心逻辑是：

```python
if use_cache and os.path.exists(cache_path):
    images = np.load(cache_path)
    return images

# 从图像文件读取后
if use_cache:
    np.save(cache_path, images)
```

这对 FSA 很重要，因为一次解码通常需要读取 `M*4` 张图像；如果同时解码水平和垂直方向，读取量会翻倍。缓存不会改变算法结果，只是把大量小图像文件的重复解码和磁盘读取，转换成一次 `.npy` 数组读取，便于重复调试 FFT 参数、频谱分析点、多峰扫描和极线校正。

## 4. FFT 长度策略

FFT 长度是当前主线中特别重要的实现细节。`FSM_decode_core.py` 中通过 `_get_fft_params()` 集中处理：

```python
proj_dim = self.proj_height if direction == "hor" else self.proj_width

if self.frequency_step == 1.0:
    fft_length = proj_dim
    coord_scale = 1.0
else:
    fft_length = num_frequencies
    coord_scale = proj_dim / num_frequencies
```

### 4.1 `frequency_step == 1.0`

当频率步长为 1.0 时，FFT 长度使用投影仪对应方向分辨率：

- `hor`：`fft_length = projector_height`
- `ver`：`fft_length = projector_width`

如果采集频率数量 `M` 小于投影仪分辨率，`np.fft.fft(..., n=fft_length)` 会把长度为 `M` 的频率序列零填充到 `fft_length`。零填充不会凭空增加新信息，但会让频谱在更多 bin 上被采样，相当于对频谱做插值。这样每个频谱 bin 可以直接对应一个投影仪像素坐标，`argmax` 可以直接输出投影仪坐标。

这个策略恢复自历史实现 `initial/decode_fsa.py`，也是当前主线用于避免量化阶梯的重要修复点。

### 4.2 `frequency_step != 1.0`

当频率步长不是 1.0 时，主线使用：

```text
fft_length = M
coord_scale = 投影仪对应方向分辨率 / M
```

这时 `argmax` 得到的是频率索引，需要再映射到投影仪坐标。这个分支用于非标准频率采样实验，例如频率步长变化时的测试。

### 4.3 为什么错误的 FFT 长度会产生量化阶梯

如果在 `frequency_step == 1.0` 时错误地使用 `fft_length = M`，那么频谱只有 `M` 个 bin。假设投影仪宽度是 1920，采集频率数是 340，则每个 bin 对应：

```text
1920 / 340 ≈ 5.65 个投影仪像素
```

此时 `argmax` 只能返回整数频率索引，再乘以 5.65 后，视差值只能取：

```text
0, 5.65, 11.29, 16.94, ...
```

这种离散化会传递到三维重建中，在点云深度上表现为量化阶梯。当前主线恢复“FFT 长度等于投影仪分辨率”的策略后，标准模式下视差最小步长回到 1 个投影仪像素，阶梯问题会明显减轻。

### 4.4 `_get_fft_params()` 的源码含义

当前主线把 FFT 长度和坐标映射集中放在 `FSADecoder._get_fft_params()` 中，避免标准模式、低内存模式和后处理逻辑各自写一套规则：

```python
proj_dim = self.proj_height if direction == "hor" else self.proj_width

if self.frequency_step == 1.0:
    fft_length = proj_dim
    coord_scale = 1.0
else:
    fft_length = num_frequencies
    coord_scale = proj_dim / num_frequencies

return fft_length, coord_scale, proj_dim
```

这段代码里的 `proj_dim` 表示当前方向要恢复的投影仪坐标范围：

- `direction == "hor"` 时恢复投影仪行坐标，所以使用投影仪高度 `projector_height`。
- `direction == "ver"` 时恢复投影仪列坐标，所以使用投影仪宽度 `projector_width`。

`coord_scale` 的意义是“频谱 bin 到投影仪像素坐标的换算比例”。标准模式下 `coord_scale = 1.0`，说明频谱峰值位置已经是投影仪像素坐标。非标准频率步长下，频谱峰值仍然是频率索引，需要再乘以 `coord_scale`。

### 4.5 标准模式下为什么 `M` 可以小于投影仪分辨率

在实际实验中，`NUM_FREQUENCIES` 可能是 151 或其他远小于投影仪宽高的数值。直觉上可能会疑惑：如果只采集 151 个频率，为什么能在 1920 或 1080 个投影仪像素坐标上找峰？

代码中的关键是：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
```

当 `fft_length` 大于输入序列长度 `M` 时，NumPy 会在频率序列后补零。补零后的 FFT 会给出更多频谱采样点，让峰值位置可以在投影仪分辨率尺度上表达。它不增加原始采集信息，因此不能替代更密集采集或亚像素估计；但它能避免把输出强行量化到 `M` 个粗 bin 上。

因此，当前主线中的“FFT 长度使用投影仪分辨率”更准确地说是：

```text
用投影仪像素坐标尺度采样频谱峰值位置
```

这也是 `modification_analysis.md` 中提到量化阶梯修复的根本原因。

### 4.6 计算复杂度和内存占用怎么看

旧版 `频移法说明.md` 里有计算复杂度和内存占用分析，这部分可以保留，但需要按当前主线的分块实现来理解。

对单个方向，输入图像规模是：

```text
M * 4 张图像，每张相机图像大小为 Hc * Wc
```

主要计算量来自沿频率轴的 FFT。粗略估算：

| 环节 | 规模 | 说明 |
| --- | --- | --- |
| 图像加载 | `O(M*Hc*Wc)` | 实际读取 `M*4` 张图，但常数项主要由四步相移决定。 |
| 构造 `I0/I1/I2/I3` | `O(M*Hc*Wc)` | 切片和类型转换。 |
| 构造 `D` | `O(M*Hc*Wc)` | 两次差分和一次复数组合。 |
| FFT | 约 `O(Hc*Wc*fft_length*log(fft_length))` | 每个相机像素做一次长度为 `fft_length` 的 FFT。 |
| 峰值检测 | `O(Hc*Wc*fft_length)` | 每个像素在频谱维度上找最大值。 |

标准分块模式的内存峰值主要来自当前块的 `D_block`、FFT 输出 `C` 和幅值谱 `C_magnitude`。如果块大小是 `block_size=100`，则一个块大约包含：

```text
block_h * block_w ≈ 100 * 100
```

FFT 输出的频谱规模约为：

```text
fft_length * block_h * block_w
```

这也是为什么主线默认采用分块处理，而不是直接对整幅图的所有像素同时保留完整频谱。分块模式在速度和内存之间折中；低内存模式进一步把块缩小到“单行 + 单像素循环”，换取更低内存占用。

需要注意，NumPy FFT 的实际临时数组和 dtype 可能由实现决定，所以文档中更适合给出量级判断，而不是固定说一定占用多少 GB。

## 5. 主线代码结构

### 5.1 `FSM_decode_example.py`

`FSM_decode_example.py` 是当前解码主入口，主要负责实验配置和流程组织：

- 设置投影仪宽高：`PROJECTOR_WIDTH`、`PROJECTOR_HEIGHT`。
- 设置频率数量和频率步长：`NUM_FREQUENCIES`、`FREQUENCY_STEP`。
- 设置解码方向：`DECODE_DIRECTION = "both" / "hor" / "ver"`。
- 设置输入路径：`IMAGE_DIR_HOR`、`IMAGE_DIR_VER`。
- 设置输出路径：`OUTPUT_DIR`。
- 控制缓存、序列检查、低内存模式、频谱分析点。
- 可选启用多峰像素扫描。
- 可选启用极线约束后处理校正。

实际解码由 `decode_fsa_images()` 组织。它会创建：

```python
decoder = FSADecoder(PROJECTOR_WIDTH, PROJECTOR_HEIGHT, FREQUENCY_STEP)
```

然后分别对水平和垂直方向调用：

```python
decoder.load_images(...)
decoder.decode_direction(...)
decoder.save_results(...)
```

### 5.2 `FSM_decode_core.py`

`FSM_decode_core.py` 是当前主线解码核心，主要类包括：

- `ImageLoader`：读取条纹图像序列，支持缓存和中文路径。
- `CalibrationReader`：从 `标定结果.txt` 读取相机和投影仪标定参数，供极线约束校正使用。
- `DFTProcessor`：封装复信号构造、FFT 和幅值谱计算。
- `PeakDetector`：封装最大峰值查找和简单子像素峰值细化。
- `MultiPeakDetector`：检测频谱中的多个候选峰值，用于处理误匹配。
- `EpipolarCorrector`：使用相机-投影仪极线约束，从候选峰组合中选择几何上更合理的匹配。
- `FSADecoder`：协调完整解码流程，提供 `load_images()`、`decode_direction()`、`save_results()` 等主接口。

### 5.3 `initial/decode_fsa.py`

`initial/decode_fsa.py` 是历史过程式实现，不是当前默认修改目标。它的价值在于保留了当前主线算法的原始形态：

- `decode_fsa()`：早期 FSA 解码主函数。
- `find_peaks()`：早期多峰检测函数。
- `get_peaks_position()`：指定像素候选峰提取。
- `fix_fsa_disparity()`：早期极线约束诊断与校正流程。
- `get_epipolar_line()`：早期极线计算逻辑。

当前 `FSM_decode_core.py` 可以看作对它的模块化重构，并移除了对部分历史依赖的直接绑定。

## 6. 多峰检测与极线约束校正

理想情况下，每个相机像素的频谱只有一个明显主峰，`argmax(|C|)` 就能得到正确投影仪坐标。但实际场景中可能出现多个显著峰值：

- 高反光区域可能产生混合反射。
- 遮挡边界可能让一个相机像素接收到多个表面或边缘附近的混合信号。
- 投影、采集和物体表面之间可能存在多路径干扰。
- 噪声或低纹理反射会降低主峰优势。

此时最大峰不一定是几何上正确的匹配。当前主线保留了两级后处理能力。

### 6.1 多峰检测

`MultiPeakDetector.find_peaks()` 会遍历归一化频谱，找出高于阈值的局部极大值。距离很近的峰会按幅值加权合并，得到一个候选峰位置。

对指定像素，`get_peaks_for_pixels()` 会分别在 `hor` 和 `ver` 方向上返回候选坐标：

```text
peaks_hor -> 候选投影仪行坐标 yp
peaks_ver -> 候选投影仪列坐标 xp
```

### 6.2 极线约束校正

`EpipolarCorrector` 使用相机和投影仪的标定参数构造投影矩阵，并计算相机像素在投影仪图像中的极线。对于同一个相机像素，它把水平和垂直方向候选峰做组合：

```text
(px, py) = peaks_ver × peaks_hor
```

然后计算每个候选点到投影仪极线的距离：

```text
distance = |a*x + b*y + c| / sqrt(a^2 + b^2)
```

距离最小的候选点被认为是更符合双目几何约束的匹配点。`FSM_decode_example.py` 中的 `run_epipolar_correction()` 会把校正结果保存为：

- `disp_hor_corrected.npy`
- `disp_ver_corrected.npy`

极线约束校正是可选后处理，通常只在发现明显误匹配、多峰像素或复杂反射区域时启用。

### 6.3 多峰检测的源码片段

`MultiPeakDetector.find_peaks()` 不只找最大峰，而是遍历整条归一化频谱，保留高于阈值的局部极大值：

```python
peaks = []

for i in range(len(spectrum)):
    value = float(spectrum[i])
    if value < limit:
        continue

    left_peak = (i == 0) or (spectrum[i] > spectrum[i - 1])
    right_peak = (i == len(spectrum) - 1) or (spectrum[i] > spectrum[i + 1])
    is_peak = left_peak and right_peak

    if is_peak:
        if len(peaks) > 0 and (i - peaks[-1][0]) < merge_distance:
            prev_pos, prev_val = peaks[-1]
            ki = value / (value + prev_val)
            new_pos = ki * i + (1.0 - ki) * prev_pos
            peaks[-1] = (new_pos, max(value, prev_val))
        else:
            peaks.append((float(i), value))
```

这段逻辑有三个判断：

- 先用 `limit` 去掉弱峰，减少噪声候选。
- 再用左右邻域判断局部极大值。
- 如果两个峰距离小于 `merge_distance`，说明它们可能属于同一个宽峰或旁瓣附近的重复检测，程序用幅值加权位置合并。

因此，多峰检测返回的是一组候选投影仪坐标，而不是最终坐标。最终坐标仍需要结合几何约束或人工诊断判断。

### 6.4 极线约束校正的源码片段

极线校正的核心是：把水平候选 `peaks_hor` 和垂直候选 `peaks_ver` 组合成投影仪候选点，然后选距离极线最近的点。

```python
pys = np.repeat(pk_hor, len(pk_ver))
pxs = np.tile(pk_ver, len(pk_hor))

proj_points = np.column_stack([pxs, pys]).astype(np.float64)
undis_points = cv2.undistortPoints(
    proj_points, self.Kp, self.dp, P=self.Kp
)
undis_points = undis_points.reshape(-1, 2)

distances = np.abs(
    lp[0] * undis_points[:, 0] + lp[1] * undis_points[:, 1] + lp[2]
) / np.sqrt(lp[0]**2 + lp[1]**2)

best_idx = np.argmin(distances)
```

这里的 `lp` 是投影仪图像中的极线方程：

```text
lp[0] * x + lp[1] * y + lp[2] = 0
```

如果一个候选投影仪点是真实匹配，它应该落在这条极线附近。程序通过点到直线距离公式为每个候选点打分，距离越小，几何一致性越好。

校正结果保存的字段中：

- `px` 是校正后的投影仪列坐标。
- `py` 是校正后的投影仪行坐标。
- `distance` 是该候选点到极线的距离。
- `all_candidates` 保留所有候选点及其距离，便于后续诊断。

### 6.5 为什么多峰和极线约束不是默认替代主解码

主解码的 `argmax` 是全图批量处理，速度快、流程稳定，适合作为默认输出。多峰检测和极线约束需要重新分析指定像素或扫描全图，并且依赖标定参数。它更适合作为后处理：

- 当峰值图显示某些区域质量低时，用于定位问题像素。
- 当频谱分析图显示多个强峰时，用于判断 `argmax` 是否选错。
- 当重建点云在局部区域出现明显异常时，用于修正少量可疑像素。

因此，当前主线把它设计为可选流程，而不是每次解码都强制执行。

## 7. 频谱分析输出

`FSM_decode_example.py` 支持通过 `SPECTRUM_POINTS` 指定一个或多个相机像素进行频谱分析。程序会保存该像素的：

- 原始四相移灰度序列 `I0/I1/I2/I3`
- 差分信号 `Dc/Ds`
- 复信号相关数据
- 完整 FFT 幅值谱
- 峰值位置和峰值强度
- 前若干个强峰信息

这些结果用于检查某个像素为什么被解码到某个投影仪坐标，也适合为论文或实验记录提供中间过程图。

## 8. 解码质量、误差来源与排查

旧版 `频移法说明.md` 中的质量评估部分值得保留，但需要理解为经验诊断，而不是严格的物理判据。当前主线里的 `peak` 定义是：

```text
peak = max(abs(C)) / fft_length
```

它反映的是某个像素频谱主峰的强度。通常来说，主峰越高、次峰越低，该像素的投影仪坐标越明确。

### 8.1 峰值强度的经验分级

`FSM_decode_example.py` 当前输出质量统计时使用了类似下面的分组：

| 峰值强度范围 | 经验含义 | 使用建议 |
| --- | --- | --- |
| `peak > 1.0` | 主峰明显，信号较强 | 通常可优先保留。 |
| `0.5 < peak <= 1.0` | 信号中等 | 可用，但建议结合点云结果和频谱图检查。 |
| `peak <= 0.5` | 信号较弱 | 更容易出现误匹配，可考虑过滤或重点诊断。 |

这个阈值和投影亮度、相机曝光、表面反射、图像灰度范围、FFT 归一化方式都有关系。实际使用时不应机械套用，而应结合当前数据集的峰值直方图、视差图连续性和三维重建结果来调整。

### 8.2 频谱图怎么看

对指定像素的频谱分析图，可以重点看三件事：

- 最大峰值位置是否与视差图中的坐标一致。
- 主峰是否明显高于其他峰。
- 是否存在多个强峰，尤其是在反光、遮挡边界或深度突变区域。

如果前几个峰值强度接近，说明该像素对应关系存在多义性。此时单纯 `argmax` 可能选到强度最大但几何不正确的点，需要结合 `MultiPeakDetector` 和 `EpipolarCorrector` 做后处理判断。

### 8.3 常见误差来源

| 误差来源 | 可能表现 | 缓解方式 |
| --- | --- | --- |
| 投影仪或相机标定误差 | 点云整体偏移、局部深度不准 | 重新检查标定参数和重投影误差。 |
| 投影仪光学畸变 | 投影仪坐标对应关系局部弯曲 | 使用正确的投影仪内参与畸变参数。 |
| 相机曝光不合适 | 峰值普遍偏低或饱和区域异常 | 调整曝光、增益和投影亮度。 |
| 表面反光或多路径反射 | 频谱多峰、局部误匹配 | 使用多峰检测和极线约束校正。 |
| 遮挡边界 | 视差突变、点云边缘毛刺 | 结合峰值图、深度连续性和边界过滤。 |
| 条纹序列顺序错误 | 大范围解码错乱 | 启用 `VERIFY_SEQUENCE` 或检查图像命名。 |
| FFT 长度设置不合理 | 视差量化阶梯 | 保持标准模式下 `fft_length = projector dimension`。 |

### 8.4 常见排查顺序

当解码结果异常时，建议按下面顺序排查：

1. 检查输入图像数量是否为 `NUM_FREQUENCIES * 4`。
2. 检查图像命名是否符合 `0.png, 1.png, 2.png, 3.png, ...` 的频率优先顺序。
3. 启用 `VERIFY_SEQUENCE`，确认四个相移图之间确实有灰度变化。
4. 查看 `peak_hor.npy` 和 `peak_ver.npy`，判断是整体信号弱还是局部异常。
5. 对异常点设置 `SPECTRUM_POINTS`，检查频谱主峰和次峰。
6. 如果有多峰，尝试多峰扫描和极线约束校正。
7. 如果点云呈现层状台阶，优先检查 `FREQUENCY_STEP` 和 FFT 长度策略。

## 9. 与三维重建的关系

FSM 解码本身不直接输出三维点云。它输出的是相机像素到投影仪像素的二维对应关系：

```text
相机像素 (xc, yc)
投影仪像素 (xp, yp)
```

其中：

```text
xp = disp_ver[yc, xc]
yp = disp_hor[yc, xc]
```

后续 `FSM_reconstruct_example.py` 和 `FSM_reconstruct_core.py` 会读取这些对应关系，再结合相机-投影仪标定参数做三角化，生成三维点云。峰值图 `peak_hor.npy` 和 `peak_ver.npy` 可用于过滤低质量像素，避免把置信度低的对应点带入三维重建。

## 10. 当前主线的完整数据流

```text
投影多频率、多相移条纹
        |
        v
相机采集水平 / 垂直条纹图像序列
        |
        v
按频率和相位整理为 0.png, 1.png, 2.png, 3.png, ...
        |
        v
FSM_decode_example.py 读取配置并调用 FSADecoder
        |
        v
ImageLoader 加载 M*4 张图像
        |
        v
提取 I0/I1/I2/I3
        |
        v
构造 D = (I0-I2) + j*(I3-I1)
        |
        v
对每个相机像素沿频率轴做 FFT
        |
        v
取频谱最大峰值作为投影仪坐标
        |
        v
保存 disp_hor / disp_ver / peak_hor / peak_ver
        |
        v
可选：频谱分析、多峰扫描、极线约束校正
        |
        v
三维重建读取对应关系并三角化生成点云
```

这就是当前仓库中 FSM / FSA 频移法从条纹图像到投影仪坐标对应关系，再到三维重建输入的完整工作机制。
