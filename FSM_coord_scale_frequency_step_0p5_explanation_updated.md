# FSM / FSA 中 `coord_scale` 与 `frequency_step=0.5` 改进方案详细说明

本文档整理了前面对话中关于 FSM / FSA 频移法解码里 `coord_scale`、`frequency_step`、`fft_length` 的讨论，重点说明：

- `coord_scale` 到底是什么意思；
- 为什么标准情况下 `coord_scale = 1.0`；
- 当投影频率步长从 `1.0` 改为 `0.5` 时，为什么不能简单沿用原来的非标准分支；
- 为什么推荐让 `fft_length = 2 * proj_dim`；
- FFT 输出 `2 * proj_dim` 个 bin 后，哪些 bin 是有效投影仪坐标；
- 实际代码应该怎样修改。

本文讨论的频率变化是从：

```text
0, 1, 2, 3, ..., 150
```

改为：

```text
0, 0.5, 1, 1.5, 2, 2.5, ..., 150
```

---

## 1. 原始代码中的关键逻辑

当前代码中有类似下面的逻辑：

```python
if frequency_step == 1.0:
    fft_length = proj_dim
    coord_scale = 1.0
else:
    fft_length = num_frequencies
    coord_scale = proj_dim / num_frequencies
```

其中：

```text
frequency_step  = 投影频率步长
proj_dim         = 当前方向对应的投影仪分辨率
num_frequencies = 采集频率数量
fft_length       = FFT 输出长度
coord_scale      = FFT bin 到投影仪坐标的映射比例
```

对于不同解码方向：

```text
ver 方向：解码投影仪列坐标 xp，proj_dim = projector_width
hor 方向：解码投影仪行坐标 yp，proj_dim = projector_height
```

---

## 2. `freq_index` 和 `coord_scale` 分别是什么

代码中：

```python
freq_index = np.argmax(C_magnitude, axis=0)
peak = np.max(C_magnitude, axis=0) / fft_length
```

这里的 `freq_index` 不是投影仪分辨率，而是：

```text
当前相机像素的频谱最大峰值所在的 FFT bin 编号
```

例如垂直方向解码投影仪 X 坐标，假设：

```text
projector_width = 1920
fft_length = 1920
```

FFT 输出：

```text
C[0], C[1], C[2], ..., C[1919]
```

如果某个像素得到：

```python
freq_index = 834
```

含义是：

```text
该相机像素最匹配投影仪第 834 列
```

所以：

```text
xp = 834
```

而 `coord_scale` 的作用是：

```text
把 FFT 峰值 bin 编号换算成投影仪像素坐标
```

即：

```python
disparity = freq_index * coord_scale
```

可以把它理解成：

```text
一个 FFT bin 代表多少个投影仪像素
```

---

## 3. 为什么标准情况下 `coord_scale = 1.0`

标准频率序列是：

```text
0, 1, 2, 3, ..., 150
```

此时：

```text
frequency_step = 1.0
```

对于某个相机像素，如果它对应投影仪坐标 `u`，理想复数信号可以写成：

```text
D[k] ≈ A * exp(j * 2π * k * u / N)
```

其中：

```text
k = 频率编号
u = 投影仪坐标
N = 投影仪该方向分辨率，也就是 proj_dim
```

如果设置：

```python
fft_length = N
```

那么 FFT 的第 `q` 个 bin 对应的候选旋转模式是：

```text
exp(j * 2π * q * k / N)
```

当：

```text
q = u
```

时，候选旋转模式和真实信号匹配，频谱峰值就出现在第 `u` 个 bin。

因此：

```text
freq_index = u
```

也就是：

```text
FFT bin 编号 = 投影仪像素坐标
```

所以标准情况下：

```python
coord_scale = 1.0
disparity = freq_index
```

---

## 4. 你对 `coord_scale` 的疑惑总结

你原来的疑惑可以概括为：

> 如果 `coord_scale = 1.0` 时，FFT 输出的 bin 已经可以和投影仪坐标对应，那为什么还会有 `coord_scale != 1.0` 的情况？  
> 如果 `coord_scale != 1.0`，为什么要用 `disparity = freq_index * coord_scale`？

答案是：

```text
当 fft_length 等于投影仪分辨率 proj_dim 时，一个 FFT bin 正好对应一个投影仪像素，所以 coord_scale = 1.0。

当 fft_length 不等于 proj_dim 时，FFT bin 的数量和投影仪像素数量不一致，freq_index 只是频谱索引，还不是投影仪坐标，所以需要乘 coord_scale 做尺度映射。
```

例如：

```text
proj_dim = 1920
fft_length = 301
```

此时 FFT 只有 301 个 bin，但投影仪有 1920 个像素坐标。

如果直接把：

```text
freq_index = 100
```

当作投影仪坐标，就会明显错误。

所以需要映射：

```text
投影仪坐标 ≈ freq_index * 1920 / 301
```

这就是 `coord_scale` 的来源。

---

## 5. 当前 `frequency_step != 1.0` 分支的问题

当前非标准分支是：

```python
fft_length = num_frequencies
coord_scale = proj_dim / num_frequencies
```

假设你把频率改成：

```text
0, 0.5, 1, 1.5, ..., 150
```

则：

```text
num_frequencies = 301
```

如果投影仪宽度为：

```text
projector_width = 1920
```

那么当前代码会得到：

```text
fft_length = 301
coord_scale = 1920 / 301 ≈ 6.38
```

这意味着：

```text
一个 FFT bin 大约对应 6.38 个投影仪像素
```

最终输出坐标会类似：

```text
0
6.38
12.76
19.14
25.52
...
```

这会产生明显的坐标量化阶梯。

对于点云重建来说，这种量化会导致点云出现层状、台阶状问题。

因此，如果你希望保持像素级坐标精度，不建议在 `frequency_step = 0.5` 的情况下继续使用：

```python
fft_length = num_frequencies
coord_scale = proj_dim / num_frequencies
```

---

## 6. 为什么频率步长改成 0.5 后，`frequency_step` 不能仍然写 1

你提出过一个想法：

> 虽然频率步长变成了 0.5，但最高频率还是 150，所以 `frequency_step` 是不是仍然可以设置为 1？

结论是：

```text
不可以。
```

原因是 FSM / FSA 解码不是只看最高频率，而是看：

```text
D[k] 随着频率编号 k 增加时，相位转得有多快
```

当频率为：

```text
0, 1, 2, 3, ...
```

第 `k` 个采样点的真实频率是：

```text
f_k = k
```

但当频率为：

```text
0, 0.5, 1, 1.5, ...
```

第 `k` 个采样点的真实频率是：

```text
f_k = 0.5 * k
```

所以同样的 `k -> k+1`，真实频率只增加了 0.5，而不是 1。

这会导致：

```text
同一个投影仪坐标 u 对应的相位旋转速度变成原来的一半
```

因此，真实投影频率步长是多少，`frequency_step` 就应该设置为多少：

```python
frequency_step = 0.5
```

不能继续伪装成：

```python
frequency_step = 1.0
```

否则坐标比例会错。

---

## 7. `frequency_step = 0.5` 时信号模型如何变化

标准情况：

```text
frequency_step = 1.0
f_k = k
```

复数信号为：

```text
D[k] ≈ A * exp(j * 2π * k * u / N)
```

当频率步长变成：

```text
frequency_step = 0.5
```

真实频率为：

```text
f_k = 0.5 * k
```

复数信号变成：

```text
D[k] ≈ A * exp(j * 2π * 0.5 * k * u / N)
```

也可以写成：

```text
D[k] ≈ A * exp(j * 2π * k * u / (2N))
```

这句话非常关键。

它说明：

```text
当 frequency_step = 0.5 时，
同一个投影仪坐标 u 在频率编号 k 上表现出来的相位旋转速度变成原来的一半。
```

如果你仍然使用：

```python
fft_length = N
```

那么 FFT 峰值大约会出现在：

```text
q = 0.5 * u
```

而不是：

```text
q = u
```

例如真实坐标为：

```text
u = 800
```

标准步长下峰值大约在：

```text
q = 800
```

但步长 0.5 且 FFT 长度仍为 `N` 时，峰值大约在：

```text
q = 400
```

所以如果直接输出：

```python
disparity = freq_index
```

坐标会大约缩小一半。

---

## 8. 通用坐标映射公式

假设真实频率序列为：

```text
f_k = s * k
```

其中：

```text
s = frequency_step
```

则理想复数信号为：

```text
D[k] ≈ A * exp(j * 2π * s * k * u / N)
```

FFT 长度为 `L` 时，第 `q` 个 bin 对应的候选旋转模式为：

```text
exp(j * 2π * q * k / L)
```

匹配条件是：

```text
q / L ≈ s * u / N
```

所以：

```text
q ≈ L * s * u / N
```

反过来得到：

```text
u ≈ q * N / (L * s)
```

因此更通用的坐标映射公式是：

```python
coord_scale = proj_dim / (fft_length * frequency_step)
disparity = freq_index * coord_scale
```

也就是说：

```text
投影仪坐标 = FFT峰值bin编号 × proj_dim / (fft_length × frequency_step)
```

这个公式比当前代码里的：

```python
coord_scale = proj_dim / num_frequencies
```

更通用。

---

## 9. 为什么推荐 `fft_length = 2 * proj_dim`

对于：

```text
frequency_step = 0.5
```

如果希望：

```text
FFT峰值bin编号 = 投影仪像素坐标
```

也就是希望：

```python
coord_scale = 1.0
```

根据通用公式：

```python
coord_scale = proj_dim / (fft_length * frequency_step)
```

令：

```text
coord_scale = 1.0
```

得到：

```text
fft_length = proj_dim / frequency_step
```

当：

```text
frequency_step = 0.5
```

时：

```text
fft_length = proj_dim / 0.5 = 2 * proj_dim
```

所以推荐：

```python
fft_length = int(round(proj_dim / frequency_step))
coord_scale = 1.0
```

对于 `frequency_step = 0.5`，就是：

```python
fft_length = 2 * proj_dim
coord_scale = 1.0
```

这样做的目的不是让投影仪坐标范围翻倍，而是让：

```text
FFT bin 编号重新和投影仪像素坐标对齐
```

---

## 10. FFT 长度变成 `2 * proj_dim` 后，输出是什么样

假设：

```text
proj_dim = N = 1920
frequency_step = 0.5
```

则：

```text
fft_length = 2N = 3840
```

FFT 输出为：

```text
C[0], C[1], C[2], ..., C[3839]
```

也就是：

```text
一共有 3840 个 bin
```

但投影仪真实有效坐标只有：

```text
0 ~ 1919
```

所以有效投影仪坐标对应：

```text
bin 0    -> 投影仪坐标 0
bin 1    -> 投影仪坐标 1
bin 2    -> 投影仪坐标 2
...
bin 1919 -> 投影仪坐标 1919
```

也就是：

```text
bin 0 ~ bin N-1
```

这一段。

---

## 11. 大于 `proj_dim` 的 bin 怎么处理

当：

```text
fft_length = 2N
```

FFT 输出范围为：

```text
bin 0 ~ bin 2N-1
```

但真实投影仪坐标范围只有：

```text
0 ~ N-1
```

所以：

```text
bin N ~ bin 2N-1
```

不能作为有效投影仪坐标输出。

从 DFT 角度看，这一段可以理解为负频率区域：

```text
bin 0        -> 0
bin 1        -> +1
bin 2        -> +2
...
bin N-1      -> +(N-1)

bin N        -> 边界频率
bin N+1      -> -(N-1)
bin N+2      -> -(N-2)
...
bin 2N-1     -> -1
```

例如：

```text
N = 1920
fft_length = 3840
```

可以理解为：

```text
bin 0       -> 坐标 0，有效
bin 1       -> 坐标 1，有效
...
bin 1919    -> 坐标 1919，有效

bin 1920    -> 边界频率，不是有效投影仪坐标
bin 1921    -> 负方向坐标 -1919
bin 1922    -> 负方向坐标 -1918
...
bin 3839    -> 负方向坐标 -1
```

所以：

```text
bin >= proj_dim 的部分不应该作为投影仪坐标。
```

这些 bin 可以在调试时观察完整频谱，但不应该参与最终坐标输出。

---

## 12. 为什么不能在全部 `2 * proj_dim` 个 bin 中直接 `argmax`

如果直接写：

```python
C = np.fft.fft(D_block, n=2 * proj_dim, axis=0)
C_magnitude = np.abs(C)

freq_index = np.argmax(C_magnitude, axis=0)
```

那么 `argmax` 会在全部：

```text
0 ~ 2 * proj_dim - 1
```

中寻找最大值。

这可能得到：

```text
freq_index >= proj_dim
```

例如：

```text
freq_index = 2500
```

如果投影仪宽度只有 1920，那么这个结果显然不是有效投影仪坐标。

这些超出范围的峰可能来自：

```text
负频率区域
噪声
反光
多路径反射
遮挡边界
异常像素
```

所以使用：

```python
fft_length = 2 * proj_dim
```

时，应该只在前 `proj_dim` 个 bin 中找峰值。

---

## 13. 正确的有效频谱截取方式

推荐写法：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

valid_spectrum = C_magnitude[:proj_dim, :, :]

freq_index = np.argmax(valid_spectrum, axis=0)
peak = np.max(valid_spectrum, axis=0) / fft_length

disparity = freq_index
```

这里：

```python
valid_spectrum = C_magnitude[:proj_dim, :, :]
```

表示：

```text
只保留 bin 0 ~ bin proj_dim-1
```

即：

```text
只在真实投影仪坐标范围内找峰值
```

这样可以保证：

```text
freq_index ∈ [0, proj_dim - 1]
```

最终输出的 `disparity` 也一定在合法投影仪坐标范围内。

---

## 14. 一个简单例子：`proj_dim = 8`

假设：

```text
proj_dim = N = 8
frequency_step = 0.5
fft_length = 16
```

FFT 输出 bin 为：

```text
0, 1, 2, ..., 15
```

投影仪有效坐标为：

```text
0, 1, 2, 3, 4, 5, 6, 7
```

对应关系为：

```text
bin 0  -> 坐标 0，有效
bin 1  -> 坐标 1，有效
bin 2  -> 坐标 2，有效
bin 3  -> 坐标 3，有效
bin 4  -> 坐标 4，有效
bin 5  -> 坐标 5，有效
bin 6  -> 坐标 6，有效
bin 7  -> 坐标 7，有效

bin 8  -> 边界频率，不是有效投影仪坐标
bin 9  -> 负方向坐标 -7
bin 10 -> 负方向坐标 -6
bin 11 -> 负方向坐标 -5
bin 12 -> 负方向坐标 -4
bin 13 -> 负方向坐标 -3
bin 14 -> 负方向坐标 -2
bin 15 -> 负方向坐标 -1
```

如果某个像素真实对应：

```text
u = 5
```

理论上峰值应该出现在：

```text
bin 5
```

最终输出：

```text
disparity = 5
```

而不是去后半段寻找坐标。

---

## 15. `NUM_FREQUENCIES` 应该如何设置

原来频率序列为：

```text
0, 1, 2, ..., 150
```

频率数量为：

```text
151
```

如果改成：

```text
0, 0.5, 1, 1.5, ..., 150
```

则频率数量为：

```text
150 / 0.5 + 1 = 301
```

所以应该设置：

```python
NUM_FREQUENCIES = 301
FREQUENCY_STEP = 0.5
```

每个频率仍然有 4 张相移图像：

```text
0°, 90°, 180°, 270°
```

因此总图像数为：

```text
301 * 4 = 1204
```

图像组织方式为：

```text
0.png, 1.png, 2.png, 3.png       -> 频率 0
4.png, 5.png, 6.png, 7.png       -> 频率 0.5
8.png, 9.png, 10.png, 11.png     -> 频率 1.0
12.png, 13.png, 14.png, 15.png   -> 频率 1.5
...
```

频率编号仍然是：

```text
k = 0, 1, 2, ..., 300
```

但真实频率是：

```text
f_k = 0.5 * k
```

---

## 16. 推荐修改方案一：专门支持 0.5 步长

如果你目前只打算支持：

```text
frequency_step = 1.0
frequency_step = 0.5
```

可以采用下面的逻辑：

```python
if frequency_step == 1.0:
    fft_length = proj_dim
    valid_length = proj_dim
    coord_scale = 1.0

elif frequency_step == 0.5:
    fft_length = 2 * proj_dim
    valid_length = proj_dim
    coord_scale = 1.0

else:
    raise ValueError("当前只支持 frequency_step = 1.0 或 0.5")
```

然后解码时：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

C_valid = C_magnitude[:valid_length, :, :]

freq_index = np.argmax(C_valid, axis=0)
peak = np.max(C_valid, axis=0) / fft_length

disparity = freq_index * coord_scale
```

对于 `frequency_step = 0.5`：

```text
fft_length = 2 * proj_dim
valid_length = proj_dim
coord_scale = 1.0
```

所以：

```text
disparity = freq_index
```

仍然成立。

---

## 17. 推荐修改方案二：更通用的写法

更推荐使用通用公式：

```python
fft_length = int(round(proj_dim / frequency_step))
valid_length = proj_dim
coord_scale = proj_dim / (fft_length * frequency_step)
```

解码时：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

C_valid = C_magnitude[:valid_length, :, :]

freq_index = np.argmax(C_valid, axis=0)
peak = np.max(C_valid, axis=0) / fft_length

disparity = freq_index * coord_scale
```

代入不同步长：

### 当 `frequency_step = 1.0`

```text
fft_length = proj_dim
coord_scale = 1.0
```

### 当 `frequency_step = 0.5`

```text
fft_length = 2 * proj_dim
coord_scale = 1.0
```

### 当 `frequency_step = 0.25`

```text
fft_length = 4 * proj_dim
coord_scale = 1.0
```

这种方式可以自然扩展到更小的频率步长。

---

## 18. 建议修改 `_get_fft_params`

可以将 `_get_fft_params` 改成类似：

```python
def _get_fft_params(self, direction, num_frequencies):
    if direction == 'hor':
        proj_dim = self.projector_height
    elif direction == 'ver':
        proj_dim = self.projector_width
    else:
        raise ValueError("direction must be 'hor' or 'ver'")

    s = self.frequency_step

    fft_length = int(round(proj_dim / s))
    valid_length = proj_dim
    coord_scale = proj_dim / (fft_length * s)

    return fft_length, coord_scale, valid_length
```

这里新增了：

```python
valid_length
```

用于控制最终 `argmax` 的有效范围。

---

## 19. 建议修改解码部分

原来的逻辑可能是：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

freq_index = np.argmax(C_magnitude, axis=0)

if coord_scale == 1.0:
    disparity[ty:ey, tx:ex] = freq_index
else:
    disparity[ty:ey, tx:ex] = freq_index * coord_scale

peak[ty:ey, tx:ex] = np.max(C_magnitude, axis=0) / fft_length
```

建议改成：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

C_valid = C_magnitude[:valid_length, :, :]

freq_index = np.argmax(C_valid, axis=0)

if coord_scale == 1.0:
    disparity[ty:ey, tx:ex] = freq_index
else:
    disparity[ty:ey, tx:ex] = freq_index * coord_scale

peak[ty:ey, tx:ex] = np.max(C_valid, axis=0) / fft_length
```

核心变化是：

```text
FFT 可以算更多 bin，
但找峰值只在有效投影仪坐标范围内找。
```

---

## 20. 关于 `peak` 的归一化

代码中：

```python
peak = np.max(C_valid, axis=0) / fft_length
```

`peak` 表示最大频谱峰值强度的归一化结果，用作解码质量或置信度。

当：

```text
fft_length = 2 * proj_dim
```

时，除以 `fft_length` 可以保持与 FFT 长度相关的归一化。

但需要注意：

```text
改变 fft_length 后，peak 的数值尺度可能和旧版本略有不同。
```

如果后续代码中使用了固定阈值，例如：

```text
peak > 1.0
peak > 0.5
```

建议重新检查这些阈值是否仍然合适。

---

## 21. 三种方案对比

### 方案 A：不推荐，错误设置

实际投影频率：

```text
0, 0.5, 1, 1.5, ..., 150
```

但设置：

```python
FREQUENCY_STEP = 1.0
NUM_FREQUENCIES = 301
```

问题：

```text
程序会把半步频率序列误认为整步频率序列。
相位旋转速度解释错误。
解码坐标会出现比例偏差。
```

不推荐。

---

### 方案 B：不推荐，当前非标准分支

设置：

```python
FREQUENCY_STEP = 0.5
NUM_FREQUENCIES = 301

fft_length = num_frequencies
coord_scale = proj_dim / num_frequencies
```

问题：

```text
FFT bin 数量太少。
一个 bin 对应多个投影仪像素。
输出坐标明显量化。
点云容易出现层状台阶。
```

不推荐用于高精度重建。

---

### 方案 C：推荐

设置：

```python
FREQUENCY_STEP = 0.5
NUM_FREQUENCIES = 301

fft_length = 2 * proj_dim
valid_length = proj_dim
coord_scale = 1.0
```

优点：

```text
真实频率步长被正确建模。
FFT bin 编号继续直接对应投影仪像素坐标。
避免 coord_scale = proj_dim / num_frequencies 带来的粗量化。
最终坐标仍然落在 0 ~ proj_dim-1。
```

推荐使用。

---

## 22. 使用 `fft_length = 2 * proj_dim` 的代价

这种方案也有代价：

```text
FFT 输出长度变为原来的 2 倍。
计算量增加。
内存占用增加。
频谱数组第一维变大。
```

例如：

```text
projector_width = 1920
frequency_step = 0.5
垂直方向 fft_length = 3840
```

```text
projector_height = 1080
frequency_step = 0.5
水平方向 fft_length = 2160
```

如果图像尺寸大，建议：

```text
减小 block_size
或启用 low_memory_mode
```

以避免内存压力过大。

---

## 23. 频率步长变小是否一定提高几何精度

把频率从：

```text
0, 1, 2, ..., 150
```

改为：

```text
0, 0.5, 1, 1.5, ..., 150
```

本质上是在相同最高频率范围内采样得更密。

它可能带来：

```text
频率轴采样更多；
抗噪声能力可能更好；
频谱峰值可能更稳定；
相位旋转轨迹采样更细。
```

但它不意味着：

```text
投影仪坐标范围变成两倍；
输出坐标可以超过投影仪分辨率；
几何分辨率一定翻倍。
```

最终投影仪坐标仍然是：

```text
0 ~ proj_dim - 1
```

`fft_length = 2 * proj_dim` 的目的不是扩大坐标范围，而是适配：

```text
frequency_step = 0.5 导致相位旋转速度减半
```

从而让：

```text
FFT bin 重新和投影仪坐标对齐。
```

---

## 24. 最终推荐配置

如果你将频率步长改为 0.5，推荐设置为：

```python
FREQUENCY_STEP = 0.5
NUM_FREQUENCIES = 301
```

推荐 FFT 参数为：

```python
fft_length = int(round(proj_dim / FREQUENCY_STEP))
valid_length = proj_dim
coord_scale = proj_dim / (fft_length * FREQUENCY_STEP)
```

对于 `FREQUENCY_STEP = 0.5`，即：

```python
fft_length = 2 * proj_dim
valid_length = proj_dim
coord_scale = 1.0
```

推荐解码方式：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

C_valid = C_magnitude[:valid_length, :, :]

freq_index = np.argmax(C_valid, axis=0)
peak = np.max(C_valid, axis=0) / fft_length

disparity = freq_index * coord_scale
```

在 `frequency_step = 0.5` 情况下：

```python
disparity = freq_index
```

但前提是：

```text
freq_index 是在前 proj_dim 个有效 bin 中找出来的。
```

---


---

## 25. 补充疑问一：`fft_length = 2 * proj_dim` 后，bin 800 是否对应真实坐标 400？

这个疑问非常关键。容易出现的误解是：

```text
frequency_step = 0.5
fft_length = 2 * proj_dim

所以是不是：
FFT 输出 bin 800 -> 真实投影仪坐标 400
FFT 输出 bin 3800 -> 真实投影仪坐标 1900
```

这个理解是不正确的。

在推荐方案中：

```python
frequency_step = 0.5
fft_length = 2 * proj_dim
coord_scale = 1.0
```

目的正是让：

```text
FFT bin 编号 = 投影仪真实坐标
```

也就是说：

```text
bin 400  -> 投影仪坐标 400
bin 800  -> 投影仪坐标 800
bin 1900 -> 投影仪坐标 1900
```

而不是：

```text
bin 800  -> 投影仪坐标 400
bin 3800 -> 投影仪坐标 1900
```

---

### 25.1 为什么不是除以 2？

设：

```text
N = proj_dim
s = frequency_step
L = fft_length
u = 真实投影仪坐标
q = FFT 峰值 bin
```

一般关系是：

```text
q ≈ L * s * u / N
```

反过来：

```text
u ≈ q * N / (L * s)
```

也就是：

```python
coord_scale = N / (L * s)
disparity = q * coord_scale
```

当：

```text
s = 0.5
L = 2N
```

代入：

```text
u = q * N / (2N * 0.5)
```

因为：

```text
2N * 0.5 = N
```

所以：

```text
u = q
```

因此在推荐方案中：

```text
投影仪真实坐标 = FFT bin 编号
```

所以：

```text
bin 800 对应坐标 800
bin 1900 对应坐标 1900
```

而不是除以 2。

---

### 25.2 用具体数字验证

假设：

```text
proj_dim = N = 2000
frequency_step = 0.5
fft_length = 2N = 4000
```

如果真实投影仪坐标是：

```text
u = 400
```

则峰值 bin 为：

```text
q = L * s * u / N
q = 4000 * 0.5 * 400 / 2000
q = 400
```

所以：

```text
真实坐标 400 -> FFT 峰值 bin 400
```

不是：

```text
真实坐标 400 -> FFT 峰值 bin 800
```

再看真实坐标：

```text
u = 1900
```

则：

```text
q = 4000 * 0.5 * 1900 / 2000
q = 1900
```

所以：

```text
真实坐标 1900 -> FFT 峰值 bin 1900
```

不是：

```text
真实坐标 1900 -> FFT 峰值 bin 3800
```

因此，如果只保留：

```text
bin 0 ~ bin proj_dim-1
```

不会把真实坐标 1900 截掉。

因为真实坐标 1900 本来就对应：

```text
bin 1900
```

而不是：

```text
bin 3800
```

---

### 25.3 bin 3800 到底代表什么？

如果：

```text
proj_dim = 2000
fft_length = 4000
```

那么 FFT 输出为：

```text
bin 0 ~ bin 3999
```

其中：

```text
bin 0 ~ bin 1999
```

对应合法的正向投影仪坐标：

```text
0 ~ 1999
```

而：

```text
bin 2000 ~ bin 3999
```

属于 DFT 的负频率区域。

例如：

```text
bin 3999 -> -1
bin 3998 -> -2
bin 3800 -> -200
```

所以：

```text
bin 3800 不是投影仪坐标 1900
```

它更接近表示负方向频率，例如：

```text
-200
```

如果某个像素在 bin 3800 附近出现强峰，不能把它解释成：

```text
投影仪坐标 1900
```

而应考虑：

```text
相位符号是否反了；
投影条纹方向是否和代码假设相反；
四步相移构造中的虚部符号是否需要反过来；
图像序列是否有问题；
该像素是否存在反光、遮挡或多路径干扰。
```

---

### 25.4 如果大量峰值出现在后半段，可能是什么原因？

如果在实际测试中发现很多正常像素的最大峰都落在：

```text
bin proj_dim ~ bin 2*proj_dim-1
```

那么通常不是因为这些 bin 对应更大的投影仪坐标，而可能是相位方向或符号约定反了。

例如当前代码可能使用：

```python
D = (I0 - I2) + 1j * (I3 - I1)
```

但如果实际投影条纹相位方向与代码假设相反，则峰值可能跑到负频率一侧。

此时可能需要检查是否应改成：

```python
D = (I0 - I2) + 1j * (I1 - I3)
```

或者检查投影条纹生成方向是否反了。

需要注意：

```text
这属于“相位方向/符号约定问题”，
不是 coord_scale 或 frequency_step 的比例映射问题。
```

---

## 26. 补充疑问二：既然真实峰应该在前 `proj_dim` 个 bin，是否可以不裁切，直接全局 `argmax`？

从理想数学模型来说，如果：

```text
frequency_step = 0.5
fft_length = 2 * proj_dim
```

并且图像无噪声、无反光、无遮挡、相位方向完全正确，那么真实峰确实应该落在：

```text
bin u_p
```

其中：

```text
u_p ∈ [0, proj_dim - 1]
```

所以理想情况下：

```text
前 proj_dim 个 bin 中的真实峰应该最大
后半段 bin 不应该超过真实峰
```

因此，从纯理论角度看，直接在完整频谱中 `argmax` 也可能得到正确结果。

但是在实际工程中，仍然建议裁切。

---

### 26.1 为什么工程上仍然建议裁切？

实际采集数据中，频谱不会只有一个点有值。

即使真实峰在：

```text
bin u_p
```

其他 bin 也可能有响应。

原因包括：

```text
采样频率数量有限；
图像噪声；
物体反光；
遮挡边界；
多路径反射；
局部低信噪比；
相位方向与代码约定不一致；
图像顺序错误。
```

因此实际频谱可能出现：

```text
真实峰；
旁瓣；
噪声峰；
异常峰；
后半段负频率峰。
```

对于大多数正常像素，真实峰通常最大。

但对于异常像素，后半段的某个峰有可能超过前半段真实峰。

如果直接对完整频谱：

```text
0 ~ 2*proj_dim - 1
```

做：

```python
freq_index = np.argmax(C_magnitude, axis=0)
```

就可能得到：

```text
freq_index >= proj_dim
```

这会输出非法投影仪坐标。

---

### 26.2 裁切不是丢掉真实坐标，而是加入物理约束

投影仪坐标有明确物理范围：

```text
0 ~ proj_dim - 1
```

例如投影仪宽度为 1920，则真实列坐标只能是：

```text
0 ~ 1919
```

即使 FFT 输出有：

```text
0 ~ 3839
```

后半段：

```text
1920 ~ 3839
```

也不属于合法投影仪正坐标。

因此裁切的含义不是：

```text
丢掉可能的真实坐标
```

而是：

```text
只在物理上可能的投影仪坐标范围内寻找最优解
```

推荐写法是：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

C_valid = C_magnitude[:proj_dim, :, :]

freq_index = np.argmax(C_valid, axis=0)
peak = np.max(C_valid, axis=0) / fft_length

disparity = freq_index
```

这样可以保证：

```text
freq_index ∈ [0, proj_dim - 1]
```

最终不会输出超过投影仪分辨率的坐标。

---

### 26.3 如果不裁切可能出现的问题

假设：

```text
proj_dim = 2000
fft_length = 4000
```

真实坐标为：

```text
u_p = 1900
```

理想峰应在：

```text
bin 1900
```

但某个像素因为噪声或反光，在后半段出现强峰：

```text
bin 3800
```

如果直接全局取最大值：

```python
freq_index = np.argmax(C_magnitude, axis=0)
```

可能得到：

```text
freq_index = 3800
```

这显然不是合法投影仪坐标，因为真实坐标最大只能到：

```text
1999
```

如果后续直接保存：

```python
disparity = 3800
```

三维重建一定会出现错误。

如果裁切：

```python
C_valid = C_magnitude[:2000]
freq_index = np.argmax(C_valid)
```

算法就只能在合法范围：

```text
0 ~ 1999
```

中选择最佳峰值。

这更符合物理约束。

---

### 26.4 什么时候可以不裁切？

只有在你非常确定以下条件都满足时，才可以考虑不裁切：

```text
相位方向完全正确；
图像顺序完全正确；
数据噪声很小；
没有明显反光和遮挡；
所有正常像素的峰值都稳定落在 0 ~ proj_dim-1。
```

但即使这些条件满足，不裁切也没有明显好处。

因为后半段本来就不是合法坐标范围。

所以从工程实现上，仍然推荐裁切。

---

### 26.5 对“最后输出的 bin 只会有一个”的修正

更准确地说：

```text
FFT 会输出很多个 bin；
argmax 最后只选出一个 bin。
```

也就是说，对一个像素来说，FFT 结果是：

```text
C_magnitude[0]
C_magnitude[1]
C_magnitude[2]
...
C_magnitude[2N-1]
```

这些值都存在。

理想情况下：

```text
C_magnitude[u_p]
```

最大。

但不是说其他 bin 全部为 0。

实际图像中由于噪声、有限采样和反射问题，其他 bin 也会有响应。

所以应该理解为：

```text
每个像素最终只输出一个最佳 bin，
但 FFT 频谱本身包含很多候选 bin。
```

这也是为什么需要用有效范围约束来防止异常峰被选中。

---

### 26.6 这一部分的最终结论

对于：

```text
frequency_step = 0.5
fft_length = 2 * proj_dim
```

理论理想情况下，真实峰应出现在：

```text
bin u_p
```

且：

```text
u_p ∈ [0, proj_dim - 1]
```

所以合法真实峰不会被裁切掉。

但是实际工程中，后半段可能因噪声、反光、相位符号反向等原因出现强峰。

因此推荐始终使用：

```python
C_valid = C_magnitude[:proj_dim]
freq_index = np.argmax(C_valid, axis=0)
```

而不是：

```python
freq_index = np.argmax(C_magnitude, axis=0)
```

简洁结论：

```text
理想情况下可以不裁切；
实际工程中强烈建议裁切；
裁切不会丢掉合法真实坐标，只会排除物理上无效的 bin。
```



## 27. 最终总结

### `coord_scale` 是什么？

```text
coord_scale 是 FFT bin 编号到投影仪像素坐标的换算比例。
```

公式：

```python
disparity = freq_index * coord_scale
```

通用形式：

```python
coord_scale = proj_dim / (fft_length * frequency_step)
```

---

### 为什么 `frequency_step = 0.5` 时不能继续设置为 1？

因为真实频率序列变成：

```text
f_k = 0.5 * k
```

相位随 `k` 的旋转速度变成原来的一半。

如果仍然设置：

```python
frequency_step = 1.0
```

程序会误解频率序列，导致坐标比例错误。

---

### 为什么推荐 `fft_length = 2 * proj_dim`？

因为当：

```text
frequency_step = 0.5
```

理想信号为：

```text
D[k] ≈ A * exp(j * 2π * k * u / (2 * proj_dim))
```

为了让 FFT 的第 `u` 个 bin 正好对应投影仪坐标 `u`，需要：

```python
fft_length = 2 * proj_dim
```

---

### 大于 `proj_dim` 的 bin 怎么处理？

当：

```python
fft_length = 2 * proj_dim
```

FFT 输出：

```text
0 ~ 2 * proj_dim - 1
```

但有效投影仪坐标只有：

```text
0 ~ proj_dim - 1
```

所以应当：

```python
C_valid = C_magnitude[:proj_dim]
freq_index = np.argmax(C_valid, axis=0)
```

也就是：

```text
只在前 proj_dim 个 bin 内找峰值。
```

---

### 最终推荐方案

```python
FREQUENCY_STEP = 0.5
NUM_FREQUENCIES = 301

fft_length = 2 * proj_dim
valid_length = proj_dim
coord_scale = 1.0
```

并且解码时：

```python
C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)

C_valid = C_magnitude[:proj_dim, :, :]

freq_index = np.argmax(C_valid, axis=0)
peak = np.max(C_valid, axis=0) / fft_length
disparity = freq_index
```

这样可以保证：

```text
真实频率步长被正确使用；
FFT bin 继续对应投影仪像素坐标；
最终坐标不会超过投影仪有效范围；
避免 coord_scale = proj_dim / num_frequencies 造成明显量化阶梯。
```
