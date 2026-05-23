# FSM / FSA 频移法统一公式详解

> 本文围绕频移法（FSM / FSA）中的统一公式展开，系统解释 `frequency_step`、`fft_length`、`proj_dim`、`num_frequencies`、FFT bin 编号 `q` 与最终投影仪坐标 `u_p` 之间的关系。  
> 重点回答：为什么物理公式分母是 `N`，为什么 FFT 长度可以不是采集频率数量 `M`，为什么推荐 `fft_length = round(proj_dim / frequency_step)`，以及为什么旧方案 `fft_length = M` 会造成量化阶梯和点云空白。

---

## 0. 先给总览结论

频移法中最容易混淆的三个量是：

```text
M = num_frequencies    实际采集 / 投影的频率数量
N = proj_dim           投影仪在当前解码方向上的分辨率
L = fft_length         FFT 用来搜索频谱峰值的候选 bin 数量
```

它们分别属于三个不同层面：

| 符号 | 含义 | 属于哪个层面 | 决定什么 |
|---|---|---|---|
| `M` | 采集了多少个频率样本 | 数据采集层 | 有多少个真实观测值参与求和 |
| `N` | 投影仪坐标轴长度 | 物理坐标层 | 坐标归一化周期，决定真实相位公式的分母 |
| `L` | FFT 长度 | 频谱搜索层 | 有多少个候选 `q` 用来匹配相位旋转速度 |

统一公式是：

```text
真实信号：
D[k] ≈ A · exp(j · 2π · s · k · u_p / N)

FFT 第 q 个 bin 测试：
exp(-j · 2π · q · k / L)

匹配条件：
q / L = s · u_p / N

因此：
u_p = q · N / (L · s)

coord_scale = N / (L · s)
disparity = q · coord_scale
```

如果希望 `q` 本身就等于投影仪坐标，也就是：

```text
disparity = q
```

就要让：

```text
coord_scale = 1
```

即：

```text
N / (L · s) = 1
```

推出：

```text
L = N / s
```

这就是当前推荐策略：

```python
fft_length = round(proj_dim / frequency_step)
valid_length = proj_dim
coord_scale ≈ 1.0
```

---

## 1. 从频移法的目标开始理解统一公式

结构光三维重建的核心任务是建立相机像素与投影仪像素之间的对应关系：

```text
相机像素 (x_c, y_c)  →  投影仪像素 (x_p, y_p)
```

对于一个固定的相机像素来说，频移法不是只看一张图，也不是只看一个频率，而是看这个像素在多个投影频率下形成的一串复数响应：

```text
D[0], D[1], D[2], ..., D[M-1]
```

其中：

```text
k = 0, 1, 2, ..., M-1
```

表示第 `k` 个实际采集的频率样本。

如果这个相机像素真实对应投影仪上的某个坐标 `u_p`，那么随着投影频率递增，复数 `D[k]` 的相位会不断旋转。旋转得越快，说明这个点对应的投影仪坐标越大；旋转得越慢，说明对应的投影仪坐标越小。

所以频移法的核心可以概括为一句话：

> **把投影仪坐标 `u_p` 编码成频率轴复信号 `D[k]` 的相位旋转速度，然后用 FFT 找出这个旋转速度对应的 bin。**

---

## 2. 四步相移如何得到复数信号 D[k]

对某一个投影频率，投影四张相移图：

```text
0°
90°
180°
270°
```

相机在同一个像素位置采到四个灰度值：

```text
I0, I1, I2, I3
```

理想情况下可以写成：

```text
I0 = B + R · cos(θ)
I1 = B - R · sin(θ)
I2 = B - R · cos(θ)
I3 = B + R · sin(θ)
```

其中：

```text
B = 直流亮度项，例如环境光、背景灰度、暗电流
R = 反射强度或调制度
θ = 当前频率下的条纹相位
```

通过差分：

```text
Dc = I0 - I2 = 2R · cos(θ)
Ds = I3 - I1 = 2R · sin(θ)
```

再构造复数：

```text
D = Dc + j · Ds
```

得到：

```text
D = 2R · (cos(θ) + j sin(θ))
D = 2R · exp(jθ)
```

所以四步相移的作用是：

1. 把受环境光影响的灰度值变成相对稳定的差分信号；
2. 抵消直流项 `B`；
3. 把相位信息放进复数 `D` 的角度中；
4. 为后续沿频率轴做 FFT 做准备。

对多个频率重复这个过程，就得到：

```text
D[0], D[1], D[2], ..., D[M-1]
```

这就是频移法中每个相机像素的“频率轴复信号”。

---

## 3. 为什么真实信号公式的分母是 N

现在考虑一个固定相机像素，它真实对应投影仪坐标：

```text
u_p
```

如果当前方向是垂直条纹解码，那么：

```text
u_p = x_p
N = projector_width
```

如果当前方向是水平条纹解码，那么：

```text
u_p = y_p
N = projector_height
```

投影条纹的相位本质上依赖于投影仪坐标在当前方向上的归一化位置：

```text
u_p / N
```

也就是说，投影仪坐标轴从 `0` 到 `N-1`，完整坐标周期由 `N` 决定。

当第 `k` 个频率的频率值为：

```text
f_k = s · k
```

其中：

```text
s = frequency_step
```

那么该相机像素接收到的复数相位可以写成：

```text
θ[k] = 2π · f_k · u_p / N
```

代入 `f_k = s · k`：

```text
θ[k] = 2π · s · k · u_p / N
```

因此复信号为：

```text
D[k] ≈ A · exp(j · 2π · s · k · u_p / N)
```

这里最重要的是：

> **分母 `N` 来自投影仪坐标的归一化周期，而不是来自你采集了多少个频率。**

所以：

- `M` 是采集数量；
- `N` 是坐标周期；
- 物理公式里分母应该是 `N`，不是 `M`。

---

## 4. frequency_step = s 的意义

`frequency_step` 表示相邻两个投影频率之间的间隔。

### 4.1 标准模式：s = 1.0

频率序列是：

```text
0, 1, 2, 3, ..., M-1
```

此时：

```text
f_k = k
```

真实信号为：

```text
D[k] ≈ A · exp(j · 2π · k · u_p / N)
```

这就是之前你看到的形式。

### 4.2 半步长模式：s = 0.5

频率序列是：

```text
0, 0.5, 1.0, 1.5, 2.0, ...
```

此时：

```text
f_k = 0.5 · k
```

真实信号为：

```text
D[k] ≈ A · exp(j · 2π · 0.5 · k · u_p / N)
```

同一个投影仪坐标 `u_p` 下，相位随着 `k` 的旋转速度只有标准模式的一半。因此如果还想让 FFT 的 bin 编号 `q` 直接等于投影仪坐标，就需要把 FFT 长度扩大到原来的 2 倍。

这就是：

```text
s = 0.5
L = N / s = 2N
```

的原因。

### 4.3 更小步长：s = 0.25

如果：

```text
s = 0.25
```

则：

```text
L = N / 0.25 = 4N
```

也就是说，频率步长越小，相位随 `k` 的变化越慢。为了让 FFT 的 bin 网格仍然和投影仪坐标对齐，就要使用更长的 FFT。

---

## 5. FFT 第 q 个 bin 到底在测试什么

对单个相机像素，程序做的是：

```python
C_full = np.fft.fft(D_block, n=fft_length, axis=0)
```

如果只看一个像素，可以写成：

```text
C[q] = Σ D[k] · exp(-j · 2π · q · k / L)
```

其中：

```text
L = fft_length
q = 0, 1, 2, ..., L-1
```

FFT 的第 `q` 个 bin 的基函数是：

```text
exp(-j · 2π · q · k / L)
```

它本质上是在问：

> “当前这串 `D[k]` 的相位旋转速度，是不是等于 `q / L`？”

也就是说，FFT bin 的归一化频率是：

```text
q / L
```

而真实信号的归一化旋转速度是：

```text
s · u_p / N
```

所以当二者相等时，FFT 会产生最大响应：

```text
q / L = s · u_p / N
```

这就是统一公式最核心的匹配条件。

---

## 6. 用相位斜率理解匹配条件

真实信号：

```text
D[k] ≈ A · exp(j · 2π · s · k · u_p / N)
```

可以把它看成：

```text
D[k] ≈ A · exp(j · 2π · α · k)
```

其中：

```text
α = s · u_p / N
```

`α` 就是频率轴上的相位斜率，也可以理解成“每增加一个频率样本，复数相位转多少圈”。

FFT 第 `q` 个 bin 测试的斜率是：

```text
β_q = q / L
```

当：

```text
β_q = α
```

即：

```text
q / L = s · u_p / N
```

该 bin 的基函数正好和真实信号的旋转速度相反，乘起来之后每一项的相位都会被抵消。

代入求和：

```text
C[q] = Σ A · exp(j · 2π · s · k · u_p / N) · exp(-j · 2π · q · k / L)
```

合并指数：

```text
C[q] = A · Σ exp(j · 2π · k · (s · u_p / N - q / L))
```

当：

```text
s · u_p / N - q / L = 0
```

也就是：

```text
q / L = s · u_p / N
```

每一项都变成：

```text
exp(j · 0) = 1
```

于是：

```text
C[q] = A + A + A + ... + A
```

所有项同相叠加，幅值最大。

这就是 FFT 能够通过峰值位置恢复投影仪坐标的数学原因。

---

## 7. 从匹配条件推出统一公式

从匹配条件出发：

```text
q / L = s · u_p / N
```

两边整理：

```text
q · N = L · s · u_p
```

所以：

```text
u_p = q · N / (L · s)
```

这就是从 FFT bin 编号 `q` 映射回投影仪坐标的统一公式。

因此在代码中可以写成：

```text
coord_scale = N / (L · s)
disparity = q · coord_scale
```

也就是：

```python
coord_scale = proj_dim / (fft_length * frequency_step)
disparity = freq_index * coord_scale
```

这个公式同时覆盖：

- 标准步长 `frequency_step = 1.0`
- 半步长 `frequency_step = 0.5`
- 更小步长 `frequency_step = 0.25`
- 旧方案 `fft_length = M`
- 推荐方案 `fft_length = N / s`

---

## 8. 为什么推荐让 coord_scale = 1

如果希望最终结果最直观，最好让：

```text
disparity = q
```

也就是 FFT 找到的 bin 编号本身就是投影仪坐标。

这意味着：

```text
coord_scale = 1
```

根据：

```text
coord_scale = N / (L · s)
```

令其等于 1：

```text
N / (L · s) = 1
```

推出：

```text
L = N / s
```

所以推荐：

```python
fft_length = round(proj_dim / frequency_step)
```

这样做的好处是：

1. `q` 的物理意义非常清楚：`q` 就是投影仪坐标；
2. 不需要再做额外坐标缩放；
3. 输出结果的最小量化单位是 1 个投影仪像素；
4. 可以避免 `fft_length = M` 造成的大步长量化；
5. 对 `frequency_step = 0.5` 这类非整数频率步长也自然成立。

---

## 9. 三种典型情况对比

### 9.1 情况 A：s = 1.0，L = N

这是标准推荐模式。

```text
s = 1.0
L = N
```

代入统一公式：

```text
coord_scale = N / (N · 1.0) = 1
```

所以：

```text
u_p = q
```

此时：

```python
fft_length = proj_dim
valid_length = proj_dim
coord_scale = 1.0
```

如果解码垂直条纹：

```python
fft_length = projector_width
valid_length = projector_width
```

如果解码水平条纹：

```python
fft_length = projector_height
valid_length = projector_height
```

此时 FFT 的第 `q` 个 bin 就是在测试：

```text
当前相机像素是否对应投影仪坐标 q
```

---

### 9.2 情况 B：s = 0.5，L = 2N

这是半步长频率序列的推荐模式。

```text
s = 0.5
L = N / 0.5 = 2N
```

代入统一公式：

```text
coord_scale = N / (2N · 0.5) = N / N = 1
```

所以仍然有：

```text
u_p = q
```

这说明即使频率序列是：

```text
0, 0.5, 1.0, 1.5, ...
```

只要 FFT 长度设置为：

```text
2N
```

那么 `q` 仍然可以直接当作投影仪坐标。

但是注意：因为 FFT 长度变成了 `2N`，而投影仪坐标只在：

```text
0, 1, 2, ..., N-1
```

范围内，所以只应该在前 `N` 个 bin 里找峰：

```python
C_valid = C_full[:proj_dim]
freq_index = np.argmax(np.abs(C_valid), axis=0)
```

也就是：

```python
valid_length = proj_dim
```

后面的 bin 不对应有效投影仪坐标，可能属于负频率区、周期重复区或无效搜索区，不能拿来当坐标。

---

### 9.3 情况 C：s = 1.0，L = M

这是旧方案或者容易误用的方案。

```text
s = 1.0
L = M
```

代入统一公式：

```text
coord_scale = N / M
```

所以：

```text
u_p = q · N / M
```

这时 `q` 不再是投影仪坐标，而是一个压缩后的频谱索引。

如果：

```text
N = 1920
M = 151
```

则：

```text
coord_scale = 1920 / 151 ≈ 12.72
```

输出坐标只能是：

```text
0
12.72
25.43
38.15
...
```

虽然这些值是浮点数，但它们本质上只有 `M` 个离散等级。

因此多个真实投影仪坐标会被压到同一个 `q` 中，最后映射回去时形成大台阶。

---

## 10. M 到底在哪里起作用

很多困惑都来自这个问题：

> 如果真实公式分母是 `N`，那我只投影了 `M` 个频率，`N-M` 怎么办？

关键是：真实求和里只有 `M` 个观测值。

当执行：

```python
np.fft.fft(D_block, n=L, axis=0)
```

而 `D_block` 的第一维长度是 `M` 时，如果：

```text
L > M
```

NumPy 会在频率轴后面补零：

```text
D_padded[0]     = D[0]
D_padded[1]     = D[1]
...
D_padded[M-1]   = D[M-1]
D_padded[M]     = 0
D_padded[M+1]   = 0
...
D_padded[L-1]   = 0
```

因此形式上是：

```text
C[q] = Σ(k=0 到 L-1) D_padded[k] · exp(-j · 2π · q · k / L)
```

但因为 `k = M ... L-1` 的数据都是 0，所以等价于：

```text
C[q] = Σ(k=0 到 M-1) D[k] · exp(-j · 2π · q · k / L)
```

所以：

> **`M` 决定真正参与求和的有效数据数量。**

而：

> **`L` 决定 FFT 在多少个候选 bin 上评价这个求和结果。**

二者不是同一个概念。

---

## 11. 零填充的意义与限制

当 `L > M` 时，FFT 实际上做了零填充。零填充的意义是：

1. 让频谱在更密的 bin 网格上被采样；
2. 让 `q` 的候选位置更多；
3. 让 `q` 可以和投影仪坐标建立更细的对应关系；
4. 在 `L = N / s` 时，使 `q` 和 `u_p` 对齐。

但是零填充也有明显限制：

1. 它不会增加真实采集信息；
2. 它不会凭空提高信噪比；
3. 它不会让原本很差的峰变得真实可靠；
4. 它只是把已有的 `M` 个样本放到更细的 FFT 网格上分析。

因此：

```text
M 决定信息量和峰值可靠性；
L 决定坐标输出网格和 bin 对齐方式。
```

如果 `M` 太少，即使 `L` 设得很大，频谱峰也可能比较宽、容易受噪声影响、多峰干扰更明显。

如果 `M` 足够，且 `L` 与 `N/s` 对齐，则可以得到更稳定、更细致的坐标解码结果。

---

## 12. 为什么 fft_length = M 会造成量化阶梯

当：

```text
L = M
```

统一公式变成：

```text
u_p = q · N / (M · s)
```

若标准步长 `s = 1`：

```text
u_p = q · N / M
```

这意味着相邻两个 `q` 映射回投影仪坐标时相差：

```text
Δu = N / M
```

如果：

```text
N = 1920
M = 151
```

那么：

```text
Δu ≈ 12.72 像素
```

也就是说，解码输出只能落在间隔约 `12.72` 个投影仪像素的粗网格上。

### 12.1 一个直观例子

假设真实投影仪坐标有：

```text
u_p = 0, 1, 2, ..., 1919
```

但 FFT 只有：

```text
q = 0, 1, 2, ..., 150
```

那么许多不同的 `u_p` 都会被映射到同一个 `q` 附近。

例如：

```text
u_p = 0  ~ 6      可能都被判成 q = 0
u_p = 7  ~ 19     可能都被判成 q = 1
u_p = 20 ~ 32     可能都被判成 q = 2
...
```

再乘回：

```text
N / M
```

得到的坐标就只能是：

```text
0
12.72
25.43
38.15
...
```

所以表面上 `disparity` 是浮点数，实际上它是“浮点形式的粗离散台阶”。

### 12.2 对点云的影响

点云重建时，三角化需要连续、准确的相机-投影仪对应关系：

```text
相机像素 (x_c, y_c)  ↔  投影仪坐标 (x_p, y_p)
```

如果很多相邻相机像素被解码成同一个投影仪坐标，就会出现：

1. 局部对应关系被压扁；
2. 相邻点的投影仪坐标不连续变化；
3. 一些中间投影仪坐标永远不会被输出；
4. 三角化得到的点云会出现条带、断层、空白或层状结构。

这就是你之前看到的“多个投影仪行或列对应同一个值，导致点云之间有空白”的根本原因。

---

## 13. 为什么不能简单把分母改成 M

有时会误以为：

```text
我只投影了 M 个频率，所以 D[k] 的公式是不是应该写成：
D[k] ≈ A · exp(j · 2π · k · u_p / M)
```

这个写法通常是不准确的。

原因是：

```text
u_p
```

表示的是投影仪坐标，它的定义范围是：

```text
0, 1, 2, ..., N-1
```

它的归一化周期应该是：

```text
N
```

所以真实物理信号仍然应该写成：

```text
D[k] ≈ A · exp(j · 2π · s · k · u_p / N)
```

只有在你重新定义一个压缩坐标：

```text
u' = M · s · u_p / N
```

时，才可以形式上写成：

```text
D[k] ≈ A · exp(j · 2π · k · u' / M)
```

但此时：

```text
u'
```

已经不是投影仪坐标，而是被压缩到 FFT 长度 `M` 上的频谱坐标。

所以：

```text
q ≈ u'
```

并不代表：

```text
q ≈ u_p
```

真正的投影仪坐标还要映射回去：

```text
u_p = q · N / (M · s)
```

也就是仍然回到统一公式。

---

## 14. valid_length 为什么通常等于 proj_dim

推荐策略是：

```python
fft_length = round(proj_dim / frequency_step)
valid_length = proj_dim
```

原因是最终要恢复的是投影仪坐标：

```text
u_p = 0, 1, 2, ..., N-1
```

因此有效输出坐标范围就是：

```text
[0, N-1]
```

当：

```text
s = 1.0
L = N
```

所有 bin 都有效：

```text
q = 0, 1, ..., N-1
```

当：

```text
s = 0.5
L = 2N
```

虽然 FFT 有 `2N` 个 bin，但投影仪坐标仍然只有 `N` 个：

```text
q = 0, 1, ..., N-1
```

因此只在前 `N` 个 bin 中找峰：

```python
C_valid = C_full[:proj_dim]
freq_index = np.argmax(np.abs(C_valid), axis=0)
```

如果不限制有效范围，而是在全部 `2N` 个 bin 中找峰，就可能选到不对应真实投影仪坐标的 bin，造成错误解码。

---

## 15. 统一公式与代码实现的对应关系

核心参数计算可以理解为：

```python
def calc_fft_params(proj_dim, frequency_step):
    fft_length = int(round(proj_dim / frequency_step))
    valid_length = proj_dim
    coord_scale = proj_dim / (fft_length * frequency_step)
    return fft_length, valid_length, coord_scale
```

在标准模式中：

```python
proj_dim = 1920
frequency_step = 1.0

fft_length = round(1920 / 1.0) = 1920
valid_length = 1920
coord_scale = 1920 / (1920 * 1.0) = 1.0
```

在半步长模式中：

```python
proj_dim = 1920
frequency_step = 0.5

fft_length = round(1920 / 0.5) = 3840
valid_length = 1920
coord_scale = 1920 / (3840 * 0.5) = 1.0
```

解码主流程可以理解为：

```python
# images shape: (M * 4, Hc, Wc)

I0 = images[0::4].astype(np.float32)
I1 = images[1::4].astype(np.float32)
I2 = images[2::4].astype(np.float32)
I3 = images[3::4].astype(np.float32)

Dc = I0 - I2
Ds = I3 - I1
D  = Dc + 1j * Ds

C_full = np.fft.fft(D_block, n=fft_length, axis=0)
C_valid = C_full[:valid_length]

freq_index = np.argmax(np.abs(C_valid), axis=0)
disparity = freq_index * coord_scale
```

如果采用推荐参数，通常：

```text
coord_scale ≈ 1.0
```

所以：

```python
disparity ≈ freq_index
```

也就是：

```text
FFT 峰值 bin 编号 ≈ 投影仪坐标
```

---

## 16. 统一公式如何同时解释旧方案和新方案

### 16.1 旧方案

旧方案类似：

```python
fft_length = num_frequencies   # L = M
coord_scale = proj_dim / num_frequencies
disparity = freq_index * coord_scale
```

统一公式解释为：

```text
coord_scale = N / (M · s)
```

当 `s = 1` 时：

```text
coord_scale = N / M
```

问题是：

```text
q 只有 M 个离散值
```

所以输出坐标也只有 `M` 个等级，造成量化阶梯。

### 16.2 新方案

新方案是：

```python
fft_length = round(proj_dim / frequency_step)
valid_length = proj_dim
coord_scale = proj_dim / (fft_length * frequency_step)
```

当：

```text
L = N / s
```

有：

```text
coord_scale = 1
```

所以：

```text
disparity = q
```

这时 `q` 的候选数量与投影仪坐标范围对齐，输出步长变成 1 个投影仪像素，量化阶梯显著减弱或消除。

---

## 17. M、N、L 的关系图解

可以把整个过程理解为三层坐标系统：

```text
第一层：真实采集数据
k = 0, 1, ..., M-1

第二层：FFT 搜索网格
q = 0, 1, ..., L-1

第三层：投影仪物理坐标
u_p = 0, 1, ..., N-1
```

三者之间的关系是：

```text
真实信号由 k 采样：
D[k] ≈ A · exp(j · 2π · s · k · u_p / N)

FFT 在 q 网格上搜索：
C[q] = Σ D[k] · exp(-j · 2π · q · k / L)

q 与 u_p 的映射：
u_p = q · N / (L · s)
```

所以：

```text
M 只决定 Σ 里有多少项；
L 决定 q 的候选网格；
N 决定 u_p 的物理坐标周期。
```

---

## 18. 频率数量 M 是否越大越好

在统一公式中，`M` 不直接决定坐标缩放，但它会影响解码质量。

### 18.1 M 影响峰值的可靠性

求和是：

```text
C[q] = Σ(k=0 到 M-1) D[k] · exp(-j · 2π · q · k / L)
```

当 `M` 更大时：

1. 正确峰值位置处会有更多项同相叠加；
2. 错误位置处更容易相互抵消；
3. 频谱峰通常更尖锐；
4. 抗噪声能力更强；
5. 多峰混淆概率下降。

所以 `M` 虽然不应该拿来当物理公式分母，但它对频谱峰质量非常重要。

### 18.2 M 太小会怎样

如果 `M` 太小：

1. 频谱主峰会更宽；
2. 噪声可能造成峰值偏移；
3. 不同坐标的相位旋转速度不容易区分；
4. 多峰或误峰更容易出现；
5. 后续亚像素拟合也会不稳定。

因此推荐策略不是说 `M` 不重要，而是说：

```text
M 负责观测质量；
N 负责坐标周期；
L 负责搜索网格。
```

---

## 19. 关于亚像素精度

即使用推荐策略：

```text
L = N / s
```

此时：

```text
q ≈ u_p
```

但 `argmax` 得到的 `q` 仍然是整数 bin。

因此基础输出的最小步长是：

```text
1 个投影仪像素
```

如果希望得到亚像素投影仪坐标，需要在频谱峰附近做进一步估计，例如：

1. 抛物线拟合；
2. 质心法；
3. 相邻 bin 加权；
4. 相位斜率拟合；
5. 多峰候选结合极线约束筛选。

但无论是否做亚像素优化，统一公式仍然是坐标尺度的基础：

```text
u_p = q · N / (L · s)
```

如果亚像素峰位置记为：

```text
q_sub
```

则：

```text
u_p_sub = q_sub · N / (L · s)
```

在推荐策略下：

```text
u_p_sub ≈ q_sub
```

---

## 20. 对 frequency_step > 1 的补充说明

统一公式数学上可以写成：

```text
L = N / s
```

但工程上最常用、最安全的情况是：

```text
0 < s ≤ 1
```

例如：

```text
s = 1.0
s = 0.5
s = 0.25
```

如果：

```text
s > 1
```

表示频率序列跳着采样，例如：

```text
0, 2, 4, 6, ...
```

这会让相邻投影坐标之间的相位旋转更快，更容易出现周期混叠和坐标范围压缩问题。此时简单使用：

```text
fft_length = round(N / s)
```

可能无法覆盖完整的投影仪坐标范围，需要额外设计频率序列和解码策略。

因此，在当前频移法解码语境中，更推荐把 `frequency_step` 理解为：

```text
标准频率步长或更细的子步长
```

也就是：

```text
s = 1.0, 0.5, 0.25, ...
```

---

## 21. 一组完整数值例子

假设垂直方向解码投影仪列坐标：

```text
projector_width = 1920
N = 1920
M = 301
s = 0.5
```

### 21.1 推荐方案

```text
L = N / s = 1920 / 0.5 = 3840
```

统一公式：

```text
coord_scale = N / (L · s)
coord_scale = 1920 / (3840 · 0.5)
coord_scale = 1
```

所以：

```text
u_p = q
```

有效搜索范围：

```text
q = 0, 1, 2, ..., 1919
```

如果某个像素真实对应：

```text
u_p = 1000
```

则真实相位斜率为：

```text
s · u_p / N = 0.5 · 1000 / 1920 ≈ 0.2604167
```

FFT bin `q = 1000` 的测试斜率为：

```text
q / L = 1000 / 3840 ≈ 0.2604167
```

二者一致，所以 `q = 1000` 处峰值最大。

### 21.2 旧方案

如果错误地使用：

```text
L = M = 301
```

则：

```text
coord_scale = N / (M · s)
coord_scale = 1920 / (301 · 0.5)
coord_scale ≈ 12.76
```

如果某个像素真实对应：

```text
u_p = 1000
```

匹配条件：

```text
q / 301 = 0.5 · 1000 / 1920
```

得到：

```text
q ≈ 78.39
```

`argmax` 只能得到整数：

```text
q = 78 或 79
```

映射回去：

```text
u_est = 78 · 12.76 ≈ 995.3
u_est = 79 · 12.76 ≈ 1008.0
```

这说明输出坐标只能在大约 `12.76` 像素的间隔上跳变，不能逐像素连续变化。

这就是量化阶梯的来源。

---

## 22. 从统一公式看代码中的几个关键参数

### 22.1 `proj_dim`

`proj_dim` 是当前解码方向上的投影仪分辨率：

```python
if direction == "ver":
    proj_dim = projector_width
elif direction == "hor":
    proj_dim = projector_height
```

它对应公式中的：

```text
N
```

### 22.2 `frequency_step`

`frequency_step` 对应公式中的：

```text
s
```

它决定实际频率序列：

```text
f_k = s · k
```

### 22.3 `fft_length`

`fft_length` 对应公式中的：

```text
L
```

推荐：

```python
fft_length = int(round(proj_dim / frequency_step))
```

### 22.4 `valid_length`

`valid_length` 表示有效搜索范围：

```python
valid_length = proj_dim
```

因为最终坐标只应该在：

```text
0 ~ proj_dim - 1
```

之间。

### 22.5 `coord_scale`

`coord_scale` 是从 FFT bin 到投影仪坐标的比例：

```python
coord_scale = proj_dim / (fft_length * frequency_step)
```

推荐设置下它约等于：

```text
1.0
```

---

## 23. 常见误区总结

### 误区 1：把 M 当成物理公式分母

错误理解：

```text
我投影了 M 个频率，所以公式分母应该是 M。
```

正确理解：

```text
公式分母是 N，因为 u_p 是投影仪坐标，需要用投影仪分辨率归一化。
```

---

### 误区 2：认为 fft_length 必须等于 M

错误理解：

```text
D[k] 只有 M 个点，所以 FFT 只能做 M 点。
```

正确理解：

```text
D[k] 有 M 个真实观测点，但 FFT 可以通过 n=L 进行零填充，在 L 个候选 bin 上搜索峰值。
```

---

### 误区 3：认为零填充产生了新的频率数据

错误理解：

```text
L > M 时，多出来的 N-M 个点是新的频率响应。
```

正确理解：

```text
多出来的是 0，不是新采集数据。真正有值的仍然只有 M 个频率样本。
```

---

### 误区 4：认为 q 天然就是投影仪坐标

错误理解：

```text
argmax 得到的 q 一定就是投影仪坐标。
```

正确理解：

```text
只有当 coord_scale = N / (L · s) = 1 时，q 才可以直接等于投影仪坐标。
```

---

### 误区 5：认为 `fft_length=M` 后乘 `N/M` 就能完全恢复连续坐标

错误理解：

```text
q 乘上 N/M 以后变成浮点数，所以坐标就连续了。
```

正确理解：

```text
q 本身只有 M 个整数等级，乘上 N/M 后只是变成浮点形式的 M 个离散台阶。
```

---

## 24. 工程推荐写法

建议把 FFT 参数统一封装，避免不同分支出现不一致逻辑：

```python
def _calc_fft_params(proj_dim: int, frequency_step: float):
    if frequency_step <= 0:
        raise ValueError("frequency_step must be positive")

    fft_length = int(round(proj_dim / frequency_step))
    valid_length = proj_dim
    coord_scale = proj_dim / (fft_length * frequency_step)

    return fft_length, valid_length, coord_scale
```

解码时：

```python
fft_length, valid_length, coord_scale = _calc_fft_params(
    proj_dim=proj_dim,
    frequency_step=frequency_step
)

C_full = np.fft.fft(D_block, n=fft_length, axis=0)
C_valid = C_full[:valid_length]

freq_index = np.argmax(np.abs(C_valid), axis=0)
disparity = freq_index.astype(np.float32) * coord_scale
```

如果使用推荐参数，通常可以检查：

```python
print(coord_scale)
```

应该非常接近：

```text
1.0
```

如果不是 1，则说明当前 `fft_length`、`proj_dim`、`frequency_step` 之间没有完全对齐，需要明确知道自己是在使用缩放坐标，而不是直接坐标。

---

## 25. 最终归纳

统一公式把频移法中最核心的尺度关系全部串起来了：

```text
D[k] ≈ A · exp(j · 2π · s · k · u_p / N)
```

说明真实信号的相位旋转速度由三者决定：

```text
s：频率步长
u_p：投影仪坐标
N：投影仪分辨率
```

FFT 第 `q` 个 bin 测试的是：

```text
exp(-j · 2π · q · k / L)
```

说明 FFT 的候选旋转速度由：

```text
q / L
```

决定。

当二者匹配时：

```text
q / L = s · u_p / N
```

于是：

```text
u_p = q · N / (L · s)
```

也就是：

```text
coord_scale = N / (L · s)
disparity = q · coord_scale
```

如果希望：

```text
disparity = q
```

就必须令：

```text
L = N / s
```

所以当前推荐策略是：

```python
fft_length = round(proj_dim / frequency_step)
valid_length = proj_dim
coord_scale ≈ 1.0
```

这套策略的本质是：

> **让 FFT 的 bin 网格与投影仪坐标网格对齐。**

这样，FFT 峰值位置 `q` 就不再只是一个抽象频谱索引，而是可以直接解释为投影仪坐标。

---

## 26. 一句话总结

`M` 是你实际采了多少个频率样本，决定频谱峰的可靠性；  
`N` 是投影仪坐标轴长度，决定真实相位公式的分母；  
`L` 是 FFT 搜索网格长度，决定 `q` 和投影仪坐标之间的比例关系；  
统一公式：

```text
u_p = q · N / (L · s)
```

告诉我们：只要设置

```text
L = N / s
```

就能让

```text
q = u_p
```

从而避免 `fft_length = M` 带来的粗量化阶梯，使频移法解码结果更自然地对应到投影仪像素坐标。
