# FSM 仓库说明

## 1. 仓库定位

本仓库用于结构光三维重建相关研究与实现，当前主线方法为 **FSA / FSM（Frequency Shift Analysis / 频移法）**，并保留了较完整的早期实验代码、相移法对照代码、条纹生成脚本、图像整理脚本、点云对齐与误差分析脚本。

这份文档的目标不是简单介绍“怎么运行”，而是作为：

1. 我自己后续重新理解整个仓库时的总入口。
2. 后续让 AI 分析仓库、函数、模块职责时的上下文索引。
3. 后续撰写论文、章节分析、实验说明时的结构化参考。

因此，这个 README 会同时覆盖：

- 仓库主线功能
- 主流程
- 当前文件地图
- 当前主线代码结构
- 关键说明文档
- 每个 `.py` 文件的职责
- `initial/` 历史代码的定位
- 后续论文或 AI 分析时建议从哪里切入

---

## 2. 仓库整体功能概述

从功能上看，这个仓库可以分为 6 个阶段：

1. **生成结构光条纹图案**
   - 生成正弦条纹、周期条纹、N 步相移图案。
2. **采集与整理图像**
   - 将采集结果重命名、分组、提取指定频率范围图像。
3. **FSA 解码**
   - 通过四步相移构造复信号，再做 DFT / FFT，从相机像素恢复投影仪坐标。
4. **解码结果可视化与诊断**
   - 查看视差图、峰值图、频谱图、统计图。
5. **三维重建**
   - 利用标定参数和相机-投影仪像素对应关系做三角化，生成点云。
6. **点云后处理与对比分析**
   - 将 FSM 与 PSM 点云对齐，并做误差统计和差异可视化。

一句话概括：

> 这是一个围绕“频移法结构光三维重建”的完整研究型代码仓库，包含条纹生成、图像整理、解码、重建、点云分析，以及多轮历史实验版本。

---

## 2.1 当前文件地图

按当前仓库文件来看，最重要的内容可以分为 5 类：

### 当前主线代码

- [FSM_decode_example.py](/E:/code/FSM/FSM_decode_example.py)
  - 当前 FSA / FSM 解码入口。
- [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)
  - 当前 FSA / FSM 解码核心。
- [FSM_visualize.py](/E:/code/FSM/FSM_visualize.py)
  - 解码结果可视化与频谱诊断工具。
- [FSM_reconstruct_example.py](/E:/code/FSM/FSM_reconstruct_example.py)
  - 当前三维重建入口。
- [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)
  - 当前三维重建核心。

### 说明文档

- [README.md](/E:/code/FSM/README.md)
  - 仓库总入口和模块地图。
- [文档/FSM_method_explanation.md](/E:/code/FSM/文档/FSM_method_explanation.md)
  - FSM / FSA 频移法原理、逐像素解码过程、FFT 策略和代码片段说明。
- [文档/modification_analysis.md](/E:/code/FSM/文档/modification_analysis.md)
  - 解码代码关键修改说明，重点解释 FFT 长度修复、多峰检测和极线约束恢复。
- [点云后续处理/点云误差计算原理说明.md](/E:/code/FSM/点云后续处理/点云误差计算原理说明.md)
  - 点云误差计算与比较逻辑说明。
- [文档/Memory.md](/E:/code/FSM/文档/Memory.md)
  - 长期稳定仓库上下文。
- [文档/log.md](/E:/code/FSM/文档/log.md)
  - 用户手动提交后的进度日志。

### 实验辅助工具

- [图像整理/rename_images.py](/E:/code/FSM/图像整理/rename_images.py)
  - 采集图像批量重命名。
- [图像整理/提取指定频率图像.py](/E:/code/FSM/图像整理/提取指定频率图像.py)
  - 从完整频率序列中提取指定频率范围。
- [周期条纹生成/](/E:/code/FSM/周期条纹生成)
  - 条纹图案生成和图像整理工具。
- [点云后续处理/点云对齐工具.py](/E:/code/FSM/点云后续处理/点云对齐工具.py)
  - FSM 与 PSM 点云坐标系对齐。
- [点云后续处理/点云误差比较文件.py](/E:/code/FSM/点云后续处理/点云误差比较文件.py)
  - 两个点云之间的最近邻误差统计与可视化。

### 运行产物与数据

- [results/fsa_decode](/E:/code/FSM/results/fsa_decode)
  - 当前 FSA 解码输出，包括视差图、峰值图、校正视差图和频谱分析目录。
- [results/visualizations](/E:/code/FSM/results/visualizations)
  - 可视化输出图，例如视差图、统计图、对应关系图。
- [results/pointcloud_fsa.ply](/E:/code/FSM/results/pointcloud_fsa.ply)
  - 当前 FSM / FSA 重建点云结果。
- [pointCloud.ply](/E:/code/FSM/pointCloud.ply)
  - 仓库中保留的另一个点云结果，可用于对照。

### 历史与演进代码

- [initial/](/E:/code/FSM/initial)
  - 历史实现、早期实验分支、中间重构版本和旧版重建流程。
  - 默认不作为当前修改目标，除非需要追溯算法来源或写演进分析。

---

## 3. 当前主线流程

当前真正推荐关注和使用的主线流程如下：

### 第一步：准备输入

需要准备：

- 投影仪参数
- 相机参数
- 标定结果文件 `标定结果.txt`
- 采集得到的水平 / 垂直条纹图像序列

辅助配置文件位于：

- [config/camera.json](/E:/code/FSM/config/camera.json)
- [config/projector.json](/E:/code/FSM/config/projector.json)
- [标定结果.txt](/E:/code/FSM/标定结果.txt)

### 第二步：必要时整理图像

如果采集图像命名混乱，先使用：

- [图像整理/rename_images.py](/E:/code/FSM/图像整理/rename_images.py)

如果只想提取某个频率范围的图像，使用：

- [图像整理/提取指定频率图像.py](/E:/code/FSM/图像整理/提取指定频率图像.py)

### 第三步：执行 FSA 解码

频移法原理说明：

- [文档/FSM_method_explanation.md](/E:/code/FSM/文档/FSM_method_explanation.md)

FSM / FSA 解码的核心是：对每个相机像素，读取它在多个频率、四个相移条纹图像中的灰度响应，
构造复信号 `D = (I0-I2) + j*(I3-I1)`，再沿频率维度做 FFT。
频谱峰值位置被解释为投影仪坐标，峰值强度作为该像素的解码质量指标。

主入口：

- [FSM_decode_example.py](/E:/code/FSM/FSM_decode_example.py)

核心模块：

- [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)

输出通常位于：

- [results/fsa_decode](/E:/code/FSM/results/fsa_decode)

典型输出包括：

- `disp_hor.npy`：投影仪行坐标对应图
- `disp_ver.npy`：投影仪列坐标对应图
- `peak_hor.npy`：水平方向峰值质量图
- `peak_ver.npy`：垂直方向峰值质量图

### 第四步：可视化与检查解码质量

使用：

- [FSM_visualize.py](/E:/code/FSM/FSM_visualize.py)

可以查看：

- 视差图
- 峰值强度图
- 统计图
- 对应关系图
- 指定像素的频谱图

### 第五步：执行三维重建

主入口：

- [FSM_reconstruct_example.py](/E:/code/FSM/FSM_reconstruct_example.py)

核心模块：

- [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)

输出点云通常为：

- [results/pointcloud_fsa.ply](/E:/code/FSM/results/pointcloud_fsa.ply)

### 第六步：点云对齐与误差分析

用于和旧的相移法点云对照：

- [点云后续处理/点云对齐工具.py](/E:/code/FSM/点云后续处理/点云对齐工具.py)
- [点云后续处理/点云误差比较文件.py](/E:/code/FSM/点云后续处理/点云误差比较文件.py)

---

## 4. 当前主线模块说明

这一节只描述根目录下“现在最值得读”的主线代码。

### 4.1 解码核心

#### [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)

当前 FSA 解码核心模块，已经由早期过程式脚本重构为多个类协作。

主要类：

- `ImageLoader`
  - 负责条纹图像序列加载、缓存、序列检查。
- `CalibrationReader`
  - 读取 `标定结果.txt`，提取相机与投影仪标定参数，供极线约束使用。
- `DFTProcessor`
  - 构造复信号并执行 FFT / DFT。
- `PeakDetector`
  - 从频谱中找主峰，并支持亚像素细化。
- `MultiPeakDetector`
  - 找多个候选峰，服务于极线约束校正。
- `EpipolarCorrector`
  - 根据标定参数和极线几何关系，对多峰候选进行筛选与修正。
- `FSADecoder`
  - 总协调类，负责完整解码流程、低内存模式和结果保存。

这个文件的核心价值在于：

1. 把早期 `decode_fsa.py` 的逻辑模块化。
2. 修复了关键的 **FFT 长度选择问题**。
3. 恢复了 **多峰检测** 和 **极线约束校正** 两个重要后处理能力。

#### [FSM_decode_example.py](/E:/code/FSM/FSM_decode_example.py)

当前解码主入口，负责把配置组织起来并真正执行流程。

主要职责：

- 配置投影仪尺寸、频率数、频率步长
- 配置输入输出目录
- 选择解码方向：`both / hor / ver`
- 调用 `FSADecoder`
- 选择是否启用低内存模式
- 选择是否做多峰扫描
- 选择是否做极线约束校正
- 选择是否保存和可视化频谱

如果要改参数、跑实验、做批量测试，通常先看这个文件。

#### [FSM_visualize.py](/E:/code/FSM/FSM_visualize.py)

当前解码结果可视化工具。

主要职责：

- 读取 `disp_hor / disp_ver / peak_hor / peak_ver`
- 可视化视差图
- 可视化峰值强度图
- 输出统计图
- 画相机像素到投影仪像素的对应关系图
- 查看某些点的频谱

这个文件适合：

- 诊断某组数据是否解码成功
- 判断 `peak_limit` 是否合理
- 为论文配图或中间结果分析提供直观材料

### 4.2 三维重建核心

#### [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)

当前 FSA 三维重建核心模块。

主要类：

- `CalibrationReader`
  - 读取标定结果。
- `CorrespondenceReader`
  - 读取解码得到的对应关系。
- `TriangulationEngine`
  - 负责三角化计算，核心调用 `cv2.triangulatePoints`。
- `PointCloudProcessor`
  - 点云保存、简单 PLY 输出、可视化。
- `FSAReconstructor`
  - 总协调类，组织读取数据、点云生成、保存与显示。

这个模块的关键点：

- 从 `disp_hor + disp_ver` 恢复投影仪 2D 坐标。
- 通过相机-投影仪双目几何做三角化。
- 支持 `peak_limit` 过滤。
- 支持 `oversample` 相关处理。
- 包含对新版解码输出峰值范围的补偿，使旧版阈值语义尽量保持一致。

#### [FSM_reconstruct_example.py](/E:/code/FSM/FSM_reconstruct_example.py)

当前三维重建入口脚本。

主要职责：

- 配置标定文件、图像尺寸、投影仪尺寸
- 读取 `results/fsa_decode` 下的结果
- 可选读取白光图像为点云赋色
- 调用 `FSAReconstructor` 做三维重建
- 保存点云
- 显示点云

如果只是“跑通一遍三维重建”，一般直接从这里开始。

---

## 5. 图像整理与点云后处理脚本

### [图像整理/rename_images.py](/E:/code/FSM/图像整理/rename_images.py)

图像批量重命名工具。

主要用途：

- 将采集后的图像重命名为 `0.png, 1.png, 2.png, ...`
- 支持自然排序
- 支持预览模式
- 支持备份原始文件名
- 支持对子目录批量处理

适合在“原始采集数据不规范”时先清洗文件名。

### [图像整理/提取指定频率图像.py](/E:/code/FSM/图像整理/提取指定频率图像.py)

按频率范围提取图像的工具。

主要用途：

- 从整套频移图像中抽出某段频率范围
- 自动识别水平/垂直条纹目录
- 支持多线程复制
- 可以联动重命名工具生成标准序列

适合做：

- 小范围实验
- 频率裁剪测试
- 快速构造局部数据集

### [点云后续处理/点云对齐工具.py](/E:/code/FSM/点云后续处理/点云对齐工具.py)

FSM 点云和 PSM 点云的坐标系对齐工具。

核心问题：

- PSM 点云通常在**相机坐标系**
- FSM 点云通常在**投影仪坐标系**

这个脚本通过读取标定文件中的旋转矩阵 `R` 和平移向量 `T`，在两个坐标系之间做变换，使两个点云可直接比较。

适合做：

- FSM 与 PSM 的实验对照
- 论文中“方法结果一致性”分析
- 后续误差计算前的预处理

### [点云后续处理/点云误差比较文件.py](/E:/code/FSM/点云后续处理/点云误差比较文件.py)

点云误差比较工具。

主要用途：

- 读取两个 `.ply` 点云
- 用最近邻距离比较两者误差
- 统计均值、最大值、最小值、RMSE、标准差、中位数、P95
- 把误差较大的点标成红色并导出新点云

适合做：

- 不同重建方法间误差分析
- 不同参数设置下结果比较
- 可视化论文中的差异区域

---

## 6. `周期条纹生成/` 目录说明

这个目录主要是条纹生成与图像组织的辅助工具，偏“实验准备层”。

### [周期条纹生成/generate_fringe_patterns.py](/E:/code/FSM/周期条纹生成/generate_fringe_patterns.py)

通用 N 步相移正弦条纹生成器。

功能：

- 生成水平或垂直方向条纹
- 支持 N 步相移
- 支持命令行参数
- 也支持交互式模式
- 可以快速预览方向和频率效果

适合做“快速生成一组测试条纹”。

### [周期条纹生成/图像分类整理工具.py](/E:/code/FSM/周期条纹生成/图像分类整理工具.py)

图像分类整理工具。

提供两种整理方式：

- 按相位分类：把 `4*i + 1 ~ 4*i + 4` 分类到 `phase0 ~ phase3`
- 按频率分组：每 4 张一组打包成 `f0 / f1 / f2 ...`

适合在实验中做数据重排和人工检查。

### [周期条纹生成/条纹图像生成.py](/E:/code/FSM/周期条纹生成/条纹图像生成.py)

较完整的条纹生成脚本。

特点：

- 支持按频率模式或周期模式批量生成
- 支持水平、垂直、双方向
- 支持 N 步相移
- 支持调制度控制
- 可生成 1D 数据或完整尺寸图像

这是更“研究实验型”的条纹生成器。

### [周期条纹生成/条纹图像生成_周期版.py](/E:/code/FSM/周期条纹生成/条纹图像生成_周期版.py)

以“周期数”为中心的条纹生成版本。

特点：

- 更强调周期数而非频率步长
- 支持小数周期命名
- 适合做按周期参数扫描的实验

可以把它理解为与上一个脚本并行存在的“另一种参数表达版本”。

---

## 7. `initial/` 目录说明

`initial/` 目录不是当前主线，而是项目研发过程中的历史留档、实验分支和中间重构结果。

这里的代码价值主要有三类：

1. **追溯当前主线代码从哪里演化而来**
2. **查看某些算法步骤最早的过程式实现**
3. **为论文写作提供“方法演进”和“对照实验”材料**

建议把 `initial/` 看作“研究档案层”，而不是“日常运行入口”。

---

## 8. `initial/` 下每个 `.py` 文件作用

### 8.1 早期采集、生成、通用配置

#### [initial/constants.py](/E:/code/FSM/initial/constants.py)

早期项目的全局配置文件。

内容包括：

- 数据路径
- 标定文件路径
- 相机曝光设置
- 投影仪分辨率
- 结果输出目录

很多早期脚本都依赖它。

#### [initial/project_capture.py](/E:/code/FSM/initial/project_capture.py)

早期自动投影采集脚本。

主要用于：

- 逐张投影结构光图案
- 调用硬件相机采集图像
- 保存采集结果

#### [initial/auto_project_for_calibrate.py](/E:/code/FSM/initial/auto_project_for_calibrate.py)

面向标定阶段的投影采集脚本。

与 `project_capture.py` 类似，但更偏标定流程，依赖 `pyaoi` 和大恒相机相关接口。

#### [initial/generate.py](/E:/code/FSM/initial/generate.py)

最早的简单条纹生成与频谱验证脚本。

主要作用：

- 生成基础条纹
- 抽取单点信号
- 直接观察 FFT 结果

偏算法理解与原理验证。

#### [initial/utils.py](/E:/code/FSM/initial/utils.py)

工具函数文件。

当前主要函数：

- `calcSubPixelPhase`

用于亚像素相位插值。

### 8.2 早期 FSA 解码实验

#### [initial/decode.py](/E:/code/FSM/initial/decode.py)

非常早期的解码测试脚本。

主要作用：

- 取单个像素的 `I0/I1/I2/I3`
- 构造 `Dc/Ds`
- 直接观察频谱

适合用于解释“频移法单像素解码原理”。

#### [initial/decode_check.py](/E:/code/FSM/initial/decode_check.py)

解码验证脚本。

主要用于对比：

- 原始信号
- 加窗前后
- FFT 效果差异

适合写论文中“频谱行为分析”部分的辅助说明。

#### [initial/decode_fsa.py](/E:/code/FSM/initial/decode_fsa.py)

最重要的早期 FSA 过程式脚本，也是当前主线解码代码的重要来源。

主要职责：

- 完整 FSA 解码
- 视差图生成
- 峰值检测
- 多峰候选处理
- 极线相关逻辑

如果需要追溯“当前 `FSM_decode_core.py` 的原始过程式版本”，应优先看这个文件。

#### [initial/decode_psa.py](/E:/code/FSM/initial/decode_psa.py)

早期 PSA 解码脚本。

本质上是调用外部程序做相移法解码的命令包装层。

### 8.3 早期 PSM / PSA 重建

#### [initial/reconstruct_psa.py](/E:/code/FSM/initial/reconstruct_psa.py)

PSA / PSM 点云重建脚本。

主要用途：

- 从相移法结果生成点云
- 查看和交互筛选点云
- 为后续 FSM/PSM 对照提供基线结果

#### [initial/三维重建_finally1.py](/E:/code/FSM/initial/三维重建_finally1.py)

较大型的早期结构光三维重建实现，已经有明显的模块化结构。

内部包含：

- 标定读取
- 图像加载
- 颜色提取
- 相位计算
- 解包裹
- 点云重建
- 点云过滤
- 可视化

它是理解“早期相移法完整工作流”的关键文件。

### 8.4 早期 FSM 重建版本

#### [initial/三维重建（通过点对关系）.py](/E:/code/FSM/initial/三维重建（通过点对关系）.py)

基于点对关系的 FSM 重建脚本。

特点：

- 直接从 `.npy` 视差 / 对应关系数据重建
- 依赖较少
- 更偏“直接三角化版本”

适合看“FSM 重建最直接的数学实现”。

### 8.5 重构中的解码版本

#### [initial/FSM解码过程/FSM_decode_core.py](/E:/code/FSM/initial/FSM解码过程/FSM_decode_core.py)

当前解码主线的早期重构版。

可以看作从 `decode_fsa.py` 走向当前 `FSM_decode_core.py` 的中间版本。

#### [initial/FSM解码过程/FSM_decode_example.py](/E:/code/FSM/initial/FSM解码过程/FSM_decode_example.py)

与上面早期重构版配套的示例入口。

#### [initial/FSM解码过程/FSM_visualize.py](/E:/code/FSM/initial/FSM解码过程/FSM_visualize.py)

与上面早期重构版配套的可视化工具。

### 8.6 测试“频率步长增加为 0.5”的分支

#### [initial/FSM解码过程_测试频率增加为0.5/FSM_decode_core.py](/E:/code/FSM/initial/FSM解码过程_测试频率增加为0.5/FSM_decode_core.py)

测试频率步长改为 `0.5` 的解码核心实验版。

#### [initial/FSM解码过程_测试频率增加为0.5/FSM_decode_example.py](/E:/code/FSM/initial/FSM解码过程_测试频率增加为0.5/FSM_decode_example.py)

对应的实验入口。

#### [initial/FSM解码过程_测试频率增加为0.5/FSM_visualize.py](/E:/code/FSM/initial/FSM解码过程_测试频率增加为0.5/FSM_visualize.py)

对应的实验可视化工具。

这一组文件说明仓库曾专门做过“频率步长变化”相关实验。

### 8.7 重构中的三维重建版本

#### [initial/解码后的三维重建过程/FSM_reconstruct_core.py](/E:/code/FSM/initial/解码后的三维重建过程/FSM_reconstruct_core.py)

当前重建主线的早期重构版。

#### [initial/解码后的三维重建过程/FSM_reconstruct_example.py](/E:/code/FSM/initial/解码后的三维重建过程/FSM_reconstruct_example.py)

对应的重建入口。

### 8.8 单方向重建实验

#### [initial/解码后的三维重建过程_单方向/FSM_reconstruct_core.py](/E:/code/FSM/initial/解码后的三维重建过程_单方向/FSM_reconstruct_core.py)

用于单方向重建实验的核心版本。

#### [initial/解码后的三维重建过程_单方向/FSM_reconstruct_example.py](/E:/code/FSM/initial/解码后的三维重建过程_单方向/FSM_reconstruct_example.py)

对应的实验入口。

#### [initial/解码后的三维重建过程_单方向/analyze_disparity_error.py](/E:/code/FSM/initial/解码后的三维重建过程_单方向/analyze_disparity_error.py)

分析单方向重建时视差图误差、视差范围、梯度等因素对高度信息影响的脚本。

#### [initial/解码后的三维重建过程_单方向/tempCodeRunnerFile.py](/E:/code/FSM/initial/解码后的三维重建过程_单方向/tempCodeRunnerFile.py)

编辑器临时文件，无实际业务意义。

### 8.9 结果对照类小脚本

#### [initial/check_distancemap.py](/E:/code/FSM/initial/check_distancemap.py)

对比 FSA 与 PSA 距离图剖线的脚本。

适合做快速结果 sanity check。

---

## 9. 关键数据与输出文件

### 标定相关

- [标定结果.txt](/E:/code/FSM/标定结果.txt)
  - 当前主线重建所使用的标定结果。

### 解码输出

- [results/fsa_decode](/E:/code/FSM/results/fsa_decode)
  - 保存解码得到的视差图、峰值图、频谱图等。

常见文件：

- `disp_hor.npy`
- `disp_ver.npy`
- `peak_hor.npy`
- `peak_ver.npy`
- `disp_hor_corrected.npy`
- `disp_ver_corrected.npy`
- `spectrum_point_*`
  - 指定像素频谱数据和 8 子图分析结果。
- `epipolar_spectrum/`
  - 极线约束校正像素的频谱分析结果。

### 可视化结果

- [results/visualizations](/E:/code/FSM/results/visualizations)
  - 保存可视化脚本生成的图。

常见文件：

- `disparity_hor.png`
- `disparity_ver.png`
- `statistics.png`
- `correspondence_map.png`

### 点云结果

- [results/pointcloud_fsa.ply](/E:/code/FSM/results/pointcloud_fsa.ply)
  - FSM / FSA 重建结果点云。
- [pointCloud.ply](/E:/code/FSM/pointCloud.ply)
  - 仓库中已有的另一个点云文件，通常可用于对照。

---

## 10. 推荐阅读顺序

如果后续我要重新理解整个仓库，推荐按下面顺序阅读：

### 路线 A：想先抓主线

1. [README.md](/E:/code/FSM/README.md)
2. [文档/FSM_method_explanation.md](/E:/code/FSM/文档/FSM_method_explanation.md)
3. [FSM_decode_example.py](/E:/code/FSM/FSM_decode_example.py)
4. [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)
5. [FSM_visualize.py](/E:/code/FSM/FSM_visualize.py)
6. [FSM_reconstruct_example.py](/E:/code/FSM/FSM_reconstruct_example.py)
7. [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)

### 路线 B：想追溯演进关系

1. [initial/decode_fsa.py](/E:/code/FSM/initial/decode_fsa.py)
2. [initial/FSM解码过程/FSM_decode_core.py](/E:/code/FSM/initial/FSM解码过程/FSM_decode_core.py)
3. [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)
4. [initial/三维重建（通过点对关系）.py](/E:/code/FSM/initial/三维重建（通过点对关系）.py)
5. [initial/解码后的三维重建过程/FSM_reconstruct_core.py](/E:/code/FSM/initial/解码后的三维重建过程/FSM_reconstruct_core.py)
6. [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)

### 路线 C：想做实验与论文整理

1. 本 README 先建立仓库地图
2. 看 [文档/FSM_method_explanation.md](/E:/code/FSM/文档/FSM_method_explanation.md) 理解频移法原理与代码实现
3. 看主线解码与重建入口脚本
4. 看 [文档/modification_analysis.md](/E:/code/FSM/文档/modification_analysis.md) 理解关键修改点
5. 看 [点云后续处理/点云误差计算原理说明.md](/E:/code/FSM/点云后续处理/点云误差计算原理说明.md) 理解点云误差指标
6. 看 `initial/` 中早期过程式脚本追溯算法来源
7. 看点云对齐与误差比较工具准备对照实验

---

## 11. 后续让 AI 分析仓库时的建议切入点

后续如果需要让我或其他 AI 做更深入的事情，建议都先从本 README 开始，再明确任务类型。

### 适合的分析任务

1. **分析整个仓库的作用**
   - 先看本 README 的第 2、3、4、7、8 节。
2. **分析某个主线模块的函数职责**
   - 优先从 `FSM_decode_core.py` 或 `FSM_reconstruct_core.py` 入手。
3. **梳理代码演进关系**
   - 优先比较 `initial/decode_fsa.py` 与 `FSM_decode_core.py`
   - 再比较 `initial/.../FSM_reconstruct_core.py` 与 `FSM_reconstruct_core.py`
4. **为论文撰写方法章节**
   - 重点看：
     - 本 README 第 2、3、4 节
     - [文档/FSM_method_explanation.md](/E:/code/FSM/文档/FSM_method_explanation.md)
     - [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)
     - [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)
5. **为论文撰写实验章节**
   - 重点看：
     - `results/`
     - [FSM_visualize.py](/E:/code/FSM/FSM_visualize.py)
     - [点云后续处理/点云对齐工具.py](/E:/code/FSM/点云后续处理/点云对齐工具.py)
     - [点云后续处理/点云误差比较文件.py](/E:/code/FSM/点云后续处理/点云误差比较文件.py)
     - [点云后续处理/点云误差计算原理说明.md](/E:/code/FSM/点云后续处理/点云误差计算原理说明.md)
6. **写“算法演进 / 改进点”章节**
   - 重点看：
     - [文档/modification_analysis.md](/E:/code/FSM/文档/modification_analysis.md)
     - `initial/` 下早期版本
     - 当前主线版本

### 后续如果让 AI 写论文，建议给它的任务可以是

- “根据 README + 主线代码，写系统总体设计章节”
- “根据 README + 文档/FSM_method_explanation.md + decode 核心代码，写频移法解码原理与实现章节”
- “根据 README + reconstruct 核心代码，写三维重建流程章节”
- “根据 README + initial 历史代码，写算法演进与改进分析章节”
- “根据 README + 点云比较工具 + 点云后续处理/点云误差计算原理说明.md，写实验评价指标与误差分析章节”

---

## 12. 我对这个仓库的整体归纳

这个仓库本质上由三层组成：

### 第一层：当前主线层

用于现在实际运行和继续开发：

- `FSM_decode_core.py`
- `FSM_decode_example.py`
- `FSM_visualize.py`
- `FSM_reconstruct_core.py`
- `FSM_reconstruct_example.py`

### 第二层：实验辅助层

用于准备数据、整理图像、生成条纹、比较点云：

- `图像整理/rename_images.py`
- `图像整理/提取指定频率图像.py`
- `周期条纹生成/`
- `点云后续处理/点云对齐工具.py`
- `点云后续处理/点云误差比较文件.py`

### 第三层：历史演进层

用于保留算法来源、旧版实现、实验分支和研究过程：

- `initial/`

如果以后只想“快速恢复对项目的理解”，先看第一层。

如果以后要“写论文或让 AI 做深度分析”，三层都要结合看，但仍建议从第一层开始，再回到第三层追踪来源。

---

## 13. 补充说明

### 关于编码和中文显示

仓库中的部分旧文件和旧文档存在编码历史，终端中可能会出现中文乱码，但不影响文件的结构定位和代码职责判断。

### 关于依赖

从代码看，仓库不同阶段依赖并不完全一致，常见依赖包括：

- `numpy`
- `opencv-python`
- `matplotlib`
- `open3d`
- `Pillow`
- `scipy`
- `plyfile`
- 早期代码中的 `pyaoi`
- 某些硬件采集脚本中的大恒相机接口

因此：

- **当前主线** 主要关注 `numpy / cv2 / matplotlib / open3d`
- **历史代码** 还涉及 `pyaoi` 和硬件环境

### 关于“当前最应该维护哪部分”

如果后续继续做维护、改进或用于论文实现说明，优先维护：

1. [FSM_decode_core.py](/E:/code/FSM/FSM_decode_core.py)
2. [FSM_decode_example.py](/E:/code/FSM/FSM_decode_example.py)
3. [FSM_reconstruct_core.py](/E:/code/FSM/FSM_reconstruct_core.py)
4. [FSM_reconstruct_example.py](/E:/code/FSM/FSM_reconstruct_example.py)
5. 本 README

---

## 14. 后续维护建议

如果后续继续扩展这个仓库，建议优先做这几件事：

1. 给主线代码补更稳定的函数级文档字符串。
2. 把根目录主线代码和 `initial/` 历史代码在 README 中持续区分清楚。
3. 后续如果写论文，可以在仓库中新增：
   - `docs/algorithm_notes/`
   - `docs/experiment_notes/`
   - `docs/paper_materials/`
4. 所有新的分析文档都建议先引用本 README，再细化到具体文件。

---

## 15. 最简结论

如果以后忘了这个仓库是干什么的，只记住下面这几句话就够了：

1. 这是一个 **频移法结构光三维重建** 仓库。
2. 当前主线是：**解码 -> 可视化 -> 重建 -> 点云分析**。
3. 根目录是现行主线，`initial/` 是历史演进与实验档案。
4. 后续让 AI 分析仓库、函数、算法或写论文，优先先读本 README，再进入主线代码。
