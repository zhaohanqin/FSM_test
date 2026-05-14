# Memory

## 仓库身份

- 仓库名称：`FSM`
- 核心用途：
  - 本仓库用于结构光三维重建相关研究与实现。
  - 当前主线方法是 FSA / FSM，即基于频移法的解码与重建流程。
- 主要使用场景：
  - 从条纹图像中解码相机像素到投影仪像素的对应关系。
  - 基于解码结果和标定参数进行三维点云重建。
  - 保留历史实验代码，用于算法演进分析与论文撰写。

## 稳定上下文

- 主要语言：
  - Python
- 主要运行时 / 框架：
  - Python 脚本
  - OpenCV
  - NumPy
  - Matplotlib
  - Open3D
- 主要入口文件：
  - `FSM_decode_example.py`
  - `FSM_reconstruct_example.py`
  - `FSM_visualize.py`
- 主要输出产物：
  - `results/fsa_decode/*.npy`
  - `results/visualizations/*.png`
  - `pointcloud_fsa.ply`

## 主线模块

- `FSM_decode_core.py`
  - 当前主线 FSA 解码核心
- `FSM_decode_example.py`
  - 解码流程入口
- `FSM_visualize.py`
  - 解码结果可视化与诊断
- `FSM_reconstruct_core.py`
  - 当前主线三维重建核心
- `FSM_reconstruct_example.py`
  - 三维重建流程入口

## 目录约定

- 根目录：
  - 放当前主线代码以及顶层工具脚本
- `results/`
  - 存放解码结果和可视化输出
- `config/`
  - 存放相机 / 投影仪相关配置
- `initial/`
  - 存放历史实现、实验分支和演进代码
- `周期条纹生成/`
  - 存放条纹生成和图像整理工具

## 命名规范

- `FSM_*`
  - 表示当前主线模块
- 中文命名的工具脚本
  - 通常是实验工具、分析工具或一次性辅助脚本
- `initial/`
  - 表示历史版本或归档实现，不默认视为当前主线
- `log.md`
  - 记录按时间顺序追加的进度日志
- `Memory.md`
  - 只记录长期稳定有效的仓库记忆

## 文档规则

- `README.md`
  - 记录仓库整体概览、模块地图和上手上下文
- `Memory.md`
  - 只记录长期稳定事实
  - 应存放架构、命名、工作流、仓库解释规则等长期有效内容
- `log.md`
  - 记录用户完成手动提交后的追加式进度历史
  - 记录改动内容、改动位置、功能影响和下一步计划

## 固定工作流

1. 修改前先阅读 `README.md`，理解仓库结构。
2. 再阅读 `Memory.md`，理解稳定上下文和解释规则。
3. 再查看 `log.md` 最新条目，了解最近演进。
4. 默认将根目录下的 `FSM_*` 文件视为当前主线。
5. 默认将 `initial/` 视为历史 / 演进代码，除非任务明确要求分析它。
6. 当实现已经完成且用户手动提交后：
   - 检查最新一次提交
   - 更新 `log.md`
   - 只有在长期稳定信息发生变化时才更新 `Memory.md`

## 后续 AI 的推荐阅读顺序

1. `README.md`
2. `Memory.md`
3. `log.md` 最新条目
4. `FSM_decode_example.py`
5. `FSM_decode_core.py`
6. `FSM_visualize.py`
7. `FSM_reconstruct_example.py`
8. `FSM_reconstruct_core.py`
9. `initial/` 只在追溯历史或写演进分析时再读

## 长期注意事项

- 不要把当前主线代码和 `initial/` 历史代码混为一谈。
- 一些旧文件和旧文档在终端中可能会有编码显示问题。
- `Memory.md` 只放稳定架构和长期上下文，不能混入一次性提交细节。
- 当从历史脚本推断功能时，必须明确标记该结论属于推断。
