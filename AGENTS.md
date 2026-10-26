# Repository Guidelines

## 仓库定位

本仓库用于结构光三维重建相关研究与实现，当前主线方法为 FSA / FSM 频移法解码与重建。

- 当前主线代码位于仓库根目录。
- `initial/` 目录主要保存历史实现、实验分支和演进过程，不作为默认修改目标。
- `README.md` 用于整体理解仓库。
- `文档/Memory.md` 用于记录长期稳定的仓库规则和上下文。
- `文档/log.md` 用于记录每次手动提交后的简要进度。

## 目录与模块约定

- `FSM_decode_example.py`
  - 当前解码流程入口。
- `FSM_decode_core.py`
  - 当前 FSA 解码核心。
- `FSM_visualize.py`
  - 解码结果可视化与诊断。
- `FSM_reconstruct_example.py`
  - 当前三维重建入口。
- `FSM_reconstruct_core.py`
  - 当前三维重建核心。
- `config/`
  - 相机和投影仪配置文件。
- `results/`
  - 解码结果、可视化结果和重建点云。
- `initial/`
  - 历史代码、旧版实现、实验版本。
- `周期条纹生成/`
  - 条纹图案生成工具。
- `图像整理/`
  - 采集图像重命名和按频率提取。
- `文档/`
  - 原理说明、修改分析、`Memory.md` 和 `log.md`。
- `点云后续处理/`
  - 点云对齐与误差比较。
- `项目说明PPT/`
  - 原理讲解幻灯片和生成脚本。

## 默认阅读顺序

1. `README.md`
2. `文档/Memory.md`
3. `文档/log.md`
4. `FSM_decode_example.py`
5. `FSM_decode_core.py`
6. `FSM_visualize.py`
7. `FSM_reconstruct_example.py`
8. `FSM_reconstruct_core.py`
9. `initial/` 仅在需要追溯历史实现时再看

## 运行方式

仓库没有统一打包构建流程，通常直接在仓库根目录运行 Python 脚本：

```powershell
python FSM_decode_example.py
python FSM_visualize.py
python FSM_reconstruct_example.py
```

常见依赖包括：

```powershell
pip install numpy opencv-python matplotlib open3d
```

部分历史脚本可能还依赖 `pyaoi` 或硬件 SDK，但这些不属于当前主线默认运行前提。

## 修改约定

- 优先修改根目录下的 `FSM_*` 主线文件，而不是 `initial/` 中的历史版本。
- 新增说明文档优先使用中文，便于后续仓库分析和论文撰写。
- 变更如果影响仓库长期规则、主线结构、命名规范或工作流，需要同步更新 `文档/Memory.md`。
- 用户手动提交完成后，如需记录本次改动，应更新 `文档/log.md`。
- `文档/log.md` 保持简洁，仅记录：
  - `提交`
  - `本次改动摘要`
  - `功能影响`
  - `下一步`

## 验证建议

- 修改解码逻辑后：
  - 运行 `python FSM_decode_example.py`
  - 检查 `results/fsa_decode/` 下输出是否正常
- 修改可视化逻辑后：
  - 运行 `python FSM_visualize.py`
  - 检查输出图片、统计图、频谱图是否符合预期
- 修改重建逻辑后：
  - 运行 `python FSM_reconstruct_example.py`
  - 检查 `pointcloud_fsa.ply` 是否成功生成且结果合理

## 协作规则

- 分析仓库时，默认将根目录 `FSM_*` 视为当前主线。
- 引用 `initial/` 中历史代码时，应明确说明其属于旧版实现或演进参考。
- 如果从历史脚本推断功能或设计意图，应把结论标记为推断，而不是既定事实。
- 如果修改了文档体系，应确保 `README.md`、`文档/Memory.md`、`文档/log.md` 三者职责不混淆：
  - `README.md`：整体介绍
  - `文档/Memory.md`：长期稳定记忆
  - `文档/log.md`：提交后进度记录
