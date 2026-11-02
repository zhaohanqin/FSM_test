# 2+1 相移模式资料

本目录集中存放 **2+1 相移** 的研究说明、实测结果与独立验证脚本。主线解码已集成在仓库根目录的 `FSM_decode_core.py` / `FSM_decode_example.py`（`PhaseShiftDemodulator`），此处保留证据链与长文推导。

## 从哪里读起

| 路径 | 内容 |
| --- | --- |
| [2plus1替代四步相移说明.md](2plus1替代四步相移说明.md) | 书架实测结论、整数列差 1、频谱图索引 |
| [2plus1_逐像素频谱/](2plus1_逐像素频谱/) | 误差表、`decode_ab_test.md`、抽样像素诊断图 |
| [FSM_2plus1/README.md](FSM_2plus1/README.md) | 理论推导、仿真与完整对话纪要 |
| [FSM_2plus1/scripts/verify_decode_2plus1.py](FSM_2plus1/scripts/verify_decode_2plus1.py) | 主线四步 vs 2+1 A/B 解码 |

仓库总览与配置说明见根目录 [README.md](../README.md) 第 3 步与「路线 D」。
