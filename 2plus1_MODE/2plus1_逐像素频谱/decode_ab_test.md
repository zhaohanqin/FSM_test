# 解码 A/B：四步 vs 2+1 (A4)

数据：`E:\images\before_2026_5_23\before_2025_12_27\images_FSM\place1\03_bookshelf_white_cycle1\FSM_vertical_f0_f150`  
有效掩膜：四步 `peak_ver >= 1.5`  

| 指标 | 值 |
| --- | ---: |
| 全图像素 | 5,013,504 |
| 有效像素 | 2,535,906 |
| 整数列相同 | 2,506,860 (98.85%) |
| 整数列差 1 | 29,044 (1.15%) |
| 整数列差 >1 | 2 |
| 亚像素 \|Δu\| 中位数 | 0.000061 列 |
| 亚像素 \|Δu\| 最大 | 104.9946 列 |
| \|Δu\| ≤ 1 列占比 | 99.9999% |

输出目录：`results/fsa_decode_four/` 与 `results/fsa_decode_2plus1/`（各含 `disp_ver.npy`, `peak_ver.npy`）。

实用结论：亚像素 ≤1 列 通过；整数 100% 一致 未通过。
