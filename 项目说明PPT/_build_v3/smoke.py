# -*- coding: utf-8 -*-
"""smoke.py — 版式冒烟测试：生成 7 张代表性页面，渲染后人工检查。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deck_kit as K
from deck_kit import *
from deck_special import *

K.TOTAL = 66
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke.pptx")
prs = new_deck()
T = 66

# ---------------- 1 封面 ----------------
set_ctx(1, "封面")
s = blank(prs)
cover(s,
      [("频移法（FSM / FSA）", "white"), ("原理与实现详解", "22D3EE")],
      "从条纹图像到投影仪坐标对应关系，再到三维点云",
      "S T R U C T U R E D   L I G H T   ·   D E C O D I N G",
      formula="uₚ = q · N / (L · s)",
      note="核心目标：让 FFT 的 bin 网格与投影仪坐标网格对齐",
      chips_items=["条纹采集", "四步相移", "复信号 D[k]", "频率轴 FFT", "投影仪坐标", "三维点云"],
      section="封面")

# ---------------- 2 目录 ----------------
set_ctx(2, "目录")
s = blank(prs)
toc(s,
    [("问题背景与方法目标", "结构光为什么要解码，两个方向分别解什么"),
     ("四步相移与复信号构造", "从灰度差分到携带相位的复数 D"),
     ("频率轴信号与 FFT 解码", "相位旋转速度就是投影仪坐标"),
     ("主线代码实现与数据形状", "模块地图、张量维度、分块解码"),
     ("FFT 长度策略：统一公式", "uₚ = q·N/(L·s) 与量化阶梯"),
     ("工程实现 · 后处理 · 排查", "内存、多峰检测、极线校正"),
     ("与三维重建的关系", "对应关系如何进入三角化与点云"),
     ("总结与常见误区", "三式串联与参数选择速查")],
    ["条纹采集", "四步相移", "频率轴信号", "FFT 解码", "投影仪坐标", "三维点云"],
    "核心叙事主线", "讲解路线", 2, T)

# ---------------- 3 章节页 ----------------
set_ctx(3, "章节页")
s = blank(prs)
chapter(s, "二", "四步相移与复信号构造", "Four-Step Phase Shift & Complex Signal",
        [("如何用", {}), ("四步相移差分", {"color": "EF4444"}),
         ("抵消直流亮度干扰，并把", {}), ("实数灰度", {"color": "EF4444"}),
         ("转化为", {}), ("携带相位的复数 D", {"color": "EF4444"}), ("。", {})],
        "四步相移与复信号构造", 7, T)

# ---------------- 4 要点卡页 ----------------
set_ctx(4, "公式带+要点卡")
s = blank(prs)
page(s, "五 · FFT 长度策略", "为什么推荐让 coord_scale = 1",
     "最理想的工程状态：FFT 峰值 bin 编号 q 本身就是投影仪坐标",
     "五 · FFT 长度策略", 30, T)
formula_band(s, ML, 2.06, 3.62, 0.78,
             [("coord_scale = N / (L · s)", {"size": 17, "color": "white"})],
             fill="navy", accent="navy", size=17)
arrow_h(s, ML + 3.76, 2.45, 0.62, "amber", 1.75)
formula_band(s, ML + 4.52, 2.06, 3.10, 0.78,
             [("coord_scale = 1", {"size": 17, "color": "F29F05"})],
             fill="white", accent="amber", size=17)
arrow_h(s, ML + 7.76, 2.45, 0.62, "amber", 1.75)
formula_band(s, ML + 8.52, 2.06, 3.01, 0.78,
             [("L = N / s", {"size": 17, "color": "cyan_br"})],
             fill="navy", accent="navy", size=17)
items_l = [("q 语义清楚", "argmax 输出的 bin 编号 q 直接就是投影仪坐标，不再需要额外换算。"),
           ("无需额外缩放", "disparity ≈ q，峰值位置与物理坐标一一对应。"),
           ("最小单位 1px", "输出步长约为 1 个投影仪像素，避免大步长带来的量化阶梯。")]
items_r = [("支持 s = 0.5", "半步长时 FFT 长度自动变为 2N，coord_scale 仍为 1。"),
           ("有效区固定", "只在前 proj_dim 个 bin 中搜索，负频率区不参与。"),
           ("一处公式统一", "fft_length = round(proj_dim / s) 同时覆盖两种步长，无需写两个分支。")]
ch = max(card_points_h(5.86, items_l), card_points_h(5.85, items_r)) + 0.10
card_points(s, ML, 3.22, 5.86, items_l, accent="teal", card_h=ch)
card_points(s, ML + 6.28, 3.22, 5.85, items_r, accent="amber", card_h=ch)

# ---------------- 5 代码页 ----------------
set_ctx(5, "代码卡+要点框")
s = blank(prs)
page(s, "五 · 代码落地", "FSA 参数计算与解码逻辑（代码现状）",
     "统一公式在主线代码中的落地位置：FSM_decode_core.py",
     "五 · FFT 长度策略", 41, T)
CODE1 = """def _calc_fft_params(proj_dim, frequency_step):
    if frequency_step == 1.0:
        fft_length   = proj_dim
        valid_length = proj_dim
        coord_scale  = 1.0
    else:
        fft_length   = int(round(proj_dim / frequency_step))
        valid_length = proj_dim
        coord_scale  = proj_dim / (fft_length * frequency_step)
    return fft_length, valid_length, coord_scale"""
CODE2 = """C = np.fft.fft(D_block, n=fft_length, axis=0)
C_magnitude = np.abs(C)
# 裁掉负频率区，只保留前 valid_length 个有效 bin
C_valid    = C_magnitude[:valid_length, :, :]
freq_index = np.argmax(C_valid, axis=0)
peak       = np.max(C_valid, axis=0) / fft_length
disparity  = freq_index * coord_scale"""
chh = max(code_card_h(CODE1), code_card_h(CODE2)) + 0.06
code_card(s, ML, 2.02, 5.86, chh, CODE1,
          label="① 计算 FFT 参数（真源函数）", title_color="cyan")
code_card(s, ML + 6.28, 2.02, 5.85, chh, CODE2,
          label="② 解码时裁掉负频率区", title_color="amber")
NB = ["fft_length 由 round(proj_dim / frequency_step) 统一计算，标准模式自然得到 proj_dim。",
      "coord_scale 由通用公式给出，两种步长下都约等于 1，无需为 0.5 写单独分支。",
      "argmax 之前必须裁掉负频率区，否则会输出超出投影仪范围的非法坐标。",
      "注意：_get_fft_params 返回 (fft_length, coord_scale, proj_dim, valid_length)，与模块级 _calc_fft_params 的返回顺序不同。"]
nb_h = note_box_h(CW, NB)
ny = BODY_BOTTOM - nb_h
note_box(s, ML, ny, CW, nb_h, "修改要点", NB)

# ---------------- 6 网格卡页 ----------------
set_ctx(6, "网格卡+要点框")
s = blank(prs)
page(s, "六 · 后处理", "峰值质量分级与排查顺序",
     "peak = max(|C|) / fft_length —— 与频率数 M 无关的归一化置信度",
     "六 · 工程实现 · 后处理 · 排查", 48, T)
NB2 = ["① 核对图像数量是否为 M×4，命名是否为「频率优先」顺序。",
       "② 开启 VERIFY_SEQUENCE，校验四相移序列是否完整、未错位。",
       "③ 查看 peak_*.npy 定位低质量像素，必要时过滤后再重建。",
       "④ 对异常像素设 SPECTRUM_POINTS，直接看频谱是否多峰。",
       "⑤ 启用多峰扫描 + 极线校正，处理反光、遮挡、多路径。",
       "⑥ 点云若呈台阶状，回查 fft_length 与 frequency_step 是否匹配。"]
nb2_h = note_box_h(CW, NB2)
ny2 = BODY_BOTTOM - nb2_h
note_box(s, ML, ny2, CW, nb2_h, "推荐排查顺序", NB2)
grid_cards(s, ML, 2.06, CW, ny2 - 0.20 - 2.06,
           [("peak > 1.0", "主峰明显，优先保留。"),
            ("0.5 < peak ≤ 1.0", "中等可用，需结合邻域与几何检查。"),
            ("peak ≤ 0.5", "较弱、易误匹配，建议过滤。"),
            ("阈值不可机械套用", "与亮度、曝光、反射率、归一化方式都有关。")],
           cols=2, head_size=14)

# ---------------- 7 结束页 ----------------
set_ctx(7, "结束页")
s = blank(prs)
ending(s, "频移法 FSM / FSA 原理与实现", "七章一条主线：从条纹图像到三维点云",
        [("本质", "把「相机看到的反射条纹」反解为「投影仪上的像素坐标」"),
         ("核心", "D = (I0−I2) + j(I3−I1)，沿频率轴 FFT，峰值位置即坐标"),
         ("关键", "fft_length = round(proj_dim / frequency_step)，coord_scale ≈ 1"),
         ("质量", "peak = max|C| / fft_length，配合多峰与极线校正"),
         ("下游", "disp_hor / disp_ver 进入三角化，生成三维点云")],
        "总结", 66, T)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
print("\n=== 版式自检 (%d) ===" % len(K.WARNINGS))
for w in K.WARNINGS:
    print(" ", w)
