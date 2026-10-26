# -*- coding: utf-8 -*-
"""ch_a.py — 封面 / 目录 / 阅读指南 / 第一章 / 第二章"""
from deck_kit import *
from deck_special import *


def cover_pages(d):
    s = d.newslide()
    d.nxt("封面")
    cover(s,
          [("频移法（FSM / FSA）", "white"), ("原理与实现详解", "22D3EE")],
          "从条纹图像到投影仪坐标对应关系，再到三维点云",
          "S T R U C T U R E D   L I G H T   ·   D E C O D I N G",
          formula="uₚ = q · N / (L · s)",
          note="核心目标：让 FFT 的 bin 网格与投影仪坐标网格对齐",
          chips_items=["条纹采集", "四步相移", "复信号 D[k]", "频率轴 FFT",
                       "投影仪坐标", "三维点云"],
          section="封面")

    s = d.newslide()
    p = d.nxt("讲解路线")
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
        "核心叙事主线", "讲解路线", p, d.total)

    # 阅读指南
    s = d.newslide()
    p = d.nxt("阅读指南")
    page(s, "阅读指南 · HOW TO READ", "本讲义的结构与来源说明",
         "由两套旧 PPT 合并去重而成；凡结论直接来自仓库代码的页面，右上角标注「代码现状」",
         "阅读指南", p, d.total)
    badge(s, "全篇导航", fill="teal")
    grid_cards(s, ML, 2.06, CW, 2.28,
               [("来源 A：原理详解（旧 42 页）",
                 "构成第一至四章主干：四步相移、频率轴信号、FFT 解码、数据形状。"),
                ("来源 B：统一公式详解（旧 32 页）",
                 "并入第五、六章：M/N/L 三角色、uₚ = q·N/(L·s)、量化阶梯、参数策略。"),
                ("去重处理",
                 "旧 deck 中 FFT 长度与 coord_scale 被讲了两遍，现合并为第五章一条推导线。"),
                ("新增「代码现状」章节",
                 "模块地图、真实参数表、亚像素细化、三角化与点云后处理，均以仓库代码为准。")],
               cols=2, head_size=13.5, body_size=10.5)
    NB = ["阅读顺序：第一至三章建立物理与数学直觉，第四章对齐代码，第五章给出唯一的参数公式。",
          "第六、七章偏工程与下游：内存、后处理、三角化、点云比较，可先跳过按需回看。",
          "每章首页是章节导览，正文页保持 3–5 个要点，代码只保留能说明问题的关键片段。",
          "标注「代码现状」的内容与 repo 中 FSM_*.py 一致，并给出文件与行号，可直接对照。"]
    nh = note_box_h(CW, NB)
    ny = BODY_BOTTOM - nh
    note_box(s, ML, ny, CW, nh, "使用建议", NB)


def chapter1(d):
    s = d.newslide()
    p = d.nxt("第一章 章节页")
    chapter(s, "一", "问题背景与方法目标", "Background & Objective",
            [("结构光三维重建的第一步，是把", {}), ("相机像素", {"color": "22D3EE"}),
             ("对应到", {}), ("投影仪像素", {"color": "22D3EE"}),
             ("；这件事无法从单张图直接读出。", {})],
            "问题背景与方法目标", p, d.total)

    # 1-1 为什么需要解码
    s = d.newslide()
    p = d.nxt("为什么需要解码")
    page(s, "一 · 问题背景", "结构光为什么需要「解码」",
         "投影仪投出已知条纹，相机看到的是被物体表面调制后的结果；调制过程丢失了坐标标签",
         "一 · 问题背景与方法目标", p, d.total)
    chips(s, ML, 2.06, ["投影仪投出条纹", "物体表面调制", "相机成像", "得到灰度图"],
          w=CW, h=0.56, gap=0.34, fill="fill_2", color="navy", size=12,
          arrow_color="cyan")
    cards = [("相机只记录灰度", "相机像素 (x_c, y_c) 上拿到的是一个亮度值，其中不含「它来自投影仪哪个像素」的信息。"),
             ("投影仪坐标是三角化的输入",
              "要重建空间点，必须知道两条视线：相机光线 + 投影仪光线。后者由投影仪像素 (x_p, y_p) 决定。"),
             ("所以需要主动编码",
              "FSM/FSA 通过多频率、多相移条纹，把投影仪坐标编码进相机像素随频率变化的相位里。")]
    NB = ["一句话定位：FSM 把「相机看到的反射条纹」反解为「投影仪上的像素坐标」，"
          "是几何三角化之前不可或缺的对应关系建立环节。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "本方法在链路中的位置", NB)
    card_points_fit(s, ML, 2.84, CW, BODY_BOTTOM - nh - 0.22, cards, gap=0.14,
                    accent=["teal", "amber", "teal"], head_size=12.5)

    # 1-2 两个方向
    s = d.newslide()
    p = d.nxt("两个方向")
    page(s, "一 · 方法目标", "两个方向分别恢复一个坐标",
         "条纹方向决定灰度沿投影仪的哪个轴变化，从而决定解出的是行坐标还是列坐标",
         "一 · 问题背景与方法目标", p, d.total)
    ty = table_simple(s, ML, 2.06, CW,
                 ["条纹方向", "图像目录", "条纹形态", "灰度沿…变化", "解出的投影仪坐标", "输出文件"],
                 [["水平条纹 hor", "FSM_horizontal_f0_f150", "条纹沿水平延伸", "投影仪纵向（行）",
                   "y_p（行坐标）", "disp_hor.npy"],
                  ["垂直条纹 ver", "FSM_vertical_f0_f150", "条纹沿垂直延伸", "投影仪横向（列）",
                   "x_p（列坐标）", "disp_ver.npy"]],
                 col_w=[1.15, 1.5, 1.05, 1.05, 1.15, 0.95], size=10.5,
                 head_size=10.5, row_h=0.62)
    NB = ["两个方向合起来才得到完整的二维对应关系：(x_c, y_c) → (x_p, y_p)。",
          "核对：hor 的 proj_dim 取 projector_height(1080)；ver 取 projector_width(1920)。",
          "除视差图之外还会输出 peak_hor.npy / peak_ver.npy，作为每一步的置信度图。",
          "极线约束校正需要两个方向的候选同时存在，因此必须两方向都解码。"]
    nh = note_box_h(CW, NB)
    ny = ty + 0.22
    note_box(s, ML, ny, CW, nh, "两个方向的配合", NB)
    CODE = 'proj_dim = self.proj_height if direction == "hor" else self.proj_width   # FSM_decode_core.py:910'
    code_card(s, ML, ny + nh + 0.18, CW, code_card_h(CODE, size=10.5), CODE,
              label=None, size=10.5)

    # 1-3 单张图不够
    s = d.newslide()
    p = d.nxt("单张图不够")
    page(s, "一 · 问题分析", "单张条纹图为什么不够",
         "正弦条纹在空间上是周期性的，同一个灰度值会重复出现，无法唯一定位",
         "一 · 问题背景与方法目标", p, d.total)
    NB = ["结论：必须用「一组」图像序列来消除多义性并抑制干扰。",
          "四步相移负责消除直流项 B（差分相减），多频率负责消除周期性多义性（相位随频率线性增长）。",
          "这两层序列正是 FSM/FSA 的两条设计主线，也是后续两章的主题。"]
    nh = note_box_h(CW, NB)
    ny = BODY_BOTTOM - nh
    note_box(s, ML, ny, CW, nh, "为什么必须用序列", NB)
    grid_cards(s, ML, 2.06, CW, ny - 0.18 - 2.06,
               [("周期性多义性", "同一亮度在一个周期内出现两次，且不同周期完全重复。"),
                ("环境光与暗电流", "叠加一个近似直流的背景项 B，抬高整幅灰度。"),
                ("物体颜色与反射率", "表面反射率 R 逐点不同，等价于逐点改变条纹对比度。"),
                ("曝光与传感器噪声", "曝光设置与读出噪声进一步扰动灰度，降低峰值可靠性。")],
               cols=2, head_size=13, body_size=10.5)

    # 1-4 两层序列
    s = d.newslide()
    p = d.nxt("两层序列方案")
    page(s, "一 · 总体方案", "FSM/FSA 的两层序列方案",
         "外层是频率序列，内层是四步相移序列；两层各解决一个问题",
         "一 · 问题背景与方法目标", p, d.total)
    formula_band(s, ML, 2.06, 5.86, 0.72,
                 [("外层：频率序列  k = 0, 1, …, M−1   (f_k = s·k)", {"size": 13, "color": "white"})],
                 fill="navy", accent="navy", size=13)
    formula_band(s, ML + 6.28, 2.06, 5.85, 0.72,
                 [("内层：四步相移  φ = 0°, 90°, 180°, 270°", {"size": 13, "color": "white"})],
                 fill="navy", accent="navy", size=13)
    items_l = [("四步相移层解决什么", "用差分把直流背景 B 消掉，把实数灰度变成携带相位的复数 D。"),
               ("它的产出", "每个相机像素在每个频率上得到一个复数 D[k]。")]
    items_r = [("频率序列层解决什么", "让复数 D[k] 的相位随频率线性增长，斜率正比于投影仪坐标。"),
               ("它的产出", "每个相机像素得到一条沿频率轴的一维复信号。")]
    ch = max(card_points_h(5.86, items_l), card_points_h(5.85, items_r)) + 0.04
    card_points(s, ML, 2.98, 5.86, items_l, accent="teal", head_size=13, card_h=ch)
    card_points(s, ML + 6.28, 2.98, 5.85, items_r, accent="amber", head_size=13, card_h=ch)
    NB = ["两层叠加后，每个相机像素拿到的不是「几个灰度值」，而是「一条沿频率轴的复信号」。",
          "这条信号的相位旋转速度正比于投影仪坐标 —— 这正是 FFT 能用来解码的原因。",
          "下一章先讲内层（四步相移如何造出复数），第三章再讲外层（沿频率轴做 FFT）。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "为什么是两层", NB)


def chapter2(d):
    s = d.newslide()
    p = d.nxt("第二章 章节页")
    chapter(s, "二", "四步相移与复信号构造", "Four-Step Phase Shift & Complex Signal",
            [("如何用", {}), ("四步相移差分", {"color": "EF4444"}),
             ("抵消直流亮度干扰，并把", {}), ("实数灰度", {"color": "EF4444"}),
             ("转化为", {}), ("携带相位的复数 D", {"color": "EF4444"}), ("。", {})],
            "四步相移与复信号构造", p, d.total)

    # 2-1 图像组织
    s = d.newslide()
    p = d.nxt("图像组织")
    page(s, "二 · 图像组织", "图像序列的组织与命名规则",
         "同一方向上，图像按「频率优先」排列：每 4 张一组，组内是四步相移",
         "二 · 四步相移与复信号构造", p, d.total)
    rows = [["0, 1, 2, 3", "频率 f₀ = 0", "0° / 90° / 180° / 270°"],
            ["4, 5, 6, 7", "频率 f₁ = s", "0° / 90° / 180° / 270°"],
            ["8, 9, 10, 11", "频率 f₂ = 2s", "0° / 90° / 180° / 270°"],
            ["…", "…", "…"],
            ["4k, 4k+1, 4k+2, 4k+3", "频率 f_k = s·k", "0° / 90° / 180° / 270°"]]
    table_simple(s, ML, 2.06, 5.86, ["图像编号", "对应频率", "组内相移"],
                 rows, col_w=[1.3, 1.1, 1.5], size=10.5, row_h=0.40)
    code_card(s, ML + 6.28, 2.06, 5.85, 2.30, """# 频率优先排列 → 用步长 4 的切片还原四路相位
I0 = images[0::4]   # 0°   → (M, Hc, Wc)
I1 = images[1::4]   # 90°  → (M, Hc, Wc)
I2 = images[2::4]   # 180° → (M, Hc, Wc)
I3 = images[3::4]   # 270° → (M, Hc, Wc)
# 总图像数 = M × 4  (M = NUM_FREQUENCIES)""",
              label="切片还原（FSM_decode_core.py:1086-1089）", title_color="cyan")
    NB = ["命名规则写成「频率优先」的原因是：解码时只需等步长切片，不需要任何排序或索引表。",
          "标准模式 s = 1.0 时 M = 151（频率 0…150），每方向 604 张；半步模式 s = 0.5 时 M = 301，每方向 1204 张。",
          "若命名顺序错乱，四路切片会把不同频率的相位混在一起，峰值会整体失效 —— 因此加载后要开 VERIFY_SEQUENCE 校验。"]
    nh = note_box_h(CW, NB)
    ny = BODY_BOTTOM - nh
    note_box(s, ML, ny, CW, nh, "命名规则与校验", NB)

    # 2-2 四步相移
    s = d.newslide()
    p = d.nxt("四步相移")
    page(s, "二 · 四步相移", "四步相移：四个已知的相位偏移",
         "对同一个频率，投影仪先后投出 0°、90°、180°、270° 四个相位的条纹",
         "二 · 四步相移与复信号构造", p, d.total)
    ty = table_simple(s, ML, 2.06, CW,
                 ["相移", "投影条纹", "理想灰度", "捕获变量", "作用"],
                 [["0°", "sin(θ)", "B + R·cos θ", "I0", "与 180° 相减消去 B"],
                  ["90°", "sin(θ+90°)", "B + R·sin θ", "I1", "与 270° 相减消去 B"],
                  ["180°", "sin(θ+180°)", "B − R·cos θ", "I2", "与 0° 相减消去 B"],
                  ["270°", "sin(θ+270°)", "B − R·sin θ", "I3", "与 90° 相减消去 B"]],
                 col_w=[0.75, 1.25, 1.35, 0.95, 1.85], size=11, head_size=11,
                 row_h=0.52)
    by = ty + 0.18
    formula_band(s, ML, by, CW, 0.68,
                 [("Dc = I0 − I2 ≈ 2R·cos θ        Ds = I3 − I1 ≈ 2R·sin θ        D = Dc + j·Ds",
                   {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    NB = ["四步而不是两步：两步只能得到 cos 项，无法分离 sin 项；四步才能同时拿到实部与虚部。",
          "四次采集之间物体、相机、投影仪都不动，因此 B 与 R 在四张图中视为同一常数，相减才能干净地消去 B。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, by + 0.68 + 0.16, CW, nh, "为什么是四步", NB)

    # 2-3 差分消直流
    s = d.newslide()
    p = d.nxt("差分消直流")
    page(s, "二 · 差分构造", "差分如何把直流项彻底消掉",
         "相减的两个相位相差 180°，直流背景 B 完全抵消，只剩余弦/正弦分量",
         "二 · 四步相移与复信号构造", p, d.total)
    code_card(s, ML, 2.06, 5.86, 2.42, """# ① 理想灰度模型      B = 环境光 + 暗电流 + 背景亮度
I0 = B + R*cos(theta)      # 0°
I2 = B - R*cos(theta)      # 180°
I1 = B - R*sin(theta)      # 90°
I3 = B + R*sin(theta)      # 270°

# ② 相差 180° 相减 → B 被完全消去
Dc = I0 - I2   # = 2R*cos(theta)
Ds = I3 - I1   # = 2R*sin(theta)""",
              label="① 从灰度到差分", title_color="cyan", size=10.5)
    code_card(s, ML + 6.28, 2.06, 5.85, 2.42, """# ③ 合成复数 D
D = Dc + 1j * Ds        # = 2R * (cos + j*sin)
                        # = 2R * exp(j*theta)

# 代码实现（FSM_decode_core.py:1094-1095）
Dc = I0 - I2
Ds = I3 - I1
D  = Dc + 1j * Ds""",
              label="② 合成复数 D", title_color="amber", size=10.5)
    NB = ["直流项 B 被完全消去 —— 这正是四步相移相对单张图的核心优势。",
          "剩下的幅值 2R 只反映反射强度，不含坐标信息；真正携带投影仪坐标的是相位 θ。",
          "信息载体因此从「绝对灰度」变成「复数 D 的相位」，后续所有推导都建立在这条相位上。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, 4.66, CW, nh, "关键结论", NB)

    # 2-4 复数与相位
    s = d.newslide()
    p = d.nxt("复数与相位")
    page(s, "二 · 复数语义", "复数 D 的相位与「旋转」是什么意思",
         "把 D 画在复平面上：长度是幅值，方向是相位；「旋转」指相位随频率编号 k 增加而变化",
         "二 · 四步相移与复信号构造", p, d.total)
    cards = [("D 是一个箭头", "复平面上的向量 D = |D|·exp(jθ)：长度 |D| 表示反射强度，方向角 θ 表示相位。"),
             ("幅值不含坐标信息", "|D| = 2R，只随表面反射率变化，与投影仪坐标无关，因此解码不看幅值。"),
             ("相位才是坐标载体", "θ 才是被条纹「写入」的量，它随频率编号 k 递增。"),
             ("「旋转」是相位的旋转", "指相位随 k 增加而连续变化，不是物体旋转、也不是图像旋转。")]
    ch = card_points_h(CW, cards) + 0.06
    card_points(s, ML, 2.06, CW, cards, accent=["teal", "amber", "teal", "amber"],
                card_h=ch, head_size=13, body_size=10.5)
    formula_band(s, ML, 2.06 + ch * 4 + 0.16 * 3 + 0.20, CW, 0.66,
                 [("相邻频率间的相位增量   Δθ = θ[k+1] − θ[k] = 2π·s·uₚ / N   —— 这就是「旋转速度」",
                   {"size": 13.5, "color": "white"})],
                 fill="navy", accent="navy", size=13.5)

    # 2-5 投影强度模型
    s = d.newslide()
    p = d.nxt("投影强度模型")
    page(s, "二 · 物理模型", "从投影条纹到相机灰度的完整模型",
         "把投影仪投出的正弦、物体表面反射、相机成像三步串成一个公式",
         "二 · 四步相移与复信号构造", p, d.total)
    formula_band(s, ML, 2.06, 5.86, 0.80,
                 [("I_proj(u, k, φ) = A · sin(2π·f_k·u + φ)", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    formula_band(s, ML + 6.28, 2.06, 5.85, 0.80,
                 [("I_cam(x_c, y_c, k, φ) = B + R · A · sin(2π·f_k·uₚ + φ)",
                   {"size": 14, "color": "white"})],
                 fill="navy", accent="navy", size=14)
    table_simple(s, ML, 3.04, CW,
                 ["符号", "含义", "在本方法中的地位"],
                 [["A", "投影仪条纹的振幅", "常量，投影系统决定"],
                  ["f_k = s·k", "第 k 个频率", "外层序列自变量，决定了相位旋转速度"],
                  ["u", "投影仪上的物理坐标", "投影仪自变量的连续位置"],
                  ["uₚ", "该相机像素对应的投影仪坐标", "★ 解码的最终目标"],
                  ["B(x_c, y_c)", "环境光 + 暗电流 + 背景亮度", "近似直流项，四步差分消去"],
                  ["R(x_c, y_c)", "表面反射率与局部亮度缩放", "只影响幅值，不影响相位"]],
                 col_w=[0.9, 2.0, 2.4], size=10.5, head_size=10.5, row_h=0.44)


def build(d):
    cover_pages(d)
    chapter1(d)
    chapter2(d)
