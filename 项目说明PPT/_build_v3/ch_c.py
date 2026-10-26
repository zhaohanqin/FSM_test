# -*- coding: utf-8 -*-
"""ch_c.py — 第五章 / 第六章"""
from deck_kit import *
from deck_special import *


def chapter5(d):
    s = d.newslide()
    p = d.nxt("第五章 章节页")
    chapter(s, "五", "FFT 长度策略：统一公式", "FFT Length & Unified Formula",
            [("全篇只有一条参数公式：", {}), ("uₚ = q · N / (L · s)", {"color": "22D3EE"}),
             ("。选 L = N/s，峰值 bin 编号就直接等于投影仪坐标。", {})],
            "FFT 长度策略：统一公式", p, d.total)

    # 5-1 M/N/L
    s = d.newslide()
    p = d.nxt("MNL 三符号")
    page(s, "五 · 符号澄清", "M / N / L 三个符号不能混",
         "三个字母都在公式里出现，但物理含义完全不同；混用是这一章绝大多数误区的根源",
         "五 · FFT 长度策略：统一公式", p, d.total)
    table_simple(s, ML, 2.06, CW,
                 ["符号", "名称", "物理含义", "代码对应", "典型取值"],
                 [["M", "num_frequencies", "实际采集的频率个数（真实观测样本数）", "NUM_FREQUENCIES", "151 或 301"],
                  ["N", "proj_dim", "投影仪在该方向的分辨率（坐标总数）", "PROJECTOR_WIDTH / HEIGHT", "1920 或 1080"],
                  ["L", "fft_length", "FFT 点数 = 候选 bin 的个数", "fft_length", "N 或 2N"],
                  ["s", "frequency_step", "相邻频率之间的步长", "FREQUENCY_STEP", "1.0 或 0.5"],
                  ["q", "freq_index", "峰值所在的那个 bin 编号", "np.argmax(...)", "0 … L−1"]],
                 col_w=[0.5, 1.4, 3.0, 1.6, 1.0], size=10.5, head_size=10.5,
                 mono_cols=(0, 3), row_h=0.44)
    NB = ["M 决定「测到了多少信息」：M 越大，正确峰的同相叠加项越多，峰越尖、越抗噪。",
          "L 决定「在多少个候选上评价」：L 越大，bin 网格越密，但不会增加任何新信息。",
          "N 来自投影仪硬件，是相位公式的分母 —— 它和 M 没有任何关系，这是最常被写错的一处。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "记住这三句话就够了", NB)

    # 5-2 统一公式
    s = d.newslide()
    p = d.nxt("统一公式推导")
    page(s, "五 · 核心推导", "统一公式：从匹配条件到坐标换算",
         "让 FFT 的某个 bin 恰好对上信号的相位斜率，就得到 uₚ 的表达式",
         "五 · FFT 长度策略：统一公式", p, d.total)
    table_simple(s, ML, 2.06, CW,
                 ["步骤", "表达式", "说明"],
                 [["① 理想复信号", "D[k] ≈ A · exp( j·2π·s·k·uₚ / N )", "相位随 k 线性增长，斜率为 α = s·uₚ / N"],
                  ["② FFT 的匹配斜率", "β_q = q / L", "FFT 用候选速率反向旋转这条信号"],
                  ["③ 峰值条件（斜率相等）", "q / L = s · uₚ / N", "残余相位为零 ⇒ M 项同相叠加"],
                  ["④ 移项解出坐标", "q · N = L · s · uₚ", "纯代数整理"],
                  ["⑤ 统一公式", "uₚ = q · N / (L · s)", "工程写法 disparity = q · coord_scale"]],
                 col_w=[1.65, 3.0, 3.55], size=10.5, head_size=10.5,
                 mono_cols=(1,), row_h=0.46)
    formula_band(s, ML, 4.86, CW, 0.76,
                 [("希望 q 本身就等于 uₚ   ⇒   coord_scale = N / (L·s) = 1   ⇒   L = N / s",
                   {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    NB = ["这条公式同时解释了标准模式（s = 1.0）与半步模式（s = 0.5），因此代码里不需要写两个分支。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "顺理成章的下一步", NB)

    # 5-3 匹配条件的几何含义
    s = d.newslide()
    p = d.nxt("匹配条件几何")
    page(s, "五 · 几何直觉", "匹配条件在几何上意味着什么",
         "信号在复平面上以 α 的速率旋转；FFT 用 β_q 的速率反向旋转去「对齐」它",
         "五 · FFT 长度策略：统一公式", p, d.total)
    grid_cards(s, ML, 2.06, CW, 2.20,
               [("α 小 → 慢旋转", "uₚ 小，相邻频率间相位增量小，向量几乎不转向。"),
                ("α 中 → 中等旋转", "uₚ 居中，M 项向量沿圆弧均匀铺开。"),
                ("α 大 → 快旋转", "uₚ 大，相位增量大；当 α 超过 1/2 会发生周期性折叠。"),
                ("匹配 α = β_q", "反向旋转后全部向量指向同一方向，叠加长度达到 M·|A|。")],
               cols=2, head_size=13, body_size=10.5)
    code_card(s, ML, 4.44, CW, min(1.94, BODY_BOTTOM - 4.44),
              """D[k] = A · exp( j · 2π · α · k )        相位随 k 线性增长，斜率 α = s·uₚ/N
C[q] = Σ_k D[k] · exp( −j·2π · β_q · k )  用候选速率 β_q = q/L 反向旋转
     = A · Σ_k exp( j·2π·(α − β_q)·k )     残余斜率 (α − β_q) 决定叠加结果
     →  α = β_q 时残余为 0，|C| 达到最大值""",
              label="同一件事的两种写法", title_color="cyan", size=10.5)

    # 5-4 推荐策略
    s = d.newslide()
    p = d.nxt("推荐策略")
    page(s, "五 · 推荐策略", "为什么推荐让 coord_scale = 1",
         "最理想的工程状态：FFT 峰值 bin 编号 q 本身就是投影仪坐标",
         "五 · FFT 长度策略：统一公式", p, d.total)
    formula_band(s, ML, 2.06, 3.62, 0.78,
                 [("coord_scale = N / (L·s)", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    arrow_h(s, ML + 3.76, 2.45, 0.62, "amber", 1.75)
    formula_band(s, ML + 4.52, 2.06, 3.10, 0.78,
                 [("coord_scale = 1", {"size": 15, "color": "F29F05"})],
                 fill="white", accent="amber", size=15)
    arrow_h(s, ML + 7.76, 2.45, 0.62, "amber", 1.75)
    formula_band(s, ML + 8.52, 2.06, 3.01, 0.78,
                 [("L = N / s", {"size": 15, "color": "cyan_br"})],
                 fill="navy", accent="navy", size=15)
    items_l = [("q 的语义清楚", "argmax 输出的 bin 编号 q 直接就是投影仪坐标，不需要额外换算。"),
               ("无需额外缩放", "disparity ≈ q，峰值位置与物理坐标一一对应。"),
               ("输出步长 1px", "坐标最小单位约为 1 个投影仪像素，避免大步长量化。")]
    items_r = [("支持 s = 0.5", "半步长时 FFT 长度自动变为 2N，coord_scale 仍为 1。"),
               ("有效区固定", "只在前 N 个 bin 中搜索，负频率区不参与决策。"),
               ("一处公式统一", "fft_length = round(N/s) 同时覆盖两种步长，无需两个分支。")]
    ch = max(card_points_h(5.86, items_l), card_points_h(5.85, items_r)) + 0.08
    card_points(s, ML, 3.10, 5.86, items_l, accent="teal", head_size=12.5, card_h=ch)
    card_points(s, ML + 6.28, 3.10, 5.85, items_r, accent="amber", head_size=12.5, card_h=ch)

    # 5-5 情况 A
    s = d.newslide()
    p = d.nxt("情况A 标准模式")
    page(s, "五 · 情况 A", "情况 A：s = 1.0，L = N",
         "标准模式的参数组合，也是最容易理解的一种",
         "五 · FFT 长度策略：统一公式", p, d.total)
    stat_row(s, ML, 2.06, CW,
             [("s = 1.0", "频率步长为 1，频率取 0, 1, …, M−1"),
              ("L = N", "fft_length = round(N / 1.0) = proj_dim"),
              ("coord_scale = 1", "N / (L·s) = N / (N·1) = 1")],
             accent="teal", h=1.20, value_size=19)
    ty = table_simple(s, ML, 3.42, CW,
                 ["方向", "N（proj_dim）", "L = fft_length", "valid_length", "输出的坐标"],
                 [["水平条纹 hor", "1080（投影仪高）", "1080", "1080", "投影仪行坐标 yₚ"],
                  ["垂直条纹 ver", "1920（投影仪宽）", "1920", "1920", "投影仪列坐标 xₚ"]],
                 col_w=[1.3, 1.5, 1.05, 1.15, 1.75], size=11, head_size=10.5,
                 row_h=0.52)
    NB = ["L = N 时 FFT 的 bin 0 … N−1 恰好与投影仪像素编号 0 … N−1 一一对应，q 就是坐标。",
          "结果是整数像素级定位，理论量化误差约 ±0.5 像素。",
          "这是最容易讲清、也最容易验证正确性的一种配置，推荐作为默认。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, ty + 0.22, CW, nh, "为什么这种配置最舒服", NB)

    # 5-6 情况 B
    s = d.newslide()
    p = d.nxt("情况B 半步长")
    page(s, "五 · 情况 B", "情况 B：s = 0.5，L = 2N",
         "半步长时 FFT 长度自动翻倍，但有效区仍然是前 N 个 bin",
         "五 · FFT 长度策略：统一公式", p, d.total)
    badge(s, "主线当前配置", fill="teal")
    formula_band(s, ML, 2.06, 5.86, 0.76,
                 [("L = round(N / 0.5) = 2N = 3840（ver）", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    formula_band(s, ML + 6.28, 2.06, 5.85, 0.76,
                 [("coord_scale = N / (2N · 0.5) = 1", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    ty = table_simple(s, ML, 2.98, CW,
                 ["bin 区间", "数量", "物理含义", "是否参与 argmax"],
                 [["bin 0 … N−1", "N = 1920", "有效投影仪坐标 0 … 1919", "✔ 参与"],
                  ["bin N … 2N−1", "N = 1920", "负频率 / 周期重复区，物理上无效", "✘ 必须裁掉"]],
                 col_w=[1.35, 1.0, 3.0, 1.3], size=11, head_size=10.5, row_h=0.56)
    CODE = """C       = np.fft.fft(D_block, n=fft_length, axis=0)   # (2N, bh, bw)
C_valid = np.abs(C)[:valid_length]                  # 只取前 N 个 bin，valid_length = proj_dim
idx     = np.argmax(C_valid, axis=0)                # 在有效区内搜索
disparity = idx * coord_scale                       # coord_scale = 1"""
    cy = ty + 0.24
    code_card(s, ML, cy, CW, min(code_card_h(CODE, size=11), BODY_BOTTOM - cy), CODE,
              label="裁掉负频率区是半步长下的强制性步骤", title_color="amber", size=11)

    # 5-7 量化阶梯
    s = d.newslide()
    p = d.nxt("量化阶梯")
    page(s, "五 · 情况 C（反例）", "情况 C：旧方案 L = M 带来的量化阶梯",
         "把 FFT 长度设成频率个数 M，bin 就不再对应投影仪像素，坐标被压进很粗的离散格",
         "五 · FFT 长度策略：统一公式", p, d.total)
    badge(s, "历史问题复盘", fill="red")
    stat_row(s, ML, 2.06, CW,
             [("L = M = 340", "旧配置：FFT 长度等于采集频率数"),
              ("Δu = N/M ≈ 5.65 px", "相邻可选坐标之间的最小间隔（N = 1920）"),
              ("最大误差 ≈ ±2.65 px", "理想信号下的实测偏差上限")],
             accent=["red", "amber", "amber"], h=1.20, value_size=18)
    ty = table_simple(s, ML, 3.30, CW,
                 ["真实投影仪坐标", "L = M 修复前（旧）", "L = N 修复后（推荐）", "说明"],
                 [["100", "101.65（误差 1.65）", "100（误差 0）", "被压到最近的粗 bin"],
                  ["500", "502.59（误差 2.59）", "500（误差 0）", "同上"],
                  ["957", "954.35（误差 2.65）", "957（误差 0）", "误差最大的一档"],
                  ["1500", "1502.12（误差 2.12）", "1500（误差 0）", "同上"]],
                 col_w=[1.2, 1.6, 1.6, 1.6], size=10.5, head_size=10.5, row_h=0.40)
    NB = ["量化阶梯的后果：相邻相机像素解码成同一个投影仪坐标 → 对应关系不连续 → 三角化出现条带、断层或层状结构。",
          "根本原因是 q 的取值太少，不是乘法缩放不够 —— 加一个系数改不了 bin 的疏密。",
          "半步长下更严重：M = 301 时 Δu ≈ 6.38 像素（ver 方向），hor 方向约 3.59 像素。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, ty + 0.20, CW, nh, "为什么会变成点云上的台阶", NB)

    # 5-8 零填充
    s = d.newslide()
    p = d.nxt("零填充本质")
    page(s, "五 · 零填充", "补零到底做了什么，没做什么",
         "np.fft.fft(D, n=L) 在 M < L 时自动补零；它改变的是评价网格，不是信息量",
         "五 · FFT 长度策略：统一公式", p, d.total)
    code_card(s, ML, 2.06, 5.86, 1.86, """# numpy 的补零发生在 FFT 计算内部
D_padded = [ D[0], D[1], …, D[M-1], 0, 0, …, 0 ]
             └─ M 个真实观测 ─┘  └─ L−M 个零 ─┘

C[q] = Σ_{k=0}^{L-1} D_padded[k] · exp(−j·2π·q·k / L)
     ≡ Σ_{k=0}^{M-1} D[k]       · exp(−j·2π·q·k / L)""",
              label="np.fft.fft 的补零语义", title_color="cyan", size=10.5)
    items = [("M 决定求和项数", "真正参与累加的只有 M 个真实观测，补进去的零不贡献任何信号。"),
             ("L 决定评价位置", "在 q = 0 … L−1 这 L 个候选频率上求值，网格间隔为 1/L。"),
             ("等价于频谱重采样", "补零 = 对隐式连续频谱在更密的点上重新采样（sinc 插值后的采样）。"),
             ("不提升信噪比", "也不创造新的高频信息；信息量始终由 M 决定。"),
             ("但确实提升定位精度", "对于「峰值落在哪个坐标」这件事，网格变密直接消除了粗量化。")]
    ch = card_points_h(5.85, items) + 0.04
    card_points(s, ML + 6.28, 2.06, 5.85, items, accent=["teal", "amber", "teal", "amber", "teal"],
                head_size=12, body_size=10, card_h=ch)

    # 5-9 代码现状
    s = d.newslide()
    p = d.nxt("统一公式代码现状")
    page(s, "五 · 代码落地", "统一公式在主线代码中的落地位置",
         "两处实现、一处重复：_calc_fft_params / _get_fft_params / FSM_visualize 的 _get_fft_length",
         "五 · FFT 长度策略：统一公式", p, d.total)
    badge(s, "代码现状")
    code_card(s, ML, 2.06, 5.86, 2.70, """# 模块级真源（FSM_decode_core.py:394-438）
def _calc_fft_params(proj_dim, frequency_step):
    if frequency_step == 1.0:
        fft_length   = proj_dim
        valid_length = proj_dim
        coord_scale  = 1.0
    else:
        fft_length   = int(round(proj_dim / frequency_step))
        valid_length = proj_dim
        coord_scale  = proj_dim / (fft_length * frequency_step)
    return fft_length, valid_length, coord_scale""",
              label="① 文档串已写明公式（源码注释）", title_color="cyan", size=9.5)
    code_card(s, ML + 6.28, 2.06, 5.85, 2.70, """# 类方法（FSM_decode_core.py:894-912）
def _get_fft_params(self, direction, num_frequencies):
    proj_dim = self.proj_height if direction == "hor" \\
               else self.proj_width        # :910
    ...
    return fft_length, coord_scale, proj_dim, valid_length   # :912
    #     ↑ 返回顺序与模块级函数不同：注意别接错

# 负频率裁切（没有字面的 C[:proj_dim]）
C_valid = C_magnitude[:valid_length, :, :]   # :1148""",
              label="② 类方法：返回顺序不同", title_color="amber", size=9.5)
    NB = ["源码注释 FSM_decode_core.py:401-403 原文就写着推导：q/L = s·u/N ⇒ u = q·proj_dim/(L·s)，令 coord_scale = 1 则 L = proj_dim/s。",
          "两份说明文档（FSM_method_explanation.md、modification_analysis.md）记录的仍是旧的「fft_length = M、coord_scale = proj_dim/M」两分支写法，已经滞后于代码。",
          "FSM_visualize.py:95-104 复制了一份等价逻辑，改参数策略时需要同步两处。"]
    nh = note_box_h(CW, NB)
    ny = BODY_BOTTOM - nh
    note_box(s, ML, ny, CW, nh, "阅读代码时的三个提醒", NB)

    # 5-10 有效区边界
    s = d.newslide()
    p = d.nxt("有效区边界")
    page(s, "五 · 边界条件", "统一公式成立的两个前提",
         "公式本身很短，但有两条隐含前提，破坏其中任何一条都会得到越界或错位的结果",
         "五 · FFT 长度策略：统一公式", p, d.total)
    items = [("前提一：valid_length = proj_dim",
              "有效坐标只有 0 … N−1 共 N 个。若不在 argmax 前裁掉后面 2N−N 个负频率 bin，"
              "就会输出超过投影仪范围的非法坐标（源码注释 FSM_decode_core.py:861-865 专门说明了这一点）。"),
             ("前提二：fft_length = round(N / s)",
              "步长改变的是「k 每加 1 相位转多少」，因此 FFT 长度必须同步改变。"
              "若沿用 step = 1.0 时的 L = N 去跑 s = 0.5，坐标会系统性偏移一倍。"),
             ("边界处的两个工程细节",
              "① fft_length 用 int(round(...)) 取整，coord_scale 用通式回算而不是硬写 1.0，"
              "这样 proj_dim 不能被步长整除时也不会引入系统偏差。"
              "② 亚像素质心细化时，邻域索引用 % fft_length 环绕取值，与搜索窗口 valid_length 不是同一个长度。")]
    ch = card_points_h(CW, items) + 0.08
    card_points(s, ML, 2.06, CW, items, accent=["teal", "teal", "amber"],
                head_size=13, body_size=10.5, card_h=ch)
    NB = ["结论：FFT 长度不是「越大越好」的自由参数，而是由 N 与 s 唯一确定的量。超过 N/s 只是对同一条频谱做更细的插值，"
          "既不会提高精度，也不会提高抗噪能力，只会线性增加计算量。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "一句话总结", NB)


def chapter6(d):
    s = d.newslide()
    p = d.nxt("第六章 章节页")
    chapter(s, "六", "工程实现 · 后处理 · 排查", "Engineering, Post-processing & Debugging",
            [("算得动、看得清、修得对：", {}), ("计算量与内存", {"color": "22D3EE"}),
             ("、多峰与极线后处理、以及出问题时的排查顺序。", {})],
            "工程实现 · 后处理 · 排查", p, d.total)

    # 6-1 计算量
    s = d.newslide()
    p = d.nxt("计算量与内存")
    page(s, "六 · 复杂度", "计算量与内存峰值从哪里来",
         "整条链路的开销集中在 FFT 与峰值搜索两步；分块的意义是控制内存，不是减少计算",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    table_simple(s, ML, 2.06, CW,
                 ["阶段", "复杂度", "说明"],
                 [["load_images", "O(M·4·Hc·Wc)", "读盘 + 解码 PNG，首次运行的主要耗时"],
                  ["构造 I0–I3 与 D", "O(M·Hc·Wc)", "切片与复数加法，内存中多一份复数体 D"],
                  ["沿频率轴 FFT", "O(Hc·Wc·L·log L)", "★ 主导项；L = 2N 时约为 L = N 的 2 倍多"],
                  ["argmax / max", "O(Hc·Wc·L)", "扫描有效频谱，与 FFT 同阶但常数小"],
                  ["save_results", "O(Hc·Wc)", "两张 float32 图写盘，可忽略"]],
                 col_w=[1.6, 1.7, 3.5], size=10.5, head_size=10.5,
                 mono_cols=(0, 1), row_h=0.44)
    NB = ["内存峰值 ≈ D 的复数体 + 一个块的 C 与 |C|（三份同数量级数组）。",
          "block_size = 100 时单块频谱规模约 L × 100 × 100，L = 3840 时约 1.5 亿个复数 —— 这是峰值的主要来源。",
          "step 从 1.0 改到 0.5 会让 L 翻倍，因而计算量与内存都随之上升（代码注释也提示约 2 倍）。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "取舍建议", NB)

    # 6-2 多峰成因
    s = d.newslide()
    p = d.nxt("多峰成因")
    page(s, "六 · 多峰问题", "为什么频谱上会出现第二个峰",
         "一个相机像素同时接收到多条光路时，频率轴信号变成多分量叠加，频谱就出现多峰",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    grid_cards(s, ML, 2.06, CW, 2.34,
               [("高反光 / 镜面反射", "同一像素同时看到直接反射与邻近面的二次反射，混合出两个不同投影仪坐标。"),
                ("遮挡边界", "像素落在前景与背景交界处，两次投影的条纹在同一像素上叠加。"),
                ("多路径干扰", "光线经多个面折返后进入同一像素，等价于多个 uₚ 的叠加。"),
                ("噪声削弱主峰优势", "信噪比下降时主峰相对旁瓣的优势减小，次峰更容易被判为主峰。")],
               cols=2, head_size=13, body_size=10.5)
    NB = ["后果：argmax 一旦选错峰，该像素的对应关系就整体偏移，三角化得到远离表面的飞点。",
          "处理思路不是「提高分辨率」，而是「把候选都找出来，再用几何约束挑选」—— 这就是多峰检测 + 极线校正的组合。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "为什么会出错", NB)

    # 6-3 多峰检测算法
    s = d.newslide()
    p = d.nxt("多峰检测")
    page(s, "六 · 多峰检测", "MultiPeakDetector.find_peaks 的三步",
         "阈值过滤 → 局部极大判定 → 近峰按幅值加权合并",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    badge(s, "代码现状")
    CODE = """def find_peaks(spectrum, limit, merge_distance=5):
    peaks = []
    for i, value in enumerate(spectrum):
        if value < limit:                      # ① 阈值去弱峰
            continue
        l_ok = (i == 0) or (spectrum[i] > spectrum[i-1])
        r_ok = (i == len(spectrum)-1) or (spectrum[i] > spectrum[i+1])
        if not (l_ok and r_ok):                # ② 局部极大值
            continue
        if peaks and i - peaks[-1][0] <= merge_distance:
            pos, val = peaks[-1]               # ③ 近峰合并
            ki = value / (value + val)
            peaks[-1] = (ki*i + (1-ki)*pos, max(value, val))
        else:
            peaks.append((i, value))
    return peaks"""
    items = [("输出是一组候选坐标", "不再只给一个 argmax 结果，而是若干 (位置, 幅值) 对。"),
             ("合并公式是幅值加权", "ki = value/(value+val)；new_pos = ki·i + (1−ki)·pos。"),
             ("merge_distance = 5", "相邻 5 个 bin 内的峰视为同一个峰，避免把肩部当独立峰。"),
             ("随后还有一层几何筛选", "候选组合交给极线校正，用标定几何挑出物理自洽的一对。")]
    chh = code_card_h(CODE, size=9)
    ch = card_points_h(5.85, items, head_size=12.5, body_size=10.5)
    tot = max(chh, 4 * ch + 3 * 0.16, 2.60)
    code_card(s, ML, 2.06, 5.86, tot, CODE,
              label="FSM_decode_core.py:455-494 摘录（merge_distance 默认 5）",
              title_color="cyan", size=9)
    card_points(s, ML + 6.28, 2.06, 5.85, items, accent="amber",
                head_size=12.5, body_size=10.5, card_h=(tot - 3 * 0.16) / 4)

    # 6-4 全图多峰扫描
    s = d.newslide()
    p = d.nxt("多峰全图扫描")
    page(s, "六 · 多峰扫描", "全图多峰扫描：先把可疑像素找出来",
         "scan_image_for_multi_peaks 在整幅图上找「主峰之外还有明显次峰」的像素并排序",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    badge(s, "代码现状")
    code_card(s, ML, 2.06, CW, 2.20, """primary_idx = np.argmax(C_row, axis=0)                    # 主峰位置      :622
dist = np.abs(freq_indices - primary_idx[None, :])
dist = np.minimum(dist, valid_length - dist)               # 圆周距离      :627-628
C_masked = C_row.copy()
C_masked[dist <= suppress_radius] = 0                      # 抑制主峰邻域  :632（默认 ±30 bins）
# 之后的局部极大若仍超过阈值，就记为「多峰」，输出字段：
#   x / y / primary_pos / primary_val / secondary_pos / secondary_val / ratio
# 按 secondary_val 降序，最多输出 max_report = 200 条                       :658-662""",
              label="FSM_decode_core.py:622-662 摘录", title_color="cyan", size=10)
    items = [("先抑制主峰邻域", "在主峰 ±30 个 bin 内清零，避免把同一个峰的裙边当作次峰。"),
             ("再找超过阈值的局部极大", "阈值由 MULTI_PEAK_SCAN_LIMIT 控制（默认 0.5）。"),
             ("ratio 是判据的核心", "次峰 / 主峰的幅值比越大，该像素越可能是真正的多路径混合。"),
             ("输出上限 200 条", "按次峰幅值降序，用作核查清单或极线校正的输入。")]
    grid_cards(s, ML, 4.36, CW, BODY_BOTTOM - 4.36, items,
               cols=2, head_size=13, body_size=10.5, accent=["teal", "teal", "amber", "amber"])

    # 6-5 极线原理
    s = d.newslide()
    p = d.nxt("极线约束原理")
    page(s, "六 · 极线约束", "极线约束：用标定几何给候选配对",
         "一个相机像素在投影仪图像上对应一条直线；正确的那对 (行, 列) 必然落在它附近",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    code_card(s, ML, 2.06, 5.86, 2.82, """# ① 投影矩阵（FSM_decode_core.py:757-763）
Pc[:, :3] = self.Rc;  Pc[:, -1] = self.tc;  self.Pc = self.Kc @ Pc
Pp[:, :3] = self.Rp;  Pp[:, -1] = self.tp;  self.Pp = self.Kp @ Pp

# ② 相机光心与极点                     :765-767
self.camera_center = -self.Rc.T @ self.tc
ep = self.Kp @ (self.Rp @ self.camera_center + self.tp)
self.epipole = ep / ep[-1]

# ③ 给定相机像素 xc，求投影仪图像上的极线   :779-782
xp = self.Pp @ np.linalg.pinv(self.Pc) @ xc
lp = np.cross(self.epipole, xp)     # lp = [a, b, c]""",
              label="EpipolarCorrector 的三段计算", title_color="cyan", size=9.5)
    items = [("为什么它能救多峰", "多峰给出若干行候选与若干列候选；极线只留下几何上一致的那一组。"),
             ("依赖标定精度", "Pc / Pp 来自标定文件，标定误差会直接反映为极线偏置。"),
             ("是可选后处理", "默认 ENABLE_EPIPOLAR_CORRECTION = False，不替代 argmax 主解码。"),
             ("要求两方向都解", "候选来自 hor 的行与 ver 的列，因此 DIRECTION 必须是 both。")]
    ch = card_points_h(5.85, items) + 0.04
    card_points(s, ML + 6.28, 2.06, 5.85, items, accent="teal",
                head_size=12, body_size=10, card_h=ch)

    # 6-6 极线实现
    s = d.newslide()
    p = d.nxt("极线校正实现")
    page(s, "六 · 极线校正", "候选配对与距离判据",
         "在 (hor 候选 × ver 候选) 的笛卡尔积里，选到极线距离最小的那一对",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    badge(s, "代码现状")
    code_card(s, ML, 2.06, CW, 1.86, """pys = np.repeat(pk_hor, len(pk_ver))          # hor 候选行坐标   :805
pxs = np.tile(pk_ver,  len(pk_hor))           # ver 候选列坐标   :806
proj_points = np.column_stack([pxs, pys]).astype(np.float64)
undis = cv2.undistortPoints(proj_points, self.Kp, self.dp, P=self.Kp)
distances = np.abs(lp[0]*undis[:,0] + lp[1]*undis[:,1] + lp[2]) / np.sqrt(lp[0]**2 + lp[1]**2)
best = int(np.argmin(distances))               # 点到直线的距离最小者   :815""",
              label="FSM_decode_core.py:805-815 摘录", title_color="cyan", size=10)
    table_simple(s, ML, 4.08, CW,
                 ["情况", "处理", "输出标记"],
                 [["两方向都有候选", "取点到极线距离最小的组合", "corrected = True"],
                  ["某一方向没有候选", "回退到原始视差，不做修改", "corrected = False"]],
                 col_w=[1.3, 3.0, 1.2], size=10.5, head_size=10.5, row_h=0.42)
    NB = ["结果写出 disp_hor_corrected.npy / disp_ver_corrected.npy，原始 disp_*.npy 仍然保留。",
          "校正只对 PIXELS_TO_CORRECT 指定的像素生效（解码侧默认 [(900, 495)]），是逐点复核工具而不是全图替换。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "写出什么", NB)

    # 6-7 亚像素细化
    s = d.newslide()
    p = d.nxt("亚像素细化")
    page(s, "六 · 亚像素细化", "质心法：把整数 bin 细化成小数坐标",
         "用峰值左右两个相邻点做加权质心，把定位精度从 1 像素推进到亚像素",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    badge(s, "代码现状")
    formula_band(s, ML, 2.06, 5.86, 0.94,
                 [("peak_coord = peak_idx + 0.5 · (v_r − v_l) / (v_l + v_c + v_r)",
                   {"size": 14, "color": "white"})],
                 fill="navy", accent="navy", size=14)
    code_card(s, ML + 6.28, 2.06, 5.85, 1.86, """if peak_value > 0.1:                       # 太弱的峰不细化
    vl = C_mag[(peak_idx - 1) % fft_length]
    vc = C_mag[peak_idx]
    vr = C_mag[(peak_idx + 1) % fft_length]
    peak_coord = peak_idx + 0.5 * (vr - vl) / (vl + vc + vr)""",
              label="FSM_decode_core.py:1001-1006 摘录", title_color="amber", size=10)
    NB = ["若要两条路径都获得亚像素精度，需要在分块模式里补一次向量化的质心 / 抛物线细化。"]
    items = [("★ 只在低内存路径生效", "分块向量化模式直接输出整数 bin，没有调用任何细化函数。"),
             ("阈值 0.1", "峰值归一化强度低于 0.1 的像素不做细化。"),
             ("环绕索引", "邻域用 % fft_length 取，峰值搜索用 valid_length；step=1.0 时两者相等。"),
             ("PeakDetector 同名函数未被调用", "refine_peak_subpixel 是参考实现，主路径另有内联版本。")]
    grid_cards(s, ML, 4.10, CW, BODY_BOTTOM - 4.10, items,
               cols=2, head_size=12.5, body_size=10.5,
               accent=["amber", "teal", "teal", "teal"])

    # 6-8 质量分级与排查
    s = d.newslide()
    p = d.nxt("质量分级排查")
    page(s, "六 · 质量与排查", "峰值质量分级与排查顺序",
         "peak = max|C| / fft_length 是归一化置信度；出现异常时按固定顺序逐层排查",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    NB2 = ["① 核对图像数量是否为 M×4，命名是否为「频率优先」顺序。",
           "② 开启 VERIFY_SEQUENCE，校验四步相移序列是否完整、未错位。",
           "③ 查看 peak_*.npy 定位低质量像素，必要时过滤后再重建。",
           "④ 对异常像素设 SPECTRUM_POINTS，直接看频谱是否多峰。",
           "⑤ 启用多峰扫描 + 极线校正，处理反光、遮挡、多路径。",
           "⑥ 点云若呈台阶状，回查 fft_length 与 frequency_step 是否匹配。"]
    nh2 = note_box_h(CW, NB2)
    ny2 = BODY_BOTTOM - nh2
    note_box(s, ML, ny2, CW, nh2, "推荐排查顺序", NB2)
    grid_cards(s, ML, 2.06, CW, ny2 - 0.20 - 2.06,
               [("peak > 1.0", "主峰明显，优先保留。"),
                ("0.5 < peak ≤ 1.0", "中等可用，需结合邻域与几何检查。"),
                ("peak ≤ 0.5", "较弱、易误匹配，建议过滤。"),
                ("阈值不可机械套用", "与亮度、曝光、反射率、归一化方式都有关。")],
               cols=2, head_size=13, body_size=10.5)

    # 6-9 误区澄清
    s = d.newslide()
    p = d.nxt("误区澄清")
    page(s, "六 · 误区澄清", "四条常见误解",
         "这四条几乎覆盖了所有「改了参数但结果没变好」的情况",
         "六 · 工程实现 · 后处理 · 排查", p, d.total)
    table_simple(s, ML, 2.06, CW,
                 ["误解", "事实"],
                 [["补零能提高采集精度",
                   "补零只在更密的网格上重采样同一条频谱，信息量仍由 M 决定，不提高信噪比。"],
                  ["改了 frequency_step 但 fft_length 不动",
                   "步长改变相位旋转速率，L 必须同步改为 round(N/s)，否则坐标系统性偏移。"],
                  ["fft_length 越大重建越准",
                   "超过 N/s 只是对同一频谱做更细插值，精度不再提升，计算量与内存却线性上涨。"],
                  ["负频率 bin 也可以当坐标用",
                   "bin ≥ N 属于负频率/周期重复区，物理上无效，必须在 argmax 前裁掉。"]],
                 col_w=[1.9, 4.3], size=10.5, head_size=10.5, row_h=0.62)
    NB = ["再补一条：阈值 0.5 / 0.8 这类数字是经验值，与亮度、曝光、反射率和归一化方式都有关，不应机械套用。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "还有一条", NB)


def build(d):
    chapter5(d)
    chapter6(d)
