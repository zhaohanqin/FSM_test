# -*- coding: utf-8 -*-
"""ch_b.py — 第三章 / 第四章"""
from deck_kit import *
from deck_special import *


def chapter3(d):
    s = d.newslide()
    p = d.nxt("第三章 章节页")
    chapter(s, "三", "频率轴信号与 FFT 解码", "Frequency-Axis Signal & FFT Decoding",
            [("对每个相机像素，D[k] 的", {}), ("相位旋转速度", {"color": "22D3EE"}),
             ("正比于投影仪坐标；FFT 就是把「旋转速度」测出来的工具。", {})],
            "频率轴信号与 FFT 解码", p, d.total)

    # 3-1 频率轴信号
    s = d.newslide()
    p = d.nxt("频率轴信号")
    page(s, "三 · 频率轴信号", "什么叫「沿频率轴的一维复信号」",
         "固定一个相机像素，把所有频率上的复数 D[k] 排成一列，就得到这条信号",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    cards = [("从图像空间到频率轴", "固定像素 (x, y)，取 signal = D[:, y, x]，得到长度 M 的一维复数序列。"),
             ("自变量是频率编号 k", "序列下标 k = 0…M−1 对应频率 f_k = s·k，而不是图像的行或列。"),
             ("每个像素各有自己的信号", "整幅图有 Hc×Wc 条独立信号，解码就是对它们逐条做同样的处理。"),
             ("维度顺序不能弄反", "复数数组 D 的形状是 (M, Hc, Wc)：axis=0 是频率轴，不是图像空间轴。")]
    CODE = """signal = D[:, y, x]              # 长度 M 的一维复信号
C = np.fft.fft(signal, n=fft_length)   # 沿频率轴做 FFT（源码中为 axis=0）"""
    code_card(s, ML, 2.06, CW, code_card_h(CODE, size=11.5),
              CODE, label="单像素信号与它的频谱", title_color="cyan", size=11.5)
    gy = 2.06 + code_card_h(CODE, size=11.5) + 0.24
    grid_cards(s, ML, gy, CW, BODY_BOTTOM - gy, cards,
               cols=2, head_size=13, body_size=10.5)

    # 3-2 相位旋转速度
    s = d.newslide()
    p = d.nxt("相位旋转速度")
    page(s, "三 · 编码原理", "投影仪坐标 = 相位的旋转速度",
         "相位随频率编号线性增长，增长速率由投影仪坐标唯一决定",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    formula_band(s, ML, 2.06, CW, 0.86,
                 [("D[k] ≈ A · exp( j · 2π · s · k · uₚ / N )",
                   {"size": 19, "color": "white"})],
                 fill="navy", accent="navy", size=19)
    ty = table_simple(s, ML, 3.00, CW,
                 ["符号", "含义", "取值来源"],
                 [["s", "frequency_step，频率步长", "配置项 FREQUENCY_STEP，1.0 或 0.5"],
                  ["k", "频率编号 0 … M−1", "图像序列中的第 k 组四步相移"],
                  ["uₚ", "该像素对应的投影仪坐标", "★ 待解目标"],
                  ["N", "proj_dim，该方向投影仪分辨率", "hor → 1080（高），ver → 1920（宽）"]],
                 col_w=[0.65, 2.9, 2.4], size=10.5, head_size=10.5, row_h=0.42)
    NB = ["关键判断：分母是 N（投影仪坐标的归一化周期），不是采集频率数 M。",
          "uₚ 越大 → 每增加一个频率相位转得越快 → 信号在频率轴上「振荡」得越急。",
          "解码任务：Δθ = 相邻频率的相位增量 = 2π·s·uₚ/N，斜率正比于 uₚ/N —— 测出这个旋转速率即可换回坐标。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, ty + 0.22, CW, nh, "读这个公式时最容易错的地方", NB)

    # 3-3 FFT 在测试什么
    s = d.newslide()
    p = d.nxt("FFT在测试什么")
    page(s, "三 · FFT 原理", "FFT 的每个 bin 都在测试一种候选旋转速度",
         "把信号反向旋转一个候选速率，同向则叠加成峰，不同向则相互抵消",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    formula_band(s, ML, 2.06, 5.86, 0.80,
                 [("C[q] = Σ_k  D[k] · exp( −j·2π·q·k / L )", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    formula_band(s, ML + 6.28, 2.06, 5.85, 0.80,
                 [("测试的斜率   β_q = q / L", {"size": 16, "color": "white"})],
                 fill="navy", accent="navy", size=16)
    items_l = [("匹配时", "反向旋转项与信号相位精确抵消，所有 D[k] 变成同方向向量，相加得到最大幅值主峰。"),
               ("不匹配时", "残余相位 2π(uₚ−q)k/N ≠ 0 随 k 增大而旋转，复数相加互相抵消，幅值很小。")]
    items_r = [("所以峰值位置就是坐标", "argmax|C| 的 bin 编号即候选速率中最接近真实 uₚ 的那个。"),
               ("这一步不需要迭代", "一次 FFT 同时评价所有候选 q，是整条链路唯一的「搜索」动作。")]
    ch = max(card_points_h(5.86, items_l), card_points_h(5.85, items_r)) + 0.06
    card_points(s, ML, 2.98, 5.86, items_l, accent="teal", head_size=13, card_h=ch)
    card_points(s, ML + 6.28, 2.98, 5.85, items_r, accent="amber", head_size=13, card_h=ch)
    NB = ["一句话：频谱峰值位置 = 投影仪坐标。这句话的成立前提是 FFT 长度与频率步长匹配（见第五章）。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "一句话总结", NB)

    # 3-4 数学细节
    s = d.newslide()
    p = d.nxt("数学细节")
    page(s, "三 · 数学细节", "为什么 q = uₚ 时会出现最大峰",
         "把 D[k] 代进 FFT 定义式，指数上只剩一个与 k 成正比的残余相位",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    code_card(s, ML, 2.06, CW, 2.60, """# 代入 D[k] = A·exp(j·2π·s·k·uₚ / N)，取 L = N、s = 1 的简单情形
C[q] = Σ_{k=0}^{M-1}  A · exp( j·2π·k·(uₚ - q) / N )

# 情况 1：q = uₚ  → 指数项 = exp(j·0) = 1，每一项都同向
|C[uₚ]| = M · |A|        ← 最大峰（M 项同相叠加）

# 情况 2：q ≠ uₚ → 残余相位 2π·(uₚ-q)·k/N ≠ 0，随 k 递增而旋转
#          复数向量在单位圆上均匀铺开 → 相互抵消 → |C[q]| 很小""",
              label="推导（N = proj_dim，M = 频率数）", title_color="cyan", size=10.5)
    NB = ["「同相叠加」是全部直觉的来源：对齐时 M 项相加，不对齐时 M 项相消。这也解释了为什么 M 越大峰越尖、抗噪越强。",
          "注意 uₚ 与 q 都是整数时，N 点 FFT 的网格里恰好包含 q = uₚ，因此峰值可以取到精确坐标。",
          "若 N 与 L 不匹配，q = uₚ 这个 bin 根本不存在，峰值只能落到最近的 bin 上 —— 这就是量化阶梯的来源。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "从推导能直接读出的三件事", NB)

    # 3-5 代码语义
    s = d.newslide()
    p = d.nxt("代码语义")
    page(s, "三 · 代码语义", "三行 numpy 分别对应什么语义",
         "np.fft.fft / np.abs / np.argmax 一一对应公式中的三个动作",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    CODE_SEM = """C          = np.fft.fft(D_block, n=fft_length, axis=0)   # 对所有候选 q 求值
C_magnitude = np.abs(C)                                     # 对齐强度
freq_index  = np.argmax(C_valid, axis=0)                    # 峰值 bin = 投影仪坐标
peak        = np.max(C_valid, axis=0) / fft_length          # 归一化置信度"""
    ty = table_simple(s, ML, 2.06, CW,
                 ["代码", "对应数学量", "语义"],
                 [["np.fft.fft(signal, n=fft_length)", "C[q]", "一次性测试全部候选旋转速度，得到整条频谱"],
                  ["np.abs(C)", "|C[q]|", "计算每个候选的对齐强度（只看幅值，不看相位）"],
                  ["np.argmax(...)", "argmax |C[q]|", "选出对齐最强的候选 bin → 投影仪坐标"],
                  ["np.max(...) / fft_length", "归一化峰值", "作为该像素的置信度 / 质量指标"]],
                 col_w=[2.0, 1.1, 2.6], size=10.5, head_size=10.5,
                 mono_cols=(0,), row_h=0.50)
    cy = ty + 0.22
    code_card(s, ML, cy, CW, min(code_card_h(CODE_SEM, size=10.5), BODY_BOTTOM - cy),
              CODE_SEM,
              label="主线代码（FSM_decode_core.py:1144-1157 摘录）", title_color="amber", size=10.5)

    # 3-6 单像素流程
    s = d.newslide()
    p = d.nxt("单像素流程")
    page(s, "三 · 流程收口", "单像素解码的完整流程",
         "读四步灰度 → 差分构造复信号 → 沿频率轴 FFT → 取峰值作为坐标",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    chips(s, ML, 2.06, ["读 4 张相移图", "差分 Dc / Ds", "合成复数 D",
                        "沿频率轴 FFT", "取峰值 bin", "乘 coord_scale"],
          w=CW, h=0.56, gap=0.28, fill="fill_2", color="navy", size=11.5,
          arrow_color="cyan")
    code_card(s, ML, 2.92, CW, 3.30, """signal = []
for k in range(M):
    I0 = image[4*k + 0][y, x]        # 0°
    I1 = image[4*k + 1][y, x]        # 90°
    I2 = image[4*k + 2][y, x]        # 180°
    I3 = image[4*k + 3][y, x]        # 270°
    signal.append((I0 - I2) + 1j * (I3 - I1))   # 频率轴复信号 D[k]

C       = np.fft.fft(signal, n=fft_length)      # 频谱
mag     = np.abs(C)[:valid_length]              # 裁掉负频率区
q       = np.argmax(mag)                        # 峰值 bin
coord   = q * coord_scale                       # 投影仪坐标
quality = mag[q] / fft_length                   # 置信度""",
              label="完整伪代码（与 FSM_decode_core.py 一致）", title_color="cyan", size=10.5)

    # 3-7 峰值图
    s = d.newslide()
    p = d.nxt("峰值质量图")
    page(s, "三 · 质量图", "peak 图：解码质量的置信度图",
         "peak = max|C| / fft_length，值越大表示主峰越突出、解码越可信",
         "三 · 频率轴信号与 FFT 解码", p, d.total)
    stat_row(s, ML, 2.06, CW,
             [("> 1.0", "主峰明显、压过旁瓣，优先保留该像素"),
              ("0.5 – 1.0", "中等可用，需结合邻域一致性与几何检查"),
              ("≤ 0.5", "主峰不突出、易误匹配，建议过滤后再重建")],
             accent=["teal", "amber", "red"], h=1.28, value_size=21)
    NB = ["peak 的分母是 fft_length 而不是 M，因此它是「幅值 / FFT 点数」的归一化量，与频率数 M 无关。",
          "主线代码在重建前用 PEAK_LIMIT（重建 example 中为 0.8）过滤低质量像素，避免错误对应关系进入三角化。",
          "常见低峰值成因：高反光导致主峰分裂、遮挡边界混合信号、多路径干扰、曝光不足导致信噪比下降。"] 
    nh = note_box_h(CW, NB)
    note_box(s, ML, 3.60, CW, nh, "怎么用它", NB)
    code_card(s, ML, 3.60 + nh + 0.20, CW, min(0.92, BODY_BOTTOM - 3.60 - nh - 0.20),
              'peak[ty:ey, tx:ex] = np.max(C_valid, axis=0) / fft_length    # FSM_decode_core.py:1157',
              label=None, size=11)


def chapter4(d):
    s = d.newslide()
    p = d.nxt("第四章 章节页")
    chapter(s, "四", "主线代码实现与数据形状", "Code Map & Tensor Shapes",
            [("把前两章的公式落到仓库里：", {}), ("谁调用谁", {"color": "22D3EE"}),
             ("、张量怎么变形、分块与内存怎么取舍。", {})],
            "主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状", fill="amber")

    # 4-1 模块地图
    s = d.newslide()
    p = d.nxt("模块地图")
    page(s, "四 · 代码地图", "主线模块地图",
         "六个文件构成完整链路；解码侧与重建侧各自独立，通过 .npy 文件交接",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    grid_cards(s, ML, 2.06, CW, 3.50,
               [("FSM_decode_example.py（667 行）",
                 "配置与入口：main() 定义全部参数，串联加载→解码→保存。"),
                ("FSM_decode_core.py（1272 行）",
                 "解码核心：ImageLoader / _calc_fft_params / MultiPeakDetector / EpipolarCorrector / FSADecoder。"),
                ("FSM_visualize.py（593 行）",
                 "诊断绘图：FSAVisualizer 输出视差图、统计图、对应关系图。"),
                ("FSM_reconstruct_core.py（1240 行）",
                 "重建核心：CalibrationReader / CorrespondenceReader / TriangulationEngine / FSAReconstructor。"),
                ("FSM_reconstruct_example.py（305 行）",
                 "重建入口：13 个配置常量，输出 pointcloud_fsa.ply。"),
                ("点云后续处理/（2 个脚本）",
                 "点云对齐工具.py（坐标系转换）与点云误差比较文件.py（KD-Tree 误差统计）。")],
               cols=2, head_size=12, body_size=10)
    NB = ["交接契约：results/fsa_decode/ 下的 disp_hor.npy / disp_ver.npy / peak_*.npy 是解码与重建之间唯一接口。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "模块之间的边界", NB)

    # 4-2 数据形状
    s = d.newslide()
    p = d.nxt("数据形状")
    page(s, "四 · 张量维度", "从读图到频谱的数据形状变化",
         "唯一容易出错的地方：axis=0 是频率轴",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    table_simple(s, ML, 2.06, CW,
                 ["变量", "形状", "含义"],
                 [["images", "(M·4, Hc, Wc)", "一次读入的全部条纹图（灰度）"],
                  ["I0 / I1 / I2 / I3", "(M, Hc, Wc)", "用步长 4 切片还原的四路相位，axis=0 是频率"],
                  ["D = Dc + 1j·Ds", "(M, Hc, Wc)", "复数信号体；D[:, y, x] 就是该像素的频率轴信号"],
                  ["D_block", "(M, bh, bw)", "分块后的切片，bh/bw ≤ block_size"],
                  ["C = fft(D_block, axis=0)", "(fft_length, bh, bw)", "沿频率轴的频谱"],
                  ["C_valid = C[:valid_length]", "(proj_dim, bh, bw)", "裁掉负频率区后的有效频谱"],
                  ["disp_* / peak_*", "(Hc, Wc)", "float32 结果图，索引为 [y, x] 与相机像素对齐"]],
                 col_w=[1.55, 1.35, 2.9], size=10.5, head_size=10.5,
                 mono_cols=(0, 1), row_h=0.40)
    NB = ["最容易踩的坑：把 axis=0 当成图像高度。在 D 与 C 里 axis=0 是频率轴，axis=1/2 才是图像空间。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "唯一容易错的地方", NB)

    # 4-3 分块解码
    s = d.newslide()
    p = d.nxt("分块解码")
    page(s, "四 · 分块解码", "分块向量化：标准模式怎么跑",
         "把图像切成 block_size × block_size 的小块，一块一次 FFT",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    CODE = """Dc = I0 - I2
Ds = I3 - I1
D  = Dc + 1j * Ds          # (M, Hc, Wc)
del images, I0, I1, I2, I3, Dc, Ds     # 及时释放

for ty in range(0, D.shape[1], block_size):
    for tx in range(0, D.shape[2], block_size):
        ey = min(ty + block_size, D.shape[1])
        ex = min(tx + block_size, D.shape[2])
        D_block = D[:, ty:ey, tx:ex]           # (M, bh, bw)
        C = np.fft.fft(D_block, n=fft_length, axis=0)
        C_valid    = np.abs(C)[:valid_length]
        freq_index = np.argmax(C_valid, axis=0)
        disparity[ty:ey, tx:ex] = freq_index * coord_scale
        peak[ty:ey, tx:ex]      = np.max(C_valid, axis=0) / fft_length"""
    items = [("block_size 默认 100", "只影响内存与缓存局部性，不影响结果。"),
             ("一块一次 FFT", "整块三维数组沿 axis=0 一起变换，靠向量化提速。"),
             ("内存峰值来源", "D 的复数体 + 每块的 C 与 |C| 三份同数量级数组。"),
             ("标准模式更快", "低内存逐像素做 FFT，Python 循环开销远大于收益。")]
    chh = code_card_h(CODE, size=9.5)
    ch = card_points_h(5.85, items, head_size=12.5, body_size=10.5)
    tot = max(chh, 4 * ch + 3 * 0.16, 2.60)
    code_card(s, ML, 2.06, 5.86, tot, CODE,
              label="FSM_decode_core.py:1094-1157 摘录", title_color="cyan", size=9.5)
    card_points(s, ML + 6.28, 2.06, 5.85, items, accent="amber",
                head_size=12.5, body_size=10.5, card_h=(tot - 3 * 0.16) / 4)

    # 4-4 低内存模式
    s = d.newslide()
    p = d.nxt("低内存模式")
    page(s, "四 · 低内存模式", "低内存模式与图像缓存",
         "LOW_MEMORY_MODE = True 时逐行逐列解码；图像缓存把大量小图读盘变成一次大数组读盘",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    CODE = """for row in range(height):                     # 逐行
    I0_row = images[0::4, row, :]             # (M, Wc)
    I1_row = images[1::4, row, :]
    I2_row = images[2::4, row, :]
    I3_row = images[3::4, row, :]
    D_row  = (I0_row - I2_row) + 1j * (I3_row - I1_row)
    for col in range(width):                  # 逐列
        C = np.fft.fft(D_row[:, col], n=fft_length)
        C_valid = np.abs(C)[:valid_length]
        peak_idx = int(np.argmax(C_valid))
        peak_value = float(C_valid[peak_idx] / fft_length)
        # 亚像素质心细化（阈值 0.1）
        if peak_value > 0.1:
            vl = C_mag[(peak_idx - 1) % fft_length]
            vc = C_mag[peak_idx]
            vr = C_mag[(peak_idx + 1) % fft_length]
            peak_coord = peak_idx + 0.5 * (vr - vl) / (vl + vc + vr)"""
    items = [("内存 O(1) 级", "每次只保留一行四路相位与单条频谱。"),
             ("代价是更慢", "双层 Python 循环 + 逐点 FFT，明显慢于向量化分块。"),
             ("★ 只有这里做亚像素", "分块模式输出整数 bin，低内存模式额外用质心法细化。"),
             ("图像缓存", "默认写 <image_dir>/images_cache.npy，避免重复读上千张小图。")]
    chh = code_card_h(CODE, size=9.5)
    ch = card_points_h(5.85, items, head_size=12.5, body_size=10.5)
    tot = max(chh, 4 * ch + 3 * 0.16, 2.60)
    code_card(s, ML, 2.06, 5.86, tot, CODE,
              label="FSM_decode_core.py:943-1013 摘录", title_color="amber", size=9.5)
    card_points(s, ML + 6.28, 2.06, 5.85, items, accent="teal",
                head_size=12.5, body_size=10.5, card_h=(tot - 3 * 0.16) / 4)

    # 4-5 真实参数表
    s = d.newslide()
    p = d.nxt("真实参数")
    page(s, "四 · 参数现状", "主线代码里的真实配置",
         "以下取值直接取自 FSM_decode_example.py 的 main()，不是示例值",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    table_simple(s, ML, 2.06, 5.86,
                 ["配置项", "取值", "行号"],
                 [["PROJECTOR_WIDTH", "1920", ":553"],
                  ["PROJECTOR_HEIGHT", "1080", ":554"],
                  ["NUM_FREQUENCIES", "301", ":557"],
                  ["FREQUENCY_STEP", "0.5", ":558"],
                  ["DECODE_DIRECTION", '"both"', ":577"],
                  ["OUTPUT_DIR", "results\\fsa_decode", ":582"]],
                 col_w=[1.7, 1.1, 0.55], size=10.5, head_size=10.5,
                 mono_cols=(0, 1), row_h=0.36)
    table_simple(s, ML + 6.28, 2.06, 5.85,
                 ["配置项", "取值", "行号"],
                 [["BLOCK_SIZE", "100", ":585"],
                  ["USE_CACHE / VERIFY_SEQUENCE", "True / True", ":586-587"],
                  ["LOW_MEMORY_MODE", "True", ":590"],
                  ["SPECTRUM_POINTS", "(1200, 1000)", ":595"],
                  ["ENABLE_MULTI_PEAK_SCAN", "False", ":598"],
                  ["ENABLE_EPIPOLAR_CORRECTION", "False", ":602"]],
                 col_w=[2.0, 1.05, 0.6], size=10.5, head_size=10.5,
                 mono_cols=(0, 1), row_h=0.36)
    NB = ["STEP = 0.5 且 NUM_FREQUENCIES = 301 ⇒ 频率覆盖 0 … 150 Hz，每方向 301×4 = 1204 张图。",
          "STEP = 0.5 时 fft_length = round(1920 / 0.5) = 3840（ver 方向），valid_length 仍为 1920。",
          "多峰扫描与极线校正默认关闭，需要时手动打开 —— 它们是可选后处理，不替代 argmax 主解码。"]
    nh = note_box_h(CW, NB)
    ny = BODY_BOTTOM - nh
    note_box(s, ML, ny, CW, nh, "从这张表能读出的三件事", NB)

    # 4-6 未接线的参考实现
    s = d.newslide()
    p = d.nxt("未接线的类")
    page(s, "四 · 代码现状", "两个「看起来重要但没有被调用」的类",
         "DFTProcessor 与 PeakDetector 保留了公式的教科书式实现，但主解码路径并未使用",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    items = [("DFTProcessor（FSM_decode_core.py:330）",
              "construct_complex_signal / perform_dft / compute_magnitude_spectrum —— 最直白的公式翻译。"),
             ("PeakDetector（FSM_decode_core.py:353）",
              "find_peak_index / compute_peak_value / refine_peak_subpixel（质心法）—— 峰值处理的参考实现。"),
             ("主路径实际用的是向量化代码",
              "FSADecoder.decode_direction 直接内联 np.fft.fft + argmax，跳过了这两个类的封装。"),
             ("阅读建议",
              "想快速理解公式就读这两个类；想理解实际行为必须读 decode_direction 与低内存分支。")]
    grid_cards(s, ML, 2.06, CW, 2.30, items,
               cols=2, head_size=12.5, body_size=10.5,
               accent=["amber", "amber", "teal", "teal"])
    NB = ["同一现象还有一处：FSM_visualize.py:95-104 复制了一份 _get_fft_length，逻辑与 _calc_fft_params 相同，"
          "但没有复用 —— 修改参数策略时需要同时改两处。",
          "另有返回顺序不一致：模块级 _calc_fft_params 返回 (fft_length, valid_length, coord_scale)，"
          "而 FSADecoder._get_fft_params 返回 (fft_length, coord_scale, proj_dim, valid_length)。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, 4.58, CW, nh, "同一类「代码现状」提醒", NB)

    # 4-7 可视化产物
    s = d.newslide()
    p = d.nxt("可视化产物")
    page(s, "四 · 诊断输出", "解码之后能看到的诊断图",
         "FSM_visualize.py 与 FSM_decode_example.py 各自输出一组诊断图",
         "四 · 主线代码实现与数据形状", p, d.total)
    badge(s, "代码现状")
    table_simple(s, ML, 2.06, CW,
                 ["产出文件", "生成者", "内容"],
                 [["disparity_ver.png", "FSAVisualizer", "垂直条纹解出的投影仪列坐标（jet 伪彩）+ 峰值强度（hot）"],
                  ["disparity_hor.png", "FSAVisualizer", "水平条纹解出的投影仪行坐标 + 峰值强度"],
                  ["statistics.png", "FSAVisualizer", "2×3 统计图：视差直方图、峰值直方图（含 0.5 / 1.0 阈值线）、质量分级柱状图"],
                  ["correspondence_map.png", "FSAVisualizer", "R = 投影仪 X，G = 投影仪 Y，B = 质量；另附 X/Y 坐标图"],
                  ["subplots/（8 张独立 PNG）", "visualize_spectrum", "单点频谱的 8 个分解子图：强度序列、Dc/Ds、频谱、峰值坐标"],
                  ["epipolar_spectrum/pixel_x_y/", "run_epipolar_correction", "极线校正前后每个待校正像素的频谱与 PNG"]],
                 col_w=[1.75, 1.5, 3.4], size=10, head_size=10.5, row_h=0.44)
    NB = ["load_and_inspect_results() 会检查视差图的唯一整数个数：少于 100 时告警「可能存在量化阶梯」。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "自带的量化阶梯告警", NB)


def build(d):
    chapter3(d)
    chapter4(d)
