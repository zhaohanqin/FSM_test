# -*- coding: utf-8 -*-
"""ch_d.py — 第七章 / 第八章 / 收尾"""
from deck_kit import *
from deck_special import *
from deck_special import _wave
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt


def qa_page(s, section, pageno, total):
    """Q & A 页"""
    bg_dark(s)
    left_bar(s)
    _wave(s, 5.55, 0.14, "22D3EE", 1.0, 0.20, 0.0, 0.0, SW)
    _wave(s, 6.10, 0.10, "22D3EE", 0.75, 0.14, 1.9, 0.0, SW)
    text(s, ML, 2.44, 8.0, 0.34, "Q  &  A", size=13, bold=True, color="cyan_br",
         label="qa-eyebrow")
    text(s, ML, 2.92, 11.0, 1.20, "提问与讨论", size=44, bold=True, color="white",
         label="qa-title")
    line_h(s, ML, 4.22, 3.30, "0E7490", 1.5)
    text(s, ML, 4.52, 10.5, 0.60,
         "关于参数取舍、后处理策略或代码细节，欢迎随时交流。",
         size=14, color="94A3B8", label="qa-sub")
    chips(s, ML, 5.32, ["参数为什么是 0.5 步长", "多峰怎么调", "点云台阶怎么查",
                        "重建侧接口"],
          w=CW, h=0.50, gap=0.24, fill="0D2138", color="D7E6F2", size=11, arrows=False)
    footer(s, section, pageno, total)


def thanks_page(s, section, pageno, total):
    bg_dark(s)
    left_bar(s)
    _wave(s, 5.40, 0.16, "22D3EE", 1.0, 0.22, 0.0, 0.0, SW)
    _wave(s, 6.05, 0.11, "22D3EE", 0.75, 0.15, 2.1, 0.0, SW)
    text(s, ML, 2.42, 11.0, 1.30, "谢谢观看", size=48, bold=True, color="white",
         label="thanks-title")
    text(s, ML, 3.86, 11.0, 0.60,
         [("频移法 FSM / FSA · 原理与实现详解", {"color": "22D3EE"})],
         size=16, label="thanks-sub")
    line_h(s, ML, 4.52, 4.60, "0E7490", 1.5)
    text(s, ML, 4.78, 11.5, 0.90,
         "核心公式：uₚ = q · N / (L · s)　|　推荐策略：L = N / s　|　质量指标：peak = max|C| / fft_length",
         size=12, color="94A3B8", label="thanks-note")
    footer(s, section, pageno, total)


def chapter7(d):
    s = d.newslide()
    p = d.nxt("第七章 章节页")
    chapter(s, "七", "与三维重建的关系", "From Correspondence to Point Cloud",
            [("解码只交出", {}), ("相机像素 → 投影仪像素", {"color": "22D3EE"}),
             ("的对应关系；放进标定好的三角化框架，它才变成三维点云。", {})],
            "与三维重建的关系", p, d.total)

    # 7-1 数据流
    s = d.newslide()
    p = d.nxt("对应关系到点云")
    page(s, "七 · 数据流", "从对应关系到三维点云",
         "FSM 解码的输出是两幅「坐标图」，它们与标定参数一起构成三角化的全部输入",
         "七 · 与三维重建的关系", p, d.total)
    vflow(s, ML, 2.06, 5.86,
          ["投影 + 采集：M×4 张条纹图",
           "解码：disp_ver / disp_hor 两幅坐标图",
           "过滤：peak ≥ PEAK_LIMIT 的像素才保留",
           "配对：xp = disp_ver[yc,xc]，yp = disp_hor[yc,xc]",
           "三角化：相机射线 × 投影仪射线求交",
           "点云：pointcloud_fsa.ply"],
          step=0.66, accent="cyan", box_h=0.50, size=11)
    code_card(s, ML + 6.28, 2.06, 5.85, 2.60, """# 解码输出（results/fsa_decode/）
disp_ver.npy   # (Hc, Wc) float32 —— 投影仪列坐标 xp
disp_hor.npy   # (Hc, Wc) float32 —— 投影仪行坐标 yp
peak_hor.npy / peak_ver.npy          # 置信度

# 重建入口读取（FSM_reconstruct_example.py）
CALIBRATION_FILE = "标定结果2.txt"
FREQUENCY_STEP   = 0.5     NUM_FREQUENCIES = 301
PEAK_LIMIT       = 0.8     OVERSAMPLE     = 4
OUTPUT_PLY       = "pointcloud_fsa.ply\"""",
              label="交接契约（代码现状）", title_color="cyan", size=10)
    NB = ["解码与重建之间只有 .npy 文件这一层接口，两侧可独立替换或重跑。",
          "peak 阈值越松，点云越密但飞点越多 —— 它是最重要的一个闸门。"]
    nh = note_box_h(5.85, NB)
    note_box(s, ML + 6.28, 4.80, 5.85, nh + 0.16, "两个工程事实", NB)

    # 7-2 三角化
    s = d.newslide()
    p = d.nxt("三角化原理")
    page(s, "七 · 三角化", "三角化：两条射线求交",
         "同一物点既被相机看到、也被投影仪「照亮」，两条射线在三维空间中交于该点",
         "七 · 与三维重建的关系", p, d.total)
    grid_cards(s, ML, 2.06, CW, 2.30,
               [("① 相机射线", "从相机光心出发、穿过像素 (xc, yc) 的空间直线，由相机内参 Kc 与位姿 [Rc|tc] 确定。"),
                ("② 投影仪射线", "从投影仪光心出发、穿过像素 (xp, yp) 的空间直线，由 Kp 与 [Rp|tp] 确定。"),
                ("③ 交汇", "两条射线在理想情况下交于物体表面一点，即该像素的三维坐标。"),
                ("④ 实际求解", "因为有噪声与量化误差，射线通常不相交，用线性三角化求最小二乘意义下的最优解。")],
               cols=2, head_size=13, body_size=10.5)
    CODE = """cam_und  = cv2.undistortPoints(cam_pix,  Kc, dc, P=Kc).reshape(-1, 2)   # 去畸变并回投原内参 :992
proj_und = cv2.undistortPoints(proj_pix, Kp, dp, P=Kp).reshape(-1, 2)   # 投影仪侧同构      :999
P_cam  = Kc @ np.hstack([np.eye(3), np.zeros((3, 1))])                  # 3×4 投影矩阵      :1008
P_proj = Kp @ np.hstack([R, T])                                         # 投影仪投影矩阵    :1014
pts4 = cv2.triangulatePoints(P_cam, P_proj, cam_und.T, proj_und.T)      # 齐次最小二乘      :1018
pts3 = (pts4[:3] / pts4[3]).T                     # (N, 3) 相机坐标系                :1024
pts3 = (R.T @ (pts3 - T.flatten()).T).T           # → 投影仪坐标系（内联，未调用同名方法）:1050"""
    code_card(s, ML, 4.56, CW, min(code_card_h(CODE, size=10.5), BODY_BOTTOM - 4.56),
              CODE, label="FSM_reconstruct_core.py 三角化实现（代码现状）", title_color="cyan", size=10.5)

    # 7-3 重建模块结构
    s = d.newslide()
    p = d.nxt("重建模块结构")
    page(s, "七 · 重建侧代码", "重建侧的模块结构",
         "FSM_reconstruct_core.py 用五个类把「读标定 → 读对应 → 三角化 → 存点云」串起来",
         "七 · 与三维重建的关系", p, d.total)
    badge(s, "代码现状")
    table_simple(s, ML, 2.06, CW,
                 ["类 / 文件", "职责", "输入 → 输出"],
                 [["CalibrationReader", "解析标定文件，得到相机与投影仪的内参、旋转、平移", "标定结果*.txt → K / R / t"],
                  ["CorrespondenceReader", "把四个 .npy 装载成 dict（只读文件，不做配对）", ".npy → {disp_*, peak_*}"],
                  ["TriangulationEngine", "去畸变 + 三角化 + 变换到投影仪坐标系", "对应关系 + 标定 → 三维点"],
                  ["PointCloudProcessor", "保存为 PLY 并可视化", "三维点 → pointcloud_fsa.ply"],
                  ["FSAReconstructor", "真正的协调器，串起过滤、配对与三角化", "配置 → 完整点云"]],
                 col_w=[1.75, 3.4, 1.5], size=10.5, head_size=10.5, row_h=0.46)
    NB = ["★ 代码现状：CorrespondenceReader 与 TriangulationEngine 的四个方法在主线上从未被调用，"
          "真实逻辑被内联复制在 FSAReconstructor._reconstruct_pointcloud 内（:990-1050）。",
          "配对 (xp, yp) 实际发生在 :946-949 —— xp ← disp_ver、yp ← disp_hor；CorrespondenceReader 只负责读文件。",
          "FSM_reconstruct_example.py（305 行）无函数定义，全部逻辑在 __main__ 块内，配置常量是局部变量。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "入口与两个已知落差", NB)

    # 7-4 峰值补偿
    s = d.newslide()
    p = d.nxt("峰值补偿因子")
    page(s, "七 · 补偿因子", "一个容易被忽略的补偿因子",
         "重建侧对峰值有一步与解码侧不同的归一化处理，用来把不同频率步长下的峰值拉到同一尺度",
         "七 · 与三维重建的关系", p, d.total)
    badge(s, "代码现状")
    formula_band(s, ML, 2.06, CW, 0.74,
                 [("补偿因子 = fft_length / M = proj_dim / (M · frequency_step)",
                   {"size": 17, "color": "white"})],
                 fill="navy", accent="navy", size=17)
    items = [("为什么需要它", "解码侧写出的是 peak = max|C| / fft_length，而 fft_length = round(proj_dim/step)；"
                              "step 由 1.0 改为 0.5 时 fft_length 翻倍，同一物理幅值的 peak 数值直接掉一半。"),
             ("它修正的是什么", "重建侧再乘 fft_length / M，把「按 fft_length 归一化」换回「按频率数 M 归一化」，"
                              "使 PEAK_LIMIT = 0.8 的绝对语义在任意步长下保持一致。"),
             ("本仓库实配", "proj_height = 1080、step = 0.5、M = 301 ⇒ 水平补偿因子 = 2160 / 301 ≈ 7.176"
                          "（执行代码 FSM_reconstruct_core.py:852-864）。"),
             ("残留风险", "未显式传入 num_frequencies 时，L882 的回退分支把绝对阈值改成 peak_limit × raw_peak_max 的相对阈值，"
                        "与入口注释「阈值含义与旧版一致」相矛盾。")]
    card_points_fit(s, ML, 2.94, CW, BODY_BOTTOM, items, accent=["teal", "amber", "teal", "amber"])

    # 7-5 坐标系对齐
    s = d.newslide()
    p = d.nxt("坐标系对齐")
    page(s, "七 · 点云后处理 ①", "两套点云的坐标系不是同一个",
         "FSM 输出的点云在投影仪坐标系，另一路方法输出的在相机坐标系；直接比较会整体错位",
         "七 · 与三维重建的关系", p, d.total)
    badge(s, "代码现状")
    formula_band(s, ML, 2.06, 5.86, 0.78,
                 [("P_cam = R @ P_proj + T", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    formula_band(s, ML + 6.28, 2.06, 5.85, 0.78,
                 [("P_proj = Rᵀ @ (P_cam − T)", {"size": 15, "color": "white"})],
                 fill="navy", accent="navy", size=15)
    grid_cards(s, ML, 3.06, CW, 2.32,
               [("问题现象", "两份点云各自看起来正常，叠在一起却有近似等于 |T| 的固定整体偏移，而不是局部噪声。"),
                ("根因", "FSM 在三角化之后多做了一次刚体变换到投影仪坐标系（:1047-1050）；另一路方法直接输出三角化结果，即相机坐标系。"),
                ("解决方式", "用点云对齐工具.py 按标定文件里的 R 与 T 做互逆变换，把一侧统一到另一侧坐标系后再比较。"),
                ("为什么必须做", "不做这一步，KD-Tree 最近邻误差会把「整体位姿差」误算成「形状误差」，结论完全失真。")],
               cols=2, head_size=13, body_size=10.5)
    NB = ["变换式：P_cam = R @ P_proj + T（点云对齐工具.py:217） ⇄ P_proj = Rᵀ @ (P_cam − T)（:236）。",
          "★ 代码现状：TriangulationEngine 的 docstring（:401-402）写「相机坐标系」，实际返回投影仪坐标系（:1061）。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, 5.50, CW, nh + 0.14, "坐标系与一句话提醒", NB)

    # 7-6 误差比较
    s = d.newslide()
    p = d.nxt("点云误差比较")
    page(s, "七 · 点云后处理 ②", "误差怎么算、看哪些指标",
         "以一份点云为参考，对另一份的每个点找 KD-Tree 最近邻，统计偏差分布",
         "七 · 与三维重建的关系", p, d.total)
    badge(s, "代码现状")
    table_simple(s, ML, 2.06, 5.86,
                 ["环节", "做法"],
                 [["最近邻检索", "cKDTree(ref) + tree.query(cmp, k=1, workers=-1)  :91-93"],
                  ["单点误差", "两点欧氏距离 ‖pᵢ − qᵢ*‖₂，单位与点云一致（本仓库为 mm）"],
                  ["统计汇总", "对全部单点误差算 9 项指标（见右表）"],
                  ["阈值标红", "distances > THRESHOLD（默认 1.5）的点染成红色 (255,0,0)  :168-169"],
                  ["输出", "output_colored.ply（ASCII 格式）+ 终端统计表"]],
                 col_w=[1.4, 3.6], size=10, head_size=10.5, row_h=0.50)
    table_simple(s, ML + 6.28, 2.06, 5.85,
                 ["9 项指标", "含义"],
                 [["mean / median", "平均误差与中位数"],
                  ["rmse / std", "均方根误差与标准差"],
                  ["min / max", "最小与最大偏差"],
                  ["p95", "95 百分位，反映尾部（飞点）程度"],
                  ["n_diff / ratio", "超阈值点数与占比（%）"]],
                 col_w=[1.7, 3.3], size=10, head_size=10.5, row_h=0.50)
    NB = ["看误差不能只看 mean：mean 小但 p95 大，说明整体贴合但存在少量飞点，往往是多峰误匹配的残留。",
          "文档落差：点云误差计算原理说明.md 提到的「统计块 2（Diff Mean / RMSE_D / Std_D / P95_D）」"
          "与 point_cloud_compare.py / .m 在本仓库中均不存在，点云后续处理/ 实际只有 3 个文件。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "读指标的顺序", NB)

    # 7-7 标定文件
    s = d.newslide()
    p = d.nxt("标定文件")
    page(s, "七 · 标定与瓶颈", "标定文件的解析与其中的风险",
         "标定文件是纯文本、按固定行号读取；格式一旦变动，解析会静默出错",
         "七 · 与三维重建的关系", p, d.total)
    badge(s, "代码现状")
    code_card(s, ML, 2.06, 5.86, 2.30, """# 解析实现重复两份：FSM_reconstruct_core.py:107-145
#                   FSM_decode_core.py:243-270
lines[4:12]    # 相机 Kc / dc
lines[17:25]   # 投影仪 Kp / dp
lines[27:30]   # 旋转矩阵 R
lines[32]      # 平移向量 T
# 全文共 38 行；第 35-38 行的基础矩阵 F 从未被读取""",
              label="标定文件的两处解析（硬编码行号）", title_color="amber", size=9.5)
    items = [("两份文件结构一致", "标定结果.txt 与 标定结果2.txt 都是 38 行、逐行同构；代码却只认行号，不认字段名。"),
             ("基础矩阵未被使用", "L35-38 的 F 已标定但主线从不解析。历史版本曾用 range(35, 38) 读取它，用于单方向重建"
                              "（过程文件/initial/解码后的三维重建过程_单方向/）。"),
             ("缺少行数校验", "未检查 len(lines)；若格式变动导致空行错位，解析会静默取到错误数字而不报错。"),
             ("换文件要改三处", "解码侧默认 标定结果.txt、重建侧默认 标定结果2.txt；点云对齐工具.py:78-84 用关键字匹配，较健壮。")]
    NB = ["瓶颈排序：标定精度 > 极线 / 对应关系质量 > FFT bin 量化。"]
    nh = note_box_h(5.86, NB)
    note_box(s, ML, 4.48, 5.86, nh + 0.16, "瓶颈在哪里", NB)
    card_points_fit(s, ML + 6.28, 2.06, 5.85, BODY_BOTTOM, items,
                    accent=["amber", "amber", "teal", "teal"],
                    head_size=11.5, body_size=9.5)


def chapter8(d):
    s = d.newslide()
    p = d.nxt("第八章 章节页")
    chapter(s, "八", "总结与常见误区", "Summary & Pitfalls",
            [("把三个符号、一条公式、两种步长、四条误区收在一处，", {}),
             ("方便随时回查", {"color": "22D3EE"}), ("。", {})],
            "总结与常见误区", p, d.total)

    # 8-1 三式串联
    s = d.newslide()
    p = d.nxt("三式串联")
    page(s, "八 · 全篇公式", "三式串联：全篇只需要记住这三行",
         "信号怎么来、峰值意味着什么、坐标怎么算 —— 三行覆盖整条链路",
         "八 · 总结与常见误区", p, d.total)
    grid_cards(s, ML, 2.06, CW, 2.30,
               [("① 信号构造", "D[k] = (I0 − I2) + j·(I3 − I1)，差分消掉直流背景，只留携带相位的复信号。"),
                ("② 物理模型", "D[k] ≈ A · exp( j·2π·s·k·uₚ / N )，相位随频率线性增长，斜率正比于 uₚ / N。"),
                ("③ 统一公式", "uₚ = q · N / (L · s)，其中 q = argmax|FFT(D)|；取 L = N/s 则 uₚ = q。"),
                ("④ 质量指标", "peak = max|C| / fft_length，作为后续过滤与后处理的置信度。")],
               cols=2, head_size=13, body_size=10.5)
    table_simple(s, ML, 4.56, CW,
                 ["符号", "角色", "一句话"],
                 [["M", "决定信息量", "真实观测数，越大峰越尖、抗噪越强"],
                  ["N", "决定坐标尺度", "投影仪分辨率，是相位归一化的分母"],
                  ["L", "决定评价网格", "FFT 点数；取 N/s 时 bin 编号等于坐标"]],
                 col_w=[0.7, 1.6, 6.6], size=10.5, head_size=10.5, row_h=0.40)

    # 8-2 参数速查
    s = d.newslide()
    p = d.nxt("参数速查")
    page(s, "八 · 参数速查", "两种步长的参数对照表",
         "改 frequency_step 时，只需要记住两个量会跟着变",
         "八 · 总结与常见误区", p, d.total)
    table_simple(s, ML, 2.06, CW,
                 ["量", "s = 1.0", "s = 0.5", "说明"],
                 [["频率覆盖", "0 … 150 Hz（步长 1）", "0 … 150 Hz（步长 0.5）", "两者覆盖同样的频率范围"],
                  ["频率个数 M", "151", "301", "半步长需要双倍的频率数"],
                  ["每方向图像数", "151 × 4 = 604", "301 × 4 = 1204", "每频率 4 张相移图"],
                  ["fft_length L", "N（= proj_dim）", "2N", "L = round(N / s)"],
                  ["valid_length", "N", "N", "★ 恒为 proj_dim，负频率区必须裁掉"],
                  ["coord_scale", "1.0", "1.0", "N / (L·s)，两种步长都等于 1"],
                  ["坐标最小单位", "≈ 1 投影仪像素", "≈ 1 投影仪像素", "量化误差约 ±0.5 像素"]],
                 col_w=[1.25, 1.75, 1.75, 3.4], size=10.5, head_size=10.5, row_h=0.42)
    NB = ["半步长的收益：相位旋转速率减半，在同样的 M 下能表示的坐标范围翻倍，"
          "因此可以在不牺牲坐标范围的前提下增加频率数、提升抗噪能力。",
          "代价：图像数、FFT 长度、内存与时间全部翻倍 —— 这是它默认关闭的原因。"]
    nh = note_box_h(CW, NB)
    note_box(s, ML, BODY_BOTTOM - nh, CW, nh, "半步长真正换来了什么", NB)

    # 8-3 误区总表
    s = d.newslide()
    p = d.nxt("误区总表")
    page(s, "八 · 常见误区", "八条最容易犯的错误",
         "每一条都对应过真实的调试弯路",
         "八 · 总结与常见误区", p, d.total)
    table_simple(s, ML, 2.06, CW,
                 ["#", "错误做法 / 想法", "正确认识"],
                 [["1", "D[k] ≈ A·exp(j·2π·k·uₚ / M)", "分母是投影仪分辨率 N，不是频率个数 M"],
                  ["2", "fft_length 设成频率数 M", "应设为 round(N / s)，否则 bin 不对应坐标，出现量化阶梯"],
                  ["3", "改了 frequency_step 但不动 fft_length", "两者必须同步改，否则坐标系统性偏移"],
                  ["4", "把负频率 bin 也当作坐标", "只在前 proj_dim 个 bin 内 argmax，超出范围的一律无效"],
                  ["5", "以为补零能提高采集精度", "补零只加密评价网格，不增加信息量、不提高信噪比"],
                  ["6", "以为 fft_length 越大越准", "超过 N/s 只是更细地插值同一条频谱，徒增计算量"],
                  ["7", "点云误差大时先怀疑算法", "先确认两份点云是否在同一坐标系（投影仪 vs 相机）"],
                  ["8", "机械套用 peak 阈值 0.5 / 0.8", "阈值与亮度、曝光、反射率、归一化方式都有关"]],
                 col_w=[0.4, 3.1, 3.7], size=10, head_size=10.5, row_h=0.40)

    # 8-4 术语表
    s = d.newslide()
    p = d.nxt("术语表")
    page(s, "八 · 术语表", "关键术语速查",
         "全篇出现的符号与工程术语，一处对照",
         "八 · 总结与常见误区", p, d.total)
    table_simple(s, ML, 2.06, 5.86,
                 ["术语", "含义"],
                 [["FSM / FSA", "频移法（Frequency Shift Method / Analysis），本项目主线方法"],
                  ["四步相移", "每个频率采集 0°/90°/180°/270° 四张图"],
                  ["复信号 D[k]", "(I0−I2) + j·(I3−I1)，消去直流后的频率轴信号"],
                  ["频率轴", "复数数组的 axis=0，下标 k 对应频率 f_k = s·k"],
                  ["零填充", "np.fft.fft(D, n=L) 在 L > M 时内部补零"],
                  ["量化阶梯", "bin 太疏导致相邻像素解出同一坐标，点云出现层状结构"],
                  ["多峰", "同一像素收到多条光路，频谱出现多个明显峰"]],
                 col_w=[1.5, 3.5], size=10.5, head_size=10.5, row_h=0.40)
    table_simple(s, ML + 6.28, 2.06, 5.85,
                 ["术语", "含义"],
                 [["极线约束", "相机像素在投影仪图像上对应一条直线，用于筛选候选配对"],
                  ["亚像素细化", "用峰值邻域做质心/抛物线插值，得到小数坐标"],
                  ["peak 图", "max|C| / fft_length，逐像素置信度图"],
                  ["disp_hor / disp_ver", "解出的投影仪行坐标 / 列坐标图，形状 (Hc, Wc)"],
                  ["proj_dim", "该方向的投影仪分辨率（hor → 1080，ver → 1920）"],
                  ["OVERSAMPLE", "重建侧采样密度参数，默认 4"],
                  ["PEAK_LIMIT", "重建前的峰值过滤阈值，默认 0.8"]],
                 col_w=[1.6, 3.4], size=10.5, head_size=10.5, row_h=0.40)

    # 8-5 一页总结
    s = d.newslide()
    p = d.nxt("一页总结")
    ending(s, "频移法 FSM / FSA 原理与实现",
            "八章一条主线：从条纹图像到三维点云",
            [("本质", "把「相机看到的反射条纹」反解为「投影仪上的像素坐标」"),
             ("核心", "D = (I0−I2) + j·(I3−I1)，沿频率轴 FFT，峰值位置即坐标"),
             ("公式", "uₚ = q · N / (L · s)；取 L = N / s 时 coord_scale = 1，q 就是坐标"),
             ("质量", "peak = max|C| / fft_length；低质量像素靠多峰检测与极线校正兜底"),
             ("下游", "disp_hor / disp_ver 与标定参数一起进入三角化，生成三维点云"),
             ("陷阱", "M / N / L 混用、步长与 FFT 长度不同步、负频率未裁、坐标系未对齐")],
            "八 · 总结与常见误区", p, d.total)


def build(d):
    chapter7(d)
    chapter8(d)
    s = d.newslide()
    p = d.nxt("Q&A")
    qa_page(s, "Q & A", p, d.total)
    s = d.newslide()
    p = d.nxt("谢谢观看")
    thanks_page(s, "谢谢观看", p, d.total)
