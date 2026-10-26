# -*- coding: utf-8 -*-
"""
deck_kit.py — 频移法 PPT 设计系统与版式原语
================================================
供 build_deck.py 使用。提供统一的配色、字体、版式度量，
以及 cover / toc / chapter / cards / code / table / ending 等版式原语。

坐标单位全部为英寸（内部换算为 EMU）。
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import copy
import measure

# 版式自检：记录所有「内容高度 > 容器高度」的文本，构建结束后统一打印
WARNINGS = []
_page_ctx = {"no": 0, "name": ""}


def set_ctx(no, name):
    _page_ctx["no"] = no
    _page_ctx["name"] = name


def _warn(kind, need, box, txt):
    WARNINGS.append("[p%02d %s] %s 需要 %.2f\" > 容器 %.2f\"  :: %s"
                    % (_page_ctx["no"], _page_ctx["name"], kind, need, box,
                       txt[:56].replace("\n", "⏎")))

# ----------------------------------------------------------------------------
# 一、设计 token
# ----------------------------------------------------------------------------

C = {
    "navy":      "0E2A47",   # 主深蓝
    "navy_deep": "0A1A2F",   # 封面/章节底
    "navy_2":    "102A43",
    "navy_3":    "16395C",
    "ink":       "1E293B",   # 正文
    "teal":      "0E7490",   # 主强调青
    "cyan":      "00A9C8",   # 亮青
    "cyan_br":   "22D3EE",   # 封面亮青
    "cyan_pale": "D7E6F2",
    "amber":     "F29F05",   # 琥珀强调
    "amber_2":   "F59E0B",
    "gray":      "64748B",   # 次要文字
    "gray_2":    "94A3B8",
    "gray_3":    "8AA4B8",
    "line":      "E2E8F0",   # 边线
    "line_2":    "CBD5E1",
    "fill":      "F2F5F9",   # 浅底卡
    "fill_2":    "EAF0F6",
    "fill_3":    "D9E2EC",
    "red":       "EF4444",
    "red_2":     "D94F4F",
    "green":     "2CA58D",
    "blue":      "3B82F6",
    "white":     "FFFFFF",
}


def rgb(key):
    """接受调色板键名，或直接传 6 位十六进制字符串。"""
    v = C.get(key, key)
    if isinstance(v, str) and len(v) == 6:
        return RGBColor.from_string(v.upper())
    return v


FONT   = "Noto Sans SC"          # 系统中已安装；同时写入 latin / ea / cs
FONT_MONO = "Consolas"
FONT_MONO_ALT = "Cascadia Mono"

# 版式度量（英寸）
SW, SH = 13.3333, 7.5            # 16:9
ML, MR = 0.60, 0.60              # 左右页边距
CW = SW - ML - MR                # 内容宽 12.133
BAR_X, BAR_W = 0.285, 0.05       # 左侧竖青条

EYEBROW_Y, EYEBROW_H = 0.44, 0.26
TITLE_Y,   TITLE_H   = 0.72, 0.58
SUB_Y,     SUB_H     = 1.33, 0.34
RULE_Y               = 1.78
BODY_Y               = 1.96      # 正文起点
BODY_BOTTOM          = 6.62      # 正文底部（footer 之上）
FOOTER_Y             = 6.98

TOTAL = 0                        # 由 build 脚本回填，用于 "NN / TT"


# ----------------------------------------------------------------------------
# 二、底层工具
# ----------------------------------------------------------------------------

def _set_ea(run, font_name):
    """同时设置 latin / ea / cs 字体，保证中文不回退。"""
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font_name)


def _no_shadow(shape):
    """关闭继承来的形状阴影。"""
    spPr = shape._element.spPr
    if spPr.find(qn("a:effectLst")) is None:
        spPr.append(spPr.makeelement(qn("a:effectLst"), {}))


def _set_radius(shape, frac):
    """圆角矩形的圆角比例（0~0.5）。"""
    try:
        shape.adjustments[0] = frac
    except Exception:
        pass


def strip_style(run):
    """去掉 run 上的默认主题化设置残留。"""
    pass


def new_deck():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ----------------------------------------------------------------------------
# 三、形状原语
# ----------------------------------------------------------------------------

def rect(s, x, y, w, h, fill=None, line=None, line_w=0.75, radius=None, shape=None):
    shp = s.shapes.add_shape(shape or MSO_SHAPE.RECTANGLE,
                             Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None:
        shp.adjustments[0] = radius
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(line_w)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    _no_shadow(shp)
    shp.text_frame.word_wrap = True
    return shp


def rrect(s, x, y, w, h, fill=None, line=None, radius=0.06, line_w=0.75):
    return rect(s, x, y, w, h, fill=fill, line=line, line_w=line_w,
                radius=radius, shape=MSO_SHAPE.ROUNDED_RECTANGLE)


def line_h(s, x, y, w, color="line", w_pt=0.75):
    ln = s.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(w_pt)
    return ln


def arrow_h(s, x, y, w, color="cyan", w_pt=1.5, vertical=False):
    """一条带箭头的细线（水平或垂直）。"""
    if vertical:
        ln = s.shapes.add_connector(1, Inches(x), Inches(y),
                                    Inches(x), Inches(y + w))
    else:
        ln = s.shapes.add_connector(1, Inches(x), Inches(y),
                                    Inches(x + w), Inches(y))
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(w_pt)
    ln.line._get_or_add_ln().append(
        ln.line._get_or_add_ln().makeelement(qn("a:tailEnd"),
                                             {"type": "triangle", "w": "med", "len": "med"}))
    return ln


def text(s, x, y, w, h, runs, size=12, bold=False, color="ink",
         align="l", anchor="t", spacing=1.0, font=None, space_after=0,
         wrap=True, shrink=False, check=True, base=1.2, label="text"):
    """
    runs: 字符串，或 [(text, {opts}), ...]
          opts 可含 color / bold / size / font / italic
    """
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE,
                          "b": MSO_ANCHOR.BOTTOM}[anchor]
    if shrink:
        tf.auto_size = None
    p = tf.paragraphs[0]
    p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER,
                   "r": PP_ALIGN.RIGHT, "j": PP_ALIGN.JUSTIFY}[align]
    p.line_spacing = spacing
    if space_after:
        p.space_after = Pt(space_after)

    plain = runs if isinstance(runs, str) else "".join(
        (it if isinstance(it, str) else it[0]) for it in runs)

    if isinstance(runs, str):
        runs = [(runs, {})]
    max_size = size
    joined = []
    for item in runs:
        if isinstance(item, str):
            item = (item, {})
        txt, o = item
        max_size = max(max_size, o.get("size", size))
        joined.append(txt)
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", False)
        r.font.color.rgb = rgb(o.get("color", color))
        _set_ea(r, o.get("font", font or FONT))

    if check and wrap and h > 0:
        need = measure.needs_height_in("".join(joined), w, max_size, spacing,
                                       base, mono=(font == FONT_MONO), bold=bold)
        if need > h + 0.012:
            _warn(label, need, h, plain)
    return tb


def para(tf_or_box, runs, size=12, bold=False, color="ink", align="l",
         spacing=1.0, space_before=0, space_after=4, font=None,
         bullet=False, bullet_color=None, indent=0.0):
    """在已有 text_frame 上追加一段。"""
    tf = tf_or_box.text_frame if hasattr(tf_or_box, "text_frame") else tf_or_box
    p = tf.add_paragraph()
    p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER,
                   "r": PP_ALIGN.RIGHT, "j": PP_ALIGN.JUSTIFY}[align]
    p.line_spacing = spacing
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if indent:
        p.left_indent = Inches(indent)
    if isinstance(runs, str):
        runs = [(runs, {})]
    if bullet:
        runs = [("▪  ", {"color": bullet_color or "cyan", "bold": True})] + list(runs)
    for item in runs:
        if isinstance(item, str):
            item = (item, {})
        txt, o = item
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", False)
        r.font.color.rgb = rgb(o.get("color", color))
        _set_ea(r, o.get("font", font or FONT))
    return p


# ----------------------------------------------------------------------------
# 四、代码块（带轻量语法高亮）
# ----------------------------------------------------------------------------

import re

_KEYWORDS = {
    "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and",
    "or", "import", "from", "as", "class", "with", "try", "except", "None",
    "True", "False", "lambda", "yield", "global", "assert", "break", "continue",
    "pass", "is", "print",
}
_HL = [
    (re.compile(r"#[^\n]*"),                              "code_comment"),
    (re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""), "code_str"),
    (re.compile(r"\b\d+\.?\d*\b"),                        "code_num"),
]


def _hl_line(txt):
    """返回 [(text, color_key)]，颜色键映射到调色板。"""
    runs = [(txt, None)]
    for cre, tag in _HL:
        out = []
        for seg, t in runs:
            if t is not None:
                out.append((seg, t))
                continue
            pos = 0
            for m in cre.finditer(seg):
                if m.start() > pos:
                    out.append((seg[pos:m.start()], None))
                out.append((m.group(0), tag))
                pos = m.end()
            if pos < len(seg):
                out.append((seg[pos:], None))
        runs = out
    # 关键词
    out = []
    for seg, t in runs:
        if t is not None:
            out.append((seg, t))
            continue
        parts = re.split(r"(\b[A-Za-z_][A-Za-z_0-9]*\b)", seg)
        for i, pp in enumerate(parts):
            if i % 2 == 1 and pp in _KEYWORDS:
                out.append((pp, "code_kw"))
            else:
                out.append((pp, None))
    return out


def nlines_est(code):
    return len(code.strip("\n").split("\n"))


def code_card_h(code, size=10.5, label_h=0.34, pad=0.30):
    """按代码行数估算 code_card 需要的总高度（含 label）。"""
    return nlines_est(code) * size * 1.16 * 1.22 / 72.0 + pad + label_h


def code_card(s, x, y, w, h, code, label=None, title_color="cyan",
              size=10.5, fill="navy", label_h=0.34):
    """深蓝实底代码卡；label 为卡片上方的小标题（在卡片外上方）。"""
    top = y
    if label:
        text(s, x, y, w, 0.26, label, size=11.5, bold=True, color=title_color)
        top = y + label_h
        h -= label_h
    card = rrect(s, x, top, w, max(h, 0.5), fill=fill, radius=0.05)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.16)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.12)
    tf.margin_bottom = Inches(0.10)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    # 代码卡自检：行数 × 行高 + 边距
    need_h = nlines_est(code) * size * 1.16 * 1.22 / 72.0 + 0.24
    widest = max([measure.text_width_in(l, size, mono=True) for l in code.strip("\n").split("\n")] or [0])
    if need_h > h + 0.02:
        _warn("code-height", need_h, h, code.strip().split("\n")[0])
    if widest > w - 0.30:
        _warn("code-width", widest, w - 0.30, code.strip().split("\n")[0])
    cmap = {"code_comment": "8AA4B8", "code_str": "F5C46B",
            "code_num": "F29F05", "code_kw": "22D3EE"}
    first = True
    nlines = 0
    for raw in code.strip("\n").split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.16
        p.space_after = Pt(0)
        nlines += 1
        for seg, tag in _hl_line(raw):
            if seg == "":
                continue
            r = p.add_run()
            r.text = seg
            r.font.size = Pt(size)
            r.font.color.rgb = rgb(cmap.get(tag, "E9F1F8") if tag else "E9F1F8")
            _set_ea(r, FONT_MONO)
    return card


# ----------------------------------------------------------------------------
# 五、页面骨架
# ----------------------------------------------------------------------------

def bg_light(s):
    rect(s, -0.02, -0.02, SW + 0.04, SH + 0.04, fill="white")


def bg_dark(s, deep=False):
    rect(s, -0.02, -0.02, SW + 0.04, SH + 0.04,
         fill="navy_deep" if deep else "navy")


def left_bar(s, color="teal", y=0.0, h=SH):
    rect(s, BAR_X, y, BAR_W, h, fill=color)


def eyebrow(s, txt, color="teal", y=EYEBROW_Y):
    """眉标：■ 五 · 推荐实现"""
    t = text(s, ML, y, 7.6, EYEBROW_H, txt, size=11.5, bold=True, color=color)
    # 手动字距
    for r in t.text_frame.paragraphs[0].runs:
        r._r.get_or_add_rPr().set("spc", "120")
    return t


def header(s, eyebrow_txt, title, subtitle=None, title_size=27,
           eb_color="teal", title_color="navy", rule=True):
    eyebrow(s, eyebrow_txt, color=eb_color)
    text(s, ML, TITLE_Y, CW, TITLE_H, title, size=title_size, bold=True,
         color=title_color, spacing=0.95)
    if subtitle:
        text(s, ML, SUB_Y, CW, SUB_H, subtitle, size=12, color="gray", spacing=1.05)
    if rule:
        line_h(s, ML, RULE_Y, CW, "line", 0.9)


def footer(s, section, page, total=None):
    total = total or TOTAL
    text(s, ML, FOOTER_Y, 4.6, 0.26, "频移法 FSM / FSA · 原理与实现详解",
         size=8.5, color="gray_2")
    text(s, ML + 4.8, FOOTER_Y, 3.4, 0.26, section, size=8.5,
         color="gray_2", align="c")
    t = text(s, SW - MR - 2.0, FOOTER_Y, 2.0, 0.26,
             "%02d / %d" % (page, total), size=8.5, color="gray_2", align="r")
    # 页码加粗
    for r in t.text_frame.paragraphs[0].runs:
        r.font.bold = True
        r.font.color.rgb = rgb("gray")


def page(s, eyebrow_txt, title, subtitle, section, pageno,
         total=None, title_size=27):
    """标准内容页骨架（浅底 + 竖青条 + 页眉 + 页脚）。"""
    bg_light(s)
    left_bar(s, "teal")
    header(s, eyebrow_txt, title, subtitle, title_size=title_size)
    footer(s, section, pageno, total)


# ----------------------------------------------------------------------------
# 六、通用内容块
# ----------------------------------------------------------------------------

def note_box(s, x, y, w, h, title, bullets, fill="navy",
             title_color="cyan", body_color="E9F1F8", size=11.5,
             title_size=13, bullet_char="•"):
    """深蓝要点框。bullets 为字符串或 [(text, opts)] 列表。"""
    card = rrect(s, x, y, w, h, fill=fill, radius=0.05)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.20)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.13)
    tf.margin_bottom = Inches(0.10)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.LEFT
    p0.line_spacing = 1.1
    p0.space_after = Pt(6)
    r = p0.add_run()
    r.text = title
    r.font.size = Pt(title_size)
    r.font.bold = True
    r.font.color.rgb = rgb(title_color)
    _set_ea(r, FONT)

    # 溢出预估
    inner_w = w - 0.38 - 0.16
    need = title_size * 1.1 * 1.2 / 72.0 + 6 / 72.0
    for b in bullets:
        if isinstance(b, str):
            b = [(b, {})]
        bt = "".join((it if isinstance(it, str) else it[0]) for it in b)
        bs = max([size] + [it[1].get("size", size) for it in b if not isinstance(it, str)])
        nb = measure.wrap_count(bullet_char + " " + bt, inner_w, bs)
        need += nb * bs * 1.22 * 1.2 / 72.0 + 3 / 72.0
    need += 0.13 + 0.10
    if need > h + 0.02:
        _warn("note-height", need, h, title)

    for b in bullets:
        if isinstance(b, str):
            b = [(b, {})]
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.22
        p.space_after = Pt(3)
        rb = p.add_run()
        rb.text = bullet_char + " "
        rb.font.size = Pt(size)
        rb.font.bold = True
        rb.font.color.rgb = rgb(title_color)
        _set_ea(rb, FONT)
        for item in b:
            if isinstance(item, str):
                item = (item, {})
            txt, o = item
            rr = p.add_run()
            rr.text = txt
            rr.font.size = Pt(o.get("size", size))
            rr.font.bold = o.get("bold", False)
            rr.font.color.rgb = rgb(o.get("color", body_color))
            _set_ea(rr, o.get("font", FONT))
    return card


def note_box_h(w, bullets, title_size=13, size=11.5, pad=0.30):
    """按内容估算 note_box 需要的高度（英寸）。"""
    inner_w = w - 0.54
    need = title_size * 1.1 * 1.2 / 72.0 + 6 / 72.0 + pad
    for b in bullets:
        if isinstance(b, str):
            b = [(b, {})]
        bt = "".join((it if isinstance(it, str) else it[0]) for it in b)
        bs = max([size] + [it[1].get("size", size) for it in b if not isinstance(it, str)])
        nb = measure.wrap_count("• " + bt, inner_w, bs)
        need += nb * bs * 1.22 * 1.2 / 72.0 + 3 / 72.0
    return need


def card_points(s, x, y, w, items, gap=0.16, card_h=0.86, accent="teal",
                head_size=13, body_size=10.5, bar_w=0.055, numbered=None):
    """竖向排列的「标题 + 说明」浅底卡，左侧带彩条。"""
    cy = y
    for i, (head, desc) in enumerate(items):
        col = accent if not isinstance(accent, (list, tuple)) else accent[i % len(accent)]
        rrect(s, x, cy, w, card_h, fill="white", line="line", radius=0.07)
        rect(s, x, cy + 0.08, bar_w, card_h - 0.16, fill=col)
        tx = x + 0.24
        if numbered is not None:
            circ = rect(s, x + 0.15, cy + card_h / 2 - 0.16, 0.32, 0.32,
                        fill=col, shape=MSO_SHAPE.OVAL)
            text(s, x + 0.15, cy + card_h / 2 - 0.115, 0.32, 0.24, numbered[i],
                 size=11, bold=True, color="white", align="c")
            tx = x + 0.58
        text(s, tx, cy + 0.13, w - (tx - x) - 0.22, 0.3, head,
             size=head_size, bold=True, color="navy")
        if desc:
            text(s, tx, cy + 0.42, w - (tx - x) - 0.22, card_h - 0.5, desc,
                 size=body_size, color="gray", spacing=1.16, label="card-desc")
        cy += card_h + gap
    return cy


def card_points_h(w, items, gap=0.16, head_size=13, body_size=10.5,
                  pad=0.60, bar_w=0.055):
    """按内容估算一组 card_points 每张卡需要的高度（取最大值）。"""
    inner = w - 0.24 - 0.22
    need = 0.0
    for head, desc in items:
        hh = measure.wrap_count(head, inner, head_size, bold=True)
        h = 0.13 + hh * head_size * 0.95 * 1.2 / 72.0 + 0.05
        if desc:
            nd = measure.wrap_count(desc, inner, body_size)
            h += nd * body_size * 1.16 * 1.2 / 72.0
        need = max(need, h + 0.20)
    return need


def card_points_fit(s, x, y, w, bottom, items, gap=0.16, accent="teal",
                    head_size=13, body_size=10.5, bar_w=0.055, numbered=None,
                    slack=0.0):
    """card_points 的自动压缩版：按 bottom 反推 card_h，保证不压到下方 note_box。

    若压缩后仍小于内容所需高度，记录 card-height 告警。"""
    n = len(items)
    ch = (bottom - y - slack - (n - 1) * gap) / n
    need = card_points_h(w, items, gap=gap, head_size=head_size, body_size=body_size)
    if ch + 1e-6 < need:
        _warn("card-height", need, ch, items[0][0])
    return card_points(s, x, y, w, items, gap=gap, card_h=ch, accent=accent,
                       head_size=head_size, body_size=body_size, bar_w=bar_w,
                       numbered=numbered)


def grid_cards(s, x, y, w, h, items, cols=2, gap_x=0.30, gap_y=0.20,
               accent=None, head_size=15, body_size=11, head_color="navy"):
    """n×m 网格卡（用于 4 宫格 / 6 宫格）。"""
    n = len(items)
    rows = (n + cols - 1) // cols
    cw = (w - gap_x * (cols - 1)) / cols
    ch = (h - gap_y * (rows - 1)) / rows
    accents = accent or ["teal", "amber", "blue", "green"]
    for i, (head, desc) in enumerate(items):
        r, c = divmod(i, cols)
        cx = x + c * (cw + gap_x)
        cy = y + r * (ch + gap_y)
        col = accents[i % len(accents)] if isinstance(accents, (list, tuple)) else accents
        rrect(s, cx, cy, cw, ch, fill="fill", line="line", radius=0.06)
        rect(s, cx, cy, 0.055, ch, fill=col)
        tf_h = cy + 0.18
        text(s, cx + 0.26, tf_h, cw - 0.46, 0.34, head, size=head_size,
             bold=True, color=head_color)
        if desc:
            text(s, cx + 0.26, tf_h + 0.40, cw - 0.46, ch - 0.62, desc,
                 size=body_size, color="gray", spacing=1.22, label="grid-desc")


def chips(s, x, y, items, w=None, h=0.42, gap=0.34, fill="navy",
          color="cyan_br", size=11, arrows=True, arrow_color="cyan"):
    """一排 chip，用箭头串联。"""
    n = len(items)
    if w is None:
        w = CW
    cw = (w - gap * (n - 1)) / n
    cx = x
    for i, it in enumerate(items):
        rrect(s, cx, y, cw, h, fill=fill, radius=0.16)
        text(s, cx, y + h / 2 - 0.115, cw, 0.24, it, size=size, bold=True,
             color=color, align="c")
        if arrows and i < n - 1:
            arrow_h(s, cx + cw + 0.06, y + h / 2, gap - 0.12, arrow_color, 1.25)
        cx += cw + gap


def formula_band(s, x, y, w, h, runs, fill="white", line="line",
                 accent="amber", size=20, align="c", bar=None):
    """带左侧彩条的公式带。浅底时自动改为彩色描边（避免游离彩条）。"""
    light = fill in ("white", "fill", "fill_2", "fill_3", "cyan_pale")
    if bar is None:
        bar = not light
    ln = accent if light else line
    rrect(s, x, y, w, h, fill=fill, line=ln, radius=0.08,
          line_w=1.2 if light else 0.75)
    if bar:
        rect(s, x + 0.03, y + 0.09, 0.055, h - 0.18, fill=accent)
    tx = x + 0.26 if bar else x + 0.14
    text(s, tx, y, w - (tx - x) - 0.14, h, runs, size=size, align=align,
         anchor="m", bold=False)


def badge(s, txt, x=None, y=None, w=None, fill="amber", color="white",
          size=9.5, h=0.28):
    """右上角小标签，例如「代码现状」。"""
    if x is None:
        x = SW - MR
    tw = measure.text_width_in(txt, size, bold=True) + 0.34
    if w:
        tw = w
    x = x - tw
    if y is None:
        y = EYEBROW_Y + 0.02
    rrect(s, x, y, tw, h, fill=fill, radius=0.14)
    text(s, x, y + h / 2 - 0.10, tw, 0.22, txt, size=size, bold=True,
         color=color, align="c")
    return x


def table_simple(s, x, y, w, headers, rows, col_w=None, row_h=0.42,
                 head_h=0.40, head_fill="navy", head_color="white",
                 size=11, head_size=11, zebra=True, align=None,
                 body_color="ink", mono_cols=(), row_min_h=None):
    """简洁表格。rows 为 list[list[str]]；col_w 为相对宽度列表。"""
    ncol = len(headers)
    if col_w is None:
        col_w = [1.0] * ncol
    tot = sum(col_w)
    cws = [w * c / tot for c in col_w]
    # 行高按内容自适应
    hs = []
    for r in rows:
        hh = row_min_h or row_h
        for i, cell in enumerate(r):
            f = FONT_MONO if i in mono_cols else FONT
            need = measure.needs_height_in(str(cell), cws[i] - 0.24, size, 1.1,
                                           1.2, mono=(i in mono_cols)) + 0.20
            hh = max(hh, need)
        hs.append(hh)
    rrect(s, x, y, w, head_h, fill=head_fill, radius=0.0)
    tx = x
    for i, hd in enumerate(headers):
        text(s, tx + 0.12, y, cws[i] - 0.24, head_h, hd, size=head_size,
             bold=True, color=head_color, anchor="m",
             align=(align[i] if align else "l"))
        tx += cws[i]
    cy = y + head_h
    for ri, r in enumerate(rows):
        if zebra and ri % 2 == 1:
            rect(s, x, cy, w, hs[ri], fill="fill_2")
        line_h(s, x, cy + hs[ri], w, "line", 0.75)
        tx = x
        for i, cell in enumerate(r):
            f = FONT_MONO if i in mono_cols else FONT
            text(s, tx + 0.12, cy, cws[i] - 0.24, hs[ri], str(cell), size=size,
                 color=body_color, anchor="m", font=f,
                 align=(align[i] if align else "l"), spacing=1.1,
                 label="cell")
            tx += cws[i]
        cy += hs[ri]
    return cy


def stat_row(s, x, y, w, items, gap=0.24, h=1.0, accent="teal",
             value_size=24, label_size=10.5, fill="white"):
    """一排「大数字 + 说明」卡。items = [(value, label), ...]"""
    n = len(items)
    cw = (w - gap * (n - 1)) / n
    accents = accent if isinstance(accent, (list, tuple)) else [accent] * n
    for i, (val, lab) in enumerate(items):
        cx = x + i * (cw + gap)
        rrect(s, cx, y, cw, h, fill=fill, line="line", radius=0.07)
        rect(s, cx, y + 0.08, 0.055, h - 0.16, fill=accents[i % len(accents)])
        text(s, cx + 0.20, y + 0.10, cw - 0.36, 0.44, val, size=value_size,
             bold=True, color="navy")
        text(s, cx + 0.20, y + 0.58, cw - 0.36, h - 0.68, lab, size=label_size,
             color="gray", spacing=1.15, label="stat-label")


def vflow(s, x, y, w, items, step=0.62, accent="cyan", box_h=0.46,
          size=11.5, fill="0D2138", color="D7E6F2", gap_arrow=True):
    """竖向流程：每项一个 box，向下箭头串联。items 为字符串列表。"""
    cy = y
    for i, it in enumerate(items):
        rrect(s, x, cy, w, box_h, fill=fill, radius=0.06)
        text(s, x + 0.18, cy, w - 0.36, box_h, it, size=size, bold=True,
             color=color, anchor="m")
        if i < len(items) - 1:
            arrow_h(s, x + w / 2 - 0.22, cy + box_h + 0.02, 0.44, accent, 1.3,
                    vertical=True)
        cy += step
    return cy


NO = 0
