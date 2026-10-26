# -*- coding: utf-8 -*-
"""measure.py — 用 PIL/FreeType 精确测量文本尺寸，用于自动防溢出。

PPT 里的字号是 pt，版式是英寸：1 pt = 1/72 in。
"""
import os
from PIL import ImageFont

FONT_DIRS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\Windows\Fonts"),
    r"C:\Windows\Fonts",
]

CAND = {
    ("sans", False): ["NotoSansSC-VF.ttf", "msyh.ttc", "simhei.ttf"],
    ("sans", True):  ["NotoSansSC-VF.ttf", "msyhbd.ttc", "simhei.ttf"],
    ("mono", False): ["consola.ttf", "cour.ttf"],
    ("mono", True):  ["consolab.ttf", "courbd.ttf"],
}

_cache = {}
_path_cache = {}


def _find(name):
    if name in _path_cache:
        return _path_cache[name]
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            _path_cache[name] = p
            return p
    _path_cache[name] = None
    return None


def _font(size_pt, mono=False, bold=False):
    key = (round(size_pt * 4), mono, bold)
    if key in _cache:
        return _cache[key]
    f = None
    for name in CAND[("mono" if mono else "sans", bold)]:
        p = _find(name)
        if not p:
            continue
        try:
            f = ImageFont.truetype(p, int(round(size_pt * 4)))  # 4x 超采样提高精度
        except Exception:
            f = None
        if f is not None:
            if bold:
                try:
                    f.set_variation_by_name("Bold")
                except Exception:
                    try:
                        f.set_variation_by_axes([700])
                    except Exception:
                        pass
            break
    _cache[key] = f
    return f


def text_width_in(txt, size_pt, mono=False, bold=False):
    """返回文本宽度（英寸）。"""
    if not txt:
        return 0.0
    f = _font(size_pt, mono, bold)
    if f is None:
        # 回退估算：中日韩按 1em，其余按 0.55em
        w = 0.0
        for ch in txt:
            w += 1.0 if ord(ch) > 0x2E80 else 0.55
        return w * size_pt / 72.0
    return f.getlength(txt) / 4.0 / 72.0


def wrap_count(txt, width_in, size_pt, mono=False, bold=False):
    """估算文本在给定宽度下的行数（按空格与字符断行，模拟 CJK 断行）。"""
    if not txt:
        return 0
    if width_in <= 0.05:
        return 99
    f = _font(size_pt, mono, bold)
    total = 0
    for hard in txt.split("\n"):
        if not hard:
            total += 1
            continue
        if f is not None:
            total += _wrap_one(hard, width_in, f)
        else:
            total += max(1, int(text_width_in(hard, size_pt, mono, bold) / width_in) + 1)
    return total


def _width(f, s):
    return f.getlength(s) / 4.0 / 72.0


def _wrap_one(line, width_in, f):
    """贪心断行：优先在空格断，CJK 可逐字断。"""
    if _width(f, line) <= width_in:
        return 1
    n = 0
    cur = ""
    for ch in line:
        if _width(f, cur + ch) <= width_in:
            cur += ch
        else:
            # 允许在空格处回退
            n += 1
            cur = ch
    if cur:
        n += 1
    return max(1, n)


def needs_height_in(txt, width_in, size_pt, line_spacing=1.0, base=1.2,
                    mono=False, bold=False, pad=0.0):
    """返回容纳该文本所需的高度（英寸）。base 是字体行高系数。"""
    lines = wrap_count(txt, width_in, size_pt, mono, bold)
    return lines * size_pt * line_spacing * base / 72.0 + pad


def fits(txt, w_in, h_in, size_pt, line_spacing=1.0, base=1.2,
         mono=False, bold=False):
    return needs_height_in(txt, w_in, size_pt, line_spacing, base, mono, bold) <= h_in + 1e-6
