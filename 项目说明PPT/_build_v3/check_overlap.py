"""check_overlap.py — 形状级碰撞检查。

自检（deck_kit._warn）只能发现"内容超出自己的容器"，发现不了"两个形状互相覆盖"。
本脚本读取生成好的 pptx，把所有**带文字的**形状两两比较包围盒，报告显著重叠的对，
用来兜住表格/代码卡/要点框被另一个元素压住的这类缺陷。

用法：python check_overlap.py <pptx> [容差英寸，默认 0.06]
"""
import sys

from pptx import Presentation

EMU = 914400.0


def box(sh):
    return (sh.left / EMU, sh.top / EMU,
            (sh.left + sh.width) / EMU, (sh.top + sh.height) / EMU)


def report(path, tol=0.06, quiet=False):
    """返回重叠条数；把每条重叠打印出来。供 CLI 与 build_deck 共用。"""
    prs = Presentation(path)
    hits = []
    for idx, slide in enumerate(prs.slides, 1):
        items = []
        for sh in slide.shapes:
            if sh.shape_type is not None and str(sh.shape_type).startswith("GROUP"):
                continue
            if not sh.has_text_frame:
                continue
            txt = sh.text_frame.text.strip()
            if not txt:
                continue
            l, t, r, b = box(sh)
            if r - l <= 0.02 or b - t <= 0.02:
                continue
            items.append((l, t, r, b, txt.replace("\n", " ⏎ ")[:46]))
        for a in range(len(items)):
            for c in range(a + 1, len(items)):
                l1, t1, r1, b1, s1 = items[a]
                l2, t2, r2, b2, s2 = items[c]
                ox = min(r1, r2) - max(l1, l2)
                oy = min(b1, b2) - max(t1, t2)
                if ox > tol and oy > tol:
                    hits.append((idx, ox, oy, s1, s2))

    if not quiet:
        for idx, ox, oy, s1, s2 in hits:
            print(f"[p{idx}] 重叠 {ox:.2f}x{oy:.2f}in :: {s1!r} <-> {s2!r}")
        print(f"=== 形状重叠检查 ({len(hits)}) ===")
    return len(hits)


def main():
    if len(sys.argv) < 2:
        print("usage: python check_overlap.py <pptx> [tol]")
        return 2
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
    report(sys.argv[1], tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

