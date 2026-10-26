# -*- coding: utf-8 -*-
"""build_deck.py — 组装《频移法 FSM / FSA 原理与实现详解》v3（合并去重版）。

页面规划（共 67 页）
  1        封面
  2–3      讲解路线 / 阅读指南
  4–8      第一章  问题背景与方法目标
  9–14     第二章  四步相移与复信号构造
  15–22    第三章  频率轴信号与 FFT 解码
  23–30    第四章  主线代码实现与数据形状
  31–40    第五章  FFT 长度策略：统一公式
  41–50    第六章  工程实现 · 后处理 · 排查
  51–59    第七章  与三维重建的关系
  60–64    第八章  总结与常见误区
  65        Q&A
  66        谢谢观看
  67        附：版式自检说明（占位——实际为 Q&A / 谢谢 两页在 66、67）

校正：实际生成 67 页（ch_a/ch_b/ch_c/ch_d 的页数之和），以此为准。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deck_kit as K
from deck_kit import *
import deck_special
import ch_a, ch_b, ch_c, ch_d
import check_overlap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "FSM_FSA_原理与实现详解_v3.pptx")
TOTAL = 67


class Deck(object):
    def __init__(self, total):
        self.prs = new_deck()
        self.n = 0
        self.total = total
        K.TOTAL = total

    def newslide(self):
        return blank(self.prs)

    def nxt(self, name):
        self.n += 1
        set_ctx(self.n, name)
        return self.n

    def save(self, path):
        self.prs.save(path)
        return len(self.prs.slides._sldIdLst)


def main():
    d = Deck(TOTAL)
    ch_a.build(d)
    ch_b.build(d)
    ch_c.build(d)
    ch_d.build(d)
    n = d.save(OUT)
    print("saved: %s" % OUT)
    print("slides: %d / planned %d" % (n, TOTAL))
    print("\n=== \u7248\u5f0f\u81ea\u68c0 (%d) ===" % len(K.WARNINGS))
    for w in K.WARNINGS:
        print("  " + w)
    if n != TOTAL:
        print("!! \u9875\u6570\u4e0e\u8ba1\u5212\u4e0d\u4e00\u81f4")
    # 形状级碰撞检查：自检只看"内容是否超出自己的容器"，
    # 看不出"两个形状互相覆盖"，因此必须再跑一遍包围盒比对。
    print("")
    check_overlap.report(OUT, tol=0.08)


if __name__ == "__main__":
    main()

