# -*- coding: utf-8 -*-
"""
E 线 Bayesian UQ · P0 对照图（Fig Y）：本文贝叶斯后验均值 vs 第一篇监督 NMF 点估计
================================================================================
⚠ 诚实定位：A（第一篇 dz-unmix-cg）的 case study 是 **4 个盆地级、文献锚定的合成样本**
（松辽营城 / 海拉尔 / 二连 / NCC 北缘），**不是**本文（E）的 15 个真实个体样本。两套非
同一批样本，故这是**物源格局层面的框架一致性对照，不是逐样本验证**（图注写死）。

图：三元图（S1 arc / S2 CAOB / S3 craton），叠加
  · E 的 15 个真实样本后验均值（NCC 北缘组 = 品红方块，松辽组 = 灰圆）
  · A 的 4 个盆地监督 NMF 点估计（黑菱形，带标签）
看同一盆地区两框架落点是否落在一致的物源区。

跑法：& "C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe" fig_bayes_vs_firstpaper.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SKILL = r"C:\Users\Lenovo\.claude\skills\geo-figures\scripts"
if SKILL not in sys.path:
    sys.path.insert(0, SKILL)
from geo_primitives import (apply_journal_rcparams, save_three_formats,
                            OKABE_ITO, GJ_DOUBLE_COL_IN)

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "figures")
os.makedirs(OUTDIR, exist_ok=True)
OI = OKABE_ITO

# ---- data -------------------------------------------------------------------
# A (first paper) supervised-NMF point estimates, 4 literature-grounded basins
# (case_study_results.json). pred_w = (arc S1, CAOB S2, craton S3).
A_BASINS = [
    ("Songliao Yingcheng", (0.7095, 0.1761, 0.1144)),
    ("Hailar Damoguaihe",  (0.5345, 0.3699, 0.0956)),
    ("Erlian Bayanhua",    (0.3145, 0.4480, 0.2376)),
    ("NCC N-margin",       (0.1086, 0.2851, 0.6063)),
]

# E (this study) Bayesian posterior mean weights, 15 real samples (Table 1).
# (name, group, (w1,w2,w3))
E_SAMPLES = [
    ("05FW007", "NCC", (0.133, 0.436, 0.431)),
    ("05FW008", "NCC", (0.940, 0.034, 0.026)),
    ("100821-1", "NCC", (0.012, 0.545, 0.443)),
    ("100821-3", "NCC", (0.009, 0.304, 0.687)),
    ("SL0929", "Songliao", (0.326, 0.587, 0.087)),
    ("SL0930", "Songliao", (0.337, 0.560, 0.103)),
    ("SL0931", "Songliao", (0.214, 0.343, 0.443)),
    ("SL0932", "Songliao", (0.071, 0.057, 0.872)),
    ("R31354-S12518", "Songliao", (0.606, 0.379, 0.015)),
    ("R31354-S12519", "Songliao", (0.690, 0.298, 0.012)),
    ("R31370-S12650", "Songliao", (0.460, 0.528, 0.012)),
    ("R31370-S12651", "Songliao", (0.567, 0.418, 0.015)),
    ("R31370-S12652", "Songliao", (0.247, 0.659, 0.093)),
    ("R31370-S12653", "Songliao", (0.214, 0.195, 0.591)),
    ("R31370-S12654", "Songliao", (0.719, 0.264, 0.016)),
]

# ---- ternary geometry: (w1 top, w2 bottom-left, w3 bottom-right) -------------
SQ3 = np.sqrt(3.0)
TOP = (0.5, SQ3 / 2.0)      # S1 arc
LEFT = (0.0, 0.0)           # S2 CAOB
RIGHT = (1.0, 0.0)          # S3 craton


def b2c(w):
    w1, w2, w3 = w
    s = w1 + w2 + w3
    w1, w2, w3 = w1 / s, w2 / s, w3 / s
    x = w3 + 0.5 * w1
    y = (SQ3 / 2.0) * w1
    return x, y


def main():
    apply_journal_rcparams("GJ")
    grey = OI.get("gray", "#999999")

    fig, ax = plt.subplots(figsize=(GJ_DOUBLE_COL_IN * 0.62, GJ_DOUBLE_COL_IN * 0.58))
    ax.set_aspect("equal"); ax.axis("off")

    # triangle outline
    tri = np.array([TOP, LEFT, RIGHT, TOP])
    ax.plot(tri[:, 0], tri[:, 1], color=OI["black"], lw=1.2, zorder=2)
    # gridlines every 20%
    for f in (0.2, 0.4, 0.6, 0.8):
        # lines parallel to each edge
        ax.plot(*zip(b2c((f, 1 - f, 0)), b2c((f, 0, 1 - f))), color=grey, lw=0.4, zorder=1)
        ax.plot(*zip(b2c((1 - f, f, 0)), b2c((0, f, 1 - f))), color=grey, lw=0.4, zorder=1)
        ax.plot(*zip(b2c((1 - f, 0, f)), b2c((0, 1 - f, f)), ), color=grey, lw=0.4, zorder=1)

    # corner labels
    ax.text(TOP[0], TOP[1] + 0.045, "S1 arc (~130 Ma)", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=OI["orange"])
    ax.text(LEFT[0] - 0.03, LEFT[1] - 0.03, "S2 CAOB\n(~280 Ma)", ha="right", va="top",
            fontsize=8, fontweight="bold", color=OI["bluish_green"])
    ax.text(RIGHT[0] + 0.03, RIGHT[1] - 0.03, "S3 craton\n(~1900 Ma)", ha="left", va="top",
            fontsize=8, fontweight="bold", color=OI["blue"])

    # E samples
    for nm, grp, w in E_SAMPLES:
        x, y = b2c(w)
        if grp == "NCC":
            ax.scatter(x, y, s=34, marker="s", facecolor=OI["reddish_purple"],
                       edgecolor="white", lw=0.5, zorder=5)
        else:
            ax.scatter(x, y, s=30, marker="o", facecolor=grey,
                       edgecolor="white", lw=0.5, zorder=4)

    # A basin point estimates
    for nm, w in A_BASINS:
        x, y = b2c(w)
        ax.scatter(x, y, s=70, marker="D", facecolor=OI["vermillion"],
                   edgecolor=OI["black"], lw=0.9, zorder=6)
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(7, 4),
                    fontsize=6.5, color=OI["black"], zorder=7,
                    fontstyle="italic")

    # legend (manual)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=OI["reddish_purple"],
               markeredgecolor="white", markersize=7,
               label="Bayesian mean · NCC N-margin (n=4)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=grey,
               markeredgecolor="white", markersize=7,
               label="Bayesian mean · Songliao (n=11)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=OI["vermillion"],
               markeredgecolor=OI["black"], markersize=8,
               label="Supervised NMF (first paper) · basins (n=4)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.08, 1.15),
              fontsize=6.8, frameon=False, handletextpad=0.4, labelspacing=0.55)

    ax.set_xlim(-0.20, 1.20); ax.set_ylim(-0.18, 1.18)
    ax.text(0.5, -0.14,
            "⚠ Different sample sets: first-paper basins are literature-grounded syntheses,\n"
            "this study uses 15 real OneDZ/Puetz samples — provenance-pattern consistency, not per-sample validation.",
            ha="center", va="top", fontsize=6.2, color=grey, style="italic",
            transform=ax.transData)

    save_three_formats(fig, OUTDIR, "FigY_bayes_vs_nmf")
    plt.close(fig)
    print("saved FigY_bayes_vs_nmf.{pdf,png,svg} ->", OUTDIR)


if __name__ == "__main__":
    main()
