# -*- coding: utf-8 -*-
"""
第二篇方法论文 · Fig 1 框架示意图（概念图，非数据图）
=====================================================
上排：解混 pipeline（源 → 混样 → KDE → NMF 解混 → Top-1 打分）
下排：本文系统扫的三个数据难度轴的微缩卡通（K / 噪声 / 可分性 Δ/σ）
      R1（α、bandwidth）灰显标注为 companion 论文已刻画

沿用全套出图惯例：Okabe-Ito、pdf/ps fonttype=42、300 dpi、三格式。
跑法: & "C:/Users/Administrator/miniconda3/python.exe" make_fig1_framework.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#808080"}
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.linewidth": 0.8,
})


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def box(ax, cx, cy, w, h, text, fc, ec=OI["black"]):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec=ec, lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.5)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=OI["black"], lw=1.2))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # ---- 上排：pipeline ----
    ax.text(5, 5.75, "Synthetic detrital-zircon unmixing pipeline",
            ha="center", fontsize=9, fontweight="bold")
    yb = 4.7
    pales = ["#DCE9F5", "#DCE9F5", "#DCE9F5", "#FCE7D2", "#E2F2EC"]
    labels = [
        "$K$ Gaussian\nsources",
        "Mixture sample\n$w\\sim$Dir($\\alpha$)\n+ noise",
        "KDE density\n(fixed bw)",
        "NMF unmix\nsupervised /\nunsupervised",
        "Top-1\naccuracy",
    ]
    cxs = [1.0, 2.975, 4.95, 6.925, 8.9]
    for cx, lab, fc in zip(cxs, labels, pales):
        box(ax, cx, yb, 1.55, 1.15, lab, fc)
    for x0, x1 in zip(cxs[:-1], cxs[1:]):
        arrow(ax, x0 + 0.8, x1 - 0.8, yb)

    # ---- 下排：三个数据难度轴卡通（本文）----
    ax.text(5, 3.35, "Data-difficulty axes swept at HPC scale (this work)",
            ha="center", fontsize=9, fontweight="bold", color=OI["vermillion"])
    xg = np.linspace(0, 10, 200)

    # R2 源数 K
    a1 = ax.inset_axes([0.06, 0.06, 0.25, 0.30])
    for mu, c in [(2.2, OI["blue"]), (4.2, OI["blue"]), (6.5, OI["blue"]),
                  (8.0, OI["skyblue"]), (5.3, OI["skyblue"])]:
        a1.plot(xg, gauss(xg, mu, 0.6), color=c, lw=1.3)
    a1.set_title("R2  source number $K$ (2–6)", fontsize=7, color=OI["vermillion"])

    # R3 噪声
    a2 = ax.inset_axes([0.38, 0.06, 0.25, 0.30])
    a2.plot(xg, gauss(xg, 5, 1.0), color=OI["blue"], lw=1.5, label="clean")
    rng = np.random.default_rng(1)
    noisy = gauss(xg, 5, 1.0) * (1 + 0.18 * rng.standard_normal(xg.size))
    a2.plot(xg, np.clip(noisy, 0, None), color=OI["vermillion"], lw=1.0, label="+noise")
    a2.set_title("R3  measurement noise (0–20%)", fontsize=7, color=OI["vermillion"])

    # R4 可分性 Δ/σ（头条）
    a3 = ax.inset_axes([0.70, 0.06, 0.25, 0.30])
    a3.plot(xg, gauss(xg, 3.2, 0.7) + gauss(xg, 6.8, 0.7), color=OI["green"], lw=1.5)
    a3.plot(xg, gauss(xg, 4.6, 0.7) + gauss(xg, 5.6, 0.7), color=OI["vermillion"], lw=1.5)
    a3.set_title("R4  separability $\\Delta/\\sigma$ (cliff)", fontsize=7, color=OI["vermillion"])

    for a in (a1, a2, a3):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(OI["grey"]); s.set_linewidth(0.7)

    # R1 灰显注记
    ax.text(5, 0.12,
            "R1 (mixing unevenness $\\alpha$, KDE bandwidth): characterized in a companion study",
            ha="center", fontsize=6.8, color=OI["grey"], style="italic")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"Fig1_framework.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig1_framework.pdf/.png/.svg")


if __name__ == "__main__":
    main()
