# -*- coding: utf-8 -*-
"""
第二篇方法论文 · 发表级图生成
================================
读 results/applicability_{srcK,noise,sep}.csv（每行 = 一个 (axis_value, seed) 的 5-fold 均值），
按 axis_value 跨 50 seed 聚合（均值 ± 标准差误差带），出 4 张图：

  F2  准确率 vs 源数 K        （R2，2 面板：top1 全集 / top1 dominant）
  F3  准确率 vs 测量噪声       （R3）
  F4  准确率 vs 可分性 Δ/σ     （R4，头条：失效悬崖 + 临界 Δ/σ 内插标注）
  F5  监督优势 sup−uns 三轴总览（方法-难度相图）

沿用第一篇出图惯例：Okabe-Ito 色盲友好、pdf/ps fonttype=42（矢量可编辑）、300 dpi、三格式。

跑法（本地或集群 headless）:
  & "C:/Users/Administrator/miniconda3/python.exe" make_paper2_figures.py
"""
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# Okabe-Ito 色盲友好色板（与第一篇一致）
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    # 顶刊版式：去 top/right spine、tick 朝外、长度统一
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    # 图例极淡边框（措辞/内容不变，仅样式统一）
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8", "legend.fancybox": False,
})

THRESH = 0.85  # 任务书阈值，沿用第一篇


def _lgd(ax, **kw):
    """统一图例样式：极淡边框、细线宽。措辞由调用方给定，一字不改。"""
    leg = ax.legend(**kw)
    fr = leg.get_frame()
    fr.set_linewidth(0.6)
    fr.set_edgecolor("0.8")
    return leg


def load_axis(axis):
    """读一个轴的 CSV，按 axis_value 跨 seed 聚合 → 均值 + 标准差。"""
    rows = []
    with open(os.path.join(RESDIR, f"applicability_{axis}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    g = defaultdict(list)
    for r in rows:
        g[float(r["axis_value"])].append(r)
    xs = sorted(g)

    def col(rs, k):
        out = []
        for rr in rs:
            v = rr.get(k, "")
            if v not in (None, "", "nan", "None"):
                out.append(float(v))
        return np.array(out, dtype=float)

    res = {"x": np.array(xs), "n_seeds": len(g[xs[0]])}
    for k in ["sup_top1_mean", "sup_top1_dom_mean", "uns_top1_mean",
              "uns_top1_dom_mean", "chance_top1"]:
        m, s = [], []
        for x in xs:
            v = col(g[x], k)
            m.append(np.nanmean(v) if v.size else np.nan)
            s.append(np.nanstd(v) if v.size else np.nan)
        res[k] = np.array(m)
        res[k + "_sd"] = np.array(s)
    return res


def _band(ax, x, m, sd, color, label, ls="-", marker=None):
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=4.5,
            markeredgecolor="white", markeredgewidth=0.6, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, linewidth=0)


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.pdf/.png/.svg")


def crossing(x, y, level):
    """线性内插 y 首次（自下而上）跨过 level 的 x。无则 None。"""
    for i in range(1, len(x)):
        if (y[i-1] < level) and (y[i] >= level):
            t = (level - y[i-1]) / (y[i] - y[i-1])
            return x[i-1] + t * (x[i] - x[i-1])
    return None


# ----------------------------------------------------------------------------
# F2 — 源数 K（R2）：2 面板（全集 / dominant）
# ----------------------------------------------------------------------------
def fig2_srcK(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    x = d["x"]
    # 左：top1（全集）+ chance
    _band(a1, x, d["sup_top1_mean"], d["sup_top1_mean_sd"], OI["blue"], "supervised", marker="o")
    _band(a1, x, d["uns_top1_mean"], d["uns_top1_mean_sd"], OI["vermillion"], "unsupervised", marker="s")
    a1.plot(x, d["chance_top1"], color=OI["black"], ls=":", lw=1.0, label="chance (1/K)")
    a1.set_xlabel("Number of sources $K$"); a1.set_ylabel("Top-1 accuracy (all samples)")
    a1.set_title("(a) All samples"); a1.set_xticks(x); _lgd(a1, loc="lower left")
    a1.set_ylim(0.0, 1.02)
    # 右：top1 dominant
    _band(a2, x, d["sup_top1_dom_mean"], d["sup_top1_dom_mean_sd"], OI["blue"], "supervised", marker="o")
    _band(a2, x, d["uns_top1_dom_mean"], d["uns_top1_dom_mean_sd"], OI["vermillion"], "unsupervised", marker="s")
    a2.set_xlabel("Number of sources $K$"); a2.set_ylabel("Top-1 accuracy (dominant subset)")
    a2.set_title("(b) Dominant subset"); a2.set_xticks(x); _lgd(a2, loc="lower right")
    a2.set_ylim(0.94, 1.004)  # 全在阈值 0.85 之上，故略去阈值线，放大有效区间
    fig.tight_layout()
    _save(fig, "Fig2_source_number_K")


# ----------------------------------------------------------------------------
# F3 — 测量噪声（R3）
# ----------------------------------------------------------------------------
def fig3_noise(d):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    x = d["x"] * 100.0  # → 百分比
    _band(ax, x, d["sup_top1_dom_mean"], d["sup_top1_dom_mean_sd"], OI["blue"], "supervised (dominant)")
    _band(ax, x, d["sup_top1_mean"], d["sup_top1_mean_sd"], OI["skyblue"], "supervised (all)", ls="--")
    _band(ax, x, d["uns_top1_mean"], d["uns_top1_mean_sd"], OI["vermillion"], "unsupervised (all)")
    ax.axhline(THRESH, color=OI["green"], ls=":", lw=1.0, label=f"threshold {THRESH}")
    ax.set_xlabel("Measurement noise (%, multiplicative $2\\sigma$)")
    ax.set_ylabel("Top-1 accuracy")
    # 图级标题移除(2026-07-08):期刊惯例图名归 caption,图内不烤总标题
    ax.set_ylim(0.80, 1.005); _lgd(ax, loc="center right", bbox_to_anchor=(0.99, 0.42))
    fig.tight_layout()
    _save(fig, "Fig3_noise_robustness")


# ----------------------------------------------------------------------------
# F4 — 可分性 Δ/σ（R4，头条：失效悬崖）
# ----------------------------------------------------------------------------
def fig4_separability(d):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    x = d["x"]
    xc = crossing(x, d["sup_top1_dom_mean"], THRESH)
    # 失效区阴影
    if xc is not None:
        ax.axvspan(x.min(), xc, color=OI["vermillion"], alpha=0.08)
    _band(ax, x, d["sup_top1_dom_mean"], d["sup_top1_dom_mean_sd"], OI["blue"], "supervised (dominant)")
    _band(ax, x, d["uns_top1_mean"], d["uns_top1_mean_sd"], OI["vermillion"], "unsupervised (all)")
    ax.plot(x, d["chance_top1"], color=OI["black"], ls=":", lw=1.0, label="chance (1/3)")
    ax.axhline(THRESH, color=OI["green"], ls="--", lw=1.0, label=f"threshold {THRESH}")
    if xc is not None:
        ax.axvline(xc, color=OI["orange"], lw=1.2)
        ax.annotate(f"critical $\\Delta/\\sigma$ ≈ {xc:.2f}",
                    xy=(xc, THRESH), xytext=(xc + 0.7, 0.66),
                    fontsize=7, color=OI["orange"],
                    arrowprops=dict(arrowstyle="->", color=OI["orange"], lw=0.9))
        ax.text((x.min()+xc)/2, 0.45, "failure\nzone", ha="center", va="center",
                fontsize=7, color=OI["vermillion"])
    ax.set_xlabel("Source separability  $\\Delta/\\sigma$  (peak spacing / width)")
    ax.set_ylabel("Top-1 accuracy")
    # 图级标题移除(2026-07-08):同 Fig3,图名归 caption
    ax.set_ylim(0.20, 1.02); _lgd(ax, loc="lower right")
    fig.tight_layout()
    _save(fig, "Fig4_separability_cliff")
    return xc


# ----------------------------------------------------------------------------
# F5 — 监督优势 sup−uns 三轴总览
# ----------------------------------------------------------------------------
def fig5_overview(dK, dN, dS):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    specs = [
        (axes[0], dK, dK["x"],          "Number of sources $K$", "(a) source number"),
        (axes[1], dN, dN["x"] * 100.0,  "Noise (%)",             "(b) measurement noise"),
        (axes[2], dS, dS["x"],          "Separability $\\Delta/\\sigma$", "(c) separability"),
    ]
    for ax, d, x, xlab, title in specs:
        adv = d["sup_top1_mean"] - d["uns_top1_mean"]
        ax.plot(x, adv, color=OI["purple"], marker="o", ms=4.5,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.fill_between(x, 0, adv, color=OI["purple"], alpha=0.15)
        ax.axhline(0, color="gray", lw=1.0, ls="--")
        ax.set_xlabel(xlab); ax.set_title(title)
    axes[0].set_ylabel("Supervised advantage\n(Top-1$_{sup}$ − Top-1$_{uns}$)")
    # suptitle 移除(2026-07-08):图名归 caption;面板标题 (a)(b)(c) 保留
    fig.tight_layout()
    _save(fig, "Fig5_supervised_advantage")


def main():
    dK = load_axis("srcK")
    dN = load_axis("noise")
    dS = load_axis("sep")
    print(f"loaded: srcK {len(dK['x'])} pts, noise {len(dN['x'])} pts, sep {len(dS['x'])} pts "
          f"(each {dK['n_seeds']} seeds)")
    fig2_srcK(dK)
    fig3_noise(dN)
    xc = fig4_separability(dS)
    fig5_overview(dK, dN, dS)
    print(f"\n临界可分性 Δ/σ (sup dominant 跨过 {THRESH}) ≈ {xc:.3f}")


if __name__ == "__main__":
    main()
