# -*- coding: utf-8 -*-
"""
make_fig6_phase_atlas.py — Fig 6 (3D phase atlas, K × sep × noise capstone)
============================================================================
第二篇方法论文 §3.5 配图（路线甲升级新增）。读 phase3d.csv (D107 数据 mirror)，
按 (K, sep, noise) 3-axis 切 3 个 noise panel = {0.05, 0.20, 0.40}，每 panel
= 11K × 25sep heatmap of supervised Top-1 dominant accuracy。
0.85 阈值用虚线等高线标。颜色对比本身展示 supervised 在高噪+高K区维持
~0.2-0.5 而 unsupervised 降到 chance ~1/K（详见 caption 与正文 §3.5）。

输出 PDF/PNG/SVG 三格式到 figures/，沿用 make_paper2_figures.py 配色 (Okabe-Ito)
+ 矢量可编辑 (pdf fonttype=42) + 三格式发表级。

数据：D107 ~/dz-unmix/paper2/phase3d.csv (534M, 467/1000 task 覆盖率,
~1.54M 行 = ~1541 cells × 1000 seeds)；scp 到本地 scan/phase3d.csv 后跑此脚本。

跑法（本地 headless）:
  & "C:/Users/Administrator/miniconda3/python.exe" make_fig6_phase_atlas.py
"""
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "scan", "phase3d.csv")
OUTDIR = os.path.join(BASE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# Okabe-Ito 色盲友好色板（沿用 make_paper2_figures.py 一致）
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

THRESH = 0.85          # supervised pass threshold (沿用 §3)
NOISE_TARGETS = [0.05, 0.20, 0.40]   # 三个 panel（用户 2026-06-01 路线甲拍板）


def load_aggregate():
    """读 phase3d.csv, 按 (K, sep, noise) 跨 seed 聚合 sup_dom/uns_dom mean."""
    print(f"Loading {CSV} ...", flush=True)
    if not os.path.exists(CSV):
        raise FileNotFoundError(
            f"{CSV} 不存在。先 scp D107: "
            f"scp panlujia@192.168.108.200:~/dz-unmix/paper2/phase3d.csv {CSV}"
        )
    agg = defaultdict(lambda: {"sup_dom": [], "uns_dom": []})
    n_rows = 0
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            K = int(r["n_sources_request"])
            sep = float(r["sep_index"])
            noise = float(r["noise_pct"])
            agg[(K, sep, noise)]["sup_dom"].append(float(r["sup_top1_dom_mean"]))
            agg[(K, sep, noise)]["uns_dom"].append(float(r["uns_top1_dom_mean"]))
            n_rows += 1
    print(f"  {n_rows} rows -> {len(agg)} unique (K, sep, noise) cells", flush=True)
    return agg


def closest(target, options):
    """Find the available value closest to target."""
    return min(options, key=lambda x: abs(x - target))


def build_grid(agg, K_vals, sep_vals, noise_target):
    """Return (sup_grid, uns_grid, n_grid) for fixed noise; NaN if uncovered."""
    sup = np.full((len(K_vals), len(sep_vals)), np.nan)
    uns = np.full((len(K_vals), len(sep_vals)), np.nan)
    n = np.zeros((len(K_vals), len(sep_vals)), dtype=int)
    for i, K in enumerate(K_vals):
        for j, sep in enumerate(sep_vals):
            cell = agg.get((K, sep, noise_target))
            if cell and cell["sup_dom"]:
                sup[i, j] = float(np.mean(cell["sup_dom"]))
                uns[i, j] = float(np.mean(cell["uns_dom"]))
                n[i, j] = len(cell["sup_dom"])
    return sup, uns, n


def main():
    agg = load_aggregate()

    # 索引轴
    K_vals = sorted({k[0] for k in agg.keys()})
    sep_vals = sorted({k[1] for k in agg.keys()})
    available_noises = sorted({k[2] for k in agg.keys()})

    print(f"  K range: {K_vals}", flush=True)
    print(f"  sep range: {sep_vals[0]:.2f}..{sep_vals[-1]:.2f} "
          f"({len(sep_vals)} values)", flush=True)
    print(f"  available noise: {available_noises}", flush=True)

    selected_noises = [closest(t, available_noises) for t in NOISE_TARGETS]
    print(f"  selected noise panels: {selected_noises}", flush=True)

    # 颜色映射：viridis 色盲安全 + perceptually uniform，覆盖 [0, 1] sup_dom
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")  # NaN = 未覆盖 cell（534 个 SLURM timeout 的部分）

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)
    im = None

    panel_letters = "abc"
    for ax, n_target in zip(axes, selected_noises):
        sup, uns, n = build_grid(agg, K_vals, sep_vals, n_target)
        coverage = int(np.sum(~np.isnan(sup)))
        total = sup.size
        print(f"  noise={n_target}: {coverage}/{total} cells covered, "
              f"mean n_seeds={np.mean(n[n > 0]):.0f}", flush=True)

        # imshow with K vertical, sep horizontal
        # extent: x = sep range (categorical 但 imshow 用线性，OK 给 reader 直观)
        im = ax.imshow(
            sup, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
            extent=[sep_vals[0] - 0.5*(sep_vals[1]-sep_vals[0]),
                    sep_vals[-1] + 0.5*(sep_vals[-1]-sep_vals[-2]),
                    K_vals[0] - 0.5, K_vals[-1] + 0.5],
        )
        # 0.85 阈值等高线（沿用 §3 关键阈值）
        if np.any(~np.isnan(sup)):
            sep_grid, K_grid = np.meshgrid(sep_vals, K_vals)
            with np.errstate(invalid="ignore"):
                try:
                    ax.contour(sep_grid, K_grid, sup, levels=[THRESH],
                               colors=OI["black"], linewidths=1.0,
                               linestyles="--")
                except Exception as e:
                    print(f"    [warn] contour failed: {e}", flush=True)

        ax.set_title(f"noise = {n_target:.2f}", fontsize=9, pad=6)
        ax.text(0.0, 1.045, f"({panel_letters[list(axes).index(ax)]})",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
        ax.set_xlabel(r"Source separability $\Delta/\sigma$")
        if ax is axes[0]:
            ax.set_ylabel(r"Number of sources $K$")
        ax.set_yticks(K_vals)
        ax.set_xlim(sep_vals[0]-0.1, sep_vals[-1]+0.1)

    # Colorbar (shared, 右侧)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, fraction=0.025, pad=0.02)
    cbar.set_label(r"Supervised Top-1 dominant accuracy")
    # 0.85 mark 在 colorbar
    cbar.ax.axhline(THRESH, color=OI["black"], linestyle="--", linewidth=0.8)
    cbar.ax.text(1.6, THRESH, f" {THRESH}", va="center",
                 transform=cbar.ax.get_yaxis_transform(), fontsize=7)

    # suptitle 用一行；panel title 自带 pad=4 隔开
    # suptitle 移除(2026-07-08):"Fig. 6" 字样不得烤进图(与排版 caption 编号冲突),图名归 caption

    for ext in ("pdf", "png", "svg"):
        out = os.path.join(OUTDIR, f"Fig6_phase_atlas.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        size_kb = os.path.getsize(out) / 1024
        print(f"  wrote {out}  ({size_kb:.1f} KB)", flush=True)

    plt.close(fig)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
