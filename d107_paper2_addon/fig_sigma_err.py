# -*- coding: utf-8 -*-
"""
fig_sigma_err.py — M1 main-theorem figure: constrained-MLE estimation error
sigma_err(K, rho, N) and the three rho-regimes (ill / LAN / saturated).

Reads sigma_err_scan.py's JSON output and draws, per K, log-log sigma_err vs rho
for each sample size N, with:
  - filled markers = LAN cells, open markers = ill-conditioned cells,
  - a dashed reference line of the theoretical LAN slope -(K-1),
  - the fitted LAN-window slope annotated per N.

Self-contained (inlines GJ rcParams + Okabe-Ito), renders headless (Agg).

Run:  & "C:/Users/Administrator/miniconda3/python.exe" fig_sigma_err.py \
          --json out_prelim/sigma_err_slope_numpy.json --out figures/fig_sigma_err_v1
"""
import matplotlib
matplotlib.use("Agg")
import argparse, json, math, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
})
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
NCOL = {}  # filled per-run from the data (small -> warm, large -> cool)
NPALETTE = [OI["vermillion"], OI["orange"], OI["green"], OI["blue"], OI["purple"]]


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["cells"], d["fits"], d["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out_prelim/sigma_err_slope_numpy.json")
    ap.add_argument("--out", default="figures/fig_sigma_err_v1")
    a = ap.parse_args()
    cells, fits, meta = load(a.json)
    Ks = sorted({c["K"] for c in cells})
    Ns = sorted({c["N"] for c in cells})
    for i, N in enumerate(Ns):
        NCOL[N] = NPALETTE[i % len(NPALETTE)]
    fitmap = {(f["K"], f["N"]): f for f in fits}

    fig, axes = plt.subplots(1, len(Ks), figsize=(3.5 * len(Ks), 3.6), squeeze=False)
    axes = axes[0]
    for ax, K in zip(axes, Ks):
        for N in Ns:
            pts = sorted([c for c in cells if c["K"] == K and c["N"] == N], key=lambda c: c["rho"])
            if not pts:
                continue
            rho = np.array([p["rho"] for p in pts]); se = np.array([p["sigma_err"] for p in pts])
            ax.plot(rho, se, "-", color=NCOL[N], lw=0.9, zorder=2)
            for p in pts:
                lan = p["regime"] == "LAN"
                ax.plot(p["rho"], p["sigma_err"], "o", ms=4.5, zorder=3,
                        mfc=NCOL[N] if lan else "white", mec=NCOL[N], mew=1.0)
        # theory reference slope -(K-1): anchor on the largest-N curve's mid LAN point
        ref_pts = sorted([c for c in cells if c["K"] == K and c["N"] == max(Ns) and c["regime"] == "LAN"],
                         key=lambda c: c["rho"])
        if len(ref_pts) >= 1:
            anchor = ref_pts[len(ref_pts) // 2]
            r0, s0, e = anchor["rho"], anchor["sigma_err"], (K - 1)
            rr = np.array([min(c["rho"] for c in cells if c["K"] == K),
                           max(c["rho"] for c in cells if c["K"] == K)])
            ax.plot(rr, s0 * (rr / r0) ** (-e), "--", color=OI["black"], lw=0.9, alpha=0.55,
                    zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"endmember KS distance  $\rho$")
        ax.set_title(f"$K={K}$")
        ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.4)
        # per-N fitted LAN slope annotation
        lines = []
        for N in Ns:
            f = fitmap.get((K, N))
            if f and not math.isnan(f.get("slope", float("nan"))):
                lines.append(f"N={N:,}:  slope={f['slope']:+.2f}  [{f['n_points']}pt]")
            else:
                lines.append(f"N={N:,}:  (no LAN window)")
        ax.text(0.03, 0.03, "\n".join(lines), transform=ax.transAxes, fontsize=6.6,
                va="bottom", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    axes[0].set_ylabel(r"$\sigma_{\mathrm{err}}$  (worst tangent-direction std)")

    # shared legend: N colours + regime marker meaning
    handles = [Line2D([0], [0], color=NCOL[N], marker="o", lw=1.2, ms=5, mfc=NCOL[N],
                      label=f"N = {N:,}") for N in Ns]
    handles += [
        Line2D([0], [0], color="0.4", marker="o", lw=0, ms=5, mfc="0.4", label="LAN cell"),
        Line2D([0], [0], color="0.4", marker="o", lw=0, ms=5, mfc="white", mec="0.4", label="ill-conditioned"),
        Line2D([0], [0], color=OI["black"], ls="--", lw=0.9, alpha=0.55, label=r"theory $\rho^{-(K-1)}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.2, handletextpad=0.4)
    fig.suptitle("M1 main theorem — supervised KL-MLE estimation error vs endmember separation",
                 y=1.07, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}.png / .pdf")


if __name__ == "__main__":
    main()
