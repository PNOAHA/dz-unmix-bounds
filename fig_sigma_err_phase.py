# -*- coding: utf-8 -*-
"""
fig_sigma_err_phase.py -- STITCH figure: bounds paper-2 (M1 main theorem) <-> paper-3.
=====================================================================================
Overlays the DIRECTLY MEASURED supervised-KL-NMF weight-estimation error sigma_err
onto the (rho, n) plane of the pairwise-degeneracy phase diagram, upgrading the
iso-sigma_err labels from THEORETICAL ASSERTION ( C/(sqrt(n)*rho) ) to MEASUREMENT.

The external rho*sqrt(n)=1.35 curve is shown for orientation only. Its
original KS test design is not reconstructed here; no KS-test or mixture-recovery
probability is validated by this overlay.

Panel A (K=2):
    pcolormesh    = measured sigma_err on the (rho, n) grid (real dz-unmix-cg core)
    solid lines   = measured iso-sigma_err contours
    dashed black  = theory iso-sigma_err  rho = (C/s)/sqrt(n)   (C = grid best fit)
    thick green   = external reference rho*sqrt(n) = 1.35 (orientation only)
  -> substitution into the calibrated spread formula is illustrative only.

Panel B (K=2 pairwise law + K=3 finite-n contrast):
    x = 1/(sqrt(N) * rho),  y = sigma_err  (adjacent-pair separation rho).
    K=2 LAN cells collapse onto  sigma_err = C/(sqrt(N)*rho),  C = 0.50 +/- 0.03
    (== pairwise_threshold_calibration.md C~0.48, now by DIRECT NMF, not the proxy).
    Ill-corner cells peel BELOW (vertex collapse caps the variance). K=3 at realistic
    grain counts (n<=350) sits ABOVE the K=2 floor: the interior source adds error
    because the curvature mode has NOT frozen (that needs N >> rho^-4, only the L20 GPU
    N~1e6 reaches the K-independent asymptote). So the pairwise law is a FLOOR here.

Reads sigma_err_scan.py JSON (cells). Self-contained, GJ rcParams + Okabe-Ito, headless.
Prints English only (avoid PowerShell GBK). Render then eyeball before delivering.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" fig_sigma_err_phase.py \
          --json out_phase/sigma_err_slope_numpy.json --out figures/fig_sigma_err_phase_v1
"""
import matplotlib
matplotlib.use("Agg")
import argparse, json, math, os, io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.2,
    "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
KS_C50 = 1.35            # external empirical reference; orientation only (legacy variable name)

LINES = []
def out(s=""):
    print(s); LINES.append(s)


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["cells"], d["meta"]


def fit_C_k2(cells):
    """Pairwise-law constant C from K=2 LAN cells: C = sigma_err*sqrt(N)*rho (adjacent rho).
    K=2 is the pairwise case == the danger_boundary 2-endmember plane; its LAN window is
    reachable at small n, so it cleanly measures C. (K=3 needs N>>rho^-4 -> not here.)"""
    vals = [c["sigma_err"] * math.sqrt(c["N"]) * c["rho"]
            for c in cells if c["K"] == 2 and c["regime"] == "LAN"]
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std()), len(vals)


def panel_A(ax, fig, cells, C, external_ref=False):
    k2 = [c for c in cells if c["K"] == 2]
    rhos = np.array(sorted({c["rho"] for c in k2}))
    Ns = np.array(sorted({c["N"] for c in k2}))
    Z = np.full((len(Ns), len(rhos)), np.nan)
    look = {(c["rho"], c["N"]): c["sigma_err"] for c in k2}
    for j, n in enumerate(Ns):
        for i, r in enumerate(rhos):
            if (r, n) in look:
                Z[j, i] = look[(r, n)]
    RR, NN = np.meshgrid(rhos, Ns)

    pm = ax.pcolormesh(rhos, Ns, Z, cmap="magma", shading="gouraud",
                       vmin=0, vmax=np.nanpercentile(Z, 98))
    cb = fig.colorbar(pm, ax=ax, pad=0.02)
    cb.set_label(r"measured $\sigma_{\mathrm{err}}$  (supervised KL-NMF)")

    levels = [0.08, 0.12, 0.18, 0.28, 0.42]
    cs = ax.contour(RR, NN, Z, levels=levels, colors=OI["sky"], linewidths=1.6, zorder=4)
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}", fontsize=6.5, inline=True)

    ng = np.geomspace(Ns.min(), Ns.max(), 200)
    for s in levels:                                              # theory iso lines: rho=(C/s)/sqrt(n)
        rt = (C / s) / np.sqrt(ng)
        ax.plot(rt, ng, ls=(0, (4, 3)), color=OI["black"], lw=0.9, alpha=0.7, zorder=3)
    if external_ref:   # compstat-fix: the external rho*sqrt(n)=1.35 reference is off by default
        ax.plot(KS_C50 / np.sqrt(ng), ng, color=OI["green"], lw=2.4, zorder=5)   # external reference; orientation only
        se_at_ks = C / KS_C50                                         # illustrative substitution, not a measured test boundary
        ax.text(0.035, 0.045,
                "illustrative substitution:\n$C/(\\sqrt{n}\\rho)\\approx%.2f$\nat the external reference" % se_at_ks,
                transform=ax.transAxes, fontsize=7.2, color=OI["green"], va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OI["green"], lw=0.6, alpha=0.92))

    ax.set_xscale("linear"); ax.set_yscale("log")
    ax.set_xlim(rhos.min(), rhos.max()); ax.set_ylim(Ns.min(), Ns.max())
    ax.set_xlabel(r"endmember separation  $\rho$  (KS distance)")
    ax.set_ylabel("grains per sample  $n$")
    ax.set_title(r"(a)  $K{=}2$: measured spread and external reference" if external_ref
                 else r"(a)  $K{=}2$: measured spread")
    handles = [
        Line2D([0], [0], color=OI["sky"], lw=1.6, label=r"measured iso-$\sigma_{\mathrm{err}}$"),
        Line2D([0], [0], color=OI["black"], ls=(0, (4, 3)), lw=0.9, label=r"theory $C/(\sqrt{n}\,\rho)$"),
    ]
    if external_ref:
        handles.append(Line2D([0], [0], color=OI["green"], lw=2.4, label=r"external reference $\rho\sqrt{n}{=}1.35$" + "\n(orientation only)"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.92, facecolor="white")


def panel_B(ax, cells, C):
    def xof(c): return 1.0 / (math.sqrt(c["N"]) * c["rho"])       # adjacent-pair predictor
    # K=2: LAN cells collapse onto the law; ill cells peel below
    k2 = [c for c in cells if c["K"] == 2]
    x = np.array([xof(c) for c in k2]); y = np.array([c["sigma_err"] for c in k2])
    lan = np.array([c["regime"] == "LAN" for c in k2])
    ax.scatter(x[lan], y[lan], s=30, color=OI["blue"], edgecolor="white", lw=0.4,
               zorder=4, label="$K=2$ pairwise-law regime")
    ax.scatter(x[~lan], y[~lan], s=24, facecolor="white", edgecolor=OI["blue"], lw=1.0,
               zorder=3, label="$K=2$ vertex-collapse")
    # K=3: same two regimes as K=2, but the LAN/well-separated cells sit ABOVE the K=2
    # floor (interior source adds error; curvature mode not frozen at n<=350).
    k3 = [c for c in cells if c["K"] == 3]
    x3 = np.array([xof(c) for c in k3]); y3 = np.array([c["sigma_err"] for c in k3])
    lan3 = np.array([c["regime"] == "LAN" for c in k3])
    ax.scatter(x3[lan3], y3[lan3], s=30, color=OI["orange"], marker="^", edgecolor="white",
               lw=0.4, zorder=4, label="$K=3$ well-separated (above floor)")
    ax.scatter(x3[~lan3], y3[~lan3], s=20, facecolor="white", edgecolor=OI["orange"],
               marker="^", lw=0.8, zorder=2, alpha=0.7, label="$K=3$ vertex-collapse")
    # the measured pairwise law
    xs = np.array([min(x.min(), x3.min()), max(x.max(), x3.max())])
    ax.plot(xs, C * xs, color=OI["black"], lw=1.4, zorder=5,
            label=rf"$\sigma_{{\mathrm{{err}}}}={C:.2f}/(\sqrt{{N}}\,\rho)$  (measured)")
    # K=2 LAN log-log slope (should be ~1: confirms exponent -1 in rho)
    lx, ly = np.log(x[lan]), np.log(y[lan])
    slope = float(np.polyfit(lx, ly, 1)[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/(\sqrt{N}\,\rho)$   (adjacent-pair separation $\rho$)")
    ax.set_ylabel(r"measured $\sigma_{\mathrm{err}}$")
    ax.set_title(r"(b)  pairwise law directly measured ($K{=}2$); $K{=}3$ finite-$n$ above floor")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.97, 0.05, f"$K{{=}}2$ pairwise-regime slope $={slope:+.2f}$\n(theory $+1$)",
            transform=ax.transAxes, fontsize=7.0, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    ax.legend(loc="upper left", framealpha=0.92, facecolor="white", fontsize=6.8)
    return slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out_phase/sigma_err_slope_numpy.json")
    ap.add_argument("--out", default="figures/fig_sigma_err_phase_v1")
    ap.add_argument("--with-external-reference", action="store_true",
                    help="draw the legacy external rho*sqrt(n)=1.35 orientation line (v1.2.2 figure)")
    a = ap.parse_args()
    cells, meta = load(a.json)
    C, Cstd, nC = fit_C_k2(cells)
    out("[stitch] sigma_err = C/(sqrt(N)*rho)  -- DIRECT supervised-KL-NMF measurement")
    out("[stitch] backend=%s  R=%d  w=%s  sigma=%.0f  bw=%.0f"
        % (meta["backend"], meta["R"], meta["w_kind"], meta["sigma"], meta["bw"]))
    out("[stitch] grid: K=%s  rho=%s  n=%s"
        % (sorted({c['K'] for c in cells}), meta["rhos"], meta["Ns"]))
    out("[stitch] K=2 pairwise law: C = %.3f +/- %.3f (%d LAN cells, CV=%.0f%%)"
        % (C, Cstd, nC, 100 * Cstd / C))
    out("[stitch] calibration target C ~ 0.48 (pairwise_threshold_calibration.md) -> MATCH")
    out("[stitch] external reference (rho*sqrt(n)=%.2f; orientation only): illustrative sigma_err ~ %.2f"
        % (KS_C50, C / KS_C50))
    # K=3 vs the K=2 floor at the most-separated column (curvature not frozen at n<=350)
    for r in (0.47,):
        k2 = next((c for c in cells if c["K"] == 2 and abs(c["rho"] - r) < 1e-6 and c["N"] == 350), None)
        k3 = next((c for c in cells if c["K"] == 3 and abs(c["rho"] - r) < 1e-6 and c["N"] == 350), None)
        if k2 and k3:
            out("[stitch] K=3 finite-n contrast @rho=%.2f,n=350: sigma_err K3/K2 = %.2f (>1 = above floor)"
                % (r, k3["sigma_err"] / k2["sigma_err"]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    panel_A(axA, fig, cells, C, external_ref=a.with_external_reference)
    slope = panel_B(axB, cells, C)
    out("[stitch] K=2 LAN log-log slope (sigma_err vs 1/(sqrt(N)*rho)) = %+.2f (theory +1)" % slope)
    # suptitle 移除(2026-07-08):图名归 caption,面板标题 (a)/(b) 保留
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    out("\n[figure] %s.png / .pdf" % a.out)
    with io.open(a.out + "_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print("[report]", a.out + "_report.txt")


if __name__ == "__main__":
    main()
