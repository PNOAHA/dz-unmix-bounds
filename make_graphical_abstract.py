# -*- coding: utf-8 -*-
"""
Graphical abstract (2-panel) for the bounds paper.
Left  : MEASURED (not schematic) weight-estimate error vs separability rho for K=3
        (supervised KL-NMF, raw-sigma dictionary, bw=12 Ma, sigma=25 Ma, fixed endpoint
        weights w*=(5,4,3)/12, R=600 replicates per cell, noise 0), read from
        deposit_bundle/out_phase/sigma_err_slope_numpy.json (the Fig. 3 harness):
          * sigma_err = sqrt(lambda_max Cov(w_hat)) = sampling SD (bias EXCLUDED)
          * RMSE = sqrt(tr Cov + ||E w_hat - w*||^2) (bias INCLUDED)
          * pinned branch C/(sqrt(N) rho_1K), C=0.50, drawn only on its domain rho<rho_c
        F03 fix 2026-10-19: replaces the hand-drawn peak*(rho/rho_c)^0.7 curve, which rose
        with rho and contradicted the pinned branch C/(sqrt(N) rho_1K).
Right : the empirical separability cliff -- supervised vs unsupervised Top-1
        dominant accuracy vs Delta/sigma (read from results/applicability_sep.csv),
        with the critical Delta/sigma ~ 1.1 and the 0.85 pass threshold.
Okabe-Ito palette, headless, English-only prints (PowerShell GBK safe).
Run: & "C:/Users/Administrator/miniconda3/python.exe" make_graphical_abstract.py
"""
import matplotlib
matplotlib.use("Agg")
import os, csv, collections, statistics as st, re, json, math
from scipy.special import erf, erfinv
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
      "green": "#009E73", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7", "grey": "#808080"}
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)

# ---------------- Left: MEASURED sampling SD vs RMSE (K=3) ----------------
JSON = os.environ.get("SIGMA_ERR_JSON") or next(
    (p for p in (os.path.join(BASE, "out_phase", "sigma_err_slope_numpy.json"),
                 os.path.join(BASE, "deposit_bundle", "out_phase", "sigma_err_slope_numpy.json"))
     if os.path.exists(p)), os.path.join(BASE, "out_phase", "sigma_err_slope_numpy.json"))
d = json.load(open(JSON, encoding="utf-8"))
assert d["meta"]["w_kind"] == "endpoint" and abs(d["meta"]["bw"] - 12.0) < 1e-9
K = 3
wstar = np.linspace(1.0, 0.6, K); wstar = wstar / wstar.sum()      # endpoint design
C_PIN = 0.50                                                       # Section 3.4 calibration
RHO_C = 0.32                                                       # release threshold (D~0.33)
styles = {350: dict(color=OI["blue"], ms=4.2), 160: dict(color=OI["skyblue"], ms=3.6)}
for N, sty in styles.items():
    cells = sorted([c for c in d["cells"] if c["K"] == K and c["N"] == N], key=lambda c: c["rho"])
    r = np.array([c["rho"] for c in cells])
    sd = np.array([c["sigma_err"] for c in cells])
    tr = np.array([max(sum(c["sigma_err_tangent_eigs"]), 0.0) for c in cells])
    bias = np.array([np.linalg.norm(np.array(c["what_mean"]) - wstar) for c in cells])
    rmse = np.sqrt(tr + bias ** 2)
    axL.plot(r, rmse, ls="--", marker="s", mfc="white", color=sty["color"], ms=sty["ms"],
             lw=1.3, label=f"RMSE, $N$={N}")
    axL.plot(r, sd, ls="-", marker="o", color=sty["color"], ms=sty["ms"], lw=1.6,
             label=rf"sampling SD $\sigma_{{\mathrm{{err}}}}$, $N$={N}")
    print(f"K=3 N={N}: rho={list(r)}")
    print("   sd  =", np.round(sd, 3).tolist())
    print("   bias=", np.round(bias, 3).tolist())
    print("   rmse=", np.round(rmse, 3).tolist())
# pinned branch, ONLY on its stated domain rho < rho_c (Section 3.3)
rr = np.linspace(0.05, RHO_C, 100)
rho1K = erf((K - 1) * erfinv(rr))          # extreme-pair KS distance, equispaced equal-sigma
axL.plot(rr, C_PIN / (math.sqrt(350) * rho1K), color=OI["black"], lw=1.0, ls=":",
         label=r"pinned branch $C/(\sqrt{N}\,\rho_{1K})$, $N$=350")
axL.axvline(RHO_C, color=OI["grey"], lw=0.8, ls="-.")
axL.text(RHO_C + 0.006, 0.012, r"$\rho_c\approx0.32$", fontsize=6.8, color="0.3", va="bottom")
axL.text(0.07, 0.36, "pinned: $\\hat w_{\\mathrm{mid}}\\to0$,\nbias $\\approx0.4$ persists",
         fontsize=6.6, color=OI["vermillion"], ha="left", va="bottom")
axL.set_xlim(0.03, 0.49); axL.set_ylim(0, 0.53)
axL.set_xlabel(r"adjacent-pair separability $\rho$ (KS distance)")
axL.set_ylabel("weight-estimate error ($K=3$)")
axL.set_title("Measured: sampling SD vs total error", pad=24)
axL.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=5.6,
           handlelength=2.0, borderaxespad=0.2, labelspacing=0.3)
secx = axL.secondary_xaxis("top", functions=(lambda x: 2 * np.sqrt(2) * erfinv(np.clip(x, 0, 0.999)),
                                             lambda s: erf(np.asarray(s) / (2 * np.sqrt(2)))))
secx.set_xlabel(r"$\Delta/\sigma$", fontsize=7, labelpad=2)
secx.tick_params(labelsize=6.5, length=2.5)
axL.text(-0.14, 1.02, "(a)", transform=axL.transAxes, fontsize=10,
         fontweight="bold", ha="left", va="bottom")

# ---------------- Right: empirical separability cliff ----------------
def load_sep():
    agg = collections.defaultdict(list); aggu = collections.defaultdict(list)
    with open(os.path.join(BASE, "results", "applicability_sep.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sv = r.get("sep_index", "")
            try:
                sv = float(sv)
            except Exception:
                m = re.findall(r"[-\d.]+", r.get("axis_value", "")); sv = float(m[0]) if m else None
            if sv is None:
                continue
            agg[sv].append(float(r["sup_top1_dom_mean"]))
            aggu[sv].append(float(r["uns_top1_dom_mean"]))
    xs = sorted(agg)
    return (np.array(xs),
            np.array([st.mean(agg[x]) for x in xs]),
            np.array([st.mean(aggu[x]) for x in xs]))

x, sup, uns = load_sep()
axR.plot(x, sup, "-o", color=OI["blue"], ms=3.5, label="supervised (known signatures)")
axR.plot(x, uns, "--s", color=OI["vermillion"], ms=3.0, label="unsupervised (blind)")
axR.axhline(0.85, color=OI["black"], lw=1.0, ls=":")
axR.text(x.max(), 0.86, "0.85 pass", ha="right", va="bottom", fontsize=6.8)
axR.axvline(1.1, color=OI["green"], lw=1.0, ls="--")
axR.text(1.15, 0.45, r"critical $\Delta/\sigma\approx1.1$", color=OI["green"],
         fontsize=7.0, rotation=90, va="center")
axR.set_xlim(left=0); axR.set_ylim(0.2, 1.02)
axR.set_xlabel(r"source separability $\Delta/\sigma$")
axR.set_ylabel("Top-1 dominant accuracy")
axR.set_title("Empirical separability cliff (HPC)", pad=24)
axR.legend(loc="lower right", frameon=False, fontsize=6.8)
axR.text(-0.14, 1.02, "(b)", transform=axR.transAxes, fontsize=10,
         fontweight="bold", ha="left", va="bottom")

# --- shared cosmetic polish: despine + outward ticks ---
for _ax in (axL, axR):
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.tick_params(direction="out", length=3.0, width=0.8)

for ext in ("png", "pdf"):
    out = os.path.join(BASE, "figures", f"GraphicalAbstract.{ext}")
    fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.1f} KB)")
plt.close(fig)
print("sep points:", len(x), "| sup range %.3f-%.3f" % (sup.min(), sup.max()),
      "| uns range %.3f-%.3f" % (uns.min(), uns.max()))
print("done.")
