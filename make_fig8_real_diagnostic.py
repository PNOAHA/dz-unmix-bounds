# -*- coding: utf-8 -*-
"""
第二篇方法论文 · Fig8 真实物源配置判据诊断（SedGeo 改投新增）
==============================================================
读 dz-global-db 的 15 个真实 K1 样品（Songliao 11 + NCC north margin 4，
OneDZ/Puetz CC-BY 编译，逐样品 n_grains 见各 csv 注释头），对 DZ-Unmix
实际工作字典（S1 130±15 / S2 280±30 / S3 1900±100 Ma；只读引用
dz-unmix-cg DEFAULT_SOURCES，不改原码）做可识别性判据诊断：

  (a) 年龄轴：三端元签名 pdf + 15 个真实样品 KDE（灰细线）
  (b) 诊断平面 (N, ρ)：KS 不可分辨边界 ρ√N=1.35 双曲线 + 悬崖 ρ≈0.42
      横线；真实端元对（ρ≈1.0）与假想弧内细分对（130 vs 145 Ma）的落位

同时输出 fig8_diagnostic_numbers.json —— §5.3 正文引用的全部数字以脚本
输出为准（教训：勿凭印象写图注，见 verify_phase3d_claims.py）。

跑法:
  & "C:/Users/Administrator/miniconda3/python.exe" make_fig8_real_diagnostic.py
"""
import os
import json
import glob as globmod

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.special import erfinv

BASE = os.path.dirname(os.path.abspath(__file__))
# 数据目录：repo 布局（v1.1 起自带 data/）优先，本地工作布局（dz-global-db）兜底
_DATA_CANDIDATES = [
    os.path.join(BASE, "data", "realdata_k1_samples"),
    os.path.normpath(os.path.join(
        BASE, "..", "..", "dz-global-db", "data", "realdata_k1_samples")),
]
DATADIR = next(p for p in _DATA_CANDIDATES if os.path.isdir(p))
OUTDIR = os.path.join(BASE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8", "legend.fancybox": False,
})

# --- 字典（来源: dz-unmix-cg DEFAULT_SOURCES，只读） -----------------------
SOURCES = [
    ("S1 Mesozoic arc", 130.0, 15.0, OI["blue"]),
    ("S2 Palaeozoic CAOB", 280.0, 30.0, OI["green"]),
    ("S3 Archean\u2013Proterozoic craton", 1900.0, 100.0, OI["vermillion"]),
]

# --- 判据常数（与正文 §3/§5.2 严格一致） -----------------------------------
KS_BOUNDARY = 1.35        # KS 不可分辨边界 ρ√N ≈ 1.35（50% 可分性常数）
CLIFF_RHO = 0.42          # 悬崖 Δ/σ≈1.10 ⇔ ρ≈0.42（等 σ 对）
C_CURV_LO, C_CURV_HI = 0.70, 0.85   # 非钉扎支前因子（K=3 数值认证窗口）


def ks_gauss_pair(mu1, s1, mu2, s2):
    """两高斯的 KS 距离 = max_t |Φ1 − Φ2|（细网格数值求极值）"""
    t = np.linspace(min(mu1, mu2) - 6 * max(s1, s2),
                    max(mu1, mu2) + 6 * max(s1, s2), 200001)
    return float(np.max(np.abs(norm.cdf(t, mu1, s1) - norm.cdf(t, mu2, s2))))


def load_samples():
    out = []
    for fp in sorted(globmod.glob(os.path.join(DATADIR, "*.csv"))):
        meta, ages = {}, []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    if ":" in line:
                        k, v = line[1:].split(":", 1)
                        meta[k.strip()] = v.strip()
                elif line and line != "age_Ma":
                    ages.append(float(line))
        out.append({"file": os.path.basename(fp),
                    "sample": meta.get("sample_name", "?"),
                    "basin": meta.get("basin", "?"),
                    "n": len(ages),
                    "ages": np.asarray(ages)})
    return out


def main():
    samples = load_samples()
    ns = np.array([s["n"] for s in samples])

    # 真实端元对的 KS ρ
    pair_rho = {}
    for i in range(3):
        for j in range(i + 1, 3):
            key = f"{SOURCES[i][0].split()[0]}-{SOURCES[j][0].split()[0]}"
            pair_rho[key] = ks_gauss_pair(SOURCES[i][1], SOURCES[i][2],
                                          SOURCES[j][1], SOURCES[j][2])

    # 假想弧内细分：130±15 vs 145±15（Δ/σ=1.0，悬崖之下）与 130 vs 160（Δ/σ=2.0，之上）
    rho_split_145 = ks_gauss_pair(130.0, 15.0, 145.0, 15.0)
    rho_split_160 = ks_gauss_pair(130.0, 15.0, 160.0, 15.0)

    # 粒数受限层面的误差预测（well-separated regime, ρ≈1 → σ_err ≈ C_curv/√N）
    sig_lo = C_CURV_LO / np.sqrt(ns.max())
    sig_hi = C_CURV_HI / np.sqrt(ns.min())

    # 各样品的 KS 可分辨临界 ρ 与等效峰间距（等 σ 对: ρ = erf(Δ/(2√2σ)))
    rho_crit = KS_BOUNDARY / np.sqrt(ns)
    dmu_crit_s15 = 2 * np.sqrt(2) * 15.0 * erfinv(np.clip(rho_crit, 0, 0.999999))
    dmu_crit_s30 = 2 * np.sqrt(2) * 30.0 * erfinv(np.clip(rho_crit, 0, 0.999999))

    numbers = {
        "n_samples": len(samples),
        "basins": sorted({s["basin"] for s in samples}),
        "n_grains": {"min": int(ns.min()), "max": int(ns.max()),
                     "median": float(np.median(ns)), "total": int(ns.sum())},
        "pair_rho": {k: round(v, 6) for k, v in pair_rho.items()},
        "rho_split_130v145": round(rho_split_145, 4),
        "rho_split_130v160": round(rho_split_160, 4),
        "sigma_err_pred_range": [round(float(sig_lo), 4), round(float(sig_hi), 4)],
        "rho_crit_range": [round(float(rho_crit.min()), 4), round(float(rho_crit.max()), 4)],
        "dmu_crit_Ma_sigma15": [round(float(dmu_crit_s15.min()), 2), round(float(dmu_crit_s15.max()), 2)],
        "dmu_crit_Ma_sigma30": [round(float(dmu_crit_s30.min()), 2), round(float(dmu_crit_s30.max()), 2)],
        "split145_rho_sqrtN": [round(float(rho_split_145 * np.sqrt(ns.min())), 2),
                               round(float(rho_split_145 * np.sqrt(ns.max())), 2)],
        "per_sample": [{"sample": s["sample"], "basin": s["basin"], "n": s["n"]}
                       for s in samples],
    }
    with open(os.path.join(BASE, "fig8_diagnostic_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2, ensure_ascii=False)

    # ---------------- 图 ----------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) 年龄轴：端元签名 + 真实样品 KDE
    tgrid = np.geomspace(50, 2600, 1200)
    for s in samples:
        kde = gaussian_kde(s["ages"], bw_method="scott")
        d = kde(tgrid)
        ax_a.plot(tgrid, d / d.max(), color="0.75", lw=0.6, zorder=1)
    for name, mu, sg, col in SOURCES:
        d = norm.pdf(tgrid, mu, sg)
        ax_a.plot(tgrid, d / d.max(), color=col, lw=1.8, zorder=3, label=name)
    ax_a.set_xscale("log")
    ax_a.set_xlim(50, 2600)
    ax_a.set_ylim(0, 1.08)
    ax_a.set_xlabel("U\u2013Pb age (Ma)")
    ax_a.set_ylabel("normalized density")
    ax_a.plot([], [], color="0.75", lw=0.6,
              label=f"real samples (N={len(samples)})")
    leg = ax_a.legend(loc="upper right", handlelength=1.4)
    fr = leg.get_frame(); fr.set_linewidth(0.6); fr.set_edgecolor("0.8")
    ax_a.text(0.02, 0.96, "a", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="top")

    # (b) 诊断平面 (N, ρ)
    ngrid = np.geomspace(20, 600, 400)
    hyper = KS_BOUNDARY / np.sqrt(ngrid)
    ax_b.fill_between(ngrid, 0, np.minimum(hyper, 1.05), color="0.88", zorder=0)
    ax_b.plot(ngrid, hyper, color=OI["black"], lw=1.4)
    ax_b.axhline(CLIFF_RHO, color=OI["orange"], lw=1.4, ls="--")

    # 单侧向下 jitter：KS 距离 ρ≤1，任何点不得画到 1.0 之上
    jit = np.linspace(-0.024, 0.0, len(ns))
    ax_b.scatter(ns, np.minimum(np.full_like(ns, pair_rho["S1-S2"], dtype=float) + jit, 1.0),
                 s=14, color=OI["blue"], zorder=4)
    ax_b.scatter(ns, np.full_like(ns, rho_split_160, dtype=float), s=14,
                 marker="^", color=OI["green"], zorder=4)
    ax_b.scatter(ns, np.full_like(ns, rho_split_145, dtype=float), s=14,
                 marker="v", color=OI["vermillion"], zorder=4)

    # \u884c\u5185\u76f4\u6807\uff08\u66ff\u4ee3\u56fe\u4f8b\uff0c\u907f\u514d\u906e\u6321\uff09
    ax_b.text(130, 1.005, "real pairs S1\u2013S2, S2\u2013S3, S1\u2013S3\n"
              r"($\rho\approx1.0$)", fontsize=6.5, color=OI["blue"], va="center")
    ax_b.text(130, rho_split_160, "hypothetical arc split\n130 vs 160 Ma",
              fontsize=6.5, color=OI["green"], va="center")
    ax_b.text(170, rho_split_145 - 0.055, "hypothetical arc split\n130 vs 145 Ma",
              fontsize=6.5, color=OI["vermillion"], va="top")
    ax_b.text(300, KS_BOUNDARY / np.sqrt(300) + 0.025,
              r"$\rho\sqrt{N}=1.35$", fontsize=6.5, color=OI["black"],
              ha="center")
    ax_b.text(580, CLIFF_RHO + 0.025,
              r"accuracy cliff $\rho\approx0.42$ ($\Delta/\sigma\approx1.1$)",
              fontsize=6.5, color=OI["orange"], ha="right")

    ax_b.text(23, 0.06, "indistinguishable", fontsize=6.5, color="0.35")
    ax_b.text(26, 0.30, "distinguishable,\nweights unreliable",
              fontsize=6.5, color="0.35", va="center")
    ax_b.text(23, 0.62, "far from overlap regime", fontsize=6.5, color="0.35")

    ax_b.set_xscale("log")
    ax_b.set_xlim(20, 600)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_xlabel("grains per sample $N$")
    ax_b.set_ylabel(r"pairwise KS distance $\rho$")
    ax_b.text(0.02, 0.96, "b", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="top")

    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"Fig8_real_diagnostic.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
