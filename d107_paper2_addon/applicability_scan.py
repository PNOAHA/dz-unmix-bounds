# -*- coding: utf-8 -*-
"""
第二篇方法论文 · DZ-Unmix 适用边界扫描 — 核心模块
====================================================
系统刻画 DZ-Unmix 在三个**数据难度轴**上的行为（方法边界），
完全**复用**第一篇 / QN2025106 的 `dz_unmixing_experiment.py`（已过 11/11 单测），
只补它没暴露的上层旋钮——**不修改那个审稿中/软著的文件一个字**。

三个轴（其它轴各自固定在 base case）:
  srcK  — 源数 K（2..6）           : 反问题维度 / endmember 个数
  noise — 测量噪声 noise_pct（0..0.2）: U-Pb 分析误差稳健性
  sep   — 可分性 sep_index = Δ/σ    : 源峰间距相对峰宽，越小越难分

每个 config = 一次 5-fold CV（重生成数据 + 重算 KDE），同时跑
supervised（已知 endmember H）与 unsupervised（NMF 学 H + 对齐）两法，
以便讲"难度↑时监督优势的保持/崩塌"。

注: R1（α×bandwidth 超参稳健性）已归第一篇（集群论文 §6.5/Fig6），本台架不重做。

本地冒烟测试（小网格，每轴几格）:
  & "C:/Users/Administrator/miniconda3/python.exe" applicability_scan.py --smoke

作者: 潘路加 — 2026-05-21
"""

import os
import sys
import argparse

import numpy as np
from sklearn.model_selection import KFold


# ----------------------------------------------------------------------------
# 0. 把第一篇核心模块挂上 import 路径（兼容两种布局）
#    - 集群扁平部署：dz_unmixing_experiment.py 与本文件同目录 → 正常 import 即可
#    - 本地子目录：本文件在 第二篇_*/scan/，核心在项目根 → 回退到 ../..（或 DZ_CORE_DIR）
# ----------------------------------------------------------------------------
def _ensure_core_on_path():
    try:
        import dz_unmixing_experiment  # noqa: F401
        return
    except ImportError:
        pass
    cand = os.environ.get("DZ_CORE_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if cand not in sys.path:
        sys.path.insert(0, cand)


_ensure_core_on_path()

from dz_unmixing_experiment import (   # noqa: E402  (must follow path bootstrap)
    Source,
    DEFAULT_SOURCES,
    T_GRID,
    generate_sample,
    kde_density,
    build_source_endmembers,
    supervised_unmix,
    fit_nmf,
    align_endmembers,
    predict_weights,
    report,
)


# ----------------------------------------------------------------------------
# 1. base case（其它轴扫描时把这些固定住）
# ----------------------------------------------------------------------------
BASE_ALPHA      = 1.0     # Dirichlet α（混合比例不均度），中性
BASE_NOISE      = 0.05    # 测量噪声基准（5%，QN2025106 默认）
BASE_BW         = 12.0    # KDE 带宽（QN2025106 默认）
BASE_SEP_INDEX  = 3.0     # 可分性基准 Δ/σ（峰间距 3 倍峰宽，易分）
BASE_SIGMA      = 25.0    # 等距源统一峰宽（Ma）
BASE_START_AGE  = 150.0   # 等距源起始峰位（Ma）
N_SAMPLES       = 1000
N_FOLDS         = 5


# ----------------------------------------------------------------------------
# 2. 可调难度的源工厂 + 扩展版数据生成器（第一篇没暴露的上层旋钮）
# ----------------------------------------------------------------------------
def make_equispaced_sources(k, sep_index=BASE_SEP_INDEX, sigma=BASE_SIGMA,
                            start_age=BASE_START_AGE):
    """k 个等距高斯源：峰间距 Δ = sep_index·σ，统一峰宽 σ。
    sep_index 小 → 峰挨得近 → 难分；大 → 易分。
    最大峰 = start_age + (k-1)·Δ，需落在 T_GRID 内（默认参数下 k≤6 / sep≤6 安全）。"""
    delta = sep_index * sigma
    src = []
    for i in range(k):
        mean_age = start_age + i * delta
        if mean_age >= T_GRID[-1]:
            raise ValueError(f"源 {i+1} 峰位 {mean_age} 超出年龄网格上限 {T_GRID[-1]}")
        src.append(Source(f"S{i+1}_syn", float(mean_age), float(sigma), f"synthetic-{i+1}"))
    return src


def generate_dataset_ext(n_samples=N_SAMPLES, sources=None, dirichlet_alpha=BASE_ALPHA,
                         noise_pct=BASE_NOISE, n_grains_range=(100, 300), seed=42):
    """generate_dataset 的扩展版——透传 `sources` 与 `noise_pct`（底层 generate_sample 早已支持）。
    返回 (ages_list, true_weights)，与原版同形。"""
    if sources is None:
        sources = DEFAULT_SOURCES
    rng = np.random.default_rng(seed)
    n_src = len(sources)
    true_w = rng.dirichlet(alpha=np.ones(n_src) * dirichlet_alpha, size=n_samples)
    ages_list = [
        generate_sample(w, n_grains_range=n_grains_range, noise_pct=noise_pct,
                        sources=sources, rng=rng)
        for w in true_w
    ]
    return ages_list, true_w


# ----------------------------------------------------------------------------
# 3. 单 config：5-fold CV，supervised + unsupervised 两法
# ----------------------------------------------------------------------------
def _agg(d, prefix):
    o = {}
    for key, arr in d.items():
        a = np.array(arr, dtype=float)
        o[f"{prefix}_{key}_mean"] = float(np.nanmean(a)) if a.size else float("nan")
        o[f"{prefix}_{key}_std"] = float(np.nanstd(a)) if a.size else float("nan")
    return o


def run_one_ext(sources, dirichlet_alpha=BASE_ALPHA, noise_pct=BASE_NOISE,
                bandwidth=BASE_BW, seed=42, n_samples=N_SAMPLES, n_folds=N_FOLDS,
                methods=("supervised", "unsupervised")):
    """单一难度 config 的 5-fold CV 平均（重生成数据、重算 KDE）。
    返回扁平 dict，含 sup_* / uns_* 指标。"""
    k = len(sources)
    ages_list, true_w = generate_dataset_ext(
        n_samples=n_samples, sources=sources,
        dirichlet_alpha=dirichlet_alpha, noise_pct=noise_pct, seed=seed,
    )
    X = np.array([kde_density(a, bandwidth=bandwidth) for a in ages_list])

    out = {
        "n_sources": k, "dirichlet_alpha": float(dirichlet_alpha),
        "noise_pct": float(noise_pct), "bandwidth": float(bandwidth),
        "seed": int(seed), "chance_top1": 1.0 / k,
    }
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if "supervised" in methods:
        H = build_source_endmembers(sources)
        acc = {"top1": [], "top1_dom": [], "mae": [], "r2": []}
        for _, te_idx in kf.split(X):
            r = report(true_w[te_idx], supervised_unmix(X[te_idx], H))
            acc["top1"].append(r["top1_accuracy"])
            acc["top1_dom"].append(r["top1_accuracy_dominant"])
            acc["mae"].append(r["weights_MAE"])
            acc["r2"].append(r["weights_R2"])
        out.update(_agg(acc, "sup"))

    if "unsupervised" in methods:
        acc = {"top1": [], "top1_dom": [], "mae": [], "r2": []}
        for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
            nmf = fit_nmf(X[tr_idx], n_components=k, seed=seed + fold_i)
            perm = align_endmembers(nmf, sources)
            pred = predict_weights(X[te_idx], nmf)[:, perm]
            r = report(true_w[te_idx], pred)
            acc["top1"].append(r["top1_accuracy"])
            acc["top1_dom"].append(r["top1_accuracy_dominant"])
            acc["mae"].append(r["weights_MAE"])
            acc["r2"].append(r["weights_R2"])
        out.update(_agg(acc, "uns"))

    # 主指标沿用第一篇口径：supervised dominant 子集 Top-1 ≥ 0.85
    primary = out.get("sup_top1_dom_mean", out.get("uns_top1_dom_mean", float("nan")))
    out["threshold_met"] = bool(primary >= 0.85) if primary == primary else None  # nan-safe
    return out


# ----------------------------------------------------------------------------
# 4. 三个轴的 config 列表构造（HPC 版与本地版共用）
# ----------------------------------------------------------------------------
def build_axis_configs(axis, seeds, *, k_values=range(2, 7), noise_grid=None,
                       sep_grid=None, sigma=BASE_SIGMA, dirichlet_alpha=BASE_ALPHA,
                       bandwidth=BASE_BW):
    """按轴生成 config 列表。每个 config 是 run_one_ext 的 kwargs + 标注字段。
    其它轴固定在 base case。"""
    cfgs = []
    if axis == "srcK":
        for k in k_values:
            for s in seeds:
                cfgs.append(dict(
                    axis="srcK", axis_value=int(k), sep_index=BASE_SEP_INDEX,
                    sources=make_equispaced_sources(k, BASE_SEP_INDEX, sigma),
                    dirichlet_alpha=dirichlet_alpha, noise_pct=BASE_NOISE,
                    bandwidth=bandwidth, seed=s,
                ))
    elif axis == "noise":
        grid = noise_grid if noise_grid is not None else np.linspace(0.0, 0.20, 40)
        for npct in grid:
            for s in seeds:
                cfgs.append(dict(
                    axis="noise", axis_value=round(float(npct), 5), sep_index=None,
                    sources=DEFAULT_SOURCES,   # 噪声轴用真实 NE-Asia 3 源
                    dirichlet_alpha=dirichlet_alpha, noise_pct=round(float(npct), 5),
                    bandwidth=bandwidth, seed=s,
                ))
    elif axis == "sep":
        grid = sep_grid if sep_grid is not None else np.linspace(0.3, 6.0, 40)
        for sep in grid:
            for s in seeds:
                cfgs.append(dict(
                    axis="sep", axis_value=round(float(sep), 4), sep_index=round(float(sep), 4),
                    sources=make_equispaced_sources(3, float(sep), sigma),  # 固定 k=3，变可分性
                    dirichlet_alpha=dirichlet_alpha, noise_pct=BASE_NOISE,
                    bandwidth=bandwidth, seed=s,
                ))
    else:
        raise ValueError(f"未知 axis: {axis}（应为 srcK / noise / sep）")
    return cfgs


def run_config(cfg, methods=("supervised", "unsupervised")):
    """跑一个 build_axis_configs 出来的 cfg，结果带回 axis / axis_value / sep_index 标注。"""
    res = run_one_ext(
        sources=cfg["sources"], dirichlet_alpha=cfg["dirichlet_alpha"],
        noise_pct=cfg["noise_pct"], bandwidth=cfg["bandwidth"], seed=cfg["seed"],
        methods=methods,
    )
    res["axis"] = cfg["axis"]
    res["axis_value"] = cfg["axis_value"]
    res["sep_index"] = cfg.get("sep_index")
    return res


# ----------------------------------------------------------------------------
# 5. 本地冒烟测试（小网格，验证两法管道都通）
# ----------------------------------------------------------------------------
def _smoke():
    import time
    seeds = [1, 2]
    plans = [
        ("srcK",  dict(k_values=[2, 3, 4])),
        ("noise", dict(noise_grid=[0.0, 0.10, 0.20])),
        ("sep",   dict(sep_grid=[0.5, 2.0, 5.0])),
    ]
    for axis, kw in plans:
        cfgs = build_axis_configs(axis, seeds, **kw)
        print(f"\n=== axis={axis}  {len(cfgs)} configs ({len(seeds)} seeds) ===")
        t0 = time.time()
        for c in cfgs:
            r = run_config(c, methods=("supervised", "unsupervised"))
            print(f"  {axis}={r['axis_value']!s:>6}  k={r['n_sources']}  seed={r['seed']}  "
                  f"sup_top1={r.get('sup_top1_mean', float('nan')):.4f}  "
                  f"sup_top1_dom={r.get('sup_top1_dom_mean', float('nan')):.4f}  "
                  f"uns_top1={r.get('uns_top1_mean', float('nan')):.4f}  "
                  f"thr={r['threshold_met']}")
        print(f"  ({time.time()-t0:.1f}s)")
    print("\n[smoke] 三轴管道全通 OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DZ-Unmix 适用边界扫描（核心模块）")
    ap.add_argument("--smoke", action="store_true", help="跑小网格冒烟测试")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        print(__doc__)
        print("用 --smoke 跑冒烟测试；大规模扫描用 applicability_scan_hpc.py（HPC job array）。")
