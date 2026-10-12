# -*- coding: utf-8 -*-
"""
multi_init_smoke_test.py — 治本方案验证 (v1 current vs v2 multi-init)
======================================================================
只 monkey-patch fit_nmf，不动 production code。run_one_ext supervised
path 走 NNLS 闭式解（不调 fit_nmf），所以 monkey-patch 只影响 unsupervised
——正是 paper2 §5 selling point 关注的对比。

测试设计:
  3 cells × 2 seeds × 2 variants
  v1 = original     (nndsvda + max_iter=2000 + tol=1e-6, n_init=1)
  v2 = multi-init   (1 nndsvda + 7 random + max_iter=500 + tol=1e-4)

决策标准（脚本结尾汇总）:
  - if Δuns_dom > +0.05 in ≥2/3 cells  → v2 治本生效，建议上 production
  - if time_ratio < 3 in all cells     → 时间代价可接受
  - both must hold                      → 改 dz_unmixing_experiment.py
"""

import time
import warnings
import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

# Import production stack
# 注意顺序：applicability_scan 在 import 时会做 sys.path bootstrap
# （把 ~/dz-unmix/ 加进去让 dz_unmixing_experiment 可见）。必须先 import 它。
import applicability_scan
import dz_unmixing_experiment as dz_exp
from phase3d_scan import build_phase3d_configs, run_phase3d_config

# 保留原 fit_nmf 引用（v1）
fit_nmf_v1 = dz_exp.fit_nmf


def fit_nmf_v2(X_train, n_components=3, seed=0, n_init=8):
    """v2: multi-init NMF，取 reconstruction loss 最小者."""
    best_nmf = None
    best_loss = np.inf
    for i in range(n_init):
        nmf = NMF(
            n_components=n_components,
            init='nndsvda' if i == 0 else 'random',
            random_state=seed * 1000 + i,
            max_iter=500,
            tol=1e-4,
            solver='cd',
        )
        nmf.fit_transform(X_train)
        if nmf.reconstruction_err_ < best_loss:
            best_loss = nmf.reconstruction_err_
            best_nmf = nmf
    return best_nmf


# 3 个 high-stress cell —— 来自 21570 / 24576 失败区典型代表
TEST_CELLS = [
    dict(K=6,  sep=1.0, noise=0.20, label='medium      (K=6,  sep=1.0, noise=0.20)'),
    dict(K=10, sep=1.0, noise=0.25, label='hard        (K=10, sep=1.0, noise=0.25)'),
    dict(K=12, sep=0.6, noise=0.30, label='extreme     (K=12, sep=0.6, noise=0.30)'),
]
SEEDS_PER_CELL = 2


def patch_fit_nmf(fit_fn):
    """关键：必须 patch applicability_scan 的本地 binding，不能只 patch dz_exp"""
    dz_exp.fit_nmf = fit_fn
    applicability_scan.fit_nmf = fit_fn


def run_one_cell(cell, fit_fn, variant_label):
    cfgs = build_phase3d_configs(
        k_values=[cell['K']],
        sep_values=[cell['sep']],
        noise_values=[cell['noise']],
        seeds=list(range(1, SEEDS_PER_CELL + 1)),
    )
    patch_fit_nmf(fit_fn)

    times, results = [], []
    for cfg in cfgs:
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter('always')
            r = run_phase3d_config(cfg)
        elapsed = time.time() - t0
        n_warn = sum(1 for w in w_list
                     if issubclass(w.category, ConvergenceWarning))
        times.append(elapsed)
        results.append(r)
        print(f"    [{variant_label}] seed={cfg['seed']}  "
              f"wall={elapsed:7.1f}s  "
              f"sup_dom={r.get('sup_top1_dom_mean', float('nan')):.4f}  "
              f"uns_dom={r.get('uns_top1_dom_mean', float('nan')):.4f}  "
              f"ConvergenceWarnings={n_warn}",
              flush=True)
    return dict(
        wall_total=float(sum(times)),
        sup_dom=float(np.nanmean(
            [r.get('sup_top1_dom_mean', np.nan) for r in results])),
        uns_dom=float(np.nanmean(
            [r.get('uns_top1_dom_mean', np.nan) for r in results])),
    )


def main():
    print("=" * 82, flush=True)
    print("  multi-init smoke test", flush=True)
    print("  3 cells x 2 seeds x 2 variants", flush=True)
    print("  v1 = original   (nndsvda + max_iter=2000 + tol=1e-6, n_init=1)",
          flush=True)
    print("  v2 = multi-init (1 nndsvda + 7 random + max_iter=500 + tol=1e-4)",
          flush=True)
    print("=" * 82, flush=True)

    summary = []
    t_start = time.time()
    for cell in TEST_CELLS:
        print(f"\n--- {cell['label']} ---", flush=True)

        print("  [v1] running ...", flush=True)
        r1 = run_one_cell(cell, fit_nmf_v1, 'v1')
        print("  [v2] running ...", flush=True)
        r2 = run_one_cell(cell, fit_nmf_v2, 'v2')

        time_ratio = r2['wall_total'] / r1['wall_total']
        uns_delta = r2['uns_dom'] - r1['uns_dom']
        sup_delta = r2['sup_dom'] - r1['sup_dom']
        print(f"\n  Summary for this cell:", flush=True)
        print(f"    v1: wall={r1['wall_total']:7.1f}s   "
              f"sup_dom={r1['sup_dom']:.4f}   uns_dom={r1['uns_dom']:.4f}",
              flush=True)
        print(f"    v2: wall={r2['wall_total']:7.1f}s   "
              f"sup_dom={r2['sup_dom']:.4f}   uns_dom={r2['uns_dom']:.4f}",
              flush=True)
        print(f"    delta:  time x{time_ratio:.2f}   "
              f"Dsup={sup_delta:+.4f} (sanity~0)   "
              f"Duns={uns_delta:+.4f}",
              flush=True)
        summary.append((cell['label'], time_ratio, uns_delta, sup_delta))

    print("\n" + "=" * 82, flush=True)
    print(f"  Final summary  (total wall: {time.time()-t_start:.0f}s)",
          flush=True)
    print("=" * 82, flush=True)
    print(f"  {'cell':<48}  {'time_x':>8}  {'Duns':>8}  {'Dsup':>8}",
          flush=True)
    print(f"  {'-'*48}  {'-'*8}  {'-'*8}  {'-'*8}", flush=True)
    for label, tr, ud, sd in summary:
        print(f"  {label:<48}  {tr:>8.2f}  {ud:>+8.4f}  {sd:>+8.4f}",
              flush=True)

    print("\nDecision:", flush=True)
    duns_wins = sum(1 for _, _, ud, _ in summary if ud > 0.05)
    time_ok = all(tr < 3.0 for _, tr, _, _ in summary)
    print(f"  uns_dom improved >0.05 in {duns_wins}/3 cells", flush=True)
    print(f"  time_ratio < 3 in all cells: {time_ok}", flush=True)
    if duns_wins >= 2 and time_ok:
        print("  -> GO: v2 treats root cause and is computationally feasible",
              flush=True)
    elif duns_wins >= 2:
        print("  -> CONDITIONAL: v2 works but is expensive; "
              "consider hierarchical retry (Tier 1 fast + Tier 2 multi-init)",
              flush=True)
    else:
        print("  -> NO-GO: v2 does not significantly improve unsupervised; "
              "the root cause may be elsewhere",
              flush=True)


if __name__ == '__main__':
    main()
