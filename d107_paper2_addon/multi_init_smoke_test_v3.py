# -*- coding: utf-8 -*-
"""
multi_init_smoke_test_v3.py — Hierarchical Retry 验证 (3-variant 对照)
======================================================================
v3 是对 v2 smoke 失败案例（cell 1 K=6 medium-stress 退步 -0.13）的针对性设计。

设计思路:
  v1 (baseline):    nndsvda + max_iter=2000 + tol=1e-6, n_init=1  ← 原 production
  v3a (control):    nndsvda + max_iter=2000 + tol=1e-4, n_init=1  ← 只放宽 tol
  v3b (hierarchical): Tier 1 = v3a; 若 n_iter_<max_iter 用 Tier 1 结果；
                      否则 Tier 2 = 7 random init, 取 best reconstruction loss

为什么 v3a 必须 + 入对照:
  v3b Tier 1 用 tol=1e-4（而非 v1 的 1e-6），因为 v1 的 tol=1e-6 在 medium-
  stress cell 都 hit 不了，会让 "n_iter_<max_iter" 几乎永远 False，hierarchical
  退化成全 fallback。换 tol=1e-4 是必要修改，但这本身可能影响 medium-stress 行为
  ——v3a 隔离测试这个变量。

预期 (基于 v2 smoke 数据):
  Cell 1 (medium):  v3a ≈ v1 (~0.41)， v3b 走 Tier 1，结果 = v3a
  Cell 2 (hard):    v3a 不变差，v3b 走 Tier 2 fallback → 接近 v2 (~0.47)
  Cell 3 (extreme): 同 cell 2
"""

import time
import warnings
import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

# 必须先 import applicability_scan 触发 sys.path bootstrap
import applicability_scan
import dz_unmixing_experiment as dz_exp
from phase3d_scan import build_phase3d_configs, run_phase3d_config


# ----------------------------------------------------------------------------
# 三个 variant: 全部满足 fit_nmf(X, n_components, seed) 签名
# ----------------------------------------------------------------------------

def fit_nmf_v1(X_train, n_components=3, seed=0):
    """v1 baseline: 完全复刻 dz_unmixing_experiment.fit_nmf"""
    nmf = NMF(
        n_components=n_components, init='nndsvda', random_state=seed,
        max_iter=2000, tol=1e-6, solver='cd',
    )
    nmf.fit_transform(X_train)
    return nmf


def fit_nmf_v3a(X_train, n_components=3, seed=0):
    """v3a 对照: 只放宽 tol 到 1e-4，单 init"""
    nmf = NMF(
        n_components=n_components, init='nndsvda', random_state=seed,
        max_iter=2000, tol=1e-4, solver='cd',
    )
    nmf.fit_transform(X_train)
    _V3A_ITERS.append(nmf.n_iter_)
    return nmf


# Tier trace for v3b: 'tier1' = 收敛走 Tier 1 / 'tier2' = 退到 multi-init
_V3B_TIERS = []
_V3A_ITERS = []


def fit_nmf_v3b(X_train, n_components=3, seed=0):
    """v3b hierarchical: Tier 1 = nndsvda relaxed tol；不收敛则 Tier 2 multi-init."""
    nmf = NMF(
        n_components=n_components, init='nndsvda', random_state=seed,
        max_iter=2000, tol=1e-4, solver='cd',
    )
    nmf.fit_transform(X_train)
    if nmf.n_iter_ < 2000:
        _V3B_TIERS.append('tier1')
        return nmf

    # Tier 1 没收敛 → Tier 2 fallback: 7 个 random init 取 best
    _V3B_TIERS.append('tier2')
    best = nmf
    for i in range(7):
        nmf2 = NMF(
            n_components=n_components, init='random',
            random_state=seed * 1000 + i,
            max_iter=1000, tol=1e-4, solver='cd',
        )
        nmf2.fit_transform(X_train)
        if nmf2.reconstruction_err_ < best.reconstruction_err_:
            best = nmf2
    return best


# ----------------------------------------------------------------------------
# Test cells (same as v2 smoke for direct comparison)
# ----------------------------------------------------------------------------

TEST_CELLS = [
    dict(K=6,  sep=1.0, noise=0.20, label='medium      (K=6,  sep=1.0, noise=0.20)'),
    dict(K=10, sep=1.0, noise=0.25, label='hard        (K=10, sep=1.0, noise=0.25)'),
    dict(K=12, sep=0.6, noise=0.30, label='extreme     (K=12, sep=0.6, noise=0.30)'),
]
SEEDS_PER_CELL = 2


def patch_fit_nmf(fit_fn):
    """关键: 必须 patch 两边的 binding (Python from-import 创建本地引用)"""
    dz_exp.fit_nmf = fit_fn
    applicability_scan.fit_nmf = fit_fn


def run_one_cell(cell, fit_fn, variant_label, reset_trace=None):
    cfgs = build_phase3d_configs(
        k_values=[cell['K']],
        sep_values=[cell['sep']],
        noise_values=[cell['noise']],
        seeds=list(range(1, SEEDS_PER_CELL + 1)),
    )
    patch_fit_nmf(fit_fn)
    if reset_trace is not None:
        reset_trace.clear()

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
              f"ConvWarnings={n_warn}",
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
    print("  Hierarchical retry smoke test v3", flush=True)
    print("  3 cells x 2 seeds x 3 variants (v1 baseline, v3a relaxed-tol, v3b hierarchical)",
          flush=True)
    print("=" * 82, flush=True)

    summary = []
    t_start = time.time()
    for cell in TEST_CELLS:
        print(f"\n--- {cell['label']} ---", flush=True)

        print("  [v1] running (nndsvda + max_iter=2000 + tol=1e-6) ...", flush=True)
        r1 = run_one_cell(cell, fit_nmf_v1, 'v1 ')

        print("  [v3a] running (only relax tol -> 1e-4) ...", flush=True)
        r3a = run_one_cell(cell, fit_nmf_v3a, 'v3a', reset_trace=_V3A_ITERS)
        if _V3A_ITERS:
            iter_summary = (f"min={min(_V3A_ITERS)} "
                            f"median={int(np.median(_V3A_ITERS))} "
                            f"max={max(_V3A_ITERS)}")
            print(f"    [v3a iter stats] {iter_summary} "
                  f"(max_iter=2000; <2000 = converged)",
                  flush=True)

        print("  [v3b] running (hierarchical: tier1=v3a; tier2=7-init random fallback) ...",
              flush=True)
        r3b = run_one_cell(cell, fit_nmf_v3b, 'v3b', reset_trace=_V3B_TIERS)
        if _V3B_TIERS:
            t1 = _V3B_TIERS.count('tier1')
            t2 = _V3B_TIERS.count('tier2')
            print(f"    [v3b tier choices] tier1={t1}/{len(_V3B_TIERS)} "
                  f"  tier2(fallback)={t2}/{len(_V3B_TIERS)}",
                  flush=True)

        print(f"\n  Summary for this cell:", flush=True)
        print(f"    v1:  wall={r1['wall_total']:7.1f}s   uns_dom={r1['uns_dom']:.4f}",
              flush=True)
        print(f"    v3a: wall={r3a['wall_total']:7.1f}s   uns_dom={r3a['uns_dom']:.4f}   "
              f"D(v3a-v1)={r3a['uns_dom']-r1['uns_dom']:+.4f}",
              flush=True)
        print(f"    v3b: wall={r3b['wall_total']:7.1f}s   uns_dom={r3b['uns_dom']:.4f}   "
              f"D(v3b-v1)={r3b['uns_dom']-r1['uns_dom']:+.4f}",
              flush=True)

        summary.append(dict(
            label=cell['label'],
            uns_v1=r1['uns_dom'], uns_v3a=r3a['uns_dom'], uns_v3b=r3b['uns_dom'],
            wall_v1=r1['wall_total'], wall_v3a=r3a['wall_total'], wall_v3b=r3b['wall_total'],
            sup_v1=r1['sup_dom'], sup_v3a=r3a['sup_dom'], sup_v3b=r3b['sup_dom'],
        ))

    print("\n" + "=" * 82, flush=True)
    print(f"  Final summary (total wall: {time.time()-t_start:.0f}s)", flush=True)
    print("=" * 82, flush=True)

    print(f"\n  {'cell':<48}  {'uns_v1':>8}  {'uns_v3a':>8}  {'uns_v3b':>8}",
          flush=True)
    print(f"  {'-'*48}  {'-'*8}  {'-'*8}  {'-'*8}", flush=True)
    for s in summary:
        print(f"  {s['label']:<48}  "
              f"{s['uns_v1']:>8.4f}  {s['uns_v3a']:>8.4f}  {s['uns_v3b']:>8.4f}",
              flush=True)

    print(f"\n  Time (s):", flush=True)
    print(f"  {'cell':<48}  {'wall_v1':>8}  {'wall_v3a':>8}  {'wall_v3b':>8}",
          flush=True)
    print(f"  {'-'*48}  {'-'*8}  {'-'*8}  {'-'*8}", flush=True)
    for s in summary:
        print(f"  {s['label']:<48}  "
              f"{s['wall_v1']:>8.1f}  {s['wall_v3a']:>8.1f}  {s['wall_v3b']:>8.1f}",
              flush=True)

    print(f"\n  Sanity (sup_dom should be identical across variants):", flush=True)
    for s in summary:
        all_close = abs(s['sup_v1']-s['sup_v3a'])<1e-9 and abs(s['sup_v1']-s['sup_v3b'])<1e-9
        print(f"  {s['label']:<48}  sup_v1={s['sup_v1']:.4f}  "
              f"OK={all_close}",
              flush=True)

    print("\nDecision rubric:", flush=True)
    print("  (a) v3a(medium) >= v1(medium) - 0.03         (relaxed tol harmless)",
          flush=True)
    print("  (b) v3b(medium) >= v1(medium) - 0.03         (hierarchical preserves medium)",
          flush=True)
    print("  (c) v3b(hard) > v1(hard) + 0.10              (hierarchical fixes hard)",
          flush=True)
    print("  (d) v3b(extreme) > v1(extreme) + 0.05        (hierarchical helps extreme)",
          flush=True)

    s0, s1, s2 = summary
    a = s0['uns_v3a'] >= s0['uns_v1'] - 0.03
    b = s0['uns_v3b'] >= s0['uns_v1'] - 0.03
    c = s1['uns_v3b'] > s1['uns_v1'] + 0.10
    d = s2['uns_v3b'] > s2['uns_v1'] + 0.05
    print(f"  (a) {a}  (b) {b}  (c) {c}  (d) {d}", flush=True)
    if a and b and c and d:
        print("  -> GO: hierarchical retry is the right design. "
              "Roll out by modifying dz_unmixing_experiment.fit_nmf.",
              flush=True)
    elif a and b:
        print("  -> PARTIAL: medium preserved but hard/extreme need more work.",
              flush=True)
    else:
        print("  -> NO-GO: design has issues, investigate further.",
              flush=True)


if __name__ == '__main__':
    main()
