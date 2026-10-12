# -*- coding: utf-8 -*-
"""
第二篇方法论文 · Big job: realdata Bayesian uncertainty + endmember-LOO
========================================================================
扩展 realdata_scan：在 (sample × perturb × bootstrap) 之上加 endmember-LOO 维度。

LOO 维度（4 variants）：
  loo_idx=-1: full K=3 model (用全 3 endmember，原 realdata)
  loo_idx= 0: drop endmember 0 (S1 中生代弧) → K=2 model
  loo_idx= 1: drop endmember 1 (S2 古生代基底) → K=2 model
  loo_idx= 2: drop endmember 2 (S3 太古克拉通) → K=2 model

scientific 意义：量化"如果地质学家漏判一个源区，bayesian 推断会如何 biased"。
posterior weights 对剩余 K-1 endmember 重分配，反映 missing component 的隐性 bias。

Grid (default)：
  4 sample × 11×11 perturb × 5000 bootstrap × 4 LOO = 9.68M fits
  per task: ~75 hours; @ 1000 concurrent → ~3 day wall clock

作者：潘路加 — 2026-05-27
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from bayesian_realdata import PriorSpec, bootstrap_ages, run_mcmc  # noqa: E402
from realdata_scan import find_case_study_dir, load_all_samples, load_sample  # noqa: E402


# ----------------------------------------------------------------------------
# 1. LOO-aware prior
# ----------------------------------------------------------------------------

def make_prior_for_loo(prior_full: PriorSpec, loo_idx: int) -> PriorSpec:
    """从 full K=3 PriorSpec drop endmember loo_idx，返回 K=2 PriorSpec。

    loo_idx=-1 视为 no-op，返回原 prior。
    """
    if loo_idx < 0:
        return prior_full
    keep = [i for i in range(len(prior_full.means_center)) if i != loo_idx]
    return PriorSpec(
        means_center=prior_full.means_center[keep],
        means_sigma=prior_full.means_sigma[keep],
        sigmas_center=prior_full.sigmas_center[keep],
        sigmas_sigma=prior_full.sigmas_sigma[keep],
    )


# ----------------------------------------------------------------------------
# 2. Grid builder
# ----------------------------------------------------------------------------

DEFAULT_LOO_VARIANTS = [-1, 0, 1, 2]      # full + 3 LOO
DEFAULT_MEAN_SHIFTS = np.linspace(-0.10, 0.10, 11)
DEFAULT_SIGMA_SHIFTS = np.linspace(-0.20, 0.20, 11)
DEFAULT_N_BOOTSTRAP = 3500   # D107 实测 39 sec/fit; per-task 73h, 23h SBATCH 4d buffer


def total_configs(n_samples: int, n_loo: int, n_ms: int, n_ss: int, n_boot: int) -> int:
    return n_samples * n_loo * n_ms * n_ss * n_boot


def _config_from_idx(idx: int, samples, loo_variants, mean_shifts, sigma_shifts,
                     n_boot: int) -> dict:
    """O(1) 整数索引 → config dict。Stride 乘积分解，不需 materialize 完整 grid。

    Iteration order: sample(outer) → loo → ms → ss → boot(inner)。
    """
    n_loo = len(loo_variants)
    n_ms = len(mean_shifts)
    n_ss = len(sigma_shifts)
    stride_sample = n_loo * n_ms * n_ss * n_boot
    stride_loo = n_ms * n_ss * n_boot
    stride_ms = n_ss * n_boot
    stride_ss = n_boot

    sample_idx = idx // stride_sample
    rem = idx % stride_sample
    loo_pos = rem // stride_loo
    rem = rem % stride_loo
    ms_idx = rem // stride_ms
    rem = rem % stride_ms
    ss_idx = rem // stride_ss
    boot_idx = rem % stride_ss

    sname, slabel, sages = samples[sample_idx]
    loo_idx = int(loo_variants[loo_pos])
    ms = mean_shifts[ms_idx]
    ss = sigma_shifts[ss_idx]

    return dict(
        sample_name=sname,
        sample_label=slabel,
        n_zircons=int(len(sages)),
        loo_idx=loo_idx,
        K_used=(3 if loo_idx < 0 else 2),
        mean_shift=round(float(ms), 4),
        sigma_shift=round(float(ss), 4),
        bootstrap_idx=int(boot_idx),
        bootstrap_seed=(int(abs(hash(sname)) % (2**31))
                        ^ (int(boot_idx) * 1000 + 42)
                        ^ ((loo_idx + 2) * 7919)),
    )


def build_configs_slice(samples, *, start_idx: int, end_idx: int,
                        mean_shifts=None, sigma_shifts=None,
                        n_bootstrap=None, loo_variants=None) -> list[dict]:
    """O(slice_size) memory — 只 materialize [start_idx, end_idx) 范围内 configs。

    替代 build_configs (全量 materialize)，让 array task 内存使用从 GB 降到 MB。
    """
    if mean_shifts is None:
        mean_shifts = DEFAULT_MEAN_SHIFTS
    if sigma_shifts is None:
        sigma_shifts = DEFAULT_SIGMA_SHIFTS
    if n_bootstrap is None:
        n_bootstrap = DEFAULT_N_BOOTSTRAP
    if loo_variants is None:
        loo_variants = DEFAULT_LOO_VARIANTS

    return [
        _config_from_idx(i, samples, loo_variants, mean_shifts, sigma_shifts,
                         int(n_bootstrap))
        for i in range(start_idx, end_idx)
    ]


def build_configs(samples, *,
                  mean_shifts=None, sigma_shifts=None,
                  n_bootstrap=None, loo_variants=None) -> list[dict]:
    """全量构造（兼容老调用；smoke / 小 grid 用）。Production HPC 用 build_configs_slice。"""
    if mean_shifts is None:
        mean_shifts = DEFAULT_MEAN_SHIFTS
    if sigma_shifts is None:
        sigma_shifts = DEFAULT_SIGMA_SHIFTS
    if n_bootstrap is None:
        n_bootstrap = DEFAULT_N_BOOTSTRAP
    if loo_variants is None:
        loo_variants = DEFAULT_LOO_VARIANTS

    total = total_configs(len(samples), len(loo_variants),
                          len(mean_shifts), len(sigma_shifts), int(n_bootstrap))
    return build_configs_slice(samples, start_idx=0, end_idx=total,
                               mean_shifts=mean_shifts, sigma_shifts=sigma_shifts,
                               n_bootstrap=n_bootstrap, loo_variants=loo_variants)


# ----------------------------------------------------------------------------
# 3. Single-config runner
# ----------------------------------------------------------------------------

def run_config_bigjob(cfg: dict, ages_lookup: dict[str, np.ndarray],
                      prior_full: PriorSpec, *,
                      n_walkers: int = 64, n_steps: int = 5000,
                      burn_in: int = 1000) -> dict:
    """LOO-aware single config: bootstrap → perturb prior → maybe-drop endmember → MCMC."""
    ages = ages_lookup[cfg["sample_name"]]
    boot_ages = bootstrap_ages(ages, seed=cfg["bootstrap_seed"])
    prior_pert = prior_full.perturbed(cfg["mean_shift"], cfg["sigma_shift"])
    prior_used = make_prior_for_loo(prior_pert, cfg["loo_idx"])
    K_used = cfg["K_used"]

    t0 = time.time()
    res = run_mcmc(boot_ages, prior_used,
                   K=K_used,
                   n_walkers=n_walkers, n_steps=n_steps,
                   burn_in=burn_in, seed=cfg["bootstrap_seed"])
    dt = time.time() - t0

    out = dict(cfg)
    for k in ("weights_mean", "weights_std", "weights_q05", "weights_q95",
              "mus_mean", "mus_std", "sigmas_mean", "sigmas_std",
              "log_evidence_proxy", "acceptance_fraction"):
        out[f"bayes_{k}"] = res[k]
    out["mcmc_elapsed_sec"] = round(dt, 2)
    out["n_walkers"] = n_walkers
    out["n_steps"] = n_steps
    return out


# ----------------------------------------------------------------------------
# 4. Local smoke
# ----------------------------------------------------------------------------

def _smoke():
    """4 sample × 1 perturb × 1 boot × 4 LOO = 16 configs ~ 8 min"""
    print("=== load samples ===")
    samples = load_all_samples()
    for sn, sl, sa in samples:
        print(f"  {sn:30s} n={len(sa)}")
    ages_lookup = {sn: sa for (sn, _, sa) in samples}
    prior = PriorSpec.from_default(mean_unc_pct=0.10, sigma_unc_pct=0.20)

    print(f"\n=== build smoke grid (mean=0, sigma=0, 1 boot, 4 LOO) ===")
    configs = build_configs(samples,
                            mean_shifts=[0.0],
                            sigma_shifts=[0.0],
                            n_bootstrap=1,
                            loo_variants=[-1, 0, 1, 2])
    print(f"  {len(configs)} configs")

    print(f"\n=== run configs (n_steps=2000 quick) ===")
    t0 = time.time()
    for i, cfg in enumerate(configs):
        r = run_config_bigjob(cfg, ages_lookup, prior, n_steps=2000, burn_in=500)
        w = r["bayes_weights_mean"]
        wstr = ", ".join(f"{x:.3f}" for x in w)
        K = r["K_used"]
        loo_str = "FULL K=3" if cfg["loo_idx"] < 0 else f"DROP s{cfg['loo_idx']} K=2"
        print(f"  [{i+1:2d}/{len(configs)}] {cfg['sample_name']:25s}  "
              f"{loo_str:14s}  w=[{wstr}]  ({r['mcmc_elapsed_sec']:.1f}s)")
    print(f"\n[smoke] {len(configs)} fits in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        print(__doc__)
