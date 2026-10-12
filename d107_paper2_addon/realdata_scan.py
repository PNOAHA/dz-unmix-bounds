# -*- coding: utf-8 -*-
"""
第二篇方法论文 · 真实数据 Bayesian uncertainty scan — 核心模块
================================================================
为 4 个 NE 亚 K1 sample 构造 perturbation × bootstrap grid，每 cell 跑独立
bayesian MCMC posterior（emcee）。

Grid 结构：
  sample (4) × mean_shift (M) × sigma_shift (S) × bootstrap (B)
  = 4 × M × S × B configs，每 config 跑 1 次 64 walkers × n_steps MCMC

输出：每 config 一行 record（含 posterior weights mean/std/CI、endmember μ/σ posterior、
acceptance fraction、log_evidence proxy）。

本地冒烟：
  python realdata_scan.py --smoke

作者：潘路加 — 2026-05-27
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# 在同目录引入 bayesian_realdata
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from bayesian_realdata import PriorSpec, bootstrap_ages, run_mcmc  # noqa: E402


# ----------------------------------------------------------------------------
# 1. case_study_data 加载
# ----------------------------------------------------------------------------

CASE_STUDY_CANDIDATES = [
    "/public/home/panlujia/dz-unmix/case_study_data",                 # D107
    os.path.join(_here, "..", "case_study_data"),                     # paper2 同级（D107）
    os.path.join(_here, "..", "..", "dz-unmix-cg", "data", "case_study_data"),  # 本地 addon
]


def find_case_study_dir() -> Path:
    # DZ_SAMPLES_DIR 环境变量优先(向后兼容:未设则走原候选列表)。
    # 用于把真实数据样本目录指给本 scan,而不动合成 case_study_data。
    _env = os.environ.get("DZ_SAMPLES_DIR")
    if _env and Path(_env).is_dir() and list(Path(_env).glob("*.csv")):
        return Path(_env).resolve()
    for cand in CASE_STUDY_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and list(p.glob("*.csv")):
            return p.resolve()
    raise FileNotFoundError(
        f"找不到 case_study_data 目录。试过：{CASE_STUDY_CANDIDATES}"
    )


def load_sample(csv_path: Path) -> tuple[str, str, np.ndarray]:
    """读 sample CSV，跳过 # 注释；返回 (name, label, ages_array)。"""
    name = csv_path.stem
    label = name
    ages: list[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# label:"):
                label = line.split(":", 1)[1].strip()
            if line.startswith("#"):
                continue
            if not line[0].isdigit():
                continue
            try:
                ages.append(float(line.split(",")[0]))
            except (ValueError, IndexError):
                pass
    return name, label, np.array(ages, dtype=float)


def load_all_samples() -> list[tuple[str, str, np.ndarray]]:
    cs_dir = find_case_study_dir()
    samples = []
    for csv in sorted(cs_dir.glob("*.csv")):
        samples.append(load_sample(csv))
    return samples


# ----------------------------------------------------------------------------
# 2. Grid 构造
# ----------------------------------------------------------------------------

DEFAULT_MEAN_SHIFTS = np.linspace(-0.10, 0.10, 11)   # ±10% in 11 steps
DEFAULT_SIGMA_SHIFTS = np.linspace(-0.20, 0.20, 11)  # ±20% in 11 steps
DEFAULT_N_BOOTSTRAP = 100


def build_configs(samples, *, mean_shifts=None, sigma_shifts=None,
                  n_bootstrap=None) -> list[dict]:
    """笛卡尔积 (sample × mean_shift × sigma_shift × bootstrap)。"""
    if mean_shifts is None:
        mean_shifts = DEFAULT_MEAN_SHIFTS
    if sigma_shifts is None:
        sigma_shifts = DEFAULT_SIGMA_SHIFTS
    if n_bootstrap is None:
        n_bootstrap = DEFAULT_N_BOOTSTRAP
    configs = []
    for (sname, slabel, sages) in samples:
        for ms in mean_shifts:
            for ss in sigma_shifts:
                for b in range(int(n_bootstrap)):
                    configs.append(dict(
                        sample_name=sname,
                        sample_label=slabel,
                        n_zircons=int(len(sages)),
                        mean_shift=round(float(ms), 4),
                        sigma_shift=round(float(ss), 4),
                        bootstrap_idx=int(b),
                        # 确定性 seed：sample_hash + bootstrap_idx，保证 reruns 复现
                        bootstrap_seed=int(abs(hash(sname)) % (2**31)) ^ (b * 1000 + 42),
                    ))
    return configs


# ----------------------------------------------------------------------------
# 3. 单 config 执行
# ----------------------------------------------------------------------------

def run_config(cfg: dict, ages_lookup: dict[str, np.ndarray],
               prior_default: PriorSpec, *,
               n_walkers: int = 64, n_steps: int = 5000, burn_in: int = 1000) -> dict:
    """Single config: bootstrap → perturb prior → MCMC → 扁平 record。"""
    ages = ages_lookup[cfg["sample_name"]]
    boot_ages = bootstrap_ages(ages, seed=cfg["bootstrap_seed"])
    prior = prior_default.perturbed(cfg["mean_shift"], cfg["sigma_shift"])

    t0 = time.time()
    res = run_mcmc(boot_ages, prior,
                   n_walkers=n_walkers, n_steps=n_steps,
                   burn_in=burn_in, seed=cfg["bootstrap_seed"])
    dt = time.time() - t0

    out = dict(cfg)
    # bayesian 输出字段统一前缀 bayes_，便于后续 merge
    for k in ("weights_mean", "weights_std", "weights_q05", "weights_q95",
              "mus_mean", "mus_std", "sigmas_mean", "sigmas_std",
              "log_evidence_proxy", "acceptance_fraction"):
        out[f"bayes_{k}"] = res[k]
    out["mcmc_elapsed_sec"] = round(dt, 2)
    out["n_walkers"] = n_walkers
    out["n_steps"] = n_steps
    return out


# ----------------------------------------------------------------------------
# 4. 本地冒烟
# ----------------------------------------------------------------------------

def _smoke():
    """4 sample × 2 mean_shift × 2 sigma_shift × 1 bootstrap = 16 configs ~ 11 分钟"""
    print(f"=== load samples ===")
    samples = load_all_samples()
    for sn, sl, sa in samples:
        print(f"  {sn:30s} ({sl})  n_zircons={len(sa)}")

    ages_lookup = {sn: sa for (sn, _, sa) in samples}
    prior = PriorSpec.from_default(mean_unc_pct=0.10, sigma_unc_pct=0.20)

    print(f"\n=== build grid (smoke: 2×2×1) ===")
    configs = build_configs(samples,
                            mean_shifts=[-0.05, 0.05],
                            sigma_shifts=[-0.10, 0.10],
                            n_bootstrap=1)
    print(f"  {len(configs)} configs")

    print(f"\n=== run configs (n_steps=2000 quick) ===")
    t0 = time.time()
    results = []
    for i, cfg in enumerate(configs):
        r = run_config(cfg, ages_lookup, prior, n_steps=2000, burn_in=500)
        results.append(r)
        w = r["bayes_weights_mean"]
        print(f"  [{i+1:2d}/{len(configs)}] {cfg['sample_name']:25s}  "
              f"ms={cfg['mean_shift']:+.2f}  ss={cfg['sigma_shift']:+.2f}  "
              f"w=[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]  ({r['mcmc_elapsed_sec']:.1f}s)")
    dt_total = time.time() - t0
    print(f"\n[smoke] {len(configs)} fits in {dt_total:.1f}s ({dt_total/len(configs):.1f}s/fit)")
    print(f"[smoke] OK — pipeline 通畅")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        print(__doc__)
        print("--smoke 跑 16 configs (~10 min); 大规模 submit 用 realdata_scan_hpc.py")
