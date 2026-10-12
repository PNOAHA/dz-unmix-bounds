# -*- coding: utf-8 -*-
"""
真实数据 Bayesian uncertainty scan — HPC job array 版
======================================================
按 SLURM_ARRAY_TASK_ID 切片 (sample × perturbation × bootstrap) 大 grid。

环境变量（可在 sbatch 前覆盖）：
  DZ_N_BOOTSTRAP       每 (sample, perturb) cell 的 bootstrap 数  默认 100
  DZ_MEAN_SHIFT_N      mean shift grid 点数                       默认 11
  DZ_SIGMA_SHIFT_N     sigma shift grid 点数                      默认 11
  DZ_N_TASKS           array 总 task 数                            默认 500
  DZ_N_WALKERS         emcee walker 数                             默认 64
  DZ_N_STEPS           emcee 总 step 数                            默认 5000
  DZ_BURN_IN           burn-in step                                默认 1000
  DZ_OUTDIR            per-task 输出目录                            默认 results_realdata
  DZ_MEAN_UNC          prior μ 不确定度 (frac)                      默认 0.10
  DZ_SIGMA_UNC         prior σ 不确定度 (frac)                      默认 0.20

默认 grid：4 sample × 11 × 11 × 100 bootstrap = 48,400 fits
分到 500 array tasks → 97 fits/task → 实测每 fit ~40s → ~ 65 分钟/task
50 concurrent → ~ 10 hours wall clock
25 concurrent → ~ 20 hours wall clock
15 concurrent → ~ 33 hours wall clock ≈ 1.4 day
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from bayesian_realdata import PriorSpec  # noqa: E402
from realdata_scan import build_configs, load_all_samples, run_config  # noqa: E402


N_BOOTSTRAP = int(os.environ.get("DZ_N_BOOTSTRAP", "100"))
MEAN_SHIFT_N = int(os.environ.get("DZ_MEAN_SHIFT_N", "11"))
SIGMA_SHIFT_N = int(os.environ.get("DZ_SIGMA_SHIFT_N", "11"))
N_TASKS = int(os.environ.get("DZ_N_TASKS", "500"))
N_WALKERS = int(os.environ.get("DZ_N_WALKERS", "64"))
N_STEPS = int(os.environ.get("DZ_N_STEPS", "5000"))
BURN_IN = int(os.environ.get("DZ_BURN_IN", "1000"))
OUTDIR = os.environ.get("DZ_OUTDIR", "results_realdata")
MEAN_UNC = float(os.environ.get("DZ_MEAN_UNC", "0.10"))
SIGMA_UNC = float(os.environ.get("DZ_SIGMA_UNC", "0.20"))

MEAN_SHIFTS = np.linspace(-0.10, 0.10, MEAN_SHIFT_N)
SIGMA_SHIFTS = np.linspace(-0.20, 0.20, SIGMA_SHIFT_N)


def slice_for_task(task_id: int, n_tasks: int, n_configs: int) -> tuple[int, int]:
    per = (n_configs + n_tasks - 1) // n_tasks
    start = task_id * per
    end = min(start + per, n_configs)
    return start, end


def main():
    # Build grid once (cheap)
    samples = load_all_samples()
    ages_lookup = {sn: sa for (sn, _, sa) in samples}
    prior = PriorSpec.from_default(mean_unc_pct=MEAN_UNC, sigma_unc_pct=SIGMA_UNC)
    configs = build_configs(samples,
                            mean_shifts=MEAN_SHIFTS,
                            sigma_shifts=SIGMA_SHIFTS,
                            n_bootstrap=N_BOOTSTRAP)

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    elif len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = 0

    os.makedirs(OUTDIR, exist_ok=True)
    start, end = slice_for_task(task_id, N_TASKS, len(configs))
    if start >= len(configs):
        print(f"task {task_id}: out of range ({start} >= {len(configs)}); skip")
        return 0

    print(f"[task {task_id}] configs [{start},{end})  total={len(configs)}  "
          f"n_steps={N_STEPS}  walkers={N_WALKERS}")
    t0 = time.time()
    out = []
    for idx in range(start, end):
        r = run_config(configs[idx], ages_lookup, prior,
                       n_walkers=N_WALKERS, n_steps=N_STEPS, burn_in=BURN_IN)
        r["config_idx"] = idx
        out.append(r)
        if (idx - start + 1) % 10 == 0:
            print(f"[task {task_id}] {idx - start + 1}/{end - start} done  "
                  f"({(time.time() - t0) / (idx - start + 1):.1f} s/fit)")

    part_path = os.path.join(OUTDIR, f"part_{task_id:05d}.json")
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    dt = time.time() - t0
    print(f"[task {task_id}] DONE  {len(out)} fits in {dt:.1f}s "
          f"({dt / len(out):.1f} s/fit)  → {part_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
