# -*- coding: utf-8 -*-
"""
Big job HPC job array 版
==========================
按 SLURM_ARRAY_TASK_ID 切片 9.68M-fit grid。

环境变量：
  DZ_N_BOOTSTRAP        bootstrap reps                默认 5000
  DZ_MEAN_SHIFT_N       mean shift grid 点数           默认 11
  DZ_SIGMA_SHIFT_N      sigma shift grid 点数          默认 11
  DZ_N_TASKS            array 总 task 数 (≤ MaxArraySize) 默认 1000
  DZ_N_WALKERS          emcee walkers                 默认 64
  DZ_N_STEPS            emcee steps                   默认 5000
  DZ_BURN_IN            burn-in                       默认 1000
  DZ_OUTDIR             output dir                    默认 results_bigjob
  DZ_LOO_VARIANTS       loo idx list 逗号分隔         默认 -1,0,1,2
  DZ_MEAN_UNC           prior μ unc                   默认 0.10
  DZ_SIGMA_UNC          prior σ unc                   默认 0.20

Default grid：4 × 11×11 × 5000 × 4 = 9.68M fits / 1000 tasks = 9680 fits/task
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
from bigjob_scan import (build_configs_slice, run_config_bigjob,  # noqa: E402
                         total_configs)
from realdata_scan import load_all_samples  # noqa: E402


N_BOOTSTRAP = int(os.environ.get("DZ_N_BOOTSTRAP", "3500"))
MEAN_SHIFT_N = int(os.environ.get("DZ_MEAN_SHIFT_N", "11"))
SIGMA_SHIFT_N = int(os.environ.get("DZ_SIGMA_SHIFT_N", "11"))
N_TASKS = int(os.environ.get("DZ_N_TASKS", "1000"))
N_WALKERS = int(os.environ.get("DZ_N_WALKERS", "64"))
N_STEPS = int(os.environ.get("DZ_N_STEPS", "5000"))
BURN_IN = int(os.environ.get("DZ_BURN_IN", "1000"))
OUTDIR = os.environ.get("DZ_OUTDIR", "results_bigjob")
MEAN_UNC = float(os.environ.get("DZ_MEAN_UNC", "0.10"))
SIGMA_UNC = float(os.environ.get("DZ_SIGMA_UNC", "0.20"))
LOO_VARIANTS = [int(x) for x in os.environ.get("DZ_LOO_VARIANTS", "-1,0,1,2").split(",")]

MEAN_SHIFTS = np.linspace(-0.10, 0.10, MEAN_SHIFT_N)
SIGMA_SHIFTS = np.linspace(-0.20, 0.20, SIGMA_SHIFT_N)


def slice_for_task(task_id: int, n_tasks: int, n_configs: int) -> tuple[int, int]:
    per = (n_configs + n_tasks - 1) // n_tasks
    start = task_id * per
    end = min(start + per, n_configs)
    return start, end


def main():
    # Lazy approach: 不 materialize 全 grid，只算 total length 再 slice
    samples = load_all_samples()
    ages_lookup = {sn: sa for (sn, _, sa) in samples}
    prior = PriorSpec.from_default(mean_unc_pct=MEAN_UNC, sigma_unc_pct=SIGMA_UNC)

    n_total = total_configs(len(samples), len(LOO_VARIANTS),
                            len(MEAN_SHIFTS), len(SIGMA_SHIFTS), N_BOOTSTRAP)

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    elif len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = 0

    os.makedirs(OUTDIR, exist_ok=True)
    start, end = slice_for_task(task_id, N_TASKS, n_total)
    if start >= n_total:
        print(f"task {task_id}: out of range ({start} >= {n_total}); skip")
        return 0

    # 只构造本 task 需要的 configs subset（~ MB 内存而非 GB）
    configs_slice = build_configs_slice(
        samples, start_idx=start, end_idx=end,
        mean_shifts=MEAN_SHIFTS, sigma_shifts=SIGMA_SHIFTS,
        n_bootstrap=N_BOOTSTRAP, loo_variants=LOO_VARIANTS,
    )
    print(f"[task {task_id}] configs [{start},{end})  total={n_total}  "
          f"slice_size={len(configs_slice)}  "
          f"n_steps={N_STEPS}  walkers={N_WALKERS}  LOO={LOO_VARIANTS}")
    t0 = time.time()
    out = []
    for j, cfg in enumerate(configs_slice):
        r = run_config_bigjob(cfg, ages_lookup, prior,
                              n_walkers=N_WALKERS, n_steps=N_STEPS, burn_in=BURN_IN)
        r["config_idx"] = start + j
        out.append(r)
        if (j + 1) % 100 == 0:
            elapsed = time.time() - t0
            done = j + 1
            eta_sec = elapsed / done * (len(configs_slice) - done)
            print(f"[task {task_id}] {done}/{len(configs_slice)}  elapsed={elapsed/60:.1f}min  "
                  f"avg={elapsed/done:.1f}s/fit  ETA={eta_sec/3600:.1f}h")

    part_path = os.path.join(OUTDIR, f"part_{task_id:05d}.json")
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    dt = time.time() - t0
    print(f"[task {task_id}] DONE  {len(out)} fits in {dt/3600:.2f}h "
          f"({dt/len(out):.1f}s/fit)  → {part_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
