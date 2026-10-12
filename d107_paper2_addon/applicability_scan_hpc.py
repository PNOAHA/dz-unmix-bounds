# -*- coding: utf-8 -*-
"""
第二篇方法论文 · 适用边界扫描 — HPC job array 版
==================================================
复用 applicability_scan.run_config()（核心逻辑见该文件），
按 SLURM_ARRAY_TASK_ID 把某一个轴的大网格分片，每 array task 算一块。

一次只扫一个轴（DZ_AXIS）。其它轴固定在 base case。

环境变量:
  DZ_AXIS        srcK | noise | sep        （默认 srcK）
  DZ_N_SEEDS     每格点 seed 数            （默认 50）
  DZ_METHODS     supervised,unsupervised   （逗号分隔；默认两者都跑）
  DZ_N_TASKS     array 总 task 数          （默认 500）
  DZ_OUTDIR      per-task 输出目录         （默认 results_appl）
  ── 轴网格 ──
  srcK :  DZ_K_MIN(2)  DZ_K_MAX(6)             → K = 2..6
  noise:  DZ_NOISE_N(40)                       → linspace(0.0, 0.20, N)
  sep  :  DZ_SEP_N(40)                         → linspace(0.3, 6.0, N) 的 Δ/σ
  ── base case（被固定的轴）──
  DZ_ALPHA(1.0)  DZ_BW(12.0)  DZ_SIGMA(25.0)

本地试跑单个 task:
  DZ_AXIS=sep DZ_N_SEEDS=50 DZ_N_TASKS=500 python applicability_scan_hpc.py 0

注: R1（α×bandwidth）已归第一篇，本台架不扫。
"""

import os
import sys
import json
import time

import numpy as np

from applicability_scan import build_axis_configs, run_config


# ----------------------------------------------------------------------------
# 从 env 组装本轮扫描（一次一个轴）
# ----------------------------------------------------------------------------
AXIS     = os.environ.get("DZ_AXIS", "srcK")
N_SEEDS  = int(os.environ.get("DZ_N_SEEDS", "50"))
N_TASKS  = int(os.environ.get("DZ_N_TASKS", "500"))
OUTDIR   = os.environ.get("DZ_OUTDIR", "results_appl")
METHODS  = tuple(m.strip() for m in
                 os.environ.get("DZ_METHODS", "supervised,unsupervised").split(",")
                 if m.strip())

_ALPHA = float(os.environ.get("DZ_ALPHA", "1.0"))
_BW    = float(os.environ.get("DZ_BW", "12.0"))
_SIGMA = float(os.environ.get("DZ_SIGMA", "25.0"))

SEEDS = list(range(1, N_SEEDS + 1))

if AXIS == "srcK":
    k_min = int(os.environ.get("DZ_K_MIN", "2"))
    k_max = int(os.environ.get("DZ_K_MAX", "6"))
    _grid_kw = dict(k_values=range(k_min, k_max + 1))
elif AXIS == "noise":
    n = int(os.environ.get("DZ_NOISE_N", "40"))
    _grid_kw = dict(noise_grid=np.linspace(0.0, 0.20, n))
elif AXIS == "sep":
    n = int(os.environ.get("DZ_SEP_N", "40"))
    _grid_kw = dict(sep_grid=np.linspace(0.3, 6.0, n))
else:
    raise ValueError(f"未知 DZ_AXIS={AXIS}（应为 srcK / noise / sep）")

CONFIGS = build_axis_configs(
    AXIS, SEEDS, sigma=_SIGMA, dirichlet_alpha=_ALPHA, bandwidth=_BW, **_grid_kw
)


def slice_for_task(task_id, n_tasks, n_configs):
    """连续分块：task_id 负责 configs[start:end]。"""
    per = (n_configs + n_tasks - 1) // n_tasks
    start = task_id * per
    end = min(start + per, n_configs)
    return start, end


def main():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    elif len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = 0

    os.makedirs(OUTDIR, exist_ok=True)
    start, end = slice_for_task(task_id, N_TASKS, len(CONFIGS))
    if start >= len(CONFIGS):
        print(f"task {task_id}: 超出范围 (start={start} >= {len(CONFIGS)}), 跳过")
        return 0

    t0 = time.time()
    out = []
    for idx in range(start, end):
        r = run_config(CONFIGS[idx], methods=METHODS)
        r["config_idx"] = idx
        out.append(r)

    part_path = os.path.join(OUTDIR, f"part_{task_id:05d}.json")
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    dt = time.time() - t0
    print(f"task {task_id:5d}: axis={AXIS} configs [{start},{end}) -> {len(out)} "
          f"results in {dt:.1f}s -> {part_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
