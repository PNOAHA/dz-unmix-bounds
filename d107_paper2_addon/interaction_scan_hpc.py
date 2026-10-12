# -*- coding: utf-8 -*-
"""
K × sep interaction phase scan — HPC job array 版
==================================================
复用 interaction_scan.{build_interaction_configs, run_interaction_config}，
按 SLURM_ARRAY_TASK_ID 分片大网格，每 array task 算一块。

环境变量：
  DZ_N_SEEDS     每格点 seed 数         （默认 30）
  DZ_N_TASKS     array 总 task 数       （默认 50）
  DZ_METHODS     supervised,unsupervised
  DZ_OUTDIR      per-task 输出目录      （默认 results_interaction）
  DZ_K_VALUES    K 值（逗号分隔）        （默认 2,3,4,5,6）
  DZ_SEP_VALUES  sep 值（逗号分隔）      （默认 0.3,0.6,1.0,1.5,2.5,4.0,6.0）

默认 grid: 5 K × 7 sep × 30 seeds = 1050 configs
分到 50 array tasks → ~21 configs/task

本地试跑单个 task：
  DZ_N_SEEDS=2 DZ_N_TASKS=3 python interaction_scan_hpc.py 0

作者: 潘路加 — 2026-05-26
"""

import os
import sys
import json
import time

from interaction_scan import build_interaction_configs, run_interaction_config


N_SEEDS = int(os.environ.get("DZ_N_SEEDS", "30"))
N_TASKS = int(os.environ.get("DZ_N_TASKS", "50"))
OUTDIR  = os.environ.get("DZ_OUTDIR", "results_interaction")
METHODS = tuple(m.strip() for m in
                os.environ.get("DZ_METHODS", "supervised,unsupervised").split(",")
                if m.strip())

K_VALUES = [int(x) for x in
            os.environ.get("DZ_K_VALUES", "2,3,4,5,6").split(",")]
SEP_VALUES = [float(x) for x in
              os.environ.get("DZ_SEP_VALUES", "0.3,0.6,1.0,1.5,2.5,4.0,6.0").split(",")]

SEEDS = list(range(1, N_SEEDS + 1))
CONFIGS = build_interaction_configs(K_VALUES, SEP_VALUES, SEEDS)


def slice_for_task(task_id, n_tasks, n_configs):
    """连续分块（与 paper2 applicability_scan_hpc 同算法）。"""
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
        print(f"task {task_id}: 超出范围 ({start} >= {len(CONFIGS)}), 跳过")
        return 0

    t0 = time.time()
    out = []
    for idx in range(start, end):
        r = run_interaction_config(CONFIGS[idx], methods=METHODS)
        r["config_idx"] = idx
        out.append(r)

    part_path = os.path.join(OUTDIR, f"part_{task_id:05d}.json")
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    dt = time.time() - t0
    print(f"task {task_id:5d}: configs [{start},{end}) -> {len(out)} results "
          f"in {dt:.1f}s -> {part_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
