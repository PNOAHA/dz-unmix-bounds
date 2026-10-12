# -*- coding: utf-8 -*-
"""
phase3d (K × sep × noise) HPC array dispatcher
==============================================
与 interaction_scan_hpc.py 完全同构，只多一个 NOISE_VALUES env。

环境变量：
  DZ_N_SEEDS       每格点 seed 数      （默认 1000）
  DZ_N_TASKS       array 总 task 数    （默认 2000）
  DZ_METHODS       supervised,unsupervised
  DZ_OUTDIR        输出目录             （默认 results_phase3d）
  DZ_K_VALUES      K 值                 （默认 2,3,4,5,6,7,8,9,10,12,15）
  DZ_SEP_VALUES    sep 值               （默认 25 点密集网格）
  DZ_NOISE_VALUES  noise 值             （默认 12 点：2%–50%）

默认 grid: 11 K × 25 sep × 12 noise × 1000 seeds = 3,300,000 configs
分到 1000 array tasks → 3300 configs/task → ~6.4h/task @ 7s/config
(SLURM MaxArraySize=1001 限制)
"""

import os
import sys
import json
import time

from phase3d_scan import build_phase3d_configs, run_phase3d_config


N_SEEDS = int(os.environ.get("DZ_N_SEEDS", "1000"))
N_TASKS = int(os.environ.get("DZ_N_TASKS", "1000"))
OUTDIR  = os.environ.get("DZ_OUTDIR", "results_phase3d")
METHODS = tuple(m.strip() for m in
                os.environ.get("DZ_METHODS", "supervised,unsupervised").split(",")
                if m.strip())

K_VALUES = [int(x) for x in os.environ.get(
    "DZ_K_VALUES", "2,3,4,5,6,7,8,9,10,12,15"
).split(",")]
SEP_VALUES = [float(x) for x in os.environ.get(
    "DZ_SEP_VALUES",
    "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.6,1.8,2.0,"
    "2.2,2.5,2.8,3.2,3.6,4.0,4.5,5.0,5.5,6.0"
).split(",")]
NOISE_VALUES = [float(x) for x in os.environ.get(
    "DZ_NOISE_VALUES",
    "0.02,0.04,0.06,0.08,0.10,0.13,0.16,0.20,0.25,0.30,0.40,0.50"
).split(",")]

SEEDS = list(range(1, N_SEEDS + 1))
CONFIGS = build_phase3d_configs(K_VALUES, SEP_VALUES, NOISE_VALUES, SEEDS)


def slice_for_task(task_id, n_tasks, n_configs):
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
    # DZ_TASK_OFFSET: 跨多个 array job 拼出 > MaxArraySize(1001) 的全局 task 空间。
    # 默认 0 -> 原行为不变;global task_id 同时驱动 slice 与 part 文件名(防跨 job 重名)。
    task_id += int(os.environ.get("DZ_TASK_OFFSET", "0"))

    os.makedirs(OUTDIR, exist_ok=True)
    start, end = slice_for_task(task_id, N_TASKS, len(CONFIGS))
    if start >= len(CONFIGS):
        print(f"task {task_id}: 超出范围 ({start} >= {len(CONFIGS)}), 跳过")
        return 0

    t0 = time.time()
    out = []
    for idx in range(start, end):
        r = run_phase3d_config(CONFIGS[idx], methods=METHODS)
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
