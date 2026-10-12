#!/bin/bash
# 真实数据 Bayesian uncertainty scan — multi-day SLURM job
# ============================================================
# 4 sample × 11 mean_shift × 11 sigma_shift × 100 bootstrap = 48,400 fits
# 分到 500 array tasks → ~97 fits/task → 实测 ~40 sec/fit → ~65 min/task
#
# Multi-day wall clock 通过限制 concurrency 实现：
#   25 concurrent: ~22 hours wall clock
#   15 concurrent: ~36 hours wall clock ≈ 1.5 day
#   10 concurrent: ~54 hours wall clock ≈ 2.3 day
#   8  concurrent: ~67 hours wall clock ≈ 2.8 day  ← 推荐：~3 day target
#
# 提交：
#   sbatch realdata.slurm                          # 默认 8 concurrent ~3 day
#   sbatch --array=0-499%15 realdata.slurm         # 加速到 ~1.5 day
#
# 冒烟（3 task）：
#   sbatch --array=0-2 realdata.slurm
#
# 合并：
#   DZ_OUTDIR=results_realdata python merge_realdata.py

#SBATCH --job-name=dz-realdata
#SBATCH --partition=debug
#SBATCH --array=0-499%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/realdata_%A_%a.out
#SBATCH --error=logs/realdata_%A_%a.err

# ── 路径 ─────────────────────────────────────────────────────────────
PAPER2_DIR=${PAPER2_DIR:-/public/home/panlujia/dz-unmix/paper2}
export DZ_CORE_DIR=${DZ_CORE_DIR:-/public/home/panlujia/dz-unmix}
PYTHON=${PYTHON:-/public/software/Anaconda3/bin/python}

# ── Bayesian scan 配置（可 env 覆盖）─────────────────────────────────
export DZ_N_BOOTSTRAP=${DZ_N_BOOTSTRAP:-100}
export DZ_MEAN_SHIFT_N=${DZ_MEAN_SHIFT_N:-11}
export DZ_SIGMA_SHIFT_N=${DZ_SIGMA_SHIFT_N:-11}
export DZ_N_TASKS=${DZ_N_TASKS:-500}
export DZ_N_WALKERS=${DZ_N_WALKERS:-64}
export DZ_N_STEPS=${DZ_N_STEPS:-5000}
export DZ_BURN_IN=${DZ_BURN_IN:-1000}
export DZ_OUTDIR=${DZ_OUTDIR:-results_realdata}
export DZ_MEAN_UNC=${DZ_MEAN_UNC:-0.10}
export DZ_SIGMA_UNC=${DZ_SIGMA_UNC:-0.20}

mkdir -p "$PAPER2_DIR/logs"
cd "$PAPER2_DIR"

# Make sure emcee from --user install + paper2 module path
export PYTHONPATH="$PAPER2_DIR:${PYTHONPATH:-}"

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -Iseconds)"
"$PYTHON" realdata_scan_hpc.py
echo "[task $SLURM_ARRAY_TASK_ID] end=$(date -Iseconds)"
