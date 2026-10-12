#!/bin/bash
# Big job — 6.77M MCMC fits, ~3 day wall clock @ 1000 concurrent
# ================================================================
# 4 sample × 11×11 perturb × 3500 bootstrap × 4 endmember-LOO = 6.77M fits
# 分到 1000 array tasks → ~6776 fits/task
# D107 实测 39 sec/fit @ 5000 steps → ~73.4 hours per task ≈ 3.06 day
# 1000 concurrent → wall clock ≈ 3 day
# SBATCH --time=4-00:00:00 给 23h buffer (容错)
#
# 提交：
#   sbatch bigjob.slurm                          # 默认 1000 tasks 全 concurrent
#
# 冒烟（3 task ~ 5 分钟 each）：
#   DZ_N_BOOTSTRAP=10 DZ_N_STEPS=2000 DZ_OUTDIR=results_bigjob_smoke sbatch --array=0-2 bigjob.slurm
#
# 合并：
#   DZ_OUTDIR=results_bigjob python merge_bigjob.py

#SBATCH --job-name=dz-bigjob
#SBATCH --partition=debug
#SBATCH --array=0-999
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/bigjob_%A_%a.out
#SBATCH --error=logs/bigjob_%A_%a.err

# ── 路径 ─────────────────────────────────────────────────────────────
PAPER2_DIR=${PAPER2_DIR:-/public/home/panlujia/dz-unmix/paper2}
export DZ_CORE_DIR=${DZ_CORE_DIR:-/public/home/panlujia/dz-unmix}
PYTHON=${PYTHON:-/public/software/Anaconda3/bin/python}

# ── 配置（可 env 覆盖）──────────────────────────────────────────────
export DZ_N_BOOTSTRAP=${DZ_N_BOOTSTRAP:-3500}
export DZ_MEAN_SHIFT_N=${DZ_MEAN_SHIFT_N:-11}
export DZ_SIGMA_SHIFT_N=${DZ_SIGMA_SHIFT_N:-11}
export DZ_N_TASKS=${DZ_N_TASKS:-1000}
export DZ_N_WALKERS=${DZ_N_WALKERS:-64}
export DZ_N_STEPS=${DZ_N_STEPS:-5000}
export DZ_BURN_IN=${DZ_BURN_IN:-1000}
export DZ_OUTDIR=${DZ_OUTDIR:-results_bigjob}
export DZ_MEAN_UNC=${DZ_MEAN_UNC:-0.10}
export DZ_SIGMA_UNC=${DZ_SIGMA_UNC:-0.20}
export DZ_LOO_VARIANTS=${DZ_LOO_VARIANTS:-"-1,0,1,2"}

mkdir -p "$PAPER2_DIR/logs"
cd "$PAPER2_DIR"

export PYTHONPATH="$PAPER2_DIR:${PYTHONPATH:-}"

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -Iseconds)"
"$PYTHON" bigjob_scan_hpc.py
echo "[task $SLURM_ARRAY_TASK_ID] end=$(date -Iseconds)"
