#!/bin/bash
# 第二篇方法论文 · K × sep × noise 三维 phase scan
# interaction.slurm 的三维姐妹脚本（多 noise 轴）
#
# 默认 grid: 11 K × 25 sep × 12 noise × 1000 seeds = 3.3M configs
#          / 1000 array tasks = 3300 configs/task × 7s/config ≈ 6.4h/task
#          (受 SLURM MaxArraySize=1001 限制，单 task 厚一些)
#          预计 wall clock: 8–13h @ 500–800 cores 并发
#
# 提交（满档）：
#   sbatch scan/phase3d.slurm
#
# True reduced smoke: python scan/phase3d_scan.py --smoke (8 configs).
# A smaller array alone does not reduce the configured experiment grid.
#
# 缩 / 扩并发（不重提）：
#   scontrol update jobid=<JOBID> ArrayTaskThrottle=300    # 收紧
#   scontrol update jobid=<JOBID> ArrayTaskThrottle=800    # 放开
#
# 合并（跑完后）：
#   DZ_OUTDIR=results_phase3d /public/software/Anaconda3/bin/python merge_phase3d.py

#SBATCH --job-name=dz-phase3d
# Set your cluster partition with sbatch --partition=NAME.
#SBATCH --array=0-999%500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=14:00:00
#SBATCH --nice=25000
#SBATCH --output=scan/logs/phase3d_%A_%a.out
#SBATCH --error=scan/logs/phase3d_%A_%a.err

PAPER2_DIR=${PAPER2_DIR:-$PWD/scan}
export DZ_CORE_DIR=${DZ_CORE_DIR:-$PWD}
PYTHON=${PYTHON:-python}

export DZ_N_SEEDS=${DZ_N_SEEDS:-1000}
export DZ_N_TASKS=${DZ_N_TASKS:-1000}
export DZ_METHODS=${DZ_METHODS:-supervised,unsupervised}
export DZ_OUTDIR=${DZ_OUTDIR:-results_phase3d}

mkdir -p "$PAPER2_DIR/logs"
cd "$PAPER2_DIR"

export PYTHONPATH="$PAPER2_DIR:${PYTHONPATH:-}"

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -Iseconds)"
"$PYTHON" phase3d_scan_hpc.py
echo "[task $SLURM_ARRAY_TASK_ID] end=$(date -Iseconds)"
