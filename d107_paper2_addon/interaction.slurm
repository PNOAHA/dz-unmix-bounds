#!/bin/bash
# 第二篇方法论文 · K × sep interaction phase scan
# paper2 applicability.slurm 的姐妹脚本（不同 axis，同套部署模式）
#
# 部署：本文件 + interaction_scan.py + interaction_scan_hpc.py + merge_interaction.py
#       放在 $PAPER2_DIR；dz_unmixing_experiment.py 仍在 $DZ_CORE_DIR（第一篇部署处）
#
# 提交（默认 50 array tasks × 21 configs/task ≈ 1050 configs total）：
#   sbatch interaction.slurm                 # 50 tasks, 默认 K/sep grid
#
# 冒烟（只跑 3 task ≈ 63 configs，验证管道）：
#   sbatch --array=0-2 interaction.slurm
#
# 合并（跑完后）：
#   DZ_OUTDIR=results_interaction python merge_interaction.py

#SBATCH --job-name=dz-interact
#SBATCH --partition=debug
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/interact_%A_%a.out
#SBATCH --error=logs/interact_%A_%a.err

# ── 路径（与 applicability.slurm 同款）────────────────────────────────
PAPER2_DIR=${PAPER2_DIR:-/public/home/panlujia/dz-unmix/paper2}
export DZ_CORE_DIR=${DZ_CORE_DIR:-/public/home/panlujia/dz-unmix}
PYTHON=${PYTHON:-/public/software/Anaconda3/bin/python}

# ── 扫描参数（可在 sbatch 前用 env 覆盖）──────────────────────────────
export DZ_N_SEEDS=${DZ_N_SEEDS:-30}
export DZ_N_TASKS=${DZ_N_TASKS:-50}
export DZ_METHODS=${DZ_METHODS:-supervised,unsupervised}
export DZ_OUTDIR=${DZ_OUTDIR:-results_interaction}
export DZ_K_VALUES=${DZ_K_VALUES:-2,3,4,5,6}
export DZ_SEP_VALUES=${DZ_SEP_VALUES:-0.3,0.6,1.0,1.5,2.5,4.0,6.0}

mkdir -p "$PAPER2_DIR/logs"
cd "$PAPER2_DIR"

# 把 paper2 目录加入 path，interaction_scan 能 import applicability_scan
export PYTHONPATH="$PAPER2_DIR:${PYTHONPATH:-}"

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -Iseconds)"
"$PYTHON" interaction_scan_hpc.py
echo "[task $SLURM_ARRAY_TASK_ID] end=$(date -Iseconds)"
