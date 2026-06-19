#!/bin/bash
# 第二篇方法论文 · 适用边界扫描 — 大规模 job array
# 一次扫一个轴（DZ_AXIS = srcK | noise | sep），单核小任务塞零散空闲核，--nice 低优先级不抢占
#
# 部署（在集群 login 节点）:
#   scp 本目录 4 个文件到 $PAPER2_DIR（含本 .slurm）
#   mkdir -p $PAPER2_DIR/logs   ← 必须！Slurm 在脚本体之前就打开 --output=logs/...，
#                                  logs/ 不存在 → 作业秒挂(exit 1, 无任何日志)。脚本体里的 mkdir 来不及救
#   dz_unmixing_experiment.py 仍在 $DZ_CORE_DIR（第一篇部署处），不必复制
#
# 提交（三个轴分别跑一次）:
#   DZ_AXIS=srcK  sbatch --array=0-249 applicability.slurm     # 5 K × 50 seed = 250 config
#   DZ_AXIS=noise sbatch --array=0-499 applicability.slurm     # 40 noise × 50 seed = 2000 config
#   DZ_AXIS=sep   sbatch --array=0-499 applicability.slurm     # 40 sep × 50 seed = 2000 config
# 冒烟（只跑 3 task 验证管道）:
#   DZ_AXIS=sep sbatch --array=0-2 applicability.slurm
# 合并（每轴一次，指定该轴的 OUTDIR）:
#   DZ_OUTDIR=results_appl_sep python merge_applicability.py

#SBATCH --job-name=dz-appl
#SBATCH --partition=debug
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/appl_%A_%a.out
#SBATCH --error=logs/appl_%A_%a.err

# ── 路径（按集群实际改）────────────────────────────────────────────────
PAPER2_DIR=${PAPER2_DIR:-/public/home/panlujia/dz-unmix/paper2}
export DZ_CORE_DIR=${DZ_CORE_DIR:-/public/home/panlujia/dz-unmix}   # dz_unmixing_experiment.py 所在
PYTHON=${PYTHON:-/public/software/Anaconda3/bin/python}

# ── 扫描参数（可在 sbatch 前用 env 覆盖）──────────────────────────────
export DZ_AXIS=${DZ_AXIS:-srcK}
export DZ_N_SEEDS=${DZ_N_SEEDS:-50}
export DZ_N_TASKS=${DZ_N_TASKS:-500}
export DZ_METHODS=${DZ_METHODS:-supervised,unsupervised}
# 每个轴各自的输出目录，避免三轴结果混到一起
export DZ_OUTDIR=${DZ_OUTDIR:-results_appl_${DZ_AXIS}}

cd "$PAPER2_DIR" || exit 1
mkdir -p logs "$DZ_OUTDIR"

echo "[$(date)] axis=$DZ_AXIS array task ${SLURM_ARRAY_TASK_ID} on $(hostname)  -> $DZ_OUTDIR"
"$PYTHON" applicability_scan_hpc.py
echo "[$(date)] done rc=$?"
