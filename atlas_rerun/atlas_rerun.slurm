#!/bin/bash
# Full Fig.3 atlas rerun with solver v2 (exact KKT). One array element = one whole 32-core node.
# Submit FROM the atlas_rerun/ directory (all paths are relative to it):
#   cd /public/home/panlujia/compstat_B/atlas_rerun && sbatch atlas_rerun.slurm
# Outputs go to results_v2/units/ so job-269468 Algorithm-2 units in results/units/ stay untouched.
# Resubmitting resumes: finished units (results_v2/units/unit_XXXX.json) are skipped.
# Do NOT point --outdir at results/units — those files are truncated-KKT (Algorithm 2) and must not be resumed over.
#SBATCH --job-name=dz-atlas-v2
#SBATCH --partition=debug                 # <-- cluster partition (given: debug, no time limit)
##SBATCH --account=panlujia               # <-- account: optional; uncomment only if the cluster requires it
#SBATCH --array=0-31                      # 32 elements x 32 cores cover the 1000 work units
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive                       # whole node per array element
#SBATCH --mem=0                           # whole-node memory (default request exceeded node allocatable 123904MB)
#SBATCH --time=04:00:00                   # ~0.5-1.5 h expected per element; resubmit resumes if hit
#SBATCH --output=logs/atlas_%A_%a.out
#SBATCH --error=logs/atlas_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results_v2/units
PYTHON=${PYTHON:-/public/software/Anaconda3/bin/python3}   # cluster Anaconda; nothing is installed
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Full target set (no DZ_CONFIG_LIST). Default DZ_SOLVER=v2 in code/atlas_units.py.
unset DZ_CONFIG_LIST DZ_CONFIGS_PER_UNIT || true
export DZ_SOLVER=${DZ_SOLVER:-v2}
N_ELEMENTS=${N_ELEMENTS:-32}              # must equal the array size
echo "[element ${SLURM_ARRAY_TASK_ID}] host=$(hostname) start=$(date -Iseconds) python=$($PYTHON -V 2>&1) DZ_SOLVER=$DZ_SOLVER outdir=results_v2/units"
"$PYTHON" code/run_node.py "${SLURM_ARRAY_TASK_ID}" "${N_ELEMENTS}" --procs "${SLURM_CPUS_PER_TASK:-32}" --outdir results_v2/units
echo "[element ${SLURM_ARRAY_TASK_ID}] end=$(date -Iseconds)"
