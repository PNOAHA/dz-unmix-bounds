#!/usr/bin/env bash
# slurm/submit.sh — Submit a DZ-Unmix Bounds grid scan as a SLURM job array on D107.
#
# Usage:
#   ./slurm/submit.sh outputs/pilot_v0/configs.parquet            # default partition / QoS
#   MAX_CONCURRENT=32 ./slurm/submit.sh outputs/full_v0/configs.parquet
#
# Idempotent: runner.py skips configs whose result file already exists, so
# rerunning the same sbatch after partial completion just fills the gaps.
#
# v0 skeleton — D107-specific SLURM directives (partition, QoS, mail) are TODO.
# Run `sinfo` and `sacctmgr show qos` on D107 to fill them in before first
# production submission.

set -euo pipefail

# -------- Arguments & defaults --------
CONFIGS_PARQUET="${1:?Usage: submit.sh <configs.parquet>}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"   # cap simultaneous array tasks
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$CONFIGS_PARQUET" ]]; then
  echo "[submit][ERROR] configs parquet not found: $CONFIGS_PARQUET" >&2
  exit 1
fi

# -------- Determine task count --------
N_TASKS=$("$PYTHON_BIN" -c "import pandas; print(len(pandas.read_parquet('$CONFIGS_PARQUET')))")
if [[ -z "$N_TASKS" || "$N_TASKS" -le 0 ]]; then
  echo "[submit][ERROR] could not determine task count" >&2
  exit 1
fi
LAST_IDX=$((N_TASKS - 1))

# -------- Derive grid name (for log paths) --------
GRID_NAME=$("$PYTHON_BIN" -c "import pandas; print(pandas.read_parquet('$CONFIGS_PARQUET').iloc[0]['grid_name'])")
LOG_DIR="$PROJECT_ROOT/outputs/$GRID_NAME/logs"
mkdir -p "$LOG_DIR"

echo "[submit] grid_name      : $GRID_NAME"
echo "[submit] configs        : $CONFIGS_PARQUET"
echo "[submit] tasks          : $N_TASKS (array 0-$LAST_IDX)"
echo "[submit] max concurrent : $MAX_CONCURRENT"
echo "[submit] log dir        : $LOG_DIR"
echo "[submit] project root   : $PROJECT_ROOT"

# -------- Submit --------
# NOTE: D107-specific options (partition, QoS, account) — fill in after
# verifying on cluster. Currently using SLURM defaults.
sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dz-bounds-${GRID_NAME}
#SBATCH --array=0-${LAST_IDX}%${MAX_CONCURRENT}
#SBATCH --output=${LOG_DIR}/task_%A_%a.out
#SBATCH --error=${LOG_DIR}/task_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# #SBATCH --partition=TODO_partition
# #SBATCH --qos=TODO_qos
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=panlujia234@gmail.com

set -euo pipefail

cd "${PROJECT_ROOT}"

# Make src/ importable when running under SLURM
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8

echo "[task \${SLURM_ARRAY_TASK_ID}] host=\$(hostname) start=\$(date -Iseconds)"
"${PYTHON_BIN}" -m src.runner "${CONFIGS_PARQUET}" --task-id "\${SLURM_ARRAY_TASK_ID}"
echo "[task \${SLURM_ARRAY_TASK_ID}] end=\$(date -Iseconds)"
EOF

echo "[submit] submitted — monitor with: squeue -u \$USER -t RUNNING,PENDING"
echo "[submit] when done: tail ${LOG_DIR}/task_*.out | grep top1 | wc -l   # count completions"
