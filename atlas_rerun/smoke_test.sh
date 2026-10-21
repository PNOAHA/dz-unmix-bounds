#!/bin/bash
# Smoke test: 2 work units (unit 0 = K=2 low noise; unit 999 = K=15 high noise), first 8 configs each,
# 2 processes, written to results_smoke/ (never to results/ or results_v2/). Then post-processing in --smoke mode.
# Uses default DZ_SOLVER=v2 (exact KKT). Unset subset env vars so this is a full-target slice, not DZ_CONFIG_LIST.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-/public/software/Anaconda3/bin/python3}
rm -rf results_smoke
unset DZ_CONFIG_LIST DZ_CONFIGS_PER_UNIT || true
export DZ_SOLVER=${DZ_SOLVER:-v2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "smoke: PYTHON=$PYTHON DZ_SOLVER=$DZ_SOLVER"
time "$PYTHON" code/run_node.py 0 1 --units 0,999 --max-configs 8 --procs 2 --outdir results_smoke/units
# rerun to show restart behaviour (both units must be reported as skipped)
"$PYTHON" code/run_node.py 0 1 --units 0,999 --max-configs 8 --procs 2 --outdir results_smoke/units
"$PYTHON" code/postprocess.py --units-dir results_smoke/units --out results_smoke --smoke
