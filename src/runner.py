"""
src/runner.py — Single-config executor for the DZ-Unmix Bounds grid scan.

Reads one task row from a configs.parquet, runs the specified method, and
writes a single-row parquet to outputs/{grid_name}/results/{config_id}.parquet.

v0 SKELETON: the method call is a *deterministic mock* — see _stub_run().
This lets the full pipeline (grid → array → aggregate) be wired and tested
end-to-end before adapter_cg.py is implemented against the actual
dz-unmix-cg code in v1.

Idempotency: if the result file already exists, skip (unless --force). This
makes resume-after-failure free for SLURM array tasks — rerun the same
sbatch and only un-done configs execute.

CLI:
    python -m src.runner outputs/pilot_v0/configs.parquet --task-id 0
    python -m src.runner outputs/pilot_v0/configs.parquet --task-id $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _seed_from_config(config: dict[str, Any]) -> int:
    """Stable integer seed derived from config_id + seed for reproducible mock."""
    payload = f"{config['config_id']}-{config['seed']}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _stub_run(config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic mock method run for v0 skeleton.

    Returns plausible-shaped metrics so downstream aggregation and smoke tests
    can develop without the real adapter. Mock accuracy degrades qualitatively
    correctly with K, N, noise, and endmember similarity — so dry-run plots
    look roughly like expected real results, catching viz/aggregation bugs early.

    Replace with `adapter_cg.run(config)` in v1.
    """
    rng = np.random.default_rng(_seed_from_config(config))

    base = 0.99
    base -= 0.05 * max(0, int(config["n_endmembers"]) - 2)
    base -= 0.10 * max(0, (300 - int(config["n_zircons_per_sample"])) / 300)
    base -= 0.20 * float(config["noise_level"])
    base -= 0.30 * max(0, 0.4 - float(config["endmember_similarity"]))
    base = float(np.clip(base, 0.2, 0.999))

    top1 = float(np.clip(base + rng.normal(0, 0.02), 0.0, 1.0))
    top1_dom = float(np.clip(top1 + rng.normal(0.005, 0.01), 0.0, 1.0))
    return {
        "top1_accuracy": top1,
        "top1_accuracy_dominant": top1_dom,
        "weights_MAE": float(np.clip(0.05 + (1 - top1) * 0.5, 0.0, 1.0)),
        "weights_R2": float(np.clip(2 * top1 - 1, -1.0, 1.0)),
        "failure_flag": bool(top1 < 0.5),
        "elapsed_seconds": float(rng.uniform(0.5, 2.0)),
        "stub_v0": True,
    }


def _real_run(config: dict[str, Any]) -> dict[str, Any]:
    """Defer adapter import so stub-only paths don't require dz-unmix-cg."""
    from src.adapter_cg import run as adapter_run
    return adapter_run(config)


def run_one_task(
    configs_df: pd.DataFrame, task_id: int, use_stub: bool = False
) -> pd.DataFrame:
    """Run task at row `task_id` of configs_df → 1-row result DataFrame.

    `use_stub=True` short-circuits to the v0 mock for regression testing.
    Default is the real adapter_cg.run() against dz-unmix-cg.
    """
    if task_id < 0 or task_id >= len(configs_df):
        raise IndexError(f"task_id {task_id} out of range [0, {len(configs_df)})")
    config = configs_df.iloc[task_id].to_dict()
    t0 = time.time()
    metrics = _stub_run(config) if use_stub else _real_run(config)
    wall = time.time() - t0
    if math.isnan(metrics.get("elapsed_seconds", float("nan"))):
        metrics["elapsed_seconds"] = wall
    result = {**config, **metrics, "task_id": int(task_id), "wall_seconds": wall}
    return pd.DataFrame([result])


def _result_path(out_root: Path, config_id: str) -> Path:
    return out_root / "results" / f"{config_id}.parquet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one config task from a configs.parquet"
    )
    parser.add_argument("configs_parquet", help="Task table parquet from src.grid")
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Row index (0-based) of the task to run",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root dir (default: outputs/{grid_name}/, derived from task row)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing result file"
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use v0 deterministic mock instead of dz-unmix-cg adapter (regression mode)",
    )
    args = parser.parse_args(argv)

    configs_path = Path(args.configs_parquet)
    if not configs_path.exists():
        print(f"[ERROR] configs parquet not found: {configs_path}", file=sys.stderr)
        return 1

    configs_df = pd.read_parquet(configs_path)
    if not 0 <= args.task_id < len(configs_df):
        print(
            f"[ERROR] task-id {args.task_id} out of range [0, {len(configs_df)})",
            file=sys.stderr,
        )
        return 1

    row = configs_df.iloc[args.task_id]
    grid_name = row["grid_name"]
    config_id = row["config_id"]
    out_root = Path(args.out_root) if args.out_root else Path("outputs") / grid_name
    result_file = _result_path(out_root, config_id)

    if result_file.exists() and not args.force:
        print(f"[runner] task {args.task_id} ({config_id}) already done — skipping")
        return 0

    result_df = run_one_task(configs_df, args.task_id, use_stub=args.stub)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(result_file, index=False)
    top1 = result_df.iloc[0]["top1_accuracy"]
    mode = "stub" if args.stub else "real"
    print(
        f"[runner] task {args.task_id} ({config_id}) → {result_file} "
        f"[{mode} top1={top1:.4f}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
