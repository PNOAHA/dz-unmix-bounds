"""
scripts/run_local.py — Sequential local runner for any grid.

Runs every task in a configs.parquet on one core, in-process (no SLURM, no
subprocess startup overhead). Useful for:
  - dev iteration on small pilots before submitting to D107
  - smoke-testing a new method adapter
  - reproducing a single grid_name's results on a workstation

Per-task parquet writes are idempotent (runner.py skips done configs), so
this script doubles as a "resume-on-workstation" tool if the cluster died
partway through a run.

Usage:
    python scripts/run_local.py outputs/pilot_v0/configs.parquet
    python scripts/run_local.py outputs/pilot_v0/configs.parquet --stub
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runner import _result_path, run_one_task  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential local runner")
    parser.add_argument("configs_parquet")
    parser.add_argument("--stub", action="store_true", help="Use v0 mock instead of real adapter")
    parser.add_argument("--force", action="store_true", help="Re-run already-done tasks")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)

    configs_path = Path(args.configs_parquet)
    configs_df = pd.read_parquet(configs_path)
    grid_name = configs_df.iloc[0]["grid_name"]
    out_root = ROOT / "outputs" / grid_name
    (out_root / "results").mkdir(parents=True, exist_ok=True)

    print(f"[run_local] {len(configs_df)} tasks  grid={grid_name}  mode={'stub' if args.stub else 'real'}")

    n_done, n_skipped = 0, 0
    t0 = time.time()
    for i in range(len(configs_df)):
        cid = configs_df.iloc[i]["config_id"]
        rp = _result_path(out_root, cid)
        if rp.exists() and not args.force:
            n_skipped += 1
            continue
        result_df = run_one_task(configs_df, i, use_stub=args.stub)
        result_df.to_parquet(rp, index=False)
        n_done += 1
        if n_done % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"[run_local] {n_done}/{len(configs_df) - n_skipped} done  {rate:.1f} task/s")

    elapsed = time.time() - t0
    print(f"[run_local] finished — {n_done} new, {n_skipped} skipped ({elapsed:.1f}s)")

    # Aggregate
    paths = sorted((out_root / "results").glob("*.parquet"))
    agg = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    agg_path = out_root / "aggregated.parquet"
    agg.to_parquet(agg_path, index=False)
    print(f"[run_local] aggregated {len(agg)} rows → {agg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
