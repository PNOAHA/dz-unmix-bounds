"""
src/grid.py — Cartesian-product expansion of a parameter grid YAML to a parquet
task table for downstream consumption by runner.py and SLURM job array.

Each row in the output parquet is exactly one independent task. SLURM job array
indexes pick rows by integer offset; runner.py reads one row and emits one
result file. This *stateless task* design is the standard HPC pattern for
embarrassingly parallel workloads.

CLI:
    python -m src.grid config/grid_pilot.yaml --out outputs/pilot_v0/configs.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _stable_config_id(record: dict[str, Any]) -> str:
    """Deterministic 16-char hex hash of a config dict (sorted-key JSON-canonical).

    Same params → same id across runs, machines, and grid files. Excludes
    grid_name on purpose so two grids can share config IDs for cross-grid joins.
    """
    canonical = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _expand_values_dict(section: dict[str, Any]) -> dict[str, list[Any]]:
    """Pick {param_name: values_list} from a YAML section where each entry
    declares a `values:` key. Ignores description-only entries so the YAML
    can carry documentation freely."""
    out: dict[str, list[Any]] = {}
    for k, v in section.items():
        if isinstance(v, dict) and "values" in v:
            out[k] = list(v["values"])
    return out


def build_task_table(grid_yaml: dict[str, Any]) -> pd.DataFrame:
    """Expand parsed grid YAML into a one-row-per-task DataFrame."""
    grid_meta = grid_yaml["grid"]
    data_axes = _expand_values_dict(grid_yaml["data"])
    method_param_axes = _expand_values_dict(grid_yaml["method_params"])
    methods = [m["name"] for m in grid_yaml["methods"]]
    seeds = list(grid_yaml["control"]["seeds"])
    n_folds = grid_yaml["control"]["n_folds"]
    n_mixed_samples = grid_yaml["control"]["n_mixed_samples"]

    data_keys = list(data_axes.keys())
    data_vals = [data_axes[k] for k in data_keys]
    mp_keys = list(method_param_axes.keys())
    mp_vals = [method_param_axes[k] for k in mp_keys]

    rows: list[dict[str, Any]] = []
    for data_combo, mp_combo, method, seed in product(
        product(*data_vals), product(*mp_vals), methods, seeds
    ):
        row: dict[str, Any] = {
            "grid_name": grid_meta["name"],
            "method": method,
            "seed": int(seed),
            "n_folds": int(n_folds),
            "n_mixed_samples": int(n_mixed_samples),
        }
        row.update(dict(zip(data_keys, data_combo)))
        row.update(dict(zip(mp_keys, mp_combo)))
        id_payload = {k: row[k] for k in row if k != "grid_name"}
        row["config_id"] = _stable_config_id(id_payload)
        rows.append(row)

    df = pd.DataFrame(rows)
    leading = ["config_id", "grid_name", "method", "seed"]
    other = [c for c in df.columns if c not in leading]
    return df[leading + other]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand a grid YAML to a parquet task table",
    )
    parser.add_argument("yaml_path", help="Input grid YAML file")
    parser.add_argument(
        "--out", required=True, help="Output parquet path (parents created)"
    )
    args = parser.parse_args(argv)

    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        print(f"[ERROR] Grid YAML not found: {yaml_path}", file=sys.stderr)
        return 1

    with yaml_path.open("r", encoding="utf-8") as f:
        grid_yaml = yaml.safe_load(f)

    df = build_task_table(grid_yaml)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"[grid] {len(df)} tasks → {out_path}")
    print(f"[grid] columns: {list(df.columns)}")
    print("[grid] preview (first 3 rows):")
    print(df.head(3).to_string(index=False))
    if df["config_id"].nunique() != len(df):
        dup = len(df) - df["config_id"].nunique()
        print(f"[grid][WARN] {dup} duplicate config_id rows — check grid YAML")
    return 0


if __name__ == "__main__":
    sys.exit(main())
