"""
tests/test_dry_run.py — End-to-end smoke test for the v0 pipeline.

Wires grid → runner → per-task parquet writes in a tmp dir, with no SLURM and
no real method (stub_run). Catches schema drift, CLI signature mismatches, and
parquet round-trip issues before they propagate to the cluster.

Run:
    pytest tests/test_dry_run.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.grid import build_task_table  # noqa: E402
from src.runner import run_one_task  # noqa: E402


@pytest.fixture
def tiny_grid_yaml() -> dict:
    """Minimal grid: 2 × 2 × 1 method × 2 seeds = 8 tasks."""
    return {
        "grid": {
            "name": "tiny_test",
            "description": "smoke test",
            "version": "0.0.0",
        },
        "data": {
            "n_endmembers": {"values": [2, 3]},
            "n_zircons_per_sample": {"values": [100]},
            "endmember_similarity": {"values": [0.1, 0.4]},
            "noise_level": {"values": [0.0]},
            "dirichlet_alpha": {"values": [1.0]},
        },
        "method_params": {
            "bandwidth": {"values": [10.0]},
        },
        "methods": [
            {"name": "supervised_nmf", "description": "stub"},
        ],
        "control": {
            "seeds": [7, 13],
            "n_folds": 5,
            "n_mixed_samples": 50,
        },
        "output": {"base_dir": "outputs/tiny_test", "format": "parquet"},
    }


def test_grid_expansion_count(tiny_grid_yaml):
    """2 K × 1 N × 2 ρ × 1 ε × 1 α × 1 bw × 1 method × 2 seeds = 8 rows."""
    df = build_task_table(tiny_grid_yaml)
    assert len(df) == 8


def test_grid_has_required_columns(tiny_grid_yaml):
    df = build_task_table(tiny_grid_yaml)
    required = {
        "config_id", "grid_name", "method", "seed",
        "n_endmembers", "n_zircons_per_sample", "endmember_similarity",
        "noise_level", "dirichlet_alpha", "bandwidth",
        "n_folds", "n_mixed_samples",
    }
    assert required.issubset(set(df.columns)), f"missing: {required - set(df.columns)}"


def test_config_id_is_stable(tiny_grid_yaml):
    """Same YAML → same config_ids (sorted) across two expansions."""
    df1 = build_task_table(tiny_grid_yaml)
    df2 = build_task_table(tiny_grid_yaml)
    assert sorted(df1["config_id"].tolist()) == sorted(df2["config_id"].tolist())


def test_config_id_is_unique(tiny_grid_yaml):
    """No two task rows should share a config_id (no parameter dup)."""
    df = build_task_table(tiny_grid_yaml)
    assert df["config_id"].nunique() == len(df)


def test_runner_stub_produces_metrics(tiny_grid_yaml):
    df = build_task_table(tiny_grid_yaml)
    result = run_one_task(df, task_id=0, use_stub=True)
    assert len(result) == 1
    row = result.iloc[0]
    assert 0.0 <= row["top1_accuracy"] <= 1.0
    assert bool(row["stub_v0"]) is True  # pandas returns np.True_, not Python True
    assert row["task_id"] == 0
    assert row["config_id"] == df.iloc[0]["config_id"]


def test_runner_is_deterministic(tiny_grid_yaml):
    """Same task_id → bit-identical metrics across calls (no hidden RNG)."""
    df = build_task_table(tiny_grid_yaml)
    r1 = run_one_task(df, task_id=0, use_stub=True).iloc[0]
    r2 = run_one_task(df, task_id=0, use_stub=True).iloc[0]
    assert r1["top1_accuracy"] == r2["top1_accuracy"]
    assert r1["weights_MAE"] == r2["weights_MAE"]


def test_end_to_end_parquet_roundtrip(tmp_path, tiny_grid_yaml):
    """grid → parquet → runner (stub) → parquet → readback. Catches schema drift."""
    df = build_task_table(tiny_grid_yaml)
    configs_path = tmp_path / "configs.parquet"
    df.to_parquet(configs_path, index=False)

    configs_df = pd.read_parquet(configs_path)
    assert len(configs_df) == len(df)

    out_root = tmp_path / "outputs" / "tiny_test"
    (out_root / "results").mkdir(parents=True, exist_ok=True)

    for task_id in range(len(configs_df)):
        result_df = run_one_task(configs_df, task_id, use_stub=True)
        config_id = configs_df.iloc[task_id]["config_id"]
        result_df.to_parquet(out_root / "results" / f"{config_id}.parquet", index=False)

    written = list((out_root / "results").glob("*.parquet"))
    assert len(written) == len(df)

    aggregated = pd.concat([pd.read_parquet(p) for p in written], ignore_index=True)
    assert len(aggregated) == len(df)
    assert set(aggregated["config_id"]) == set(df["config_id"])
