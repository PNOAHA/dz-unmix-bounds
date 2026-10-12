"""
tests/test_adapter.py — Sanity checks for src/adapter_cg.py against dz-unmix-cg.

Skipped automatically if ../dz-unmix-cg/ is not present (e.g. in CI without
the upstream repo). Exercises actual NMF code, so much slower than test_dry_run.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CG_SIBLING = ROOT.parent / "dz-unmix-cg" / "dz_unmixing_experiment.py"
pytestmark = pytest.mark.skipif(
    not CG_SIBLING.exists(),
    reason=f"dz-unmix-cg not available at {CG_SIBLING}",
)


@pytest.fixture(scope="module")
def adapter():
    from src import adapter_cg
    return adapter_cg


def _base_config(**overrides) -> dict:
    """Minimal valid config row for adapter_cg.run."""
    cfg = {
        "config_id": "test_dummy",
        "grid_name": "test",
        "method": "supervised_nmf",
        "seed": 42,
        "n_endmembers": 2,
        "n_zircons_per_sample": 500,
        "endmember_similarity": 0.4,
        "noise_level": 0.0,
        "dirichlet_alpha": 1.0,
        "bandwidth": 10.0,
        "n_folds": 5,
        "n_mixed_samples": 50,
    }
    cfg.update(overrides)
    return cfg


def test_easy_case_unmixes_well(adapter):
    """K=2, N=500, ρ=0.4, ε=0 should be a very easy case for supervised NMF."""
    result = adapter.run(_base_config())
    assert result["top1_accuracy"] > 0.85, (
        f"easy case should be solvable, got top1={result['top1_accuracy']:.3f}"
    )
    assert not result["failure_flag"]
    assert result["stub_v0"] is False
    assert result["ks_distance_measured"] > 0.5  # peaks well-separated


def test_harder_case_lower_accuracy(adapter):
    """ρ=0.1 (overlapping peaks) should be measurably harder than ρ=0.4."""
    easy = adapter.run(_base_config(endmember_similarity=0.4))
    hard = adapter.run(_base_config(endmember_similarity=0.1))
    assert hard["top1_accuracy"] <= easy["top1_accuracy"] + 1e-9, (
        f"harder ρ should not be more accurate: easy={easy['top1_accuracy']:.3f} "
        f"hard={hard['top1_accuracy']:.3f}"
    )
    # And the measured KS should reflect the requested ordering
    assert hard["ks_distance_measured"] < easy["ks_distance_measured"]


def test_metrics_schema_matches_stub(adapter):
    """Adapter result must contain the same metric keys as the v0 stub so
    aggregation code doesn't have to special-case the source."""
    from src.runner import _stub_run
    cfg = _base_config()
    stub_keys = set(_stub_run(cfg).keys())
    real_keys = set(adapter.run(cfg).keys())
    missing = stub_keys - real_keys
    assert not missing, f"adapter missing keys present in stub: {missing}"


def test_seed_determinism(adapter):
    """Same config → identical metrics across runs."""
    cfg = _base_config()
    r1 = adapter.run(cfg)
    r2 = adapter.run(cfg)
    assert r1["top1_accuracy"] == r2["top1_accuracy"]
    assert r1["weights_MAE"] == r2["weights_MAE"]
    assert r1["ks_distance_measured"] == r2["ks_distance_measured"]


def test_k3_works(adapter):
    """K=3 should also run (matches dz-unmix-cg DEFAULT_SOURCES dimensionality)."""
    result = adapter.run(_base_config(n_endmembers=3))
    assert 0.0 <= result["top1_accuracy"] <= 1.0
    assert result["ks_distance_measured"] > 0.0


def test_synthetic_sources_count(adapter):
    """make_synthetic_sources should return exactly K Source objects."""
    for k in [2, 3, 5]:
        srcs = adapter.make_synthetic_sources(n_endmembers=k, endmember_similarity=0.3)
        assert len(srcs) == k


# ----------------------------------------------------------------------------
# v1.1: failure-regime tests — validates that the tightened ρ heuristic
# actually reaches the failure region (which v1 pilot dry-run could not).
# ----------------------------------------------------------------------------

def test_rho_heuristic_reaches_failure_regime(adapter):
    """ρ=0.01 → 0.1σ spacing → peaks essentially coincide → KS very small."""
    srcs = adapter.make_synthetic_sources(n_endmembers=2, endmember_similarity=0.01)
    ks = adapter.measure_avg_pairwise_ks(srcs)
    assert ks < 0.1, f"ρ=0.01 should give near-zero KS, got {ks:.3f}"


def test_rho_heuristic_reaches_easy_regime(adapter):
    """ρ=0.5 → 5σ spacing → peaks well-separated → KS near 1."""
    srcs = adapter.make_synthetic_sources(n_endmembers=2, endmember_similarity=0.5)
    ks = adapter.measure_avg_pairwise_ks(srcs)
    assert ks > 0.95, f"ρ=0.5 should give near-saturated KS, got {ks:.3f}"


def test_identical_peaks_yield_near_random(adapter):
    """ρ=0 → endmembers literally coincide → argmax is essentially random.

    For K=2, random baseline = 0.5. Supervised NMF is otherwise very robust
    to overlap (because H is given), so this is the ONE regime that
    fundamentally breaks identifiability — H[0] and H[1] are the same matrix,
    weights become non-identifiable.
    """
    result = adapter.run(_base_config(endmember_similarity=0.0))
    assert result["ks_distance_measured"] < 0.01
    assert result["top1_accuracy"] < 0.7, (
        f"K=2 identical-peak case should be near-random, got {result['top1_accuracy']:.3f}"
    )


def test_K5_tight_overlap_triggers_failure(adapter):
    """K=5 + ρ=0.01: high-dim unmixing with near-identical peaks → failure regime.

    This is the empirically discovered "interesting failure" mode — neither
    ρ↓ alone (K=2 robust) nor K↑ alone (well-separated peaks still work) is
    enough; the *interaction* breaks the method. This is the kind of result
    the paper's phase diagram should highlight.
    """
    result = adapter.run(_base_config(n_endmembers=5, endmember_similarity=0.01))
    assert result["failure_flag"], (
        f"K=5 + ρ=0.01 should trigger failure_flag, top1={result['top1_accuracy']:.3f}"
    )
    assert result["top1_accuracy"] < 0.5
