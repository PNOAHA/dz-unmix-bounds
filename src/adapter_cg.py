"""
src/adapter_cg.py — Adapter from a DZ-Unmix Bounds grid config to the
dz-unmix-cg supervised NMF method.

Treats dz-unmix-cg as a frozen upstream dependency (DOI reserved for the
first paper). Imports its source via sys.path injection — NEVER modifies
any file under ../dz-unmix-cg/.

What this adapter does that dz-unmix-cg cannot:
  - Generate synthetic endmember sets parameterized by (K, ρ) — the
    upstream DEFAULT_SOURCES is fixed at 3 NE-Asia K1 sources.
  - Generate mixed datasets with arbitrary (K, N, noise_level) — the
    upstream `generate_dataset` hardcodes these.
  - Record post-hoc measured KS distance between endmembers alongside the
    requested similarity, so the heuristic ρ-control can be independently
    audited.

What this adapter DOES NOT do (intentionally — reuses upstream verbatim):
  - The supervised NMF algorithm itself (`supervised_unmix`)
  - KDE density estimation (`kde_density`)
  - Endmember matrix construction (`build_source_endmembers`)
  - Metric definitions (`report` + sub-functions)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

# Inject dz-unmix-cg into import path. This is the only place that touches
# the upstream — adapter_cg owns the contract.
_CG_PATH = Path(__file__).resolve().parents[2] / "dz-unmix-cg"
if str(_CG_PATH) not in sys.path:
    sys.path.insert(0, str(_CG_PATH))

import dz_unmixing_experiment as cg  # noqa: E402


# ----------------------------------------------------------------------------
# Synthetic endmember construction with (K, ρ)-control
# ----------------------------------------------------------------------------

def make_synthetic_sources(
    n_endmembers: int,
    endmember_similarity: float,
    sigma: float = 30.0,
    mean_start: float = 200.0,
) -> list[cg.Source]:
    """K synthetic Gaussian endmembers with controlled spacing.

    Heuristic (v1.1): spacing between adjacent peak means = sigma × 10 × ρ.
      ρ = 0.01 → 0.1σ   (peaks essentially coincide — method must fail)
      ρ = 0.05 → 0.5σ   (heavy overlap — failure regime)
      ρ = 0.10 → 1.0σ   (critical/transition regime)
      ρ = 0.30 → 3.0σ   (mostly separable)
      ρ = 0.50 → 5.0σ   (well-separated — easy)

    Recommended grid range: ρ ∈ [0.01, 0.5] to span failure → success regimes.

    The `endmember_similarity` arg is a *control parameter*, not an exact
    KS distance — the runner records both the requested value and the
    post-hoc measured KS distance (`ks_distance_measured`) so the heuristic
    can be audited and the paper can plot using empirically measured KS
    on the x-axis rather than the synthetic control parameter.

    Why this mapping and not f(ρ) = σ(1 + 10ρ): the previous form bottomed
    out at 1σ spacing (still separable), so the failure regime was unreachable
    — pilot dry-run revealed all configs scored top1 > 0.84 even at ρ=0.1.
    Linear scaling lets the failure regime be hit at ρ < 0.1.
    """
    spacing = sigma * 10.0 * float(endmember_similarity)
    return [
        cg.Source(
            name=f"Synth_{i:02d}",
            mean_age=mean_start + i * spacing,
            sigma=sigma,
            geo_label="synthetic_v1.1",
        )
        for i in range(int(n_endmembers))
    ]


def measure_avg_pairwise_ks(
    sources: list[cg.Source], t_grid: np.ndarray = cg.T_GRID
) -> float:
    """Mean pairwise KS distance between endmember densities.

    Densities are evaluated on `t_grid`, normalized to unit area; CDFs are
    obtained by cumulative integration. KS is max|CDF_i − CDF_j| over the
    grid. For k < 2 returns 0.0.
    """
    H = cg.build_source_endmembers(sources=sources, t_grid=t_grid)
    dt = t_grid[1] - t_grid[0]
    cdfs = np.cumsum(H, axis=1) * dt
    cdfs = cdfs / cdfs[:, -1:]
    k = H.shape[0]
    if k < 2:
        return 0.0
    distances = [
        float(np.max(np.abs(cdfs[i] - cdfs[j])))
        for i in range(k)
        for j in range(i + 1, k)
    ]
    return float(np.mean(distances))


# ----------------------------------------------------------------------------
# Mixed-dataset generation bypassing upstream's hardcoded defaults
# ----------------------------------------------------------------------------

def generate_mixed_dataset(
    sources: list[cg.Source],
    n_mixed_samples: int,
    n_zircons_per_sample: int,
    noise_level: float,
    dirichlet_alpha: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build (ages_list, true_weights) for the given config.

    Bypasses cg.generate_dataset (which hardcodes n_grains_range=(100,300)
    and noise_pct=0.05); calls cg.generate_sample per mixed sample with
    explicit per-config parameters.
    """
    k = len(sources)
    true_weights = rng.dirichlet(
        alpha=np.full(k, float(dirichlet_alpha)), size=int(n_mixed_samples)
    )
    ages_list = [
        cg.generate_sample(
            weights=w,
            n_grains_range=(int(n_zircons_per_sample), int(n_zircons_per_sample)),
            noise_pct=float(noise_level),
            sources=sources,
            rng=rng,
        )
        for w in true_weights
    ]
    return ages_list, true_weights


# ----------------------------------------------------------------------------
# Top-level adapter entry point — matches stub_run's signature
# ----------------------------------------------------------------------------

def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute one config end-to-end. Returns metrics dict.

    Output schema is a strict superset of `_stub_run`'s — the runner can
    swap stub ↔ adapter without changing parquet column structure.
    """
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)

    # 1. Construct K synthetic endmembers controlled by (K, ρ)
    sources = make_synthetic_sources(
        n_endmembers=int(config["n_endmembers"]),
        endmember_similarity=float(config["endmember_similarity"]),
    )
    ks_measured = measure_avg_pairwise_ks(sources)

    # 2. Generate mixed samples with controlled (N, ε, α)
    ages_list, true_w = generate_mixed_dataset(
        sources=sources,
        n_mixed_samples=int(config["n_mixed_samples"]),
        n_zircons_per_sample=int(config["n_zircons_per_sample"]),
        noise_level=float(config["noise_level"]),
        dirichlet_alpha=float(config["dirichlet_alpha"]),
        rng=rng,
    )

    # 3. KDE → fixed-grid density vectors (bandwidth controlled)
    bandwidth = float(config["bandwidth"])
    X = np.array([cg.kde_density(a, bandwidth=bandwidth) for a in ages_list])
    H = cg.build_source_endmembers(sources=sources, t_grid=cg.T_GRID)

    # 4. K-fold CV with the upstream supervised_unmix algorithm verbatim
    n_folds = int(config["n_folds"])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold_metrics: list[dict[str, float]] = []
    for _tr_idx, te_idx in kf.split(X):
        pred = cg.supervised_unmix(X[te_idx], H)
        per_fold_metrics.append(cg.report(true_w[te_idx], pred))

    # 5. Aggregate across folds (use nanmean to tolerate top1_dominant NaN
    # when no sample has max-weight ≥ 0.5 — e.g. high α flat mixing).
    def _agg(key: str, fn) -> float:
        vals = np.array([m[key] for m in per_fold_metrics], dtype=float)
        return float(fn(vals))

    top1 = _agg("top1_accuracy", np.mean)
    return {
        "top1_accuracy": top1,
        "top1_accuracy_std": _agg("top1_accuracy", np.std),
        "top1_accuracy_dominant": _agg("top1_accuracy_dominant", np.nanmean),
        "weights_MAE": _agg("weights_MAE", np.mean),
        "weights_R2": _agg("weights_R2", np.mean),
        "ks_distance_measured": ks_measured,
        "failure_flag": bool(top1 < 0.5),
        "elapsed_seconds": float("nan"),  # filled by runner.run_one_task
        "stub_v0": False,
    }
