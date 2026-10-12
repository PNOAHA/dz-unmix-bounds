# -*- coding: utf-8 -*-
"""
probe_bw_mechanism.py -- SKEPTIC test of mechanism (C): KDE bandwidth bw acts as a
low-pass on the density, so LARGER bw -> stronger low-pass -> SMALLEST effective Fisher
for the most oscillatory (degree K-1) weight-contrast -> LARGER variance -> LARGER excess.

At fixed K in {4,5}, rho=0.47, n=350, sweep bw in {6,12,18,24}.
  - H (dictionary) is ALWAYS built from sigma via build_source_endmembers (bw-independent) -- verified.
  - ONLY core.kde_density(..., bandwidth=bw) changes -> the low-pass on the *data*.

Prediction if (C) is right:
  (i)  excess MONOTONE INCREASING in bw (toward 1 as bw -> small).
  (ii) dominant contrast degree stays K-1 across all bw.

Refute (C) if bw has no monotone effect, the wrong sign, or the worst degree flips off K-1.

Reuses sigma_err_scan real core (numpy backend). Self-contained driver; does NOT touch
diag/scan files or out_phase. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_bw_mechanism.py
"""
import math
import numpy as np
import sigma_err_scan as S

R, SIGMA, C_K2 = 600, S.SIGMA_DEFAULT, 0.500
RHO, N = 0.47, 350
BWS = [6.0, 12.0, 18.0, 24.0]
KS = [4, 5]
SEEDS = [101, 202]


def contrast_polys(K):
    """orthonormal degree-1..K-1 contrast polynomials on the sum-zero tangent."""
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean()
    B = []
    for deg in range(1, K):
        v = x ** deg; v = v - v.mean()
        for b in B:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            B.append(v / nv)
    return B


def measure(K, rho, N, bw, seed, n_iter=2000):
    """Real-core sigma_err + top-mode contrast-degree decomposition at a given KDE bw.
    H is built from sigma (bw-independent); only kde_density uses bw."""
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)         # <-- NO bw here
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=bw)   # <-- ONLY bw here
    what = core.supervised_unmix(X, H, n_iter=n_iter, tol=1e-13)
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)
    top = V[:, -1]
    B = contrast_polys(K)
    ov = [abs(float(top @ b)) for b in B]
    sig = math.sqrt(max(lam[-1], 0.0))
    floor = C_K2 / (math.sqrt(N) * rho)
    dom_deg = int(np.argmax(ov)) + 1
    return sig, sig / floor, dom_deg, ov


def hsanity():
    """Prove H is bw-invariant: build H once; confirm kde output changes with bw."""
    core = S._load_real_core()
    mus, _ = S.peak_ages(4, RHO, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(4)]
    H = core.build_source_endmembers(src, core.T_GRID)
    rng = np.random.default_rng(7)
    w = S.fixed_interior_w(4, "endpoint")
    ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
    x6 = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=6.0)
    x24 = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=24.0)
    print("[H bw-invariance check] H built from sigma only; H.shape=%s  H.sum_per_row(area)~1: %s"
          % (H.shape, np.round(np.trapezoid(H, core.T_GRID, axis=1), 4)))
    print("[KDE bw effect] ||x(bw=6) - x(bw=24)||_2 = %.4f (nonzero => bw changes the data density)"
          % np.linalg.norm(x6 - x24))
    # roughness proxy: total variation of the density (low-pass should reduce it)
    tv6 = np.abs(np.diff(x6)).sum(); tv24 = np.abs(np.diff(x24)).sum()
    print("[low-pass proxy] total-variation(density): bw=6 -> %.5f ,  bw=24 -> %.5f  (smaller=smoother)"
          % (tv6, tv24))
    print()


hsanity()

print("MECHANISM (C) bw-sweep:  rho=%.2f  n=%d   prediction: excess RISES with bw; dom_deg stays K-1" % (RHO, N))
print("%2s %5s %6s %5s | %8s %7s | %7s %6s | overlaps deg=1..K-1"
      % ("K", "bw", "seed", "nit", "sig_err", "excess", "dom_deg", "=K-1?"))
summary = {}
for K in KS:
    for bw in BWS:
        excs = []
        for seed in SEEDS:
            sig, exc, dom, ov = measure(K, RHO, N, bw, 20260615 + K * 1000 + seed)
            excs.append(exc)
            print("%2d %5.1f %6d %5d | %8.4f %7.2f | deg=%d %5s | %s"
                  % (K, bw, seed, 2000, sig, exc, dom, "YES" if dom == K - 1 else "no",
                     " ".join(f"{o:.2f}" for o in ov)))
        summary[(K, bw)] = float(np.mean(excs))
    print()

print("=== EXCESS vs bw (seed-averaged) — monotone increasing in bw is the (C) prediction ===")
for K in KS:
    row = [summary[(K, bw)] for bw in BWS]
    diffs = np.diff(row)
    mono_up = all(d > 0 for d in diffs)
    sign = "MONOTONE UP" if mono_up else ("MONOTONE DOWN" if all(d < 0 for d in diffs) else "NON-MONOTONE")
    print("K=%d  bw %s -> excess %s   d(excess)/d(bw step)=%s   [%s]"
          % (K, BWS, [round(v, 3) for v in row], [round(d, 3) for d in diffs], sign))
