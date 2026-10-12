# -*- coding: utf-8 -*-
"""
diag_w_sweep.py -- SKEPTIC test of the "worst mode = curvature at large rho" claim.

Question: is the curvature-dominance at large rho an ARTIFACT of the specific
endpoint true w = [0.417, 0.333, 0.25]?  Re-run the eigenvector diagnostic at
rho in {0.47, 0.35}, n=350, K=3 for SEVERAL true weight vectors and check whether
the top tangent eigenvector still overlaps the curvature mode u2 with C_curv ~ 0.7.

If worst-mode identity is w-fragile -> claim refuted as stated.

Reuses sigma_err_scan internals (real dz-unmix-cg core, numpy backend).
"""
import math
import numpy as np
import sigma_err_scan as S

K, R, SIGMA, BW = 3, 600, S.SIGMA_DEFAULT, S.BW_DEFAULT
C_FLOOR = 0.500

u1 = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)   # extreme pairwise (antisymmetric)
u2 = np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0)   # curvature (symmetric)

# weight vectors to test
W_SET = {
    "endpoint(base)": S.fixed_interior_w(3, "endpoint"),   # [.417,.333,.25]  baseline
    "middle":         S.fixed_interior_w(3, "middle"),     # [.25,.417,.333]  max interior
    "uniform":        np.array([1/3, 1/3, 1/3]),           # symmetric
    "asym_0.6/0.25/0.15": np.array([0.60, 0.25, 0.15]),    # strongly skewed, max at end
    "asym_0.15/0.25/0.6": np.array([0.15, 0.25, 0.60]),    # mirror, max at other end
    "interior_heavy":  np.array([0.20, 0.60, 0.20]),       # symmetric, interior-dominated
    "near_vertex":     np.array([0.80, 0.10, 0.10]),       # near a corner of the simplex
}

CELLS = [(0.47, 350), (0.35, 350)]


def what_for(rho, N, w, seed):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    return core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)


for rho, N in CELLS:
    print(f"\n=== rho={rho} n={N} K=3 ===  (u1=pairwise, u2=curvature)")
    print("%-22s | %8s %8s | %5s %5s | %7s %7s | %s"
          % ("true w", "sqrt_l1", "sqrt_l2", "|.u1|", "|.u2|", "C_u1", "C_u2", "excess"))
    floor = C_FLOOR / (math.sqrt(N) * rho)
    for name, w in W_SET.items():
        w = np.asarray(w, dtype=float); w = w / w.sum()
        seed = 20260611 + K * 1000 + int(rho * 1e4) + int(1e6 * w[0])  # vary seed by w
        what = what_for(rho, N, w, seed)
        dev = what - what.mean(0, keepdims=True)
        cov = dev.T @ dev / what.shape[0]
        lam, V = np.linalg.eigh(cov)
        l1, l2 = lam[-1], lam[-2]
        top = V[:, -1]
        o1, o2 = abs(top @ u1), abs(top @ u2)
        var_u1 = float(u1 @ cov @ u1); var_u2 = float(u2 @ cov @ u2)
        C_u1 = math.sqrt(max(var_u1, 0)) * math.sqrt(N) * rho
        C_u2 = math.sqrt(max(var_u2, 0)) * math.sqrt(N) * rho
        print("%-22s | %8.4f %8.4f | %5.2f %5.2f | %7.3f %7.3f | %.2fx"
              % (name, math.sqrt(max(l1, 0)), math.sqrt(max(l2, 0)), o1, o2,
                 C_u1, C_u2, math.sqrt(max(l1, 0)) / floor))
