# -*- coding: utf-8 -*-
"""conv_probe2.py -- direct weight-convergence + fine n_iter near the bottom for rho=0.35.

Checks: (1) max |what(n_iter) - what(80000)| per cell -> are the WEIGHTS themselves
converged at n_iter=2000?  (2) C_curv vs very-fine n_iter to locate the plateau.
"""
import math
import numpy as np
import sigma_err_scan as S

K, R, SIGMA, BW = 3, 600, S.SIGMA_DEFAULT, S.BW_DEFAULT
u2 = np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0)


def what_for(rho, N, seed, n_iter):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    return core.supervised_unmix(X, H, n_iter=n_iter, tol=0.0)


def c_curv(what, rho, N):
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    var_u2 = float(u2 @ cov @ u2)
    return math.sqrt(max(var_u2, 0)) * math.sqrt(N) * rho


for rho, N in [(0.47, 350), (0.35, 350)]:
    seed = 20260611 + K * 1000 + int(rho * 1e4)
    ref = what_for(rho, N, seed, 80000)           # gold-standard converged estimate
    print("rho=%.2f n=%d   C_curv(80000)=%.4f" % (rho, N, c_curv(ref, rho, N)))
    print("  %7s | %10s | %8s" % ("n_iter", "max|dW|", "C_curv"))
    for ni in [50, 100, 200, 500, 1000, 2000, 4000]:
        w = what_for(rho, N, seed, ni)
        maxdw = float(np.max(np.abs(w - ref)))
        print("  %7d | %.3e | %8.4f" % (ni, maxdw, c_curv(w, rho, N)))
    print()
