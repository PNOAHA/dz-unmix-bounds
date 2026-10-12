# -*- coding: utf-8 -*-
"""
diag_kge4_poly.py -- nail the K>=4 unpinned worst-mode shape: is it the HIGHEST-DEGREE
discrete contrast polynomial (degree K-1, parity (-1)^{K-1})?  Decisive test = project the
top tangent eigenvector onto an orthonormal contrast-polynomial basis P_1..P_{K-1}
(Gram-Schmidt of monomials i^1..i^{K-1} on the sum-zero tangent) and report which degree
dominates.  Plus seed stability and an n_iter convergence guard (the rho=0.35 K=4 cell
sits near rho_c and may be under-converged).

excess = sigma_err / (0.50/(sqrt(N)*rho)).  Reuses sigma_err_scan real core. English prints.
Run:  & "C:/Users/Administrator/miniconda3/python.exe" diag_kge4_poly.py
"""
import math
import numpy as np
import sigma_err_scan as S

R, SIGMA, BW, C_K2 = 600, S.SIGMA_DEFAULT, S.BW_DEFAULT, 0.500


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean()
    B = []
    for deg in range(1, K):
        v = x ** deg; v = v - v.mean()
        for b in B:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            B.append(v / nv)
    return B                                          # B[d-1] = orthonormal degree-d contrast


def measure(K, rho, N, seed, n_iter=2000):
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
    what = core.supervised_unmix(X, H, n_iter=n_iter, tol=1e-13)
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)
    top = V[:, -1]
    B = contrast_polys(K)
    ov = [abs(float(top @ b)) for b in B]             # overlap per degree 1..K-1
    sig = math.sqrt(max(lam[-1], 0.0))
    dom_deg = int(np.argmax(ov)) + 1
    return sig, sig / (C_K2 / (math.sqrt(N) * rho)), dom_deg, ov


print("Worst-mode degree test: dominant contrast degree should = K-1 (highest, most oscillatory)")
print("%2s %5s %4s %5s %5s | %8s %6s | %7s %5s | overlaps deg=1..K-1"
      % ("K", "rho", "n", "seed", "nit", "sig_err", "excess", "dom_deg", "=K-1?"))
for K in [2, 3, 4, 5]:
    for seed in [101, 202]:
        rho, N = 0.47, 350
        sig, exc, dom, ov = measure(K, rho, N, 20260615 + K * 1000 + seed)
        print("%2d %5.2f %4d %5d %5d | %8.4f %6.2f | deg=%d %5s | %s"
              % (K, rho, N, seed, 2000, sig, exc, dom, "YES" if dom == K - 1 else "no",
                 " ".join(f"{o:.2f}" for o in ov)))
    print()

print("convergence guard (rho near rho_c / high K): n_iter 2000 vs 6000")
for K, rho, N in [(5, 0.47, 350), (4, 0.35, 350), (5, 0.35, 350)]:
    for nit in [2000, 6000]:
        sig, exc, dom, ov = measure(K, rho, N, 20260615 + K * 1000 + 101, n_iter=nit)
        print("  K=%d rho=%.2f n=%d nit=%5d | sig_err=%.4f excess=%.2f dom_deg=%d(K-1=%d)"
              % (K, rho, N, nit, sig, exc, dom, K - 1))
