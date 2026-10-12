# -*- coding: utf-8 -*-
"""
diag_k3_mode.py -- identify the WORST tangent mode of K=3 supervised-NMF, to settle
why measured sigma_err sits ~1.35x above the K=2 pairwise floor at large rho.

Reuses sigma_err_scan internals (real dz-unmix-cg core). For selected (rho,n) cells it
re-measures the (R,K) weight estimates `what`, forms Cov(what), and reports:
  - the two nonzero tangent eigenvalues (simplex tangent is 2D for K=3),
  - the top eigenvector's overlap with the two orthonormal tangent modes
        u1 = (1,0,-1)/sqrt2   (extreme PAIRWISE, antisymmetric)
        u2 = (1,-2,1)/sqrt6   (CURVATURE, symmetric)
  - variance along u1 vs u2, and the implied prefactor C_mode = sqrt(var)*sqrt(N)*rho.

Hypotheses being discriminated (for the >floor excess at large rho, ~n-independent):
  (P) worst mode = pairwise extreme  -> would be BELOW floor (rho_1K>rho). reject if above.
  (C) worst mode = curvature         -> KDE-lifted Fisher ~rho^2, larger var, ABOVE floor.
  (M) mix / middle-endmember squeezed -> top eigvec ~ rotated, prefactor inflated.

English prints (PS GBK). Run:
  & "C:/Users/Administrator/miniconda3/python.exe" diag_k3_mode.py
"""
import math
import numpy as np
import sigma_err_scan as S

CELLS = [(0.47, 70), (0.47, 350), (0.35, 350), (0.19, 160), (0.10, 350)]
K, R, SIGMA, BW = 3, 600, S.SIGMA_DEFAULT, S.BW_DEFAULT
C_FLOOR = 0.500                                            # K=2 measured pairwise constant

u1 = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)          # extreme pairwise
u2 = np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0)          # curvature


def what_for(rho, N, seed):
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
    return core.supervised_unmix(X, H, n_iter=2000, tol=1e-13), w


print("K=3 worst-mode identification  (u1=pairwise (1,0,-1), u2=curvature (1,-2,1))")
print("%5s %4s | %8s %8s | %5s %5s | %7s %7s | %s"
      % ("rho", "n", "sqrt_l1", "sqrt_l2", "|.u1|", "|.u2|", "C_u1", "C_u2", "floor/excess"))
for rho, N in CELLS:
    seed = 20260611 + K * 1000 + int(rho * 1e4)
    what, w = what_for(rho, N, seed)
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)                           # ascending
    l1, l2 = lam[-1], lam[-2]                              # two largest (tangent 2D)
    top = V[:, -1]
    o1, o2 = abs(top @ u1), abs(top @ u2)                  # overlaps of top eigvec
    var_u1 = float(u1 @ cov @ u1); var_u2 = float(u2 @ cov @ u2)
    C_u1 = math.sqrt(var_u1) * math.sqrt(N) * rho
    C_u2 = math.sqrt(var_u2) * math.sqrt(N) * rho
    floor = C_FLOOR / (math.sqrt(N) * rho)
    print("%5.2f %4d | %8.4f %8.4f | %5.2f %5.2f | %7.3f %7.3f | %.4f / %.2fx"
          % (rho, N, math.sqrt(max(l1, 0)), math.sqrt(max(l2, 0)), o1, o2,
             C_u1, C_u2, floor, math.sqrt(max(l1, 0)) / floor))
