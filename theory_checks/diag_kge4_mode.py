# -*- coding: utf-8 -*-
"""
diag_kge4_mode.py -- extend the K=3 two-mode finding to K=4,5: is the large-rho worst
tangent mode still CURVATURE-like, and does the excess over the K=2 pairwise floor grow
with K?  Fills the §0.5a (II)/(III) gap "numerics only K=2/3, K>=4 is mechanism extrapolation".

For K in {2,3,4,5} at large rho (unpinned, rho > rho_c~0.32) it measures sigma_err and the
top tangent eigenvector, then characterizes that eigenvector by INDEX-REVERSAL PARITY:
    reversal R(v)_i = v_{K+1-i};  v_sym=(v+Rv)/2 (curvature-like), v_anti=(v-Rv)/2 (pairwise-like).
The extreme-pairwise mode (e1-eK) is purely antisymmetric; the curvature mode (1,-2,1,...) is
purely symmetric. sym_frac = ||v_sym||^2/||v||^2 of the top eigenvector tells which family dominates.
Also reports overlap with the explicit extreme-pair d=(e1-eK)/sqrt2 and the centered-quadratic
curvature mode q (the K-general (1,-2,1)).

excess = sigma_err / (C_K2/(sqrt(N)*rho)),  C_K2=0.50  (the K=2 measured pairwise floor).

Reuses sigma_err_scan internals (real dz-unmix-cg core). English prints. Run:
   python diag_kge4_mode.py
"""
import math
import numpy as np
import sigma_err_scan as S

KS = [2, 3, 4, 5]
CELLS = [(0.35, 350), (0.47, 350), (0.47, 105)]      # large-rho, unpinned
R, SIGMA, BW = 600, S.SIGMA_DEFAULT, S.BW_DEFAULT
C_K2 = 0.500


def parity_basis_vectors(K):
    """explicit physical modes: extreme-pair d (antisym) and centered-quadratic curvature q (sym)."""
    d = np.zeros(K); d[0], d[-1] = 1.0, -1.0; d /= np.linalg.norm(d)
    idx = np.arange(1, K + 1, dtype=float)
    q = (idx - (K + 1) / 2.0) ** 2                    # generalized (1,-2,1)
    q = q - q.mean()                                  # project to sum-zero tangent
    q /= np.linalg.norm(q)
    return d, q


def what_for(K, rho, N, seed):
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
    return core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)


print("K>=4 unpinned worst-mode test  (sym_frac: 1.0=pure curvature-like, 0.0=pure pairwise-like)")
print("%2s %5s %4s | %8s %6s | %7s %6s %6s | %6s | %s"
      % ("K", "rho", "n", "sig_err", "excess", "sym_frac", "|.q|", "|.d|", "l2/l1", "n_active(>1%l1)"))
for rho, N in CELLS:
    for K in KS:
        seed = 20260615 + K * 1000 + int(rho * 1e4)
        what = what_for(K, rho, N, seed)
        dev = what - what.mean(0, keepdims=True)
        cov = dev.T @ dev / what.shape[0]
        lam, V = np.linalg.eigh(cov)                  # ascending
        l1 = max(lam[-1], 0.0)
        sig = math.sqrt(l1)
        top = V[:, -1]
        rev = top[::-1]
        v_sym = 0.5 * (top + rev); v_anti = 0.5 * (top - rev)
        sym_frac = float(v_sym @ v_sym)               # ||v_sym||^2 (top is unit)
        d, q = parity_basis_vectors(K)
        oq, od = abs(top @ q), abs(top @ d)
        floor = C_K2 / (math.sqrt(N) * rho)
        nact = int((lam > 0.01 * l1).sum()) - 1 if l1 > 0 else 0   # tangent active modes (drop ~0 sum mode)
        l2l1 = (max(lam[-2], 0.0) / l1) if l1 > 0 else float("nan")
        sf = "n/a" if K == 2 else f"{sym_frac:.2f}"   # K=2 tangent 1D, no parity split
        print("%2d %5.2f %4d | %8.4f %6.2f | %7s %6.2f %6.2f | %6.2f | %d"
              % (K, rho, N, sig, sig / floor, sf, oq, od, l2l1, nact))
    print()
