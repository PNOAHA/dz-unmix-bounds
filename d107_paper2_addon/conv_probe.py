# -*- coding: utf-8 -*-
"""
conv_probe.py -- convergence-artifact test for the K=3 large-rho curvature dominance.

Re-measures rho=0.47 n=350 and rho=0.35 n=350 (K=3, endpoint w) with n_iter in
{500, 2000, 8000} (and a few extra), using the REAL dz-unmix-cg core supervised_unmix.
Reports per (rho, n_iter):
  sqrt_l1, sqrt_l2 (tangent eigs), |top.u1|, |top.u2|, C_u1, C_u2, excess vs K=2 floor.

If C_curv (=C_u2) and |top.u2| stay put across n_iter -> structural prefactor (claim stands).
If they drift materially -> convergence artifact (claim refuted).
"""
import math
import numpy as np
import sigma_err_scan as S

K, R, SIGMA, BW = 3, 600, S.SIGMA_DEFAULT, S.BW_DEFAULT
C_FLOOR = 0.500

u1 = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)   # extreme pairwise
u2 = np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0)   # curvature

CELLS = [(0.47, 350), (0.35, 350)]
NITERS = [200, 500, 2000, 8000, 20000]


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
    # IMPORTANT: tol=0.0 disables the every-25-iter early stop so n_iter is the true budget
    what = core.supervised_unmix(X, H, n_iter=n_iter, tol=0.0)
    return what, w, X, H


def diag(what, rho, N):
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)
    l1, l2 = lam[-1], lam[-2]
    top = V[:, -1]
    o1, o2 = abs(top @ u1), abs(top @ u2)
    var_u1 = float(u1 @ cov @ u1); var_u2 = float(u2 @ cov @ u2)
    C_u1 = math.sqrt(max(var_u1, 0)) * math.sqrt(N) * rho
    C_u2 = math.sqrt(max(var_u2, 0)) * math.sqrt(N) * rho
    floor = C_FLOOR / (math.sqrt(N) * rho)
    excess = math.sqrt(max(l1, 0)) / floor
    return dict(sqrt_l1=math.sqrt(max(l1, 0)), sqrt_l2=math.sqrt(max(l2, 0)),
                o1=o1, o2=o2, C_u1=C_u1, C_u2=C_u2, excess=excess)


print("CONVERGENCE PROBE  K=3 endpoint  R=%d  (tol=0.0 -> full n_iter budget)" % R)
print("%5s %5s %7s | %8s %8s | %5s %5s | %7s %7s | %6s | %s"
      % ("rho", "n", "n_iter", "sqrt_l1", "sqrt_l2", "|.u1|", "|.u2|",
         "C_u1", "C_u2", "excess", "KL@last"))
for rho, N in CELLS:
    seed = 20260611 + K * 1000 + int(rho * 1e4)
    for ni in NITERS:
        what, w, X, H = what_for(rho, N, seed, ni)
        d = diag(what, rho, N)
        # report a convergence indicator: mean residual KL over reps at the returned W
        # reconstruct W (unnormalized not available post-normalize); use normalized W as proxy density
        eps = 1e-10
        WH = what @ H + eps
        kl = float(np.mean(np.sum(X * np.log((X + eps) / WH) - X + WH, axis=1)))
        print("%5.2f %5d %7d | %8.5f %8.5f | %5.2f %5.2f | %7.3f %7.3f | %5.2fx | %.3e"
              % (rho, N, ni, d["sqrt_l1"], d["sqrt_l2"], d["o1"], d["o2"],
                 d["C_u1"], d["C_u2"], d["excess"], kl))
    print("-" * 100)
