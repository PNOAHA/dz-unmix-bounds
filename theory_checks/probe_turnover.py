# -*- coding: utf-8 -*-
"""
probe_turnover.py -- empirical target for the K~7 excess TURNOVER (Phase 4 / nonnegativity clipping).
Hypothesis: the unconstrained sandwich predicts excess grows monotonically, but the real KL-NMF has
w_hat >= 0; at high K + small rho the worst-mode fluctuation drives some w_hat_k against 0, the
clipping truncates the variance -> measured sigma_err saturates/turns over below the sandwich.

Measures, via the real dz-unmix-cg core (rho=0.40, n=350): the measured excess, the fraction of
w_hat entries near 0, and the min/quantiles of w_hat -- to show boundary-hitting onset coincides with
where measured excess (1.01/1.56/1.99/2.15/2.40/2.19 for K=2..7) peels below the unconstrained
sandwich (1.0/1.44/1.91/2.37/2.77/3.09).

Run:   python probe_turnover.py
"""
import math
import numpy as np
import sigma_err_scan as S

RHO, N, R, SIGMA, BW = 0.40, 350, 800, S.SIGMA_DEFAULT, S.BW_DEFAULT
C_K2 = 0.500
SANDWICH = {2: 1.00, 3: 1.44, 4: 1.91, 5: 2.37, 6: 2.77, 7: 3.09}   # unconstrained pred (compute_sandwich)


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
    return core.supervised_unmix(X, H, n_iter=2000, tol=1e-13), w


print(f"Turnover / clipping probe  rho={RHO} n={N} R={R}  (endpoint w)")
print("%2s | %8s %8s %8s | %8s %9s %9s %9s"
      % ("K", "exc_meas", "exc_sand", "meas/sand", "min_what", "frac<.01", "frac<.005", "wmin_true"))
se2 = None
for K in range(2, 9):
    what, w = what_for(K, RHO, N, 20260616 + K)
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    se = math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0))
    if K == 2: se2 = se
    exc = se / se2 if se2 else float('nan')
    sand = SANDWICH.get(K, float('nan'))
    frac01 = float((what < 0.01).mean())
    frac005 = float((what < 0.005).mean())
    print("%2d | %8.3f %8.3f %8.2f | %8.4f %9.3f %9.3f %9.4f"
          % (K, exc, sand, (exc / sand if sand == sand else float('nan')),
             what.min(), frac01, frac005, w.min()))
