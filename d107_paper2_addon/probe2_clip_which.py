# -*- coding: utf-8 -*-
"""
probe2_clip_which.py -- WHICH components actually hit the w_hat>=0 boundary, and how the
worst-MODE (not per-component marginal) variance is truncated.

probe_turnover frac<.01 is over ALL K components pooled. probe2_clip_truncgauss STEP 1 showed
the smallest-w* (endpoint) component does NOT clip (neg skew, 0 boundary mass). So the clipping
that truncates sigma_err must live on OTHER components and/or in the worst-MODE projection, not
the smallest-w* marginal. This probe finds the truth before modeling.

For each K (rho=0.40, endpoint w), via the real core:
  - per-component: w*_k, mean_hat_k, std_hat_k, frac(w_hat_k<.01), skew_k
  - worst-mode projection a = (w_hat-mean) . v   (v=unconstrained worst tangent dir): its
    std, skew, and how it compares to the unconstrained sandwich worst-dir std
  - the standardized boundary distance lambda_k = w*_k / std_hat_k (EMPIRICAL std, model-free):
    identify the binding component(s) = smallest lambda_k.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_which.py
"""
import math
import numpy as np
from scipy.special import erfinv
import sigma_err_scan as S
from probe2_clip_truncgauss import unconstrained_cov, endpoint_w

SIGMA, BW = 25.0, 12.0


def real_what(K, rho, N=350, R=2000, seed=20260616):
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
    what = core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)
    return what, w


for K in range(2, 8):
    what, w = real_what(K, 0.40)
    Cov, lam, V = unconstrained_cov(K, 0.40, n=1.0)  # n=1; we use SHAPES (v) only here
    v = V[:, -1]
    mean = what.mean(0)
    sd = what.std(0)
    print(f"\n=== K={K}  rho=0.40  endpoint w ===  (R={what.shape[0]})")
    print("  k |   w*_k    mean_k    std_k   lam_k=w*/std  frac<.01   frac<.005   skew_k   v_k")
    for k in range(K):
        col = what[:, k]
        sk = float(((col - mean[k]) ** 3).mean() / (sd[k] ** 3 + 1e-300))
        lamk = w[k] / (sd[k] + 1e-300)
        print("  %d | %6.4f  %6.4f  %7.4f  %9.2f  %8.3f  %9.3f  %+7.2f  %+5.2f"
              % (k, w[k], mean[k], sd[k], lamk, float((col < .01).mean()),
                 float((col < .005).mean()), sk, v[k]))
    # worst-mode projection
    a = (what - mean) @ v
    print("  worst-mode v.(w_hat-mean):  std=%.4f  skew=%+.3f  (kurt=%.2f)"
          % (a.std(), float((a**3).mean() / (a.std()**3 + 1e-300)),
             float((a**4).mean() / (a.std()**4 + 1e-300))))
    # which component has the most boundary mass
    fracs = (what < 0.01).mean(0)
    kbind = int(np.argmax(fracs))
    print("  most-clipped component: k=%d (frac<.01=%.3f, w*=%.4f, v_k=%+.2f)  | argmin lam_k: k=%d"
          % (kbind, fracs[kbind], w[kbind], v[kbind], int(np.argmin(w / (sd + 1e-300)))))
