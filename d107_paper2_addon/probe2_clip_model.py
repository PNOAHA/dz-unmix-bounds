# -*- coding: utf-8 -*-
"""
probe2_clip_model.py -- TRUNCATED-GAUSSIAN clipping model, CORRECTED picture (probe2_clip_which).

KEY EMPIRICAL FACTS (probe2_clip_which.py, real core, rho=0.40):
  * The clipping does NOT bind on the smallest-w* (endpoint) component; endpoints have small |v_k|
    and never hit 0. It binds on the INTERIOR components most aligned with the worst mode v
    (large |v_k|), specifically the ones the worst-mode fluctuation pushes DOWN (v_k>0 side).
  * The worst-mode coordinate a = v.(w_hat-w*) develops large POSITIVE SKEW exactly at K=5
    (skew: 0.04 0.06 -0.05 | +0.67 +0.15 +0.75 for K=2..7). The positive-skew onset == the
    meas/sand drop onset. Signature: the positive a-tail (which drives v_k>0 components to 0)
    is TRUNCATED by the nonnegativity wall; the negative a-tail is free.

MODEL (one-sided wall on the worst-mode coordinate):
  Unconstrained worst coordinate a ~ N(0, sigma_unc^2), sigma_unc = sigma_sand (worst dir).
  A positive fluctuation a>0 lowers component k by a*v_k for v_k>0. The most-exposed positive-v
  component k+ hits 0 when  a * v_{k+} >= w*_{k+}, i.e. a >= a_wall := min_{k:v_k>0} w*_k/v_k.
  Beyond a_wall the constraint activates and the realized worst-mode excursion is CLIPPED back
  (the MLE projects onto the active face), truncating a's positive tail at ~ a_wall.

  So the clipped worst coordinate is, to leading order, a one-sided-truncated/rectified Gaussian
  with standardized wall location
      Lambda(K,rho) = a_wall / sigma_unc = min_{k: v_k>0} (w*_k / v_k) / sigma_unc.
  The variance reduction factor (meas/sand)^... is the variance of a Gaussian whose POSITIVE
  tail beyond +Lambda*sigma is reflected/clipped to the wall. Using the rectified-from-above
  primitive (clip x at +c=Lambda*sigma):  y = min(x, c)+ ... we model the realized variance as
      Var_clip/sigma_unc^2 = g(Lambda),
  with g the variance of  y = x - max(0, x-c)*kappa  (kappa in [0,1] = how hard the face clips).
  For a hard one-sided wall (kappa=1, x reflected: y=min(x,c) on the upper side only, lower side free):
      one-sided-upper-clip variance, X~N(0,1), wall at +Lambda:
        let P=Phi(Lambda), p=phi(Lambda)
        E[y]   = -p + Lambda*(1-P)                       # lower half free + atom at Lambda above
        E[y^2] =  P_low_2 + Lambda^2*(1-P)
        where for the free part (x<Lambda): E[x^2 1{x<Lambda}] = P - Lambda*p
      => Var = (P - Lambda*p) + Lambda^2*(1-P) - E[y]^2
  meas/sand = sqrt(Var_clip)/sigma_unc relative to K=2 ... but excess already normalized; here
  meas/sand is the per-K reduction so meas/sand(K) = sqrt(g(Lambda_K)) / sqrt(g(Lambda_2)).

  Lambda_2 is huge (endpoints, no interior) -> g~1 -> meas/sand(2)=1 by construction.

VERIFY against probe_turnover meas/sand (0.40): 1.00 1.01 1.05 0.89 0.88 0.69 and K=5 onset.
Extend to rho=0.47.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_model.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.stats import norm
import sigma_err_scan as S
from probe2_clip_truncgauss import unconstrained_cov, endpoint_w

SIGMA, BW = 25.0, 12.0
MEAS_OVER_SAND = {0.40: {2: 1.00, 3: 1.01, 4: 1.05, 5: 0.89, 6: 0.88, 7: 0.69}}


def upper_clip_var(Lambda):
    """Var[y]/s^2 where y = min(x, Lambda) for x~N(0,1): upper tail beyond +Lambda is clipped
    to the wall (atom at Lambda), lower side free. Models the active-face projection on the
    positive (downward-pushing) worst-mode excursion."""
    P = norm.cdf(Lambda)      # P(x<Lambda)
    p = norm.pdf(Lambda)
    # free part x<Lambda: E[x 1]= -p ; E[x^2 1]= P - Lambda*p
    Ey = -p + Lambda * (1.0 - P)
    Ey2 = (P - Lambda * p) + (Lambda ** 2) * (1.0 - P)
    return max(Ey2 - Ey * Ey, 0.0)


def wall_lambda(K, rho, n):
    """Standardized wall location Lambda = a_wall/sigma_unc,
       a_wall = min over v_k>0 of w*_k/v_k. sigma_unc = sqrt(lambda_max)=worst-dir std."""
    Cov, lam, V = unconstrained_cov(K, rho, n=n)
    v = V[:, -1].copy()
    w = endpoint_w(K)
    sigma_unc = math.sqrt(max(lam[-1], 0.0))
    # choose sign of v so that the binding (smallest w*_k/|v_k|) side is the +v side:
    ratios_pos = [w[k] / v[k] for k in range(K) if v[k] > 1e-6]
    ratios_neg = [w[k] / (-v[k]) for k in range(K) if v[k] < -1e-6]
    a_pos = min(ratios_pos) if ratios_pos else np.inf
    a_neg = min(ratios_neg) if ratios_neg else np.inf
    a_wall = min(a_pos, a_neg)      # the side that binds first
    Lambda = a_wall / (sigma_unc + 1e-300)
    return Lambda, sigma_unc, v, w


def calibrate_neff(rho):
    """n_eff so the absolute K=2 worst-mode sandwich std matches the measured K=2 rep std."""
    core = S._load_real_core()
    mus, _ = S.peak_ages(2, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(2)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = S.fixed_interior_w(2, "endpoint")
    rng = np.random.default_rng(20260616)
    R = 2000
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(350, 350), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    what = core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)
    dev = what - what.mean(0, keepdims=True)
    se2_meas = math.sqrt(max(np.linalg.eigvalsh(dev.T @ dev / R)[-1], 0.0))
    _, lam, _ = unconstrained_cov(2, rho, n=1.0)
    se2_n1 = math.sqrt(max(lam[-1], 0.0))
    return (se2_n1 / se2_meas) ** 2, se2_meas


if __name__ == "__main__":
    for rho in (0.40, 0.47):
        n_eff, se2_meas = calibrate_neff(rho)
        print("=" * 86)
        print(f"rho={rho}: n_eff={n_eff:.1f} (calibrated K=2 abs std={se2_meas:.4f})")
        print("%2s | %9s %10s | %9s %9s | %8s %8s"
              % ("K", "Lambda", "sigma_unc", "g(Lam)", "ms_model", "MEAS", "err"))
        g2 = None
        for K in range(2, 8):
            Lambda, sigma_unc, v, w = wall_lambda(K, rho, n_eff)
            g = upper_clip_var(Lambda)
            if K == 2:
                g2 = g
            ms = math.sqrt(g / g2) if g2 else float('nan')
            meas = MEAS_OVER_SAND.get(rho, {}).get(K, float('nan'))
            err = (100 * (ms - meas) / meas) if meas == meas else float('nan')
            errs = ("%+7.1f%%" % err) if err == err else "    --"
            print("%2d | %9.3f %10.4f | %9.4f %9.3f | %8s %8s"
                  % (K, Lambda, sigma_unc, g, ms,
                     ("%.2f" % meas) if meas == meas else "  --", errs))
