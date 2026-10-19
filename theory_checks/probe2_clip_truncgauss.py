# -*- coding: utf-8 -*-
"""
probe2_clip_truncgauss.py -- TRUNCATED-GAUSSIAN model of the nonnegativity-clipping turnover.

Phase-1 established: the UNCONSTRAINED M-estimator sandwich Cov_unc = G^{-1} Sigma G^{-1}
predicts excess(K) growing MONOTONICALLY (no K~7 turnover). The real KL-NMF enforces w_hat>=0.
probe_turnover.py measured the clipping reduction factor meas/sand: 1.00 1.01 1.05 0.89 0.88 0.69
with the boundary-hitting frac JUMPING at K=5 (where w*_min ~ O(1/K) shrinks past the noise).

MODEL (this script):
  Unconstrained:  w_hat ~ N(w*, Cov_unc),  Cov_unc = G^{-1} Sigma G^{-1}  (tangent).
  The worst tangent direction v (degree-(K-1) contrast) has unconstrained variance
      sigma_unc^2 = v^T Cov_unc v = lambda_max = sigma_sand^2.
  Constraint w_hat_k >= 0 for the BINDING (smallest-w*) components truncates the joint
  Gaussian. Project the truncation onto v.

  Define for each component k the standardized distance to the boundary
      lambda_k = w*_k / s_k ,   s_k = sqrt( Cov_unc[k,k] )   (marginal std).
  A component "binds" when lambda_k = O(1): the smallest-w* component(s) dominate.

  ONE-SIDED-WALL variance reduction. A 1-D Gaussian N(mu, s^2) reflected/clipped at 0 has,
  conditional on the active set, a reduced variance. The exact reduction factor for the
  variance of a half-line-truncated Gaussian as a function of lambda = mu/s is the standard
  truncated-normal second-moment formula (truncation to [0, inf)):
      let a = -lambda;  Z = 1 - Phi(a) ;  z = phi(a)/Z   (inverse Mills)
      Var_trunc / s^2 = 1 + a*z - z^2        (Barr-Sherrill / truncated-normal variance)
  As lambda -> +inf this -> 1 (no clipping); as lambda -> -inf -> 0.

  But KL-NMF does NOT condition on staying positive -- it CLIPS (projects) the negative
  excursions to the boundary. The clipped variable y = max(0, x), x~N(mu,s^2), has
      E[y]   = mu*Phi(lambda) + s*phi(lambda)
      E[y^2] = (mu^2+s^2)*Phi(lambda) + mu*s*phi(lambda)
      Var[y] = E[y^2] - E[y]^2
  This rectified-Gaussian variance r(lambda) = Var[y]/s^2 is the relevant primitive
  (clip, not condition). r(+inf)=1, r(0)=1-1/pi ~ 0.682, r(-inf)=0.

  The worst direction v loads on the binding components with weights v_k. The worst-mode
  variance after clipping is reduced from sigma_unc^2 by the v-weighted average of the
  per-component rectification factors at the binding set:
      meas/sand (K,rho) ~  R(K,rho) = sum_k v_k^2 r(lambda_k) / sum_k v_k^2     (model A)
  We also test a "binding-component-only" variant (model B): the reduction is governed by
  the single smallest-w* component's loading on v:
      R_B = 1 - (v_kmin^2)*(1 - r(lambda_kmin))                                  (model B)

VERIFY: reproduce meas/sand 1.00->0.69 and the K=5 onset at rho=0.40; extend to rho=0.47.
Confirm the truncation PICTURE on real w_hat samples (skew, boundary mass) from the real core.

Run:   python probe2_clip_truncgauss.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.stats import norm

import sigma_err_scan as S

# ------- geometry (mirror compute_sandwich.py exactly) -------
SIGMA, START, BW = 25.0, 200.0, 12.0
DT = 1.0
TGRID = np.arange(50.0, 2600.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho):
    return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def gauss(t, mu, s):
    return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K)
    return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT


def smooth(f):
    return _KS @ f


def unconstrained_cov(K, rho, n=1.0):
    """Full unconstrained sandwich covariance Cov_unc = G^{-1} Sigma G^{-1} on the tangent,
    returned as the FULL KxK matrix (pinv on tangent, re-projected). Sigma scales as 1/n."""
    h, _ = endmembers(K, rho)
    w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    hq = h / q
    phi = np.array([smooth(hq[i]) for i in range(K)])
    Eq = lambda f: TRAPZ(f * q, TGRID)
    m = np.array([Eq(phi[i]) for i in range(K)])
    Sig = np.array([[Eq(phi[i] * phi[j]) - m[i] * m[j] for j in range(K)] for i in range(K)]) / n
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    St = P @ Sig @ P
    Gpi = np.linalg.pinv(Gt, rcond=1e-10)
    Cov = P @ (Gpi @ St @ Gpi) @ P
    lam, V = np.linalg.eigh(Cov)
    return Cov, lam, V


# ------- truncated/rectified-Gaussian primitives -------
def rect_factor(lam):
    """Var[max(0,X)]/s^2 for X~N(lam*s, s^2) standardized => function of lam=mu/s only.
    r(+inf)=1, r(0)=1-1/pi, r(-inf)=0."""
    Phi = norm.cdf(lam)
    phi = norm.pdf(lam)
    Ey = lam * Phi + phi               # E[y]/s
    Ey2 = (lam * lam + 1.0) * Phi + lam * phi  # E[y^2]/s^2
    return max(Ey2 - Ey * Ey, 0.0)


def trunc_cond_factor(lam):
    """Var of N truncated (conditioned) to [0,inf), /s^2, as function of lam=mu/s.
    a=-lam; z=phi(a)/(1-Phi(a)); factor = 1 + a z - z^2."""
    a = -lam
    Z = 1.0 - norm.cdf(a)
    if Z < 1e-300:
        return 1.0
    z = norm.pdf(a) / Z
    return max(1.0 + a * z - z * z, 0.0)


# ------- model R(K,rho) -------
def model_factor(K, rho, n, kind="rect", scale=1.0):
    """Predict meas/sand reduction.
    scale: empirical calibration of the effective per-rep n that sets s_k
           (the rep-to-rep sigma_err is what binds, not the per-grain Fisher 1/n)."""
    Cov, lam, V = unconstrained_cov(K, rho, n=n)
    v = V[:, -1]                       # worst tangent direction
    s = np.sqrt(np.clip(np.diag(Cov), 0.0, None)) * scale
    w = endpoint_w(K)
    lams = np.where(s > 0, w / np.maximum(s, 1e-300), np.inf)  # lambda_k = w*_k/s_k
    prim = trunc_cond_factor if kind == "trunc" else rect_factor
    r = np.array([prim(lk) for lk in lams])
    # model A: v^2-weighted average of per-component rectification
    wv = v * v
    RA = float((wv * r).sum() / wv.sum())
    # model B: single binding (smallest-w*) component
    kmin = int(np.argmin(w))
    RB = float(1.0 - wv[kmin] / wv.sum() * (1.0 - r[kmin]))
    return dict(RA=RA, RB=RB, lams=lams, r=r, v=v, s=s, kmin=kmin,
                lam_min_comp=float(lams[kmin]), w=w)


# ------- real-core w_hat samples (confirm the truncation picture) -------
def real_what(K, rho, N=350, R=800, seed=20260616):
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


# =====================================================================================
if __name__ == "__main__":
    # measured meas/sand targets (from probe_turnover.py; task table)
    MEAS_OVER_SAND = {0.40: {2: 1.00, 3: 1.01, 4: 1.05, 5: 0.89, 6: 0.88, 7: 0.69},
                      0.47: {}}  # to be predicted

    print("=" * 92)
    print("STEP 1: confirm the truncation picture on REAL w_hat samples (rho=0.40, endpoint w)")
    print("  -> at the smallest-w* component, expect: skew>0, boundary mass, mean pulled UP off w*")
    print("%2s | %8s %8s | %10s %10s %10s %10s"
          % ("K", "w*_min", "kmin", "mean_hat", "std_hat", "skew", "frac<.01"))
    for K in range(2, 8):
        what, w = real_what(K, rho=0.40)
        kmin = int(np.argmin(w))
        col = what[:, kmin]
        m, sd = col.mean(), col.std()
        sk = float(((col - m) ** 3).mean() / (sd ** 3 + 1e-300))
        print("%2d | %8.4f %8d | %10.4f %10.4f %+10.3f %10.3f"
              % (K, w[kmin], kmin, m, sd, sk, float((col < 0.01).mean())))

    print("\n" + "=" * 92)
    print("STEP 2: calibrate effective scale so lambda_k = w*_k/s_k reproduces the K=5 onset.")
    print("  The sandwich s_k is per-grain (n in Sigma). The rep-to-rep std that actually clips")
    print("  is sigma_err itself; calibrate one scalar 'scale' (and effective n) on rho=0.40.")
    # Use n so that the K=2 worst-mode sandwich std ~ the measured K=2 rep std.
    # Measured: at K=2 rho=0.40 the real rep std of w_hat (worst dir) ~ from real_what.
    what2, w2 = real_what(2, rho=0.40)
    dev = what2 - what2.mean(0, keepdims=True)
    cov2 = dev.T @ dev / what2.shape[0]
    se2_meas = math.sqrt(max(np.linalg.eigvalsh(cov2)[-1], 0.0))
    Cov2, lam2, V2 = unconstrained_cov(2, 0.40, n=1.0)
    se2_sand_n1 = math.sqrt(max(lam2[-1], 0.0))
    n_eff = (se2_sand_n1 / se2_meas) ** 2     # n that matches K=2 absolute std
    print("  measured K=2 sigma_err=%.4f ; sandwich(n=1)=%.4f => n_eff=%.1f"
          % (se2_meas, se2_sand_n1, n_eff))

    print("\n" + "=" * 92)
    print("STEP 3: model meas/sand(K) at rho=0.40 with rectified-Gaussian factor (n_eff)")
    print("%2s | %10s %8s %8s | %8s %8s | %8s"
          % ("K", "lam_min", "r_kmin", "RA", "RB", "MEAS", "RA-err"))
    for K in range(2, 8):
        mA = model_factor(K, 0.40, n=n_eff, kind="rect")
        meas = MEAS_OVER_SAND[0.40][K]
        print("%2d | %10.3f %8.3f %8.3f | %8.3f %8.3f | %+7.1f%%"
              % (K, mA["lam_min_comp"], mA["r"][mA["kmin"]], mA["RA"], mA["RB"],
                 meas, 100 * (mA["RA"] - meas) / meas))

    print("\n" + "=" * 92)
    print("STEP 4: predict meas/sand(K) at rho=0.47 (Phase-1: weak clipping expected -> ~1)")
    # recalibrate n_eff for rho=0.47 K=2 (same n_eff in principle; rho only changes geometry)
    print("%2s | %10s %8s %8s | %8s"
          % ("K", "lam_min", "r_kmin", "RA", "RB"))
    for K in range(2, 8):
        mA = model_factor(K, 0.47, n=n_eff, kind="rect")
        print("%2d | %10.3f %8.3f %8.3f | %8.3f"
              % (K, mA["lam_min_comp"], mA["r"][mA["kmin"]], mA["RA"], mA["RB"]))
