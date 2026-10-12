# -*- coding: utf-8 -*-
"""
probe2_clip_mc.py -- DIRECT Monte-Carlo validation of the truncated-Gaussian clipping picture.

The cleanest test of "nonnegativity clipping truncates the worst-mode variance": draw the
UNCONSTRAINED estimator w_hat ~ N(w*, Cov_unc) [Cov_unc=G^{-1}SigmaG^{-1}, the SANDWICH], then
apply the SAME constraint the KL-NMF enforces -- project onto the simplex tangent face with
w>=0 -- and measure how the worst-tangent-direction variance shrinks. If meas/sand from this
GAUSSIAN+CLIP pipeline reproduces probe_turnover's meas/sand (1.00..0.69, K=5 onset), the
turnover IS the clipping of the Gaussian sandwich (no other mechanism needed).

Two clip operators (both KKT-consistent reductions of the unconstrained step):
  (a) simplex_clip: w_c = max(w_hat, 0) then renormalize sum=1 (the actual MUR fixed-point
      lives on this set; this is the leading-order projection).
  (b) face_project: Euclidean projection of w_hat onto {w>=0, sum=1} (exact QP face proj).
We report both; (a) mirrors the multiplicative-update geometry more closely.

We then compare the worst-tangent-dir std of the clipped cloud to the unconstrained one
(=meas/sand per K) and to the real-core probe_turnover targets.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_mc.py
"""
import math
import numpy as np
from numpy.random import default_rng
from probe2_clip_truncgauss import unconstrained_cov, endpoint_w
from probe2_clip_model import calibrate_neff

MEAS = {0.40: {2: 1.00, 3: 1.01, 4: 1.05, 5: 0.89, 6: 0.88, 7: 0.69},
        0.47: {2: 1.04 / 1.04, 3: 1.43 / 1.43, 4: 1.62 / 1.62}}  # rho=0.47 meas~sand (weak clip)


def worst_std(cloud, v):
    a = (cloud - cloud.mean(0)) @ v
    return a.std()


def project_simplex(Y):
    """Euclidean projection of each row of Y onto {w>=0, sum w =1} (Wang & Carreira-Perpinan)."""
    n, K = Y.shape
    U = np.sort(Y, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, K + 1)
    cond = U - css / ind > 0
    rho = cond.sum(1)
    theta = css[np.arange(n), rho - 1] / rho
    return np.maximum(Y - theta[:, None], 0.0)


def run(rho, R=40000, seed=7):
    n_eff, se2_meas = calibrate_neff(rho)
    rng = default_rng(seed)
    print("=" * 92)
    print(f"rho={rho}  n_eff={n_eff:.1f}  R={R}   (Gaussian sandwich draw + nonneg clip)")
    print("%2s | %8s | %9s %9s | %9s %9s | %8s"
          % ("K", "sig_unc", "ms_clipRN", "ms_proj", "skew_clip", "skew_proj", "MEAS"))
    base = {}
    rows = []
    for K in range(2, 8):
        Cov, lam, V = unconstrained_cov(K, rho, n=n_eff)
        v = V[:, -1]
        w = endpoint_w(K)
        # symmetric sqrt for sampling on the tangent (Cov is rank K-1, PSD)
        evals, evecs = np.linalg.eigh(Cov)
        evals = np.clip(evals, 0.0, None)
        L = evecs @ np.diag(np.sqrt(evals))
        draws = w[None, :] + (rng.standard_normal((R, K)) @ L.T)
        sig_unc = worst_std(draws, v)
        # (a) clip + renormalize
        cl = np.maximum(draws, 0.0)
        cl = cl / cl.sum(1, keepdims=True)
        s_cl = worst_std(cl, v)
        a_cl = (cl - cl.mean(0)) @ v
        sk_cl = float((a_cl ** 3).mean() / (a_cl.std() ** 3 + 1e-300))
        # (b) exact simplex projection
        pr = project_simplex(draws)
        s_pr = worst_std(pr, v)
        a_pr = (pr - pr.mean(0)) @ v
        sk_pr = float((a_pr ** 3).mean() / (a_pr.std() ** 3 + 1e-300))
        rows.append((K, sig_unc, s_cl, s_pr, sk_cl, sk_pr))
        if K == 2:
            base = dict(unc=sig_unc, cl=s_cl, pr=s_pr)
    # meas/sand = (clipped std / unc std)_K / (clipped std / unc std)_2  -> reduction relative to K=2
    print("-- meas/sand = (s_clip/s_unc)_K normalized so K=2 -> matches probe_turnover def --")
    r2_cl = base["cl"] / base["unc"]
    r2_pr = base["pr"] / base["unc"]
    for (K, sig_unc, s_cl, s_pr, sk_cl, sk_pr) in rows:
        ms_cl = (s_cl / sig_unc) / r2_cl
        ms_pr = (s_pr / sig_unc) / r2_pr
        meas = MEAS.get(rho, {}).get(K, float('nan'))
        print("%2d | %8.4f | %9.3f %9.3f | %+9.3f %+9.3f | %8s"
              % (K, sig_unc, ms_cl, ms_pr, sk_cl, sk_pr,
                 ("%.2f" % meas) if meas == meas else "--"))


if __name__ == "__main__":
    run(0.40)
    run(0.47)
