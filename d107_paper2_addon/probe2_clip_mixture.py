# -*- coding: utf-8 -*-
"""
probe2_clip_mixture.py -- the CORRECTED effective-K mixture turnover model.

probe2_clip_mechanism found the naive single-drop model fails for 2 reasons:
  (1) the right mixture weight is the PER-REALIZATION n_clip distribution P(n_clip=j), not the
      per-entry frac<.01 (which under-counts ~5x at K=7);
  (2) when j endmembers clip, the active set drops to K-j, whose worst-mode excess is the SMALLER
      unconstrained exc_unc(K-j).

MIXTURE MODEL (multi-state active-set reduction):
    excess_clip(K)^2  =  sum_{j=0}^{K-2}  P(n_clip=j) * exc_unc(K-j)^2 .
This generalises probe2_clip_effectiveK's two-state form. Tested two ways:
  (A) MEASURED P(n_clip=j) from the real core  -> tests whether the active-set reduction picture is
      mechanistically correct (does exc_clip reproduce exc_meas given the true clip distribution?);
  (B) a PREDICTED P(n_clip=j) from a one-parameter onset law, to get a closed-form K*(rho).

We then predict K*(rho), and cross-check against the Phase-0.3 "sigma_err ~ w*_min" onset model.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_mixture.py
"""
import math
import numpy as np
from scipy.special import erfinv
import sigma_err_scan as S

# ---- unconstrained sandwich excess (recomputed, same as probe2_clip_effectiveK) ----
SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 1.0
TGRID = np.arange(50.0, 3200.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]
def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()
_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f
def sandwich_se(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    hq = h / q
    phi = np.array([smooth(hq[i]) for i in range(K)])
    Eq = lambda f: TRAPZ(f * q, TGRID)
    m = np.array([Eq(phi[i]) for i in range(K)])
    Sig = np.array([[Eq(phi[i] * phi[j]) - m[i] * m[j] for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt, St = P @ G @ P, P @ Sig @ P
    Gpi = np.linalg.pinv(Gt, rcond=1e-10)
    Cov = P @ (Gpi @ St @ Gpi) @ P
    return math.sqrt(max(np.linalg.eigvalsh(Cov)[-1], 0.0))
def excess_unc(rho, Kmax):
    se = {K: sandwich_se(K, rho) for K in range(2, Kmax + 1)}
    return {K: se[K] / se[2] for K in se}

# ---- measured real-core clip distribution + measured excess ----
RHO, N, R = 0.40, 350, 1500
def measure_core(K, seed, tau=0.02):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, RHO, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    what = core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)
    dev = what - what.mean(0, keepdims=True)
    se = math.sqrt(max(np.linalg.eigvalsh(dev.T @ dev / R)[-1], 0.0))
    nclip = (what < tau).sum(1)
    dist = np.bincount(nclip, minlength=K).astype(float) / R
    return se, dist


def mixture_excess(K, unc, dist):
    """excess_clip(K)^2 = sum_j P(n_clip=j) exc_unc(K-j)^2.  K-j floored at 2."""
    tot = 0.0
    for j in range(min(len(dist), K - 1)):
        Keff = max(K - j, 2)
        tot += dist[j] * unc[Keff] ** 2
    # tail mass (j beyond range) -> Keff=2
    used = dist[:min(len(dist), K - 1)].sum()
    tot += (1 - used) * unc[2] ** 2
    return math.sqrt(tot)


def main():
    unc = excess_unc(RHO, 12)
    print("=" * 92)
    print("CORRECTED EFFECTIVE-K MIXTURE  rho=0.40   excess_clip^2 = sum_j P(n_clip=j) exc_unc(K-j)^2")
    print("=" * 92)

    # measure the real-core clip distribution + excess
    se2 = None
    meas, dists = {}, {}
    for K in range(2, 9):
        se, dist = measure_core(K, 20260616 + K)
        if K == 2: se2 = se
        meas[K] = se / se2
        dists[K] = dist

    # (A) mixture with MEASURED clip distribution
    print("\n(A) MIXTURE with MEASURED P(n_clip=j):")
    print("%2s | %10s %10s %10s | %8s"
          % ("K", "exc_unc", "exc_mix", "exc_meas", "mix-meas%"))
    resid = []
    for K in range(2, 9):
        em = meas[K]
        ex = mixture_excess(K, unc, dists[K])
        rr = 100 * (ex - em) / em
        resid.append(abs(rr))
        print("%2d | %10.3f %10.3f %10.3f | %+7.1f" % (K, unc[K], ex, em, rr))
    Kstar_mix = max(range(2, 9), key=lambda K: mixture_excess(K, unc, dists[K]))
    Kstar_meas = max(meas, key=meas.get)
    print("  mean |resid| (A): %.1f%%   K*(mix,measured-dist)=%d  K*(measured)=%d"
          % (np.mean(resid), Kstar_mix, Kstar_meas))

    # (B) PREDICTED clip distribution from a one-parameter onset law -> closed-form K*.
    # mean clipped count mu(K) ~ expected # of weights whose 1-sigma worst-mode fluctuation exceeds
    # their true value. Model n_clip ~ Poisson(mu(K)) with mu(K) = (K-1)*Phi(-w*_min / s(K)),
    # s(K)=kappa*sigma_unc(K) the per-weight sd ~ exc_unc(K)*s0. Calibrate (kappa*s0) to measured mu.
    from math import erf
    def Phi(z): return 0.5 * (1 + erf(z / math.sqrt(2)))
    wmin = {K: endpoint_w(K).min() for K in range(2, 13)}
    mu_meas = {K: float(np.dot(np.arange(len(dists[K])), dists[K])) for K in range(2, 9)}
    # fit s0 by matching mu_meas at K=6 (mid turnover): mu = (K-1)*Phi(-wmin/(exc_unc*s0))
    def mu_pred(K, s0):
        return (K - 1) * Phi(-wmin[K] / (unc[K] * s0))
    # solve s0 from K=6
    from scipy.optimize import brentq
    target = mu_meas[6]
    f = lambda s0: mu_pred(6, s0) - target
    s0 = brentq(f, 1e-4, 5.0)
    print("\n(B) PREDICTED clip law  n_clip~Poisson(mu),  mu(K)=(K-1)*Phi(-w*_min/(exc_unc(K)*s0))")
    print("    calibrated s0=%.4f from mu_meas(K=6)=%.3f" % (s0, target))
    print("%2s | %10s %10s | %10s %10s | %8s"
          % ("K", "mu_meas", "mu_pred", "exc_mixB", "exc_meas", "B-meas%"))

    def poisson_dist(mu, K):
        # truncated Poisson up to K-1 (cannot clip more than K-1)
        from math import exp, factorial
        p = np.array([exp(-mu) * mu ** j / factorial(j) for j in range(K)])
        p[K - 1] += 1 - p.sum()      # dump tail into last bin (rare)
        return np.maximum(p, 0)
    residB = []
    for K in range(2, 11):
        mu = mu_pred(K, s0)
        pd = poisson_dist(mu, K)
        ex = mixture_excess(K, unc, pd)
        em = meas.get(K, float('nan'))
        mm = mu_meas.get(K, float('nan'))
        rr = 100 * (ex - em) / em if em == em else float('nan')
        if em == em and K >= 2: residB.append(abs(rr))
        print("%2d | %10.3f %10s | %10.3f %10s | %s"
              % (K, mu, ("%.3f" % mm if mm == mm else "  -"), ex,
                 ("%.3f" % em if em == em else "  -"),
                 ("%+7.1f" % rr if rr == rr else "   -")))
    Kstar_B = max(range(2, 11), key=lambda K: mixture_excess(K, unc, poisson_dist(mu_pred(K, s0), K)))
    print("  mean |resid| (B, K=2..8): %.1f%%   K*(mix,predicted-law)=%d" % (np.mean(residB), Kstar_B))

    # ---- closed-form K*: turnover where d/dK[exc_clip]=0; equivalently where the active-set
    # reduction rate overtakes the unconstrained growth rate. Report the predicted K* vs rho.
    print("\n[K*(rho) PREDICTION from law (B), cross-rho]:")
    print("%6s | %8s %8s %8s | %s" % ("rho", "K*_mixB", "Kpk_unc", "onset", "note"))
    for rho in (0.47, 0.40, 0.35, 0.30):
        u = excess_unc(rho, 14)
        wm = {K: endpoint_w(K).min() for K in range(2, 15)}
        # recalibrate s0 per rho by anchoring mu(K=6) to the SAME measured value scaling
        # (use rho=0.40 s0 as the rho-invariant per-weight noise unit; mu shifts via exc_unc & wmin)
        def muP(K): return (K - 1) * Phi(-wm[K] / (u[K] * s0))
        def pd(K):
            from math import exp, factorial
            p = np.array([exp(-muP(K)) * muP(K) ** j / factorial(j) for j in range(K)])
            p[K - 1] += 1 - p.sum(); return np.maximum(p, 0)
        Kgrid = range(2, 15)
        excB = {K: mixture_excess(K, u, pd(K)) for K in Kgrid}
        Kstar = max(Kgrid, key=lambda K: excB[K])
        Kpk_unc = max(Kgrid, key=lambda K: u[K])
        # onset: first K with muP>=0.5 (expected >=0.5 clipped components)
        onset = next((K for K in Kgrid if muP(K) >= 0.5), None)
        print("%6.2f | %8d %8d %8s | unc peaks later; clipping pulls K* down"
              % (rho, Kstar, Kpk_unc, str(onset)))


if __name__ == "__main__":
    main()
