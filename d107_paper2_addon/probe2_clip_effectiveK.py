# -*- coding: utf-8 -*-
"""
probe2_clip_effectiveK.py -- EFFECTIVE-K / active-set model of the K~7 excess turnover.

Phase-1 established: the UNCONSTRAINED sandwich excess grows monotonically (peaks at K=8/12 for
rho=0.47/0.40); it does NOT turn over. The observed K~7 turnover is nonnegativity clipping (w_hat>=0).
probe_turnover.py measured (rho=0.40, n=350, endpoint w, R=800):
    K              2     3     4     5     6     7     8
    exc_meas    1.000 1.461 2.010 2.108 2.442 2.132 2.169   (real clipped NMF)
    exc_sand    1.000 1.440 1.910 2.370 2.770 3.090  -      (unconstrained)
    meas/sand   1.00  1.01  1.05  0.89  0.88  0.69   -      (clipping reduction)
    frac<.01    0.000 0.003 0.009 0.063 0.061 0.099 0.096   (boundary-hitting fraction)

EFFECTIVE-K MODEL (this script):
When the smallest-weight endmember w*_min~O(1/K) is driven to the boundary a fraction f(K,rho) of
the time, that realization's ACTIVE set drops to K-1 endmembers, whose worst-mode excess is the
SMALLER unconstrained excess(K-1). The measured ensemble sigma_err is then a TWO-STATE MIXTURE over
realizations:
    interior state  (prob 1-f): worst-mode variance = sigma_unc(K)^2     (full active set)
    clipped state   (prob  f) : worst-mode variance = sigma_unc(K-1)^2   (active set reduced by 1)
Variance is additive over the realization ensemble (sigma_err^2 = top eigenvalue of the realization
covariance), so the mixture is on VARIANCES:
    excess_clip(K)^2 = (1-f(K)) * exc_unc(K)^2  +  f(K) * exc_unc(K-1)^2.
Equivalently the reduction factor meas/sand:
    R(K) = excess_clip(K)/exc_unc(K) = sqrt( 1 - f(K)*(1 - (exc_unc(K-1)/exc_unc(K))^2) ).

This LINKS frac->reduction with NO free parameter once f is the measured frac<.01 and exc_unc is the
sandwich. We then predict K*(rho) = argmax_K excess_clip(K) and cross-check vs probe_turnover and vs
the alternative "sigma_err ~ w*_min" onset model from Phase 0.3.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_effectiveK.py
"""
import math
import numpy as np
from scipy.special import erfinv

# ---------------------------------------------------------------------------------------
# unconstrained sandwich excess (recomputed here from compute_sandwich geometry, so K* sweep
# is not capped at the hand-typed dict). Same math as compute_sandwich.sandwich(bw_on=True).
# ---------------------------------------------------------------------------------------
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
    """Unconstrained sandwich sigma_err (worst tangent eigenvalue), bw on. n=1."""
    h = endmembers(K, rho)
    w = endpoint_w(K)
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


def excess_unc_curve(rho, Kmax):
    se = {K: sandwich_se(K, rho) for K in range(2, Kmax + 1)}
    se2 = se[2]
    return {K: se[K] / se2 for K in se}


# ---------------------------------------------------------------------------------------
# measured target (probe_turnover.py rho=0.40, n=350, endpoint w, R=800)
# ---------------------------------------------------------------------------------------
MEAS_EXC  = {2: 1.000, 3: 1.461, 4: 2.010, 5: 2.108, 6: 2.442, 7: 2.132, 8: 2.169}
MEAS_FRAC = {2: 0.000, 3: 0.003, 4: 0.009, 5: 0.063, 6: 0.061, 7: 0.099, 8: 0.096}
SAND_DICT = {2: 1.00, 3: 1.44, 4: 1.91, 5: 2.37, 6: 2.77, 7: 3.09}     # printed sandwich (cap K=7)
WMIN_TRUE = {K: endpoint_w(K).min() for K in range(2, 13)}              # smallest true weight ~O(1/K)


def clip_reduction(f, exc_K, exc_Km1):
    """meas/sand reduction factor from the two-state effective-K mixture."""
    ratio2 = (exc_Km1 / exc_K) ** 2
    return math.sqrt(max(1.0 - f * (1.0 - ratio2), 0.0))


def excess_clip_curve(rho, frac_func, Kmax):
    """excess_clip(K) = exc_unc(K) * R(K); R from the mixture with f=frac_func(K)."""
    unc = excess_unc_curve(rho, Kmax)
    out = {2: unc[2]}
    for K in range(3, Kmax + 1):
        f = frac_func(K)
        out[K] = unc[K] * clip_reduction(f, unc[K], unc[K - 1])
    return unc, out


def main():
    rho = 0.40
    Kmax = 7      # measured frac available to K=8; sandwich dict to K=7. use recomputed unc to Kmax+1.
    print("=" * 90)
    print("CLIP-EFFECTIVE-K  rho=0.40  (two-state active-set mixture on VARIANCE)")
    print("  excess_clip(K)^2 = (1-f)*exc_unc(K)^2 + f*exc_unc(K-1)^2,  f = measured frac(w_hat<.01)")
    print("=" * 90)

    # recomputed unconstrained excess vs the printed sandwich dict (sanity)
    unc = excess_unc_curve(rho, 12)
    print("\n[sanity] recomputed unconstrained excess vs printed sandwich dict:")
    print("%2s | %10s %10s" % ("K", "exc_recomp", "exc_dict"))
    for K in range(2, 8):
        print("%2d | %10.3f %10.3f" % (K, unc[K], SAND_DICT[K]))

    # ---- LINK 1: frac -> reduction, parameter-free, against measured meas/sand ----
    print("\n[LINK frac->reduction]  R_pred = sqrt(1 - f*(1-(exc(K-1)/exc(K))^2)),  f = measured frac<.01")
    print("%2s | %8s | %10s %10s | %10s %10s | %8s"
          % ("K", "f_meas", "exc_un(K-1)", "exc_un(K)", "R_meas", "R_pred", "resid%"))
    resid_R = []
    for K in range(3, 8):
        f = MEAS_FRAC[K]
        R_pred = clip_reduction(f, unc[K], unc[K - 1])
        R_meas = MEAS_EXC[K] / SAND_DICT[K]
        rr = 100 * (R_pred - R_meas) / R_meas
        resid_R.append(abs(rr))
        print("%2d | %8.3f | %10.3f %10.3f | %10.3f %10.3f | %+7.1f"
              % (K, f, unc[K - 1], unc[K], R_meas, R_pred, rr))
    print("  mean |resid| of reduction factor: %.1f%%" % np.mean(resid_R))

    # ---- LINK 2: full excess_clip(K) vs measured exc, predict turnover K* ----
    # use measured frac for K<=8, extrapolate constant beyond (flat ~0.10 plateau seen at K=7,8)
    def frac_meas(K):
        if K in MEAS_FRAC: return MEAS_FRAC[K]
        return MEAS_FRAC[8]
    unc12, clip12 = excess_clip_curve(rho, frac_meas, 12)
    print("\n[FULL excess_clip vs measured]  predicted turnover from this model:")
    print("%2s | %10s %10s %10s | %8s"
          % ("K", "exc_unc", "exc_clip", "exc_meas", "clip-meas%"))
    resid_E = []
    for K in range(2, 9):
        em = MEAS_EXC.get(K, float('nan'))
        ec = clip12[K]
        rr = 100 * (ec - em) / em if em == em else float('nan')
        if em == em: resid_E.append(abs(rr))
        print("%2d | %10.3f %10.3f %10.3f | %+7.1f"
              % (K, unc12[K], ec, em, rr))
    Kstar_clip = max(range(2, 9), key=lambda K: clip12[K])
    Kstar_meas = max(MEAS_EXC, key=MEAS_EXC.get)
    print("  mean |resid| of excess_clip vs measured (K=2..8): %.1f%%" % np.mean(resid_E))
    print("  K*(clip-effK model) = %d   |   K*(measured) = %d" % (Kstar_clip, Kstar_meas))

    # ---- CROSS-CHECK with the alternative Phase-0.3 "sigma_err ~ w*_min" onset model ----
    # onset model: clipping kicks in when the worst-mode 1-sigma fluctuation of the smallest weight
    # exceeds its true value, i.e. when sigma_err_abs(K) >~ w*_min(K). sigma_err_abs ~ exc_unc(K)*sigma0.
    # Calibrate sigma0 from the measured frac jump (frac crosses ~5% between K=4 and K=5).
    print("\n[CROSS-CHECK 'sigma_err ~ w*_min' onset model]")
    print("  onset where exc_unc(K)/w*_min(K) crosses a threshold; w*_min~O(1/K), exc_unc grows.")
    print("%2s | %10s %10s %12s | %8s"
          % ("K", "exc_unc", "w*_min", "exc/w*_min", "frac_meas"))
    ratio = {}
    for K in range(2, 9):
        r = unc12[K] / WMIN_TRUE[K]
        ratio[K] = r
        print("%2d | %10.3f %10.4f %12.3f | %8.3f"
              % (K, unc12[K], WMIN_TRUE[K], r, frac_meas(K)))
    # the onset model says K* is set where ratio passes the level at which frac saturates.
    # find K where exc_unc/w*_min first exceeds its K=5 value (the frac-jump cell)
    thr = ratio[5]
    K_onset = next((K for K in range(2, 9) if ratio[K] >= thr), None)
    print("  onset threshold (calib at frac-jump K=5): exc/w*_min >= %.2f -> onset K=%d" % (thr, K_onset))

    # The two models agree on K* if effective-K turnover and onset both land at the same K.
    print("\n[AGREEMENT]")
    print("  clip-effectiveK turnover K* = %d" % Kstar_clip)
    print("  sigma~w*_min onset K        = %d  (clipping ACTIVE for K>=this)" % K_onset)
    print("  measured turnover K*        = %d" % Kstar_meas)


if __name__ == "__main__":
    main()
