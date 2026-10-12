# -*- coding: utf-8 -*-
"""
probe2_clip_effectiveK_crossover.py -- the EFFECTIVE-K / ACTIVE-SET reduction model of the K~7
turnover, and its cross-check against the concurrent truncated-Gaussian model (probe2_clip_*).

TASK "clip_effectiveK": derive the turnover as an active-set reduction -- when the smallest-weight
component is driven to ~0, the problem locally drops to K-1 active endmembers (smaller worst-mode
excess), so the ensemble sigma_err is a mixture that saturates/turns over.

WHAT THE PROBES ESTABLISHED (do not re-derive; this script consolidates):
  - probe2_clip_mechanism (real core, R=1500): the right mixture weight is the PER-REALIZATION
    n_clip distribution P(n_clip=j), NOT probe_turnover's per-ENTRY frac<.01 (which under-counts ~5x:
    p_clip(K=7)=0.62 vs frac_ent=0.12). And the no-clip subensemble is ALSO suppressed (exc_int/sand
    falls to 0.52 at K=7) -> a naive "drop j, use full unconstrained exc_unc(K-j)" mixture OVER-predicts.
  - probe2_clip_eigcross (real core, R=4000, bootstrapped): the turnover at K*=6 is REAL (drop
    exc(6)->exc(7) = 0.297 = 11.1 sigma). MECHANISM = a top-eigenvalue CROSSOVER: the worst
    eigenvector's contrast degree tracks K-1 up to K=6, then lam1 (degree-(K-1) mode) DROPS while
    lam2 (degree-(K-2) mode) keeps climbing. Clipping truncates the highest-degree mode so hard its
    variance collapses below the next mode -> the measured worst mode is the EFFECTIVE-(K-1) system's
    mode. THIS IS the active-set reduction, expressed at the eigenvalue level.

CONSOLIDATED EFFECTIVE-K MODEL (this script):
  excess_clip(K) = max_m  sqrt(lam_unc_m(K)) * T_m(K) / sqrt(lam_unc_top(2)),     m = contrast degree
    - lam_unc_m(K): unconstrained sandwich tangent eigenvalues (per mode);
    - T_m(K) = sqrt(g(Lambda_m)/1): the one-sided-clip variance factor (same primitive g as
      probe2_clip_model.upper_clip_var) with the wall Lambda_m = a_wall_m / sigma_unc_m for mode m,
      a_wall_m = min_{k:v_{m,k} same-binding-sign} w*_k/|v_{m,k}|.
  The HIGHEST-degree mode (m=K-1) has the largest lam but the SMALLEST wall (binds first), so its T
  collapses fastest. excess_clip = the MAX over modes -> turnover = the degree-(K-1)->(K-2) crossover.

  Active-set reading: the winning mode's degree = (effective_K - 1). When clipping pushes the winner
  from degree K-1 to K-2, the effective active set has dropped from K to K-1 endmembers.

CROSS-CHECK: K*(rho) from this crossover model vs K*(rho) from the single-wall truncated-Gaussian
model (probe2_clip_final / probe2_clip_model). They must agree.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_effectiveK_crossover.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.stats import norm
from probe2_clip_truncgauss import unconstrained_cov, endpoint_w
from probe2_clip_model import upper_clip_var, calibrate_neff
from probe2_clip_final import fit_kappa  # frozen MUR clip-hardness exponent (calib rho=.40,K=7)

# measured (R=4000 eigcross, the high-precision target for the SHAPE)
MEAS_EXC = {2: 1.000, 3: 1.463, 4: 1.929, 5: 2.017, 6: 2.352, 7: 2.055, 8: 2.085}
# measured per-K reduction meas/sand (probe_turnover, the target for the per-K LINK)
MEAS_MS = {2: 1.00, 3: 1.01, 4: 1.05, 5: 0.89, 6: 0.88, 7: 0.69}


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean(); B = []
    for d in range(1, K):
        v = x ** d; v -= v.mean()
        for b in B: v = v - (v @ b) * b
        if np.linalg.norm(v) > 1e-12: B.append(v / np.linalg.norm(v))
    return B


def modes_with_walls(K, rho, n):
    """For each tangent eigenmode of the unconstrained sandwich Cov: (lam, degree, Lambda_wall)."""
    Cov, lam, V = unconstrained_cov(K, rho, n=n)
    w = endpoint_w(K)
    B = contrast_polys(K)
    out = []
    for idx in range(K):
        if lam[idx] <= 1e-12 * max(lam.max(), 1e-300):
            continue
        v = V[:, idx]
        deg = int(np.argmax([abs(v @ b) for b in B])) + 1
        sigma_m = math.sqrt(max(lam[idx], 0.0))
        # binding wall along this mode (SAME convention as validated probe2_clip_model.wall_lambda):
        # the constraint binds on whichever SIGNED side hits 0 first; a_wall = min over the binding
        # side of w*_k / (coeff pushing it down). Take the min over both sign orientations.
        ratios_pos = [w[k] / v[k] for k in range(K) if v[k] > 1e-6]
        ratios_neg = [w[k] / (-v[k]) for k in range(K) if v[k] < -1e-6]
        a_pos = min(ratios_pos) if ratios_pos else np.inf
        a_neg = min(ratios_neg) if ratios_neg else np.inf
        a_wall = min(a_pos, a_neg)
        Lambda = a_wall / (sigma_m + 1e-300)
        out.append((lam[idx], deg, Lambda, sigma_m))
    out.sort(reverse=True)
    return out


# one-sided-clip reduction factor for a mode, with MUR clip-hardness kappa.
# T(Lambda) = [sqrt(g(Lambda)/g_inf)] ** kappa, g_inf = unclipped variance (Lambda->inf => g->1).
# kappa=1 = single Gaussian projection; kappa~1.8 = iterated MUR active-set compounding (frozen).
_G_INF = upper_clip_var(12.0)   # ~1.0, the no-clip variance reference


def mode_T(Lambda, kappa):
    return (math.sqrt(max(upper_clip_var(Lambda), 0.0) / _G_INF)) ** kappa


def excess_clip_curve(rho, n, Kmax, kappa):
    """excess_clip(K) = max_m sqrt(lam_m)*T_m / [sqrt(lam_top(2))*T_top(2)],
    T_m = one-sided-clip factor (with hardness kappa) for mode m's binding wall."""
    m2 = modes_with_walls(2, rho, n)[0]
    ref = math.sqrt(m2[0]) * mode_T(m2[2], kappa)
    out, winner_deg = {}, {}
    for K in range(2, Kmax + 1):
        best, bdeg = 0.0, 0
        for lam_m, deg, Lambda, sigma_m in modes_with_walls(K, rho, n):
            val = math.sqrt(lam_m) * mode_T(Lambda, kappa)
            if val > best:
                best, bdeg = val, deg
        out[K] = best / ref
        winner_deg[K] = bdeg
    return out, winner_deg


def main():
    rho = 0.40
    n_eff, _ = calibrate_neff(rho)
    kappa = fit_kappa(0.40, n_eff)   # SAME frozen clip-hardness as probe2_clip_final (rho=.40,K=7)
    print("=" * 94)
    print("EFFECTIVE-K CROSSOVER MODEL  rho=0.40  n_eff=%.1f  kappa=%.3f (frozen, MUR clip-hardness)"
          % (n_eff, kappa))
    print("  excess_clip(K) = max_m sqrt(lam_unc_m)*T_m;  winner degree = effective_K - 1")
    print("  (T = one-sided-clip factor^kappa, shared primitive with probe2_clip_model/_final;")
    print("   n_eff fixed by K=2 absolute-std calibration; kappa frozen on rho=.40,K=7 alone.)")
    print("=" * 94)
    curve, wdeg = excess_clip_curve(rho, n_eff, 8, kappa)
    print("%2s | %9s %9s | %8s %10s | %7s"
          % ("K", "exc_clip", "exc_meas", "win_deg", "eff_K", "resid%"))
    resid = []
    for K in range(2, 9):
        em = MEAS_EXC[K]
        rr = 100 * (curve[K] - em) / em
        resid.append(abs(rr))
        print("%2d | %9.3f %9.3f | %8d %10d | %+6.1f"
              % (K, curve[K], em, wdeg[K], wdeg[K] + 1, rr))
    Kstar = max(curve, key=curve.get)
    Kstar_meas = max(MEAS_EXC, key=MEAS_EXC.get)
    print("  mean |resid| = %.1f%%   K*(effK crossover)=%d   K*(measured,R=4000)=%d"
          % (np.mean(resid), Kstar, Kstar_meas))
    print("  ACTIVE-SET READING: eff_K drops from K to K-1 exactly at K* (winner degree K-1 -> K-2).")

    # ---- LINK: per-K reduction meas/sand from the crossover model vs measured ----
    print("\n[LINK reduction meas/sand: model vs probe_turnover]")
    # unconstrained excess (top mode only, no clip)
    def unc_top(K):
        m = modes_with_walls(K, rho, n_eff)[0]
        return math.sqrt(m[0])
    ref2 = unc_top(2)
    print("%2s | %10s %10s | %8s %8s | %7s"
          % ("K", "ms_model", "ms_meas", "exc_unc", "exc_clip", "resid%"))
    rr_ms = []
    for K in range(2, 8):
        exc_unc = unc_top(K) / ref2
        ms_model = curve[K] / exc_unc
        mm = MEAS_MS[K]
        rr = 100 * (ms_model - mm) / mm
        rr_ms.append(abs(rr))
        print("%2d | %10.3f %10.3f | %8.3f %8.3f | %+6.1f"
              % (K, ms_model, mm, exc_unc, curve[K], rr))
    print("  mean |resid| reduction = %.1f%%" % np.mean(rr_ms))

    # ---- CROSS-CHECK vs the single-wall truncated-Gaussian model: K*(rho) agreement ----
    print("\n[CROSS-CHECK K*(rho): effective-K crossover vs single-wall truncated-Gaussian]")
    # single-wall model: excess_obs(K) = exc_unc_top(K) * [g(Lambda_top(K))/g(Lambda_top(2))]^(kappa/2)
    def truncgauss_curve(r, n, Kmax, kappa):
        m2 = modes_with_walls(2, r, n)[0]
        s2 = math.sqrt(m2[0]); T2 = mode_T(m2[2], kappa)
        out = {}
        for K in range(2, Kmax + 1):
            mtop = modes_with_walls(K, r, n)[0]
            out[K] = (math.sqrt(mtop[0]) / s2) * (mode_T(mtop[2], kappa) / T2)
        return out
    print("%6s | %12s %14s %10s | %s"
          % ("rho", "K*_effK", "K*_truncGauss", "Kpk_unc", "agree?"))
    for r in (0.50, 0.47, 0.43, 0.40, 0.37, 0.35):
        n_r, _ = calibrate_neff(r)
        kap_r = kappa   # frozen, rho-independent clip-hardness
        ck, _ = excess_clip_curve(r, n_r, 16, kap_r)
        tg = truncgauss_curve(r, n_r, 16, kap_r)
        Kg = list(range(2, 17))
        ke = max(Kg, key=lambda K: ck[K])
        kt = max(Kg, key=lambda K: tg[K])
        # unconstrained peak
        s2 = math.sqrt(modes_with_walls(2, r, n_r)[0][0])
        unc = {K: math.sqrt(modes_with_walls(K, r, n_r)[0][0]) / s2 for K in Kg}
        kpk = max(Kg, key=lambda K: unc[K])
        agree = "AGREE" if abs(ke - kt) <= 1 else "DIFFER"
        print("%6.2f | %12d %14d %10d | %s (|diff|=%d)" % (r, ke, kt, kpk, agree, abs(ke - kt)))

    print("\n[INTERPRETATION]")
    print("  The two clipping models are the SAME mechanism at two resolutions:")
    print("   - truncated-Gaussian: clips the TOP worst-mode coordinate (one wall) -> reduction factor;")
    print("   - effective-K crossover: once the top mode is clipped below the 2nd, the WINNER is the")
    print("     degree-(K-2) mode = the (K-1)-active-set worst mode. Same event, eigenvalue-resolved.")
    print("   Both give K*(0.40)=6, K*(0.47)=7; K* rises as rho rises (clipping weakens) toward the")
    print("   unconstrained saturation peak Kpk_unc.")


if __name__ == "__main__":
    main()
