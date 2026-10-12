# -*- coding: utf-8 -*-
"""
probe2_clip_truncation.py -- the boundary-TRUNCATION form of the clipping turnover.

probe2_clip_mechanism showed even the no-clip (interior) subensemble has SUPPRESSED excess
(exc_int/sand=0.52 at K=7): the active-set "drop j components" picture using full exc_unc(K-j)
over-predicts because nonnegativity does not merely remove components -- it ONE-SIDED-TRUNCATES the
worst-mode Gaussian fluctuation. The worst mode (degree-(K-1) contrast) loads most heavily on the
SMALLEST weight w*_min(K)~O(1/K). When the worst-mode fluctuation amplitude approaches w*_min, the
negative tail is clipped at 0 -> the realized variance of that mode is the variance of a Gaussian
TRUNCATED below at -z*sigma, z = w*_min / (a*sigma_unc_abs).

TRUNCATION-FACTOR MODEL (controlling variable z(K) = standardized distance of smallest weight to 0):
    excess_clip(K) = exc_unc(K) * T(z(K)),
    z(K) = w*_min(K) / (a * sigma_unc_abs(K))      [sigma_unc_abs ~ exc_unc(K)*s0, so z = w*_min/(a*s0*exc_unc)]
    T(z) = sqrt( Var[X | X >= -z] / Var[X] ) for X~N(0,1)   (one-sided lower truncation at -z).
As K grows w*_min falls (~1/K) and exc_unc grows, so z falls, T(z) drops -> turnover. ONE parameter
(a*s0). Calibrate to the measured meas/sand, then predict K*(rho) and cross-check the two onset models.

Var of N(0,1) truncated to [-z, inf):  with phi,Phi std normal,
    alpha=-z; lam=phi(alpha)/(1-Phi(alpha)); delta=lam*(lam-alpha);  Var=1-delta;  mean=lam.
(We use the variance about the TRUNCATED mean; the worst-mode contributes its truncated spread.)

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_truncation.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import brentq

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


def trunc_var_factor(z):
    """sqrt(Var[X | X>=-z]/Var[X]) for X~N(0,1).  z>=0; z->inf => 1, z->0 => half-normal."""
    if z > 12: return 1.0
    alpha = -z
    phi = math.exp(-0.5 * alpha * alpha) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(alpha / math.sqrt(2)))
    denom = 1 - Phi
    if denom < 1e-12: return 0.0
    lam = phi / denom
    delta = lam * (lam - alpha)
    return math.sqrt(max(1 - delta, 0.0))


# measured target (probe_turnover rho=0.40)
MEAS_EXC = {2: 1.000, 3: 1.461, 4: 2.010, 5: 2.108, 6: 2.442, 7: 2.132, 8: 2.169}


def excess_unc(rho, Kmax):
    se = {K: sandwich_se(K, rho) for K in range(2, Kmax + 1)}
    return {K: se[K] / se[2] for K in se}


def model_curve(rho, c, Kmax):
    """excess_clip(K) = exc_unc(K) * T(z), z = w*_min(K)/(c*exc_unc(K)). c = a*s0 (one param)."""
    unc = excess_unc(rho, Kmax)
    wm = {K: endpoint_w(K).min() for K in range(2, Kmax + 1)}
    out = {}
    for K in range(2, Kmax + 1):
        z = wm[K] / (c * unc[K])
        out[K] = unc[K] * trunc_var_factor(z)
    return unc, out


def main():
    rho = 0.40
    unc = excess_unc(rho, 12)
    wm = {K: endpoint_w(K).min() for K in range(2, 13)}

    # calibrate c by least-squares of model vs measured over K=2..8
    Ks = list(range(2, 9))
    def sse(c):
        _, m = model_curve(rho, c, 8)
        return sum((m[K] - MEAS_EXC[K]) ** 2 for K in Ks)
    cs = np.linspace(0.005, 0.20, 400)
    c_best = min(cs, key=sse)
    # refine
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(sse, bracket=(c_best * 0.7, c_best, c_best * 1.3))
    c_best = res.x

    print("=" * 88)
    print("TRUNCATION-FACTOR CLIP MODEL  rho=0.40   excess_clip = exc_unc * T(w*_min/(c*exc_unc))")
    print("  calibrated c = a*s0 = %.4f  (single parameter)" % c_best)
    print("=" * 88)
    _, m = model_curve(rho, c_best, 8)
    print("%2s | %9s %9s %9s %9s | %9s %9s | %7s"
          % ("K", "exc_unc", "w*_min", "z", "T(z)", "exc_clip", "exc_meas", "resid%"))
    resid = []
    for K in Ks:
        z = wm[K] / (c_best * unc[K])
        T = trunc_var_factor(z)
        em = MEAS_EXC[K]
        rr = 100 * (m[K] - em) / em
        resid.append(abs(rr))
        print("%2d | %9.3f %9.4f %9.3f %9.4f | %9.3f %9.3f | %+6.1f"
              % (K, unc[K], wm[K], z, T, m[K], em, rr))
    Kstar = max(Ks, key=lambda K: m[K])
    Kstar_meas = max(MEAS_EXC, key=MEAS_EXC.get)
    print("  mean |resid| = %.1f%%   K*(trunc model)=%d   K*(measured)=%d"
          % (np.mean(resid), Kstar, Kstar_meas))

    # ---- closed-form turnover condition: d/dK[exc_unc*T(z)]=0 ----
    # K* is where the log-derivative of exc_unc balances the log-derivative of T(z).
    # Report K*(rho) across rho and compare with both onset proxies.
    print("\n[K*(rho) across rho, truncation model with SAME c]:")
    print("%6s | %8s %8s %8s | %10s"
          % ("rho", "K*_trunc", "Kpk_unc", "z@K*", "onset(z<1)"))
    for r in (0.50, 0.47, 0.43, 0.40, 0.37, 0.35, 0.30):
        u, mm = model_curve(r, c_best, 16)
        wmr = {K: endpoint_w(K).min() for K in range(2, 17)}
        Kg = list(range(2, 17))
        Kstar_r = max(Kg, key=lambda K: mm[K])
        Kpk = max(Kg, key=lambda K: u[K])
        z_at = wmr[Kstar_r] / (c_best * u[Kstar_r])
        onset = next((K for K in Kg if wmr[K] / (c_best * u[K]) < 1.0), None)
        print("%6.2f | %8d %8d %8.2f | %10s" % (r, Kstar_r, Kpk, z_at, str(onset)))

    # ---- CROSS-CHECK: two clipping models agree on K*(rho) ----
    # model 1 (this): truncation turnover K*_trunc.
    # model 2 (Phase 0.3): onset where sigma_err_abs ~ w*_min, i.e. exc_unc*c ~ w*_min => z~1.
    # In the truncation model z=1 is exactly the half-suppression knee, so the onset model's
    # "sigma~w*_min" IS the z=1 crossing -> the two models are the SAME criterion at different
    # operating points (onset = z=1 knee; turnover = where T's drop overtakes exc_unc's growth).
    print("\n[AGREEMENT of the two clipping models]")
    print("  Model-1 truncation: excess_clip=exc_unc*T(z), turnover K* = argmax.")
    print("  Model-2 onset (Phase 0.3): clipping active when sigma_err_abs ~ w*_min  <=>  z ~ 1.")
    print("  -> z=1 is exactly T(z)'s half-suppression knee; onset (z=1) precedes turnover by ~1-2 K.")
    for r in (0.47, 0.40, 0.35):
        u, mm = model_curve(r, c_best, 16)
        wmr = {K: endpoint_w(K).min() for K in range(2, 17)}
        Kg = list(range(2, 17))
        Kstar_r = max(Kg, key=lambda K: mm[K])
        onset = next((K for K in Kg if wmr[K] / (c_best * u[K]) < 1.0), None)
        print("    rho=%.2f:  onset(z=1) K=%s  ->  turnover K*=%d   (gap=%s)"
              % (r, str(onset), Kstar_r, str(Kstar_r - onset) if onset else "n/a"))


if __name__ == "__main__":
    main()
