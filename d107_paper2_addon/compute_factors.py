# -*- coding: utf-8 -*-
"""
compute_factors.py -- decompose the validated sandwich excess into its two analytic factors,
giving precise K-scaling targets for the asymptotic derivation.

excess(K) = CR(K) * S(K)
  CR(K) = sqrt(lambda_min(G,K=2)/lambda_min(G,K))   (Cramer-Rao / no-bw envelope; the signal-Fisher part)
  S(K)  = sqrt( Var_{a~q}[g_v ⋆ K_bw] / Var_{a~q}[g_v] )   (bw suppression of the worst mode g_v=Σ v_i h_i/q)
Also reports omega_eff(K) = rms Fourier frequency of g_v, to test S(K) ≈ exp(-bw^2 omega_eff^2 / 2).

K up to 10 (deterministic linear algebra). Fits lambda_min(K), omega_eff(K), and the resulting excess.
English prints. Run:  & "C:/Users/Administrator/miniconda3/python.exe" compute_factors.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
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


def worst_mode(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    v = V[:, kmin]
    lam_min = lam[kmin]
    g = (v[:, None] * h).sum(0) / q                         # g_v = Σ v_i h_i / q
    gs = smooth(g)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    var_g = Eq(g * g) - Eq(g) ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / var_g)
    # rms Fourier frequency of g (unweighted spectrum)
    G_hat = np.fft.rfft(g - g.mean())
    freq = np.fft.rfftfreq(len(g), d=DT) * 2 * math.pi       # angular freq
    P_hat = np.abs(G_hat) ** 2
    omega_eff = math.sqrt((freq ** 2 * P_hat).sum() / P_hat.sum())
    return lam_min, S, omega_eff


for rho in (0.47, 0.40):
    print(f"\n=== rho={rho} ===")
    print("%2s | %10s %8s %8s %8s | %8s %8s %8s | %8s"
          % ("K", "lam_min", "CR_exc", "S(K)", "S_pred", "exc=CR*S", "exp_lnfit", "omega_eff", "bw*om"))
    rows = []
    lm2 = None
    for K in range(2, 11):
        lm, S, om = worst_mode(K, rho)
        if K == 2: lm2 = lm
        cr = math.sqrt(lm2 / lm)
        S_pred = math.exp(-0.5 * BW ** 2 * om ** 2)           # crude single-freq guess
        exc = cr * S
        rows.append((K, lm, cr, S, exc, om))
        print("%2d | %10.4e %8.3f %8.4f %8.4f | %8.3f %8s %8.4f | %8.3f"
              % (K, lm, cr, S, S_pred, exc, "", om, BW * om))
    # fit excess(K) to a + b ln(K-1) over K=2..7 (the measured window)
    Ks = np.array([r[0] for r in rows]); exc = np.array([r[4] for r in rows])
    m = Ks <= 7
    A = np.vstack([np.ones(m.sum()), np.log(Ks[m] - 1)]).T
    coef, *_ = np.linalg.lstsq(A, exc[m], rcond=None)
    pred = A @ coef
    ss = 1 - ((exc[m] - pred) ** 2).sum() / ((exc[m] - exc[m].mean()) ** 2).sum()
    print("  fit excess ~ a + b*ln(K-1) over K=2..7:  a=%.3f b=%.3f  R2=%.3f" % (coef[0], coef[1], ss))
    # also fit lam_min ~ c*(K-1)^p
    lm = np.array([r[1] for r in rows])
    p = np.polyfit(np.log(Ks - 1 + 1e-9)[1:], np.log(lm[1:] / lm2), 1)[0]
    print("  lam_min/lam_min(2) ~ (K-1)^%.2f   (CR_exc ~ (K-1)^%.2f)" % (p, -p / 2))
    # fit omega_eff ~ d*(K-1)^r
    om = np.array([r[5] for r in rows])
    r_om = np.polyfit(np.log(Ks - 1 + 1e-9)[1:], np.log(om[1:]), 1)[0]
    print("  omega_eff ~ (K-1)^%.2f" % r_om)
