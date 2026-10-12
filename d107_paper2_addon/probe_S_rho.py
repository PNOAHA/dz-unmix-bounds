# -*- coding: utf-8 -*-
"""
probe_S_rho.py -- pin the rho-dependence of omega_q's saturation law and give the geometric anchor,
and test the model at a THIRD rho for genuine validation.

omega_q(K) = omega_inf * sqrt((K-1)/(K+a)).
We found omega_inf ~ 1.46*Nyq (rho .47), 1.74*Nyq (rho .40), Nyq=pi/dmu(rho). So omega_inf is set by
the contrast comb's Nyquist BUT scaled up (the q-bulk is narrower than the full comb so the effective
spatial period is < dmu). Test extra rho=0.55 and 0.35 to map omega_inf(rho), a(rho).

GEOMETRIC interpretation: omega_q is the q-weighted rms gradient/amplitude of the alternating mode
g~(-1)^i in the bulk. As K->inf, g oscillates with spatial half-period ~ dmu, so the gradient scale is
~ pi/dmu = Nyq; the >1 factor comes from g being sharper than sinusoidal between peaks (cusps).
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import curve_fit

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 4500.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def worst(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    return g, q


def measure(K, rho):
    g, q = worst(K, rho)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID); fpp = np.gradient(fp, TGRID)
    den = Eq(f * f)
    m1 = Eq(fp * fp) / den; m2 = Eq(fpp * fpp) / den
    varw = max(m2 - m1 * m1, 0.0)
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
    gs = KS @ g
    S_true = math.sqrt(max(Eq(gs * gs) - Eq(gs) ** 2, 0) / den)
    return S_true, math.sqrt(m1), varw


def f_sat(K, oinf, a): return oinf * np.sqrt((K - 1) / (K + a))

print("rho-map of omega_q saturation law, plus held-out validation at rho=0.55 & 0.35")
print("=" * 100)
rows = []
for rho in (0.55, 0.47, 0.40, 0.35):
    Ks = np.arange(2, 11, dtype=float)
    Strue = np.zeros(len(Ks)); omq = np.zeros(len(Ks)); varw = np.zeros(len(Ks))
    for i, K in enumerate(Ks):
        Strue[i], omq[i], varw[i] = measure(int(K), rho)
    popt, _ = curve_fit(f_sat, Ks, omq, p0=[0.15, 10.0], maxfev=40000)
    om_c = f_sat(Ks, *popt)
    cv = np.median(varw / omq ** 4)
    S2 = np.exp(-0.5 * BW ** 2 * om_c ** 2 + 0.25 * BW ** 4 * cv * om_c ** 4)
    rmsv = math.sqrt(np.mean((100 * (S2 - Strue) / Strue) ** 2))
    nyq = math.pi / dmu(rho)
    kappa = 2 * erfinv(rho) ** 2     # dmu^2/(4 sigma^2)
    rows.append((rho, popt[0], popt[0] / nyq, popt[1], cv, rmsv, nyq, kappa))
    print("rho=%.2f  dmu=%.2f Nyq=%.4f kappa=%.4f | omega_inf=%.4f (=%.3f Nyq) a=%.2f cv=%.3f | 2mom RMS=%.2f%%"
          % (rho, dmu(rho), nyq, kappa, popt[0], popt[0] / nyq, popt[1], cv, rmsv))

print("\nomega_inf vs Nyq*g(kappa): check if omega_inf/Nyq is a smooth fn of rho (or kappa)")
print("  rho   omega_inf/Nyq   a       a/(stuff)")
for r in rows:
    print("  %.2f     %.3f       %.2f" % (r[0], r[2], r[3]))
