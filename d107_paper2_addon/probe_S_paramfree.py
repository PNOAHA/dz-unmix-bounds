# -*- coding: utf-8 -*-
"""
probe_S_paramfree.py -- FULLY parameter-free closed form (no per-rho fit), end-to-end.

CLOSED FORM:
  kappa(rho) = dmu^2/(4 sigma^2) = 2 (erfinv rho)^2 ,   Nyq = pi/dmu = pi/(2 sqrt2 erfinv(rho) sigma)
  omega_inf(rho) = C1 * Nyq / sqrt(kappa)   with C1 ~= 0.92        (omega_inf*sqrt(kappa)/Nyq ~ const)
  a(rho)         = C2 / kappa^2             with C2 ~= 2.4
  omega_q(K,rho) = omega_inf * sqrt((K-1)/(K+a))
  S(K,rho)       = exp( -bw^2 omega_q^2/2 + bw^4 c_v omega_q^4/4 ),  c_v ~= 0.09  (spectral-width corr.)

Test against TRUE S over K=2..10 at rho=0.55,0.47,0.40,0.35 with NO per-case fitting.
Report residuals. This is the falsifiable, closed prediction.
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 4500.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

C1, C2, CV = 0.92, 2.4, 0.09     # the three universal constants


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def S_true(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    Eq = lambda f: TRAPZ(f * q, TGRID)
    den = Eq(g * g) - Eq(g) ** 2
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
    gs = KS @ g
    return math.sqrt(max(Eq(gs * gs) - Eq(gs) ** 2, 0) / den)


def S_closed(K, rho):
    kappa = 2 * erfinv(rho) ** 2
    nyq = math.pi / dmu(rho)
    omega_inf = C1 * nyq / math.sqrt(kappa)
    a = C2 / kappa ** 2
    omq2 = omega_inf ** 2 * (K - 1) / (K + a)
    return math.exp(-0.5 * BW ** 2 * omq2 + 0.25 * BW ** 4 * CV * omq2 ** 2)


print("PARAMETER-FREE closed form (C1=%.2f, C2=%.1f, c_v=%.2f) vs TRUE S" % (C1, C2, CV))
print("=" * 80)
allerr = []
for rho in (0.55, 0.47, 0.40, 0.35):
    print(f"\n--- rho={rho} ---  %2s | %8s %8s %7s" % ("K", "S_true", "S_closed", "err%"))
    errs = []
    for K in range(2, 11):
        st = S_true(K, rho); sc = S_closed(K, rho)
        e = 100 * (sc - st) / st; errs.append(e); allerr.append(e)
        print("        %2d | %8.4f %8.4f %+7.2f" % (K, st, sc, e))
    print("   RMS%%=%.2f  MAX%%=%.2f" % (math.sqrt(np.mean(np.array(errs) ** 2)), np.max(np.abs(errs))))
print("\nOVERALL (4 rho x 9 K = 36 pts): RMS%%=%.2f  MAX%%=%.2f"
      % (math.sqrt(np.mean(np.array(allerr) ** 2)), np.max(np.abs(allerr))))
