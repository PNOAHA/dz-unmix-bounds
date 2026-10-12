# -*- coding: utf-8 -*-
"""
probe_lammin_variational7.py -- the GENUINE closed form is the THETA-function descent, not a power law.

  lam_min(K,rho) = A(rho) * G_K(beta),   G_K(beta) = (cosine-Rayleigh, true Gram) or the truncated
  alternating theta partial sum.  The clean two-parameter analytic model:

     lam_min(K,rho) ~ Afit * Theta_K(beta),
     Theta_K(beta) := sum_{m=-(K-1)}^{K-1} (-1)^m beta^{m^2} (1 - |m|/K)   (Fejer-tapered alternating theta)

  beta = exp(-L), L = 2 erfinv(rho)^2.   Afit fit by least squares in log over K=2..10 (one scalar / rho).

We compare THREE models per cell:
  (M1) pure power law  C (K-1)^{-p},  p = 0.74/L - 0.92            [the framework's summary law]
  (M2) Afit * Theta_K(beta)  (tapered alternating theta; ANALYTIC shape, 1 fitted scale)
  (M3) cosine-Rayleigh on true Gram (the variational UB; no fit, but needs the Gram)

The point: M2 has the SAME number of free params as M1 (one scale) yet matches far better because it
carries the correct theta-function curvature.  p(rho) is the derived log-slope SUMMARY of M2.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational7.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 5000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def Lof(rho): return 2.0 * erfinv(rho) ** 2


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.exp(-0.5 * ((TGRID[None, :] - mus[:, None]) / SIGMA) ** 2)
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def gram(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    return np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])


def tangent_lammin(G):
    K = G.shape[0]
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def cheb(K, deg):
    i = np.arange(K)
    v = np.cos(deg * math.pi * (i + 0.5) / K)
    v = v - v.mean(); return v / np.linalg.norm(v)


def R_cheb(K, rho):
    G = gram(K, rho); v = cheb(K, K - 1); return float(v @ G @ v)


def Theta_K(beta, K):
    ms = np.arange(-(K - 1), K)
    return float(np.sum(((-1.0) ** np.abs(ms)) * beta ** (ms ** 2) * (1 - np.abs(ms) / K)))


targets = [0.30, 0.35, 0.40, 0.47, 0.55]
Ks = np.arange(2, 11)
# p(rho) law from finer grid fit (reuse the derived a,b)
A_COEF, B_COEF = 0.7386, -0.9187

print("=" * 100)
print("Per-cell %% match of three models for lam_min(K,rho).  L=2 erfinv(rho)^2.")
print("  M1 = C(K-1)^{-p}, p=0.7386/L-0.9187    M2 = Afit*Theta_K(beta)    M3 = cosine-Rayleigh(Gram)")
print("=" * 100)
allres = {"M1": [], "M2": [], "M3": []}
for rho in targets:
    L = Lof(rho); beta = math.exp(-L); p = A_COEF / L + B_COEF
    lmt = np.array([tangent_lammin(gram(K, rho)) for K in Ks])
    # M1
    C1 = lmt[0]
    m1 = C1 * (Ks - 1.0) ** (-p)
    # M2: least-squares scale in log
    th = np.array([Theta_K(beta, K) for K in Ks])
    logA = np.mean(np.log(lmt) - np.log(th))
    m2 = math.exp(logA) * th
    # M3
    m3 = np.array([R_cheb(K, rho) for K in Ks])
    r1 = 100 * (m1 - lmt) / lmt
    r2 = 100 * (m2 - lmt) / lmt
    r3 = 100 * (m3 - lmt) / lmt
    allres["M1"].append(np.abs(r1)); allres["M2"].append(np.abs(r2)); allres["M3"].append(np.abs(r3))
    print("rho=%.2f L=%.4f beta=%.4f p=%.3f" % (rho, L, beta, p))
    print("  true:  " + " ".join("%9.3e" % x for x in lmt))
    print("  M1 res%%:" + " ".join("%+8.1f" % x for x in r1) + "  med=%.1f%%" % np.median(np.abs(r1)))
    print("  M2 res%%:" + " ".join("%+8.1f" % x for x in r2) + "  med=%.1f%%" % np.median(np.abs(r2)))
    print("  M3 res%%:" + " ".join("%+8.1f" % x for x in r3) + "  med=%.1f%%" % np.median(np.abs(r3)))
    print()

print("OVERALL median |resid| across all K,rho:")
for m in ("M1", "M2", "M3"):
    arr = np.concatenate(allres[m])
    print("  %s: median=%.1f%%  90th=%.1f%%  max=%.1f%%" %
          (m, np.median(arr), np.percentile(arr, 90), np.max(arr)))
