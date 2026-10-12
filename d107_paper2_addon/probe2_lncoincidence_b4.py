# -*- coding: utf-8 -*-
"""
probe2_lncoincidence_b4.py -- resolve the b3 surprise: p_M small-rho slope was ~0.79, not 2.

HYPOTHESIS: lambda_min(M) is a theta-function descent, NOT a power law. The symbol law
p_M ~ pi^2/(4b) ~ 1/erfinv^2 is the TRUE ASYMPTOTIC LOCAL SLOPE of ln lambda_min vs
theta_K^2 near theta=pi, which is only reached for LARGE K (theta_K = pi(1-1/K)). Over a
small finite K-window (K=2..6, theta_K up to ~2.6 rad < pi) at small rho, the descent has
not entered the high-frequency Gaussian-symbol regime, so the FITTED window slope p_M is
suppressed below the asymptotic 1/erfinv^2.

TEST: measure the small-rho log-log divergence slope s (p_M ~ rho^{-s}) as a function of
the K-window upper edge Kmax. If s climbs toward ~2 as Kmax grows, the symbol law's
1/erfinv^2 divergence is the genuine asymptote (claim b survives ASYMPTOTICALLY, with the
honest caveat that finite-K windows see a deflated slope). If s stays ~1 for all Kmax,
even the asymptote is closer to 1/rho and claim (b) NEEDS-REVISION.

Also: directly probe the asymptotic symbol slope by computing the LOCAL slope
d(ln lambda_min)/d(ln(K-1)) at the LARGEST available K (not a window-average), for each rho,
and test THAT against 1/erfinv^2.
Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_lncoincidence_b4.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 7000.0 + 1e-9, 0.5)   # tall grid so large K fits at small rho
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def tangent_lammin(A, K):
    P = np.eye(K) - np.ones((K, K)) / K
    At = P @ A @ P
    lam, V = np.linalg.eigh(At)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def lammin_M_series(rho, Kmax):
    Ks = np.arange(2, Kmax + 1)
    out = []; used = []
    for K in Ks:
        mus_top = START + (K - 1) * dmu(rho)
        if mus_top >= TGRID[-1] - 3 * SIGMA:
            break
        h = endmembers(int(K), rho)
        M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
        lm = tangent_lammin(M, K)
        if lm <= 1e-15 * M.trace():   # eigenvalue underflowed -> stop, beyond float resolution
            break
        out.append(lm); used.append(K)
    return np.array(used), np.array(out)


def p_M_window(rho, Kmax):
    Ks, lM = lammin_M_series(rho, Kmax)
    if len(lM) < 3: return float('nan')
    return -np.polyfit(np.log(Ks - 1)[1:], np.log(lM[1:] / lM[0]), 1)[0]


def p_M_local_topK(rho, Kmax):
    """local slope between the top two K's = closest to asymptotic theta->pi."""
    Ks, lM = lammin_M_series(rho, Kmax)
    if len(lM) < 2: return float('nan')
    return -(math.log(lM[-1]) - math.log(lM[-2])) / (math.log(Ks[-1] - 1) - math.log(Ks[-2] - 1))


RHOS_small = np.array([0.12, 0.14, 0.16, 0.18, 0.20])
print("Small-rho divergence slope s  (p_M ~ rho^{-s}) vs K-window upper edge Kmax:")
print("  (s->2 => 1/erfinv^2 ~ 1/rho^2 asymptote ;  s->1 => 1/rho)")
print("  Kmax :   s(window-fit p_M)")
for Kmax in (6, 8, 10, 14, 20):
    pvals = np.array([p_M_window(rho, Kmax) for rho in RHOS_small])
    if np.any(np.isnan(pvals)):
        print("  %4d :   (some K off-grid, skipped)" % Kmax); continue
    s = -np.polyfit(np.log(RHOS_small), np.log(pvals), 1)[0]
    print("  %4d :   %.3f" % (Kmax, s))

print("\nLOCAL top-K slope p_M (closest to theta=pi asymptote) vs 1/erfinv^2:")
print("  rho   erfinv  p_M_local(Kmax=20)  p_local*erfinv^2  p_local*rho")
RHOS = np.array([0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.55])
loc = []
for rho in RHOS:
    pl = p_M_local_topK(rho, 20)
    loc.append(pl)
    e = erfinv(rho)
    print("  %.2f  %.4f      %6.3f           %7.4f       %7.4f"
          % (rho, e, pl, pl * e**2, pl * rho))
loc = np.array(loc); erf = np.array([erfinv(r) for r in RHOS])

def cv(x): return 100*np.std(x)/abs(np.mean(x))
s_local = -np.polyfit(np.log(RHOS[:4]), np.log(loc[:4]), 1)[0]
print("\n  small-rho slope of p_M_local: s=%.3f" % s_local)
print("  flatness CV: p_local*erfinv^2=%.1f%%  p_local*rho^2=%.1f%%  p_local*rho=%.1f%%"
      % (cv(loc*erf**2), cv(loc*RHOS**2), cv(loc*RHOS)))

print("\nVERDICT NOTE: if s climbs toward 2 with Kmax and the top-K local slope ~ 1/erfinv^2,")
print("the 1/erfinv^2 law is the genuine ASYMPTOTIC (large-K) divergence and the finite-window")
print("p was just deflated -> claim (b) SURVIVES asymptotically. If s saturates ~1 for all Kmax")
print("and top-K slope tracks 1/rho, claim (b) NEEDS-REVISION to 1/rho over the usable window.")
