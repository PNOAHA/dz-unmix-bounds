# -*- coding: utf-8 -*-
"""
probe2_lncoincidence_b3.py -- FINAL discriminator for claim (b): the small-rho divergence.

1/erfinv(rho)^2 and 1/rho are nearly affinely related over rho in [0.28,0.60] (why 1/rho
won the AFFINE fit). They SEPARATE as rho->0, where erfinv(rho)~ (sqrt(pi)/2) rho so
1/erfinv^2 -> (4/pi)/rho^2 (DIVERGES LIKE 1/rho^2, NOT 1/rho). So the clean discriminator
is: does p_M diverge like 1/rho^2 (=> 1/erfinv^2) or like 1/rho as rho shrinks?

We push rho down to 0.12 (keeping K small so peaks stay on-grid) and check the LOG-LOG
slope of p_M vs rho at the small-rho end:
   p_M ~ rho^{-s_loc}.  s_loc -> 2 supports 1/erfinv^2 (and 1/rho^2); s_loc -> 1 supports 1/rho.
Also directly compare p_M*erfinv^2 vs p_M*rho and p_M*rho^2 for flatness at small rho.

This is the asymptotic regime where the rigorous symbol law was derived (theta-descent near
theta=pi), so it is the fair home turf of claim (b).
Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_lncoincidence_b3.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
# small rho -> small spacing -> peaks crowd near START; modest top is fine. Use K=2..6.
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def tangent_lammin(A, K):
    P = np.eye(K) - np.ones((K, K)) / K
    At = P @ A @ P
    lam, V = np.linalg.eigh(At)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def p_M(rho, Kmax=6):
    Ks = np.arange(2, Kmax + 1)
    lM = []
    for K in Ks:
        h = endmembers(int(K), rho)
        M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
        lM.append(tangent_lammin(M, K))
    lM = np.array(lM)
    return -np.polyfit(np.log(Ks - 1)[1:], np.log(lM[1:] / lM[0]), 1)[0]


# dense small-rho grid
RHOS = np.array([0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.32, 0.38, 0.45, 0.55])
erf = np.array([erfinv(r) for r in RHOS])
pM = np.array([p_M(rho) for rho in RHOS])

print("  rho   erfinv   p_M    p_M*erfinv^2  p_M*rho  p_M*rho^2   (4/pi)/erfinv^2-vs-1/rho^2")
for i, rho in enumerate(RHOS):
    print("  %.2f  %.4f  %6.3f    %7.4f   %7.4f  %7.4f"
          % (rho, erf[i], pM[i], pM[i]*erf[i]**2, pM[i]*rho, pM[i]*rho**2))

# log-log local slope of p_M vs rho at the SMALL-rho end (first 5 points)
sm = slice(0, 5)
s_loc = -np.polyfit(np.log(RHOS[sm]), np.log(pM[sm]), 1)[0]
# and erfinv^2 vs rho^2 small-rho identity check: erfinv(rho)^2 / rho^2 -> pi/4 = 0.785
print("\n  small-rho (rho<=0.20) log-log slope  p_M ~ rho^{-s}:  s = %.3f" % s_loc)
print("    s->2 supports 1/erfinv^2 (= (4/pi)/rho^2 as rho->0);  s->1 supports 1/rho")
print("  erfinv(rho)^2/rho^2 at smallest rho (->pi/4=0.785 confirms erfinv~(sqrt pi/2)rho):")
print("    ", np.round(erf[sm]**2 / RHOS[sm]**2, 4))

# flatness comparison across the WHOLE small-to-mid range
def cv(x): return 100 * np.std(x) / abs(np.mean(x))
print("\n  flatness (CV%, lower=better) of p_M * f(rho):")
print("    p_M*erfinv^2 : CV=%.1f%%" % cv(pM * erf**2))
print("    p_M*rho^2    : CV=%.1f%%" % cv(pM * RHOS**2))
print("    p_M*rho      : CV=%.1f%%" % cv(pM * RHOS))

print("\nVERDICT NOTE: if small-rho slope s~2 and p_M*erfinv^2 is the flattest, the DIVERGENCE")
print("structure of claim (b) is 1/erfinv^2 (~1/rho^2), and the earlier 1/rho 'win' was a")
print("narrow-window affine artifact. 1/rho cannot match a 1/rho^2 divergence as rho->0.")
