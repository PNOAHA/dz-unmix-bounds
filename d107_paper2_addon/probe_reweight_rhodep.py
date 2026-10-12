# -*- coding: utf-8 -*-
"""
probe_reweight_rhodep.py -- the reweight R(K)=lammin(G)/lammin(M) is rho-DEPENDENT.
Decompose p_G cleanly as p_G = p_M(rho) - p_R(rho) and get p_R(rho).
Then test whether p_R is ~ constant (the offset C) or scales -- this decides the closed form.

p_M from exact M Toeplitz symbol; p_R from R(K)=lammin(G)/lammin(M).
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def b_of_rho(rho): return 2.0 * erfinv(rho) ** 2
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
    return lam[tang[0]], V[:, tang[0]]


rhos = [0.50, 0.47, 0.45, 0.40, 0.38, 0.35, 0.32, 0.30]
print("rho |  p_M    p_R    p_G(=pM-pR)  p_G(direct) | b=2erfinv^2  1/erfinv^2")
pMs, pRs, pGs, invs = [], [], [], []
for rho in rhos:
    Ks = np.arange(2, 11); lgs, lms, Rs = [], [], []
    for K in range(2, 11):
        h = endmembers(K, rho); q = np.maximum((endpoint_w(K)[:, None] * h).sum(0), 1e-300)
        G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
        M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
        lg = tangent_lammin(G, K)[0]; lm = tangent_lammin(M, K)[0]
        lgs.append(lg); lms.append(lm); Rs.append(lg / lm)
    lgs = np.array(lgs); lms = np.array(lms); Rs = np.array(Rs)
    pM = -np.polyfit(np.log(Ks - 1), np.log(lms / lms[0]), 1)[0]
    pR = np.polyfit(np.log(Ks - 1), np.log(Rs / Rs[0]), 1)[0]
    pG = -np.polyfit(np.log(Ks - 1), np.log(lgs / lgs[0]), 1)[0]
    pMs.append(pM); pRs.append(pR); pGs.append(pG); invs.append(1/erfinv(rho)**2)
    print("  %.2f | %.3f  %.3f   %.3f       %.3f      | %.4f      %.3f"
          % (rho, pM, pR, pM - pR, pG, b_of_rho(rho), 1/erfinv(rho)**2))

pMs = np.array(pMs); pRs = np.array(pRs); pGs = np.array(pGs); invs = np.array(invs)
# fit p_M = aM/erfinv^2 (+cM), p_R vs rho
aM, cM = np.polyfit(invs, pMs, 1)
print("\np_M = %.4f/erfinv^2 %+.4f  (theory slope alpha*pi^2/8; alpha~0.257 -> 0.317)" % (aM, cM))
print("p_R(rho):", np.round(pRs, 3), " mean=%.3f std=%.3f" % (pRs.mean(), pRs.std()))
# is p_R ~ const or scaling with 1/erfinv^2?
aR, cR = np.polyfit(invs, pRs, 1)
print("p_R = %.4f/erfinv^2 %+.4f" % (aR, cR))
aG, cG = np.polyfit(invs, pGs, 1)
print("p_G = %.4f/erfinv^2 %+.4f  R2=%.4f"
      % (aG, cG, 1 - ((pGs-(aG*invs+cG))**2).sum()/((pGs-pGs.mean())**2).sum()))
