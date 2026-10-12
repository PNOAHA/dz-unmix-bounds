# -*- coding: utf-8 -*-
"""
probe_theta_and_p.py -- pin down (a) the effective theta of the discrete worst tangent mode,
(b) the closed-form p(rho), and (c) a direct symbol-based predictor of lammin(G).

Model:
  lammin(M,K) ~ f_M(theta_K),  f_M(theta)=c sqrt(pi/b)[exp(-(theta)^2/4b images...)] (Poisson, exact for M).
  G = D^{1/2}-like reweight, but cleanest: treat the WORST-MODE Rayleigh quotient.
  For the discrete K-point problem the smallest tangent eigenvalue of a Toeplitz matrix with
  symbol f sits near theta* = pi*(K-1)/K = pi - pi/K (highest DFT mode on K points, sum-zero).

(a) Solve theta_K from lammin(M,K) = f_M(theta_K): invert the (monotone-near-pi) symbol.
(b) Closed form for p:  ln lammin ~ ln f_M(theta_K) ~ const - theta_K^2/(4b);  theta_K=pi(1-1/K).
    p(rho) = - d ln lammin / d ln(K-1). Compute both for M-only and with the +1 from <1/q>~K.

GOAL closed form:  p(rho) ~ [pi^2/(4b)] * <local slope of (1-1/K)^2 in ln(K-1)>  -  1
    where b = Delta(rho)^2/(4 sigma^2) = (erfinv(rho))^2 * 2.  (since Delta=2 sqrt2 erfinv sigma)
    => b = 2 (erfinv(rho))^2,  pi^2/(4b) = pi^2 / (8 (erfinv rho)^2).
Verify the (1-1/K)^2-vs-ln(K-1) effective slope ~ a constant 'alpha' over K=2..10, giving
    p_M ~ alpha * pi^2/(4b),  p_G ~ p_M - 1.
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import brentq

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def b_of_rho(rho): return dmu(rho) ** 2 / (4 * SIGMA ** 2)          # = 2*erfinv(rho)^2
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def c_const(): return 1.0 / (2 * SIGMA * math.sqrt(math.pi))


def fM(theta, rho, nmax=6):
    b = b_of_rho(rho)
    s = sum(math.exp(-((theta - 2 * math.pi * n) ** 2) / (4 * b)) for n in range(-nmax, nmax + 1))
    return c_const() * math.sqrt(math.pi / b) * s


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


print("b(rho) = 2*erfinv(rho)^2 check:")
for rho in (0.47, 0.40, 0.35):
    print("  rho=%.2f  b=%.5f  2erfinv^2=%.5f  pi^2/(4b)=%.4f"
          % (rho, b_of_rho(rho), 2 * erfinv(rho) ** 2, math.pi ** 2 / (4 * b_of_rho(rho))))

print("\n(a) Effective theta_K from inverting lammin(M)=f_M(theta), vs pi-pi/K:")
for rho in (0.47, 0.40):
    print("  -- rho=%.2f --" % rho)
    for K in range(2, 11):
        M = np.array([[TRAPZ(endmembers(K, rho)[i] * endmembers(K, rho)[j], TGRID)
                       for j in range(K)] for i in range(K)])
        lm, _ = tangent_lammin(M, K)
        # invert f_M on [pi/2, pi] (monotone decreasing toward pi)
        try:
            th = brentq(lambda t: fM(t, rho) - lm, math.pi / 2.5, math.pi)
        except Exception:
            th = float('nan')
        print("    K=%2d  lammin(M)=%.4e  theta_K=%.4f  pi-pi/K=%.4f  theta/pi=%.4f"
              % (K, lm, th, math.pi - math.pi / K, th / math.pi))

print("\n(b) closed-form p: p_M = alpha*pi^2/(4b), p_G=p_M-1.  alpha=effective slope of theta_K^2/(4b) in ln(K-1)")
for rho in (0.47, 0.40, 0.35):
    b = b_of_rho(rho)
    Ks = np.arange(2, 11)
    # use theta_K = pi - pi/K
    thK = math.pi * (1 - 1.0 / Ks)
    lnf = -thK ** 2 / (4 * b)            # n=0 dominant
    pM_th = -np.polyfit(np.log(Ks - 1), lnf - lnf[0], 1)[0]
    # full symbol (all images)
    lnf_full = np.array([math.log(fM(math.pi * (1 - 1.0 / K), rho)) for K in Ks])
    pM_full = -np.polyfit(np.log(Ks - 1), lnf_full - lnf_full[0], 1)[0]
    alpha = pM_th / (math.pi ** 2 / (4 * b))
    print("  rho=%.2f  b=%.4f  pi^2/4b=%.3f  alpha=%.4f  p_M(n0)=%.3f  p_M(full)=%.3f  => p_G~p_M(full)-1=%.3f"
          % (rho, b, math.pi ** 2 / (4 * b), alpha, pM_th, pM_full, pM_full - 1))

print("\n(c) compare to TRUE p(G):")
for rho in (0.47, 0.40):
    Ks = np.arange(2, 11); lgs = []
    for K in range(2, 11):
        G = np.array([[TRAPZ(endmembers(K, rho)[i] * endmembers(K, rho)[j]
                             / np.maximum((endpoint_w(K)[:, None] * endmembers(K, rho)).sum(0), 1e-300),
                             TGRID) for j in range(K)] for i in range(K)])
        lg, _ = tangent_lammin(G, K)
        lgs.append(lg)
    lgs = np.array(lgs)
    pG = -np.polyfit(np.log(Ks - 1), np.log(lgs / lgs[0]), 1)[0]
    print("  rho=%.2f  TRUE p(G)=%.3f" % (rho, pG))
