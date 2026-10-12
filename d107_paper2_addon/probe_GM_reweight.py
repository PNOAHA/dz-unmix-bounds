# -*- coding: utf-8 -*-
"""
probe_GM_reweight.py -- understand the G/M reweighting that turns p_M (symbol, ~1.6/2.3) into the
true p (lambda_min(G), ~0.95/1.88).

G_ij = ∫ h_i h_j / q.  Write 1/q(t).  On the support of the product h_i h_j (centered at (mu_i+mu_j)/2),
q is ~ the mixture density there. The worst mode v ~ degree-(K-1) contrast = alternating-sign.
For a Toeplitz-like quadratic form v^T G v = ∫ (sum_i v_i h_i)^2 / q = ∫ g_num^2 / q where g_num=sum v_i h_i.
v^T M v = ∫ g_num^2.   So  v^T G v / v^T M v = < 1/q >_{g_num^2}  (weighted avg of 1/q by g_num^2 density).

KEY: the worst mode is the same shape for M and G (alternating). The Rayleigh ratio for the worst
eigvec gives lammin(G) ~ lammin(M) * <1/q> where <1/q> = ∫ g^2/q / ∫ g^2 for g=worst-mode numerator.

For the endpoint weights w=linspace(1,0.6)/sum, q ~ roughly uniform-ish plateau of height ~ (avg w)*h_peak
spread over the K endmembers. The plateau VALUE of q scales like 1/K (since sum w =1 spread over K bumps
of fixed width). So <1/q> ~ K. THAT is the K-enhancement that lifts the exponent:
  lammin(G) ~ lammin(M) * <1/q>,  <1/q> ~ c1 * K   =>  p = p_M - 1 (since (K-1)^{-p_M} * (K-1)^{+1}).
Test this quantitatively.
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.5)
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
    return lam[tang[0]], V[:, tang[0]]


for rho in (0.47, 0.40):
    print("=== rho=%.2f ===" % rho)
    print("%2s | %11s %11s | %9s %9s %9s | %9s"
          % ("K", "lammin(G)", "lammin(M)", "ratioGM", "<1/q>_g", "qbar_inv", "Rayl(G)"))
    Ks, lgs, lms, inv_g = [], [], [], []
    for K in range(2, 11):
        h = endmembers(K, rho); w = endpoint_w(K)
        q = np.maximum((w[:, None] * h).sum(0), 1e-300)
        G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
        M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
        lg, vg = tangent_lammin(G, K)
        lm, vm = tangent_lammin(M, K)
        # <1/q> weighted by worst-mode-of-M numerator squared
        gnum = (vm[:, None] * h).sum(0)
        num = TRAPZ(gnum ** 2 / q, TGRID)
        den = TRAPZ(gnum ** 2, TGRID)
        inv_q_g = num / den
        # plateau 1/q at center
        cidx = np.argmin(abs(TGRID - (START + (K - 1) / 2 * dmu(rho))))
        qbar_inv = 1.0 / q[cidx]
        # Rayleigh of G at vm (worst mode of M) -- should ~ lammin(M)*<1/q>
        rayl = (vm @ G @ vm)
        print("%2d | %11.4e %11.4e | %9.3f %9.3f %9.3f | %9.4e"
              % (K, lg, lm, lg / lm, inv_q_g, qbar_inv, rayl))
        Ks.append(K); lgs.append(lg); lms.append(lm); inv_g.append(inv_q_g)
    Ks = np.array(Ks, float); lgs = np.array(lgs); lms = np.array(lms); inv_g = np.array(inv_g)
    pG = -np.polyfit(np.log(Ks - 1), np.log(lgs / lgs[0]), 1)[0]
    pM = -np.polyfit(np.log(Ks - 1), np.log(lms / lms[0]), 1)[0]
    pinv = np.polyfit(np.log(Ks - 1), np.log(inv_g / inv_g[0]), 1)[0]
    print("  p(G)=%.3f  p(M)=%.3f  slope(<1/q>)=+%.3f  => p(M)-slope=%.3f (vs p(G)=%.3f)"
          % (pG, pM, pinv, pM - pinv, pG))
    # also check lammin(M)*<1/q> reconstructs lammin(G)
    recon = lms * inv_g
    err = np.abs(recon - lgs) / lgs * 100
    print("  recon lammin(M)*<1/q> vs lammin(G): max err %.1f%%, mean %.1f%%" % (err.max(), err.mean()))
