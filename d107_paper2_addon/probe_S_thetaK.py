# -*- coding: utf-8 -*-
"""
probe_S_thetaK.py -- the CORRECT closed form: worst mode is at the highest DISCRETE frequency
theta_K = pi*(K-1)/K  (degree-(K-1) contrast), NOT exactly pi. Both G and L symbols vary fast near pi,
so lambda_min = G(theta_K) and D = L(theta_K), giving S^2 = 1 - L(theta_K)/G(theta_K).

The K-dependence is ENTIRELY in theta_K -> pi. As K grows, theta_K -> pi, G(theta_K) -> G(pi) (small),
so lambda_min shrinks; the rate is set by how fast G drops near theta=pi (the curvature G''(pi)).

Symbol:  G(theta)/G_0 = sum_n cos(n theta) r^{n^2-? } ... using measured coeffs directly is cleanest.
We:
 (1) build G,L symbols from measured interior Toeplitz coeffs (many terms)
 (2) evaluate at theta_K = pi(K-1)/K and at the EXACT worst DST/DCT eigen-frequency for finite K
 (3) compare to true lambda_min, D, S over K=2..10, both rho.
 (4) Then derive the ASYMPTOTIC: near theta=pi, G(theta) ~ G(pi) + (1/2)G''(pi)(theta-pi)^2.
     theta_K - pi = -pi/K. So lambda_min ~ G(pi) + (1/2)G''(pi)(pi/K)^2.
     If G(pi)~0 (true for rho>=~0.45), lambda_min ~ (pi^2/2)|G''(pi)|/K^2 ~ K^{-2}.
     For rho=0.40 deeper, also ~K^{-2} once theta_K close to pi.
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
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f
Eqf = lambda f: TRAPZ(f, TGRID)


def matrices(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    psi = h / q
    G = np.array([[Eqf(h[i] * psi[j]) for j in range(K)] for i in range(K)])
    sp = np.stack([smooth(psi[i]) for i in range(K)])
    Bm = np.array([[Eqf(sp[i] * sp[j] * q) for j in range(K)] for i in range(K)])
    L = G - Bm
    return G, L


def worst_eig(G, K):
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return V[:, tang[0]], lam[tang[0]]


def sym(coeffs, theta):
    s = coeffs[0]
    for n in range(1, len(coeffs)):
        s += 2 * coeffs[n] * math.cos(n * theta)
    return s


for rho in (0.47, 0.40):
    print(f"\n========== rho={rho} ==========")
    Kbig = 14
    G, L = matrices(Kbig, rho)
    mid = Kbig // 2
    nb = 7
    gcoef = [G[mid, mid + n] for n in range(nb)]
    lcoef = [L[mid, mid + n] for n in range(nb)]
    print("  theta_K=pi*(K-1)/K model with measured interior symbols")
    print("  %2s | %8s | %10s %10s %8s | %10s %10s %8s | %8s"
          % ("K", "theta/pi", "lam_true", "G(thK)", "rel%", "D_true", "L(thK)", "rel%", "S_sym"))
    for K in range(2, 11):
        Gk, Lk = matrices(K, rho)
        v, lam = worst_eig(Gk, K)
        D = v @ Lk @ v
        S_true = math.sqrt(max(1 - D / lam, 0))
        thK = math.pi * (K - 1) / K
        Gth = sym(gcoef, thK); Lth = sym(lcoef, thK)
        S_sym = math.sqrt(max(1 - Lth / Gth, 0)) if Gth > 0 else float('nan')
        print("  %2d | %8.4f | %10.4e %10.4e %7.1f | %10.4e %10.4e %7.1f | %8.4f (true %.4f)"
              % (K, thK / math.pi, lam, Gth, 100 * (Gth - lam) / lam,
                 D, Lth, 100 * (Lth - D) / D, S_sym, S_true))
