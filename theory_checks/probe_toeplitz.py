# -*- coding: utf-8 -*-
"""
probe_toeplitz.py -- derive lambda_min(G,K,rho) via the Toeplitz symbol of the overlap matrix.

KEY OBJECTS
  M_ij = ∫ h_i h_j           (overlap; EXACTLY Toeplitz: m_k = (1/(2 sigma sqrt(pi))) * exp(-Delta^2 k^2 / (4 sigma^2)))
  G_ij = ∫ h_i h_j / q       (signal Fisher; q slowly varying -> G ~ near-Toeplitz, M reweighted)

DERIVATION TARGET
  lambda_min(G) on sum-zero tangent ~ symbol value of the relevant matrix near theta=pi
  (most-oscillatory degree-(K-1) mode). Get K, rho dependence and p(rho).

We probe:
  (A) m_k decay constant r = exp(-Delta^2/(4 sigma^2)), and relate to rho.
  (B) Toeplitz symbol f(theta) = sum_k m_k e^{i k theta} = theta_3-type Gaussian-comb sum.
  (C) the smallest tangent eigenvalue of M and G vs f(theta=pi - pi/K) and vs the table.
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


def matrices(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
    return G, M


def tangent_lammin(A, K):
    P = np.eye(K) - np.ones((K, K)) / K
    At = P @ A @ P
    lam, V = np.linalg.eigh(At)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]], V[:, tang[0]]


# ---- analytic overlap symbol ----
def overlap_entry_analytic(k, rho):
    # m_k = (1/(2 sigma sqrt(pi))) exp(-Delta^2 k^2/(4 sigma^2)),  Delta=dmu(rho)
    D = dmu(rho)
    return (1.0 / (2 * SIGMA * math.sqrt(math.pi))) * math.exp(-(D ** 2) * (k ** 2) / (4 * SIGMA ** 2))


def r_of_rho(rho):
    # decay base for the k=1 term relative to k=0 :  r = exp(-Delta^2/(4 sigma^2))
    D = dmu(rho)
    return math.exp(-(D ** 2) / (4 * SIGMA ** 2))


def symbol(theta, rho, kmax=40):
    # f(theta) = sum_k m_k e^{i k theta}, real & even -> m_0 + 2 sum_{k>=1} m_k cos(k theta)
    s = overlap_entry_analytic(0, rho)
    for k in range(1, kmax + 1):
        s += 2 * overlap_entry_analytic(k, rho) * math.cos(k * theta)
    return s


print("Check overlap matrix M is Toeplitz & matches analytic m_k (rho=0.47, K=6)")
G, M = matrices(6, 0.47)
print("  numerical M row0:", np.round(M[0], 5))
print("  analytic  m_k   :", np.round([overlap_entry_analytic(k, 0.47) for k in range(6)], 5))
print("  Toeplitz dev (max |M_ij - m_{|i-j|}|): %.2e" % max(
    abs(M[i, j] - overlap_entry_analytic(abs(i - j), 0.47)) for i in range(6) for j in range(6)))

print("\nr = exp(-Delta^2/(4 sigma^2)) by rho:")
for rho in (0.47, 0.40, 0.35):
    print("  rho=%.2f  Delta=%.2f  r=%.4f  -ln r = Delta^2/(4 sig^2) = %.4f"
          % (rho, dmu(rho), r_of_rho(rho), -math.log(r_of_rho(rho))))

print("\nSmallest tangent eigenvalue: M vs symbol f(pi - pi/K) vs G")
for rho in (0.47, 0.40):
    print("=== rho=%.2f ===" % rho)
    print("%2s | %11s %11s | %11s %11s | %9s"
          % ("K", "lammin(G)", "lammin(M)", "f(pi)", "f(pi-pi/K)", "G/M ratio"))
    for K in range(2, 11):
        G, M = matrices(K, rho)
        lg, _ = tangent_lammin(G, K)
        lm, _ = tangent_lammin(M, K)
        fpi = symbol(math.pi, rho)
        fpik = symbol(math.pi - math.pi / K, rho)
        print("%2d | %11.4e %11.4e | %11.4e %11.4e | %9.4f"
              % (K, lg, lm, fpi, fpik, lg / lm))
