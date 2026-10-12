# -*- coding: utf-8 -*-
"""
probe_lammin_operator.py -- CONTINUOUS-OPERATOR derivation of lambda_min and p(rho).

Picture:
  worst contrast c (degree K-1) ~ alternating (-1)^i with a slow envelope.
  delta_q = sum_i c_i h_i  is a Gaussian comb modulated at the lattice Nyquist freq
  k* = pi/d   (d = adjacent spacing).
  lambda_min = min_{c perp 1, ||c||=1}  R[c],  R[c] = int (sum c_i h_i)^2 / q.

Two analytic ingredients we test numerically here:
 (A) For the PURE alternating mode c_i=(-1)^i (normalized), how close is R[c] to the
     true lambda_min?  (tests "alternating = worst" approximation.)
 (B) The "uniform-background" model:  if q were ~ const Q over the support, then
       R[c] = (1/Q) * int (sum c_i h_i)^2 = (1/Q) * c^T M c,
       M_ij = int h_i h_j = (1/(2 sqrt(pi) sigma)) exp(-(mu_i-mu_j)^2/(4 sigma^2))
            = m0 * r^{(i-j)^2},  r = exp(-d^2/(4 sigma^2)),  m0=1/(2 sqrt(pi) sigma).
     Then lambda_min(M) on the alternating mode, divided by mean(q), is the prediction.
     The Toeplitz form M_ij = m0 r^{(i-j)^2} has a clean Fourier symbol; its alternating-mode
     eigenvalue -> m0 * theta-sum at k=pi (Poisson). We extract its K->inf limit and r-scaling.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_operator.py
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
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def true_lammin_and_v(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    return lam[kmin], V[:, kmin], G, q, h, w, mus


def rayleigh(c, G):
    P = np.eye(len(c)) - np.ones((len(c), len(c))) / len(c)
    c = P @ c
    c = c / np.linalg.norm(c)
    return c @ G @ c


print("=== (A) alternating vs true worst mode; (B) uniform-bg Toeplitz model ===")
print("r = exp(-d^2/(4 sigma^2));  m0 = 1/(2 sqrt(pi) sigma)")
m0 = 1.0 / (2 * math.sqrt(math.pi) * SIGMA)
for rho in (0.47, 0.40, 0.35, 0.30):
    d = dmu(rho)
    r = math.exp(-d ** 2 / (4 * SIGMA ** 2))
    print(f"\n--- rho={rho}  d={d:.2f}  r={r:.5f}  m0={m0:.6e} ---")
    print("%2s | %11s %11s %7s | %11s %7s | %11s %7s"
          % ("K", "true_lam", "alt_Ray", "ratio", "Mbg_alt", "ratio", "Mbg/meanq", "ratio"))
    for K in range(2, 11):
        lam, v, G, q, h, w, mus = true_lammin_and_v(K, rho)
        # (A) pure alternating Rayleigh quotient on true G
        c_alt = np.array([(-1.0) ** i for i in range(K)])
        alt = rayleigh(c_alt.copy(), G)
        # (B) uniform-background Toeplitz M_ij = m0 r^{(i-j)^2}
        M = np.array([[m0 * r ** ((i - j) ** 2) for j in range(K)] for i in range(K)])
        Pm = np.eye(K) - np.ones((K, K)) / K
        Mt = Pm @ M @ Pm
        lamM, VM = np.linalg.eigh(Mt)
        ones = np.ones(K) / math.sqrt(K)
        tangM = [k for k in np.argsort(lamM) if abs(VM[:, k] @ ones) < 0.5]
        Mbg_alt = lamM[tangM[0]]                 # smallest tangent eig of M (no q)
        # mean of q over the endmember support (between first and last mu)
        mask = (TGRID >= mus[0] - 3 * SIGMA) & (TGRID <= mus[-1] + 3 * SIGMA)
        meanq = TRAPZ(q[mask], TGRID[mask]) / (TGRID[mask][-1] - TGRID[mask][0])
        Mbg_over_q = Mbg_alt / meanq
        print("%2d | %11.4e %11.4e %7.3f | %11.4e %7.3f | %11.4e %7.3f"
              % (K, lam, alt, alt / lam, Mbg_alt, Mbg_alt / lam, Mbg_over_q, Mbg_over_q / lam))
