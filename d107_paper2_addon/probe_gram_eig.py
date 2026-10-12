# -*- coding: utf-8 -*-
"""
probe_gram_eig.py -- the CLEAN analytic core: smallest tangent eigenvalue of the
Gaussian Gram (overlap) matrix  M_ij = m0 * r^{(i-j)^2},  r = exp(-d^2/(4 sigma^2)).

This is the L2-overlap operator (NOT divided by q). We show:
  (1) the worst tangent mode of M is the degree-(K-1) DISCRETE-ORTHOGONAL contrast
      (NOT plain alternating), edge-peaked;
  (2) its eigenvalue lamM(K) ~ C(r) * (K-1)^{-p_M} for an effective range, and
      lamM has a clean K->inf law via Poisson summation on the Gaussian symbol;
  (3) p_M(r) and how it ties to true p(rho).

Symbol of the Toeplitz part: a(k) = m0 * sum_{n} r^{n^2} e^{ikn} = m0 * theta3-like.
The alternating (k=pi) value a(pi) = m0 * sum_n r^{n^2} (-1)^n is the n->inf floor.
But the WORST tangent eigenvalue sits BELOW a(pi) because of finite-K edge modes:
the discrete prolate / Chebyshev highest mode. We measure lamM(K) and fit.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_gram_eig.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA = 25.0
m0 = 1.0 / (2 * math.sqrt(math.pi) * SIGMA)


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def theta_alt(r, N=200):
    # sum_{n=-N}^{N} r^{n^2} (-1)^n   (Jacobi theta_4(0, r) when r<1)
    n = np.arange(-N, N + 1)
    return np.sum(r ** (n ** 2) * (-1.0) ** n)


def gram_min(K, r):
    M = np.array([[m0 * r ** ((i - j) ** 2) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Mt = P @ M @ P
    lam, V = np.linalg.eigh(Mt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]], V[:, tang[0]]


print("Gaussian-Gram smallest tangent eigenvalue lamM(K) = m0 * mu(K,r)")
print("a(pi)=m0*theta4 = the n->inf alternating-symbol floor")
for rho in (0.47, 0.40, 0.35, 0.30):
    r = math.exp(-dmu(rho) ** 2 / (4 * SIGMA ** 2))
    apI = m0 * theta_alt(r)
    print(f"\n--- rho={rho}  r={r:.5f}  a(pi)=m0*theta4={apI:.4e} ---")
    print("%2s | %12s | %10s | %10s" % ("K", "lamM(K)", "lamM/a(pi)", "lamM/lamM2"))
    lamM2 = None
    Ks, lams = [], []
    for K in range(2, 13):
        lm, v = gram_min(K, r)
        if K == 2: lamM2 = lm
        Ks.append(K); lams.append(lm)
        print("%2d | %12.5e | %10.4f | %10.4f" % (K, lm, lm / apI, lm / lamM2))
    Ks = np.array(Ks); lams = np.array(lams)
    # power-law fit over K=2..10 of lamM/lamM2 vs (K-1)
    m = Ks <= 10
    p = np.polyfit(np.log(Ks[m] - 1)[1:], np.log(lams[m][1:] / lamM2), 1)[0]
    print("  lamM/lamM2 ~ (K-1)^%.3f  over K=2..10" % p)
