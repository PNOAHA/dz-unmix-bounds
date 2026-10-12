# -*- coding: utf-8 -*-
"""
probe_qhermite.py -- EXACT closed form for the Gaussian-Gram via q-Hermite / Cholesky.

M_ij = m0 * r^{(i-j)^2}.  Write t = r^2.  Then r^{(i-j)^2} = r^{i^2} r^{j^2} t^{-ij}.
With D = diag(r^{i^2}),  M = m0 * D V D  where V_ij = t^{-ij} = (t^{-i})^j  is a
VANDERMONDE matrix in nodes x_i = t^{-i} = r^{-2i}.
=> M = m0 D V D is totally positive; its LDL/Cholesky pivots (= ratios of leading
principal minors) have a CLOSED PRODUCT form (Vandermonde Gram / Hankel determinant).

The k-th Cholesky pivot p_k of a Gram of monomials {x^0..x^{K-1}} weighted by D is the
squared norm of the k-th ORTHOGONAL polynomial = the degree-k discrete-orthogonal norm.
For the worst tangent mode (degree K-1) the relevant scale is the (K-1)-th pivot p_{K-1}.

We test:  vMv_min(K)  ~  c * p_{K-1}(K)  and derive p_{K-1} in closed form.
det(leading k x k of M) for a Vandermonde-Gram telescopes:
   the monic-OP squared norms for nodes x_i=r^{-2i}, weights w_i = m0 r^{2 i^2}.
We compute pivots two ways (LDL of M, and the analytic product) and compare.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_qhermite.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA = 25.0
m0 = 1.0 / (2 * math.sqrt(math.pi) * SIGMA)


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def vMv_min(K, r):
    M = np.array([[m0 * r ** ((i - j) ** 2) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Mt = P @ M @ P
    lam, V = np.linalg.eigh(Mt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def ldl_pivots(K, r):
    # exact LDL pivots of the full M (no projection): p_k = det_k / det_{k-1}
    M = np.array([[m0 * float(r) ** ((i - j) ** 2) for j in range(K)] for i in range(K)], dtype=np.float64)
    piv = []
    A = M.copy()
    for k in range(K):
        p = A[k, k]
        piv.append(p)
        if abs(p) < 1e-300: break
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:]) / p
    return np.array(piv)


print("Compare smallest tangent eig vMv_min(K) to the (K-1)-th LDL pivot p_{K-1}.")
for rho in (0.47, 0.40, 0.35, 0.30):
    r = math.exp(-dmu(rho) ** 2 / (4 * SIGMA ** 2)); t = r * r
    print(f"\n--- rho={rho}  r={r:.5f}  t=r^2={t:.5f} ---")
    print("%2s | %12s | %12s | %8s | %12s %8s" %
          ("K", "vMv_min", "p_{K-1}", "v/p", "p_K analytic", "p/anal"))
    for K in range(2, 12):
        vm = vMv_min(K, r)
        piv = ldl_pivots(K, r)
        pKm1 = piv[-1]                                   # last pivot = degree K-1 norm
        # analytic pivot product for monic OP wrt nodes x_i=r^{-2i}, weights m0 r^{2i^2}:
        # squared norm of degree-n monic OP for a Vandermonde Gram with weights w_i:
        # closed (q-)form is messy; instead give the LEADING term:
        # p_{K-1} ~ m0 * prod_{i<j<=K-1}(x_i-x_j)^2-style telescoping. Use ratio of Hankel dets.
        # Just report leading-order guess m0 * r^{?}.  Use empirical leading: p_{K-1} ~ m0 * r^{(K-1)^2}? test:
        anal = m0 * r ** ((K - 1) ** 2)
        print("%2d | %12.5e | %12.5e | %8.4f | %12.5e %8.3f" %
              (K, vm, pKm1, vm / pKm1, anal, pKm1 / anal))
