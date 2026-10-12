# -*- coding: utf-8 -*-
"""
probe_pM_law.py -- derive the analytic p_M(rho) for the Gaussian-Gram worst-mode
eigenvalue  vMv = v^T M v ~ (K-1)^{-p_M},  M_ij = m0 r^{(i-j)^2}.

KEY ANALYTIC IDEA (highest discrete-orthogonal contrast over a Gaussian comb):
The worst tangent mode v is the degree-(K-1) orthogonal polynomial in the index.
For a Gram with entries m0 r^{(i-j)^2}, this is the q-Hermite / Stieltjes-Wigert family
with q-parameter = r^2.  Their squared norms decay GEOMETRICALLY in the degree, but
over the finite window K the smallest eigenvalue is set by the EDGE (Chebyshev-like)
mode.  Empirically vMv(K)/vMv(K-1) is NOT constant -> it is between geometric and power.

Hypotheses to test for the LOCAL exponent p_M over K=2..10 as a function of
s := -ln r = d^2/(4 sigma^2) = 2 (erfinv(rho))^2 :
  (H1) p_M = 1 + a*s            (linear in s)
  (H2) p_M = b*s                (pure proportional)
  (H3) p_M = 1 + s/ln2 ... etc.
We also fit p_M against  -ln(r) and against rho directly.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_pM_law.py
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


print("s := -ln r = 2*(erfinv(rho))^2 = d^2/(4 sigma^2)")
print("%6s %8s %8s %8s %8s %8s %8s" %
      ("rho", "r", "s", "p_M(2-10)", "p_M(2-7)", "1+a*s?", "Geom?"))
rows = []
for rho in (0.50, 0.47, 0.45, 0.40, 0.37, 0.35, 0.32, 0.30, 0.27, 0.25):
    d = dmu(rho); r = math.exp(-d ** 2 / (4 * SIGMA ** 2))
    s = -math.log(r)
    Ks = np.arange(2, 11)
    vs = np.array([vMv_min(K, r) for K in Ks])
    p_full = np.polyfit(np.log(Ks - 1)[1:], np.log(vs[1:] / vs[0]), 1)[0]
    m7 = Ks <= 7
    p_7 = np.polyfit(np.log(Ks[m7] - 1)[1:], np.log(vs[m7][1:] / vs[0]), 1)[0]
    rows.append((rho, r, s, -p_full, -p_7))
    print("%6.2f %8.4f %8.4f %8.3f %8.3f" % (rho, r, s, -p_full, -p_7))

rows = np.array(rows)
s = rows[:, 2]; pM = rows[:, 3]
# fit p_M = A + B*s
A = np.vstack([np.ones_like(s), s]).T
coef, *_ = np.linalg.lstsq(A, pM, rcond=None)
pred = A @ coef
r2 = 1 - ((pM - pred) ** 2).sum() / ((pM - pM.mean()) ** 2).sum()
print("\nfit  p_M = %.4f + %.4f * s   R2=%.4f" % (coef[0], coef[1], r2))
# fit p_M = B*s (no intercept)
B = (s @ pM) / (s @ s)
pred2 = B * s
r2b = 1 - ((pM - pred2) ** 2).sum() / ((pM - pM.mean()) ** 2).sum()
print("fit  p_M = %.4f * s            R2=%.4f" % (B, r2b))
# in terms of erfinv:  s = 2*(erfinv(rho))^2  ->  p_M = A + 2B (erfinv rho)^2
print("=> p_M = %.3f + %.3f*(erfinv(rho))^2   [since s=2(erfinv rho)^2]" %
      (coef[0], 2 * coef[1]))
