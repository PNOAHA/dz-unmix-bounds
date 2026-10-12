# -*- coding: utf-8 -*-
"""
probe_S_perpair.py -- decompose BOTH lambda_min=Var_q[g] AND D=smoothing-loss into per-adjacent-pair
contributions, to get a closed form S(K)=sqrt(1 - D(K)/lambda_min(K)).

GEOMETRY in the bulk (the only region where q is non-negligible, so the only region contributing to
Var_q): with constant spacing dmu and ~constant q in the interior, the worst eigvec v_i is a near-
sinusoidal (highest discrete contrast, degree K-1) -- the Chebyshev/DST-I mode of the tridiagonal-like
Gram structure. Both forms become QUADRATIC in v through a banded matrix:
   lambda_min = v^T G v,  with  G_ij = int h_i h_j / q   (banded: dominant diag + nearest-neighbor)
   D          = v^T (G - G_s) v,  G_s,ij = int (K_bw*?)... actually D=Var_q[g]-Var_q[smooth g] is NOT
               quadratic in v through a fixed matrix because smooth acts on g=m/q (nonlinear in q).
   BUT empirically D is a quadratic form too: define the loss kernel L by
       D = v^T L v ,   L_ij := int [ (h_i/q) - K_bw*(h_i/q-stuff) ]...  -- let's just MEASURE L_ij
   numerically by polarization:  D(v) is quadratic in the COEFFICIENTS only if smooth/q linear in v.
   Since g=(sum v_i h_i)/q is LINEAR in v, and smooth is linear, and Var_q is quadratic -> YES D IS a
   quadratic form in v:  D = v^T L v with
       L_ij = <psi_i, psi_j>_q - <K_bw*psi_i, K_bw*psi_j>_q ,  psi_i := h_i/q
   (minus the mean-correction, which vanishes on tangent). So L is a fixed K x K matrix.

So: S^2 = 1 - (v^T L v)/(v^T G v),  v=argmin of (v^T G v)/(v^T v) on tangent.
This is the clean closed structure. Now characterize G and L as banded matrices and get K-scaling.
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
    psi = h / q                                   # psi_i = h_i / q  (K,T)
    G = np.array([[Eqf(h[i] * psi[j]) for j in range(K)] for i in range(K)])   # = int h_i h_j /q
    # L_ij = <psi_i,psi_j>_q - <K_bw*psi_i, K_bw*psi_j>_q  ; <a,b>_q = int a b q
    sp = np.stack([smooth(psi[i]) for i in range(K)])   # K_bw * psi_i
    A = np.array([[Eqf(psi[i] * psi[j] * q) for j in range(K)] for i in range(K)])   # <psi_i,psi_j>_q = int h_i h_j/q = G
    Bm = np.array([[Eqf(sp[i] * sp[j] * q) for j in range(K)] for i in range(K)])
    L = A - Bm
    return G, L, q, h


def worst_eig(G, K):
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return V[:, tang[0]], lam[tang[0]]


print("Verify D = v^T L v  and  lambda_min = v^T G v, then show banded structure of G and L")
print("=" * 95)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    for K in (2, 4, 6, 10):
        G, L, q, h = matrices(K, rho)
        v, lam = worst_eig(G, K)
        lam_check = v @ G @ v
        D_check = v @ L @ v
        S2 = 1 - D_check / lam_check
        print("K=%2d  v^T G v=%.4e (lam=%.4e)  v^T L v=%.4e  S=%.4f"
              % (K, lam_check, lam, D_check, math.sqrt(max(S2, 0))))
    # show banded structure for K=6
    G, L, q, h = matrices(6, rho)
    print("  G (K=6) rows (diag + neighbors dominate):")
    for i in range(6):
        print("   ", np.array2string(G[i], precision=4, suppress_small=True))
    print("  L (K=6):")
    for i in range(6):
        print("   ", np.array2string(L[i], precision=5, suppress_small=True))
