# -*- coding: utf-8 -*-
"""
probe_S_symbol.py -- the closed form via the Toeplitz SYMBOL at the worst (alternating) mode.

Both G and L are banded ~Toeplitz in the interior:
   G:  diag d_G = int h_i^2/q,   1st off a_G = int h_i h_{i+1}/q,   2nd off b_G = int h_i h_{i+2}/q, ...
   L:  diag d_L,  1st off a_L,   2nd off b_L, ...
For a Toeplitz matrix with row [..,c2,c1,c0,c1,c2,..], the eigenvalue at discrete frequency theta is the
SYMBOL  M(theta) = c0 + 2 c1 cos(theta) + 2 c2 cos(2 theta) + ...
The WORST (smallest-G) tangent mode is the MAXIMALLY OSCILLATING one: v_i ~ (-1)^i  => theta = pi.
=> lambda_min ~ G_symbol(pi) = d_G - 2 a_G + 2 b_G - 2 c_G + ...    (alternating)
   D          ~ L_symbol(pi) = d_L - 2 a_L + 2 b_L - ...
=> S^2 = 1 - L(pi)/G(pi).

The off-diagonals are Gaussian-overlap integrals. With area-normalized Gaussians spaced dmu and ~flat q
in the bulk (q ~ q0), the n-th off-diagonal of G is
   G_n = int h_i h_{i+n}/q  ~ (1/q0) int h_i h_{i+n}  = (1/q0) * [overlap of two unit Gaussians sigma,
         centers n*dmu apart] = (1/q0) * exp(-(n dmu)^2/(4 sigma^2)) / (2 sqrt(pi) sigma)
   since int N(0,s)N(nd,s) dt = exp(-(nd)^2/(4 s^2))/(2 sqrt(pi) s).
So  G_n / G_0 = exp(-(n dmu)^2/(4 sigma^2)) =: r^{n^2},  where r = exp(-dmu^2/(4 sigma^2)).
Note: rho = KS distance for spacing dmu => dmu = 2 sqrt2 erfinv(rho) sigma => dmu^2/(4 sigma^2)
      = 2 (erfinv rho)^2 =: kappa.  So r = exp(-kappa),  and the adjacent OVERLAP coefficient is r.

Then G(pi)/G_0 = sum_n (-1)^n r^{n^2} ... = theta-function-like:  1 - 2r + 2 r^4 - 2 r^9 + ...
This is the Jacobi theta:  sum_{n=-inf}^{inf} (-1)^n r^{n^2} = theta_4(0, r)  (q-series, here nome=r).
=> G(pi) = (G_0) * theta_4(0,r).
So the WORST-mode signal Fisher per-grain ~ d_G * theta_4(0, r), r=exp(-2 (erfinv rho)^2).

This script:
 (1) extracts measured Toeplitz coefficients (interior row) of G and L
 (2) builds the symbol at theta=pi, compares lambda_min, D, S to truth
 (3) tests the theta_4 / Gaussian-overlap closed form for the G coefficients
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


def symbol_pi(coeffs):
    # coeffs[0]=diag, coeffs[n]=n-th off; symbol at theta=pi: c0 + 2 sum (-1)^n c_n
    s = coeffs[0]
    for n in range(1, len(coeffs)):
        s += 2 * ((-1) ** n) * coeffs[n]
    return s


print("Toeplitz-symbol-at-pi model for lambda_min and D (interior row, large K)")
print("=" * 95)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho}  (kappa=2*erfinv(rho)^2=%.4f, r=exp(-kappa)=%.4f) ---"
          % (2 * erfinv(rho) ** 2, math.exp(-2 * erfinv(rho) ** 2)))
    # use a large K to read off interior Toeplitz coeffs
    Kbig = 12
    G, L = matrices(Kbig, rho)
    mid = Kbig // 2
    nb = 5
    gcoef = [G[mid, mid + n] for n in range(nb)]
    lcoef = [L[mid, mid + n] for n in range(nb)]
    print("  G interior coeffs (d, off1..):", np.array2string(np.array(gcoef), precision=4))
    print("  L interior coeffs (d, off1..):", np.array2string(np.array(lcoef), precision=5))
    Gpi = symbol_pi(gcoef); Lpi = symbol_pi(lcoef)
    print("  G(pi)=%.4f  L(pi)=%.5f  =>  S_symbol=sqrt(1-L(pi)/G(pi))=%.4f"
          % (Gpi, Lpi, math.sqrt(max(1 - Lpi / Gpi, 0))))
    # theta_4 prediction for G coeffs: G_n/G_0 = r^{n^2}
    r = math.exp(-2 * erfinv(rho) ** 2)
    pred_ratio = [r ** (n * n) for n in range(nb)]
    print("  G_n/G_0 measured:", np.array2string(np.array(gcoef) / gcoef[0], precision=4))
    print("  G_n/G_0 r^{n^2} :", np.array2string(np.array(pred_ratio), precision=4))
    theta4 = sum((-1) ** n * r ** (n * n) for n in range(-20, 21))
    print("  theta_4(0,r)=%.4f  => G(pi)/G_0 predicted=%.4f vs measured %.4f"
          % (theta4, theta4, Gpi / gcoef[0]))

    print("  --- compare symbol-at-pi vs true lambda_min/D over K ---")
    print("  %2s | %10s %10s | %10s %10s | %8s %8s" %
          ("K", "lam_true", "G(pi)approx", "D_true", "L(pi)approx", "S_true", "S_symbol"))
    for K in range(3, 11):
        Gk, Lk = matrices(K, rho)
        v, lam = worst_eig(Gk, K)
        D = v @ Lk @ v
        S_true = math.sqrt(max(1 - D / lam, 0))
        S_sym = math.sqrt(max(1 - Lpi / Gpi, 0))
        print("  %2d | %10.4e %10.4e | %10.4e %10.4e | %8.4f %8.4f"
              % (K, lam, Gpi, D, Lpi, S_true, S_sym))
