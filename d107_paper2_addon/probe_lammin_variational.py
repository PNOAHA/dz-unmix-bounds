# -*- coding: utf-8 -*-
"""
probe_lammin_variational.py -- VARIATIONAL / Rayleigh-quotient derivation of lambda_min(G,K,rho)
on the sum-zero tangent, with an explicit Chebyshev/cosine trial worst-vector, plus the matching
lower bound, and a numerical map of the rho-dependent exponent p(rho) in lambda_min ~ (K-1)^{-p}.

Does NOT overwrite compute_factors.py / analytic_worst_mode.py.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3500.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho):
    return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.exp(-0.5 * ((TGRID[None, :] - mus[:, None]) / SIGMA) ** 2)
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K)
    return w / w.sum()


def gram(K, rho, w=None):
    if w is None:
        w = endpoint_w(K)
    h = endmembers(K, rho)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    return G


def tangent_lammin(G):
    K = G.shape[0]
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    return lam[kmin], V[:, kmin]


# ----- Toeplitz / convolution approximation -----
# For equally spaced area-normalized Gaussians, and a SLOWLY varying weight envelope,
# G_ij depends mainly on |i-j| (Toeplitz). Build the "uniform-weight" Gram (w=const) which
# is exactly Toeplitz, then study its tangent eigenvalues via the symbol (DFT).

def gram_uniform(K, rho):
    return gram(K, rho, w=np.ones(K) / K)


def toeplitz_symbol(rho, mmax=40):
    """c_m = ∫ h_0 h_m / q dt for UNIFORM infinite lattice (approx by large K, central row)."""
    Kbig = 2 * mmax + 1
    h = endmembers(Kbig, rho)
    w = np.ones(Kbig) / Kbig
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    c = np.array([TRAPZ(h[mmax] * h[mmax + m] / q, TGRID) for m in range(-mmax, mmax + 1)])
    return c  # index m+mmax


def chebyshev_trial(K, deg):
    """Discrete cosine (Chebyshev) contrast of given degree on nodes i=0..K-1, mean-removed, unit norm."""
    i = np.arange(K)
    # cos( deg * pi * (i+0.5)/K )  -- DCT-II basis vector
    v = np.cos(deg * math.pi * (i + 0.5) / K)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-14 else v


def rayleigh(G, v):
    K = G.shape[0]
    v = v - v.mean()
    n = np.linalg.norm(v)
    if n < 1e-14:
        return float("nan")
    v = v / n
    return float(v @ G @ v)


if __name__ == "__main__":
    print("=" * 100)
    print("PART A: map p(rho) over rho in {0.30,0.35,0.40,0.47,0.55}, K=2..10  (true tangent lambda_min)")
    print("=" * 100)
    rhos = [0.30, 0.35, 0.40, 0.47, 0.55]
    table = {}
    for rho in rhos:
        lms = []
        for K in range(2, 11):
            G = gram(K, rho)
            lm, _ = tangent_lammin(G)
            lms.append(lm)
        table[rho] = np.array(lms)
        Ks = np.arange(2, 11)
        # fit lam/lam2 ~ (K-1)^{-p}  over full K=2..10
        p_full = -np.polyfit(np.log(Ks - 1)[1:], np.log(lms[1:] / lms[0]), 1)[0]
        # fit over K=2..7 (measured window)
        m = Ks <= 7
        p_w = -np.polyfit(np.log(Ks[m] - 1)[1:], np.log(np.array(lms)[m][1:] / lms[0]), 1)[0]
        print("rho=%.2f  dmu/sig=%.4f  lam_min(2..10)=%s" %
              (rho, dmu(rho) / SIGMA, " ".join("%.3e" % x for x in lms)))
        print("         p(full K2-10)=%.3f   p(window K2-7)=%.3f" % (p_full, p_w))
        table[rho] = (np.array(lms), p_full, p_w)

    print()
    print("p(rho) candidates:")
    print("%5s %8s %10s %10s %12s %12s %12s" %
          ("rho", "p_full", "1/rho", "-1/ln rho", "(sig/dmu)^2", "exp((dmu/s)^2/4)", "dmu/sig"))
    for rho in rhos:
        p = table[rho][1]
        b = dmu(rho) / SIGMA
        print("%5.2f %8.3f %10.3f %10.3f %12.3f %12.3f %12.4f" %
              (rho, p, 1.0 / rho, -1.0 / math.log(rho), (SIGMA / dmu(rho)) ** 2,
               math.exp((b ** 2) / 4), b))
