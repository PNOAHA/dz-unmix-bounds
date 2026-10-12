# -*- coding: utf-8 -*-
"""
probe_lammin_variational2.py -- VARIATIONAL bound with explicit Chebyshev/cosine trial vector
and matching lower bound via the Toeplitz SYMBOL.

THEORY
------
Equally spaced area-normalized Gaussians h_i = N(mu_0 + i*dmu, sigma).
For a roughly-uniform mixture envelope q the weight-Fisher Gram is approximately Toeplitz:
    G_ij ~ c(|i-j|),   c(m) = ∫ h_i h_{i+m}/q dt.
The numerator h_i h_{i+m} is a Gaussian product:  h_i h_{i+m} ∝ exp(-(m dmu)^2/(4 sigma^2)) * N(midpoint, sigma/sqrt2).
So before dividing by q,  ∫ h_i h_{i+m} dt = (1/(2 sqrt(pi) sigma)) * exp(-(m dmu)^2/(4 sigma^2)) = A * beta^{m^2},
   where beta = exp(-dmu^2/(4 sigma^2)) = the ADJACENT-PAIR overlap factor (Gaussian, NOT the KS rho).
Dividing by q (slowly varying, ~ const where mass sits) keeps the Gaussian-in-m decay:  c(m) ~ const * beta^{m^2}.

A symmetric Toeplitz matrix with generating coefficients c(m) has, on the K-point lattice, eigenvalues
approximately equal to the SYMBOL sampled at frequencies theta = pi*j/(K-1) (Dirichlet/Neumann modes):
    f(theta) = c(0) + 2 sum_{m>=1} c(m) cos(m theta).
The sum-zero tangent EXCLUDES theta=0 (the all-ones mode), and the SMALLEST tangent eigenvalue is
the symbol at the HIGHEST attainable frequency theta_max = pi*(K-1)/K  (the degree-(K-1) Chebyshev/cosine
contrast). Hence:
    lambda_min(K) ~ f(theta_max),  theta_max = pi (K-1)/K  ->  pi  as K->inf.

For small beta (well separated, large rho) the symbol is dominated by the m=1 term and
    f(theta) ~ c0 + 2 c1 cos theta,  lambda_min ~ c0 - 2 c1 |cos(pi/K)| ~ c0 - 2c1 (1 - pi^2/(2K^2)).
This does NOT give a power law -> small p (rho large). For beta near 1 (strong overlap, small rho) many
m contribute, the symbol near theta=pi behaves like a fractional power of (pi-theta), giving the (K-1)^{-p}
law. We test the symbol prediction directly and read off p.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational2.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 4000.0 + 1e-9, 0.5)
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
    return np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])


def tangent_lammin(G):
    K = G.shape[0]
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]], V[:, tang[0]]


def chebyshev_trial(K, deg):
    i = np.arange(K)
    v = np.cos(deg * math.pi * (i + 0.5) / K)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-14 else v


def rayleigh(G, v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    if n < 1e-14:
        return float("nan")
    v = v / n
    return float(v @ G @ v)


# ---- Toeplitz coefficients from a uniform-weight central row (clean Toeplitz) ----
def toeplitz_coeffs(rho, Kbig=41):
    h = endmembers(Kbig, rho)
    w = np.ones(Kbig) / Kbig
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    mid = Kbig // 2
    c = np.array([TRAPZ(h[mid] * h[mid + m] / q, TGRID) for m in range(0, Kbig - mid)])
    return c  # c[m], m=0,1,2,...


def symbol(c, theta):
    m = np.arange(1, len(c))
    return c[0] + 2.0 * np.sum(c[m] * np.cos(m * theta))


if __name__ == "__main__":
    rhos = [0.30, 0.35, 0.40, 0.47, 0.55]

    print("=" * 110)
    print("PART B1: Chebyshev (degree K-1) trial Rayleigh quotient  vs  true tangent lambda_min")
    print("=" * 110)
    for rho in rhos:
        print("-- rho=%.2f  beta=exp(-dmu^2/4sig^2)=%.4f" % (rho, math.exp(-(dmu(rho)/SIGMA)**2/4)))
        print("  %2s | %12s %12s %8s | %12s" % ("K", "lam_min_true", "R[Cheb K-1]", "ratio", "best_deg"))
        for K in range(2, 11):
            G = gram(K, rho)
            lm, vtrue = tangent_lammin(G)
            R = rayleigh(G, chebyshev_trial(K, K - 1))
            # also scan all degrees to find which Chebyshev degree minimizes Rayleigh
            best = min(((rayleigh(G, chebyshev_trial(K, d)), d) for d in range(1, K)),
                       key=lambda x: x[0])
            print("  %2d | %12.4e %12.4e %8.3f | deg=%d R=%.4e" %
                  (K, lm, R, R / lm, best[1], best[0]))
        print()

    print("=" * 110)
    print("PART B2: Toeplitz SYMBOL prediction  lambda_min ~ f(theta_max), theta_max=pi(K-1)/K")
    print("=" * 110)
    for rho in rhos:
        c = toeplitz_coeffs(rho)
        print("-- rho=%.2f  c[0..5]=%s" % (rho, " ".join("%.3e" % x for x in c[:6])))
        print("  %2s | %12s %12s %8s" % ("K", "lam_min_true", "f(theta_max)", "ratio"))
        lms, syms = [], []
        for K in range(2, 11):
            G = gram(K, rho)
            lm, _ = tangent_lammin(G)
            th = math.pi * (K - 1) / K
            f = symbol(c, th)
            lms.append(lm); syms.append(f)
            print("  %2d | %12.4e %12.4e %8.3f" % (K, lm, f, f / lm))
        Ks = np.arange(2, 11)
        p_true = -np.polyfit(np.log(Ks - 1)[1:], np.log(np.array(lms)[1:] / lms[0]), 1)[0]
        p_sym = -np.polyfit(np.log(Ks - 1)[1:], np.log(np.array(syms)[1:] / syms[0]), 1)[0]
        print("  p_true=%.3f  p_symbol=%.3f" % (p_true, p_sym))
        print()
