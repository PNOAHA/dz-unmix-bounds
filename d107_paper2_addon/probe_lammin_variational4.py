# -*- coding: utf-8 -*-
"""
probe_lammin_variational4.py -- the CLEAN variational closed form.

We have established (probe3, exact to machine precision):
    Toeplitz couplings  c(m) = A * q^{... } ... actually c(m) = A * beta^{m^2},  beta = exp(-L), L = dmu^2/(4 sig^2).
So the (uniform-weight) Gram is the Gaussian-Toeplitz matrix  G_ij = A beta^{(i-j)^2}.

VARIATIONAL UPPER BOUND with the highest cosine contrast v_k(i)=cos(pi (K-1)(i+1/2)/K) (degree K-1):
  R(v) = sum_{i,j} v_i v_j A beta^{(i-j)^2} / sum v_i^2
       = A * [ c0 + 2 sum_{m>=1} beta^{m^2} * rho_K(m) ],  rho_K(m)=autocorrelation of v at lag m / ||v||^2.
  For the degree-(K-1) cosine (the Nyquist mode), rho_K(m) ~ (-1)^m (alternating), so
  R = A * sum_{m=-(K-1)}^{K-1} (-1)^m beta^{m^2} * (1 - |m|/K) approx (triangular taper from finite K).
  As K->inf:  R_inf = A * sum_{m=-inf}^{inf} (-1)^m beta^{m^2} = A * theta_4(0, beta)  (Jacobi theta_4!).
  theta_4(0,beta) = prod (1-beta^{2n})(1-beta^{2n-1})^2 ; and by Jacobi imaginary transform
    theta_4(0, e^{-L}) = sqrt(pi/L) * exp(-pi^2/(4L)) * theta_2-ish  ~  sqrt(pi/L) exp(-pi^2/(4L)) for small L.
  => the K=inf floor of lam_min is  A * sqrt(pi/L) exp(-pi^2/(4L)).   This is the SATURATION value.

The FINITE-K value is the TRUNCATED/tapered theta sum:
    R_K = A * sum_{m=-(K-1)}^{K-1} (-1)^m beta^{m^2} (1 - |m|/K).
This is the explicit closed-form upper bound. We test R_K against the true lam_min, and also against the
NON-uniform-q true Gram (the A and c(m) read from the actual central row).

p(rho): from R_K we get an EXPLICIT effective exponent. We show p(rho) is governed by the single scale
    L = dmu^2/(4 sigma^2) = (dmu/sigma)^2 / 4 = ln(1/beta),
and fit p(rho) as a function of L (test p ~ a/L + b, p ~ c/sqrt(L), etc.).

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational4.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 4500.0 + 1e-9, 0.5)
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


def cheb(K, deg):
    i = np.arange(K)
    v = np.cos(deg * math.pi * (i + 0.5) / K)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n


def rayleigh(G, v):
    v = v - v.mean(); v = v / np.linalg.norm(v)
    return float(v @ G @ v)


def coeffs_from_uniform(rho, Kbig=51):
    h = endmembers(Kbig, rho); w = np.ones(Kbig) / Kbig
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    mid = Kbig // 2
    return np.array([TRAPZ(h[mid] * h[mid + m] / q, TGRID) for m in range(0, Kbig - mid)])


rhos = [0.30, 0.35, 0.40, 0.47, 0.55]

print("=" * 105)
print("PART D1: explicit truncated-theta UPPER BOUND  R_K = sum_{|m|<K} (-1)^m c(m) (1-|m|/K)")
print("         vs Chebyshev Rayleigh on true Gram vs true tangent lambda_min")
print("=" * 105)
for rho in rhos:
    c = coeffs_from_uniform(rho)
    L = (dmu(rho) / SIGMA) ** 2 / 4.0
    print("-- rho=%.2f  L=%.4f  beta=%.4f" % (rho, L, math.exp(-L)))
    print("  %2s | %12s %12s %12s | %8s %8s" %
          ("K", "lam_true", "R_cheb(Gram)", "R_K(theta)", "rt_cheb", "rt_thK"))
    for K in range(2, 11):
        G = gram(K, rho)
        lm, _ = tangent_lammin(G)
        Rc = rayleigh(G, cheb(K, K - 1))
        ms = np.arange(-(K - 1), K)
        RK = np.sum([((-1) ** abs(m)) * (c[abs(m)] if abs(m) < len(c) else 0.0) * (1 - abs(m) / K) for m in ms])
        print("  %2d | %12.4e %12.4e %12.4e | %8.3f %8.3f" %
              (K, lm, Rc, RK, Rc / lm, RK / lm))
    print()

print("=" * 105)
print("PART D2: p(rho) as a function of the single scale L=dmu^2/(4 sig^2)=ln(1/beta)")
print("=" * 105)
Ls, ps = [], []
for rho in rhos:
    L = (dmu(rho) / SIGMA) ** 2 / 4.0
    Ks = np.arange(2, 11)
    lms = []
    for K in Ks:
        G = gram(K, rho); lm, _ = tangent_lammin(G); lms.append(lm)
    lms = np.array(lms)
    p = -np.polyfit(np.log(Ks - 1)[1:], np.log(lms[1:] / lms[0]), 1)[0]
    Ls.append(L); ps.append(p)
Ls = np.array(Ls); ps = np.array(ps)
print(" rho     L      p_true   pi^2/(4L)/ln? tests:")
print("%5s %8s %8s %10s %10s %10s %10s" % ("rho", "L", "p", "c1/L", "c2/sqrtL", "a-b*L", "pi^2/(4L)/<ln(K-1)>"))
# average ln(K-1) over K=3..10  ~ the effective lever arm of the K^-2 -> (K-1)^-p mapping
Kbar = np.mean(np.log(np.arange(3, 11) - 1))
for L, p, rho in zip(Ls, ps, rhos):
    print("%5.2f %8.4f %8.3f %10.3f %10.3f %10s %10.3f" %
          (rho, L, p, 1.0 / L * 0.5, 1.0 / math.sqrt(L) * 0.8, "", (math.pi ** 2 / (4 * L)) * 0.0 + 0))
# fits
print()
# p ~ a / L
A1 = np.vstack([1.0 / Ls, np.ones_like(Ls)]).T
co1, *_ = np.linalg.lstsq(A1, ps, rcond=None)
r1 = 1 - ((ps - A1 @ co1) ** 2).sum() / ((ps - ps.mean()) ** 2).sum()
print("fit  p = %.4f / L + %.4f     R2=%.4f" % (co1[0], co1[1], r1))
# p ~ a/L + b/L^2
A2 = np.vstack([1.0 / Ls, 1.0 / Ls ** 2, np.ones_like(Ls)]).T
co2, *_ = np.linalg.lstsq(A2, ps, rcond=None)
r2 = 1 - ((ps - A2 @ co2) ** 2).sum() / ((ps - ps.mean()) ** 2).sum()
print("fit  p = %.4f/L + %.4f/L^2 + %.4f   R2=%.4f" % (co2[0], co2[1], co2[2], r2))
# p ~ a*(1/L) linear through structure: recall p_true ~ pi^2/(4L)/something
A3 = np.vstack([1.0 / Ls]).T
co3, *_ = np.linalg.lstsq(A3, ps, rcond=None)
r3 = 1 - ((ps - A3 @ co3) ** 2).sum() / ((ps - ps.mean()) ** 2).sum()
print("fit  p = %.4f / L  (no intercept)  R2=%.4f   [pi^2/4=%.4f]" % (co3[0], r3, math.pi ** 2 / 4))
# p linear in 1/L with slope pi^2/4 ?  predict
print()
print(" check p ~ (pi^2/4)/L * correction :   p*L vs L")
for L, p in zip(Ls, ps):
    print("   L=%.4f  p=%.3f  p*L=%.4f  p*4L/pi^2=%.4f" % (L, p, p * L, p * 4 * L / math.pi ** 2))
