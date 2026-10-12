# -*- coding: utf-8 -*-
"""
probe_lammin_variational6.py -- derive the local exponent p(rho) ANALYTICALLY from the theta-function
descent, and confirm 'a' ~ pi^2/4 over a finer rho grid. Also confirm the closed-form lam_min(K,rho).

CLOSED FORM (variational squeeze):
  lam_min(K,rho) ~ A(rho) * g_K(beta),   beta=exp(-L),  L=2 erfinv(rho)^2 = (dmu/sigma)^2/4,
  where the cosine-Rayleigh upper bound gives the explicit
     g_K(beta) = sum_{m=-(K-1)}^{K-1} (-1)^m beta^{m^2} * a_K(m),   a_K = autocorr of Nyquist cosine.
  As K->inf, g -> vartheta_4(0,beta) (the floor).  The PRE-ASYMPTOTIC value at finite K is what is fit
  to (K-1)^{-p}.  Local exponent:
     p(K) = - d ln lam_min / d ln(K-1).
  Using the cosine-Rayleigh model on the GRAM (which we showed is tight), we extract p and test
     p(rho) = a / L + b  with a -> pi^2/4 = 2.467 ?  (theory: the gap pi - theta_max = pi/K controls
     the descent; ln f(pi-pi/K) - ln f(pi) ~ (1/(4L_eff)) (pi/K)^2 ... mapping (K)^-2 onto (K-1)^-p
     over a window gives p ~ const/L.)

We map p over rho in {0.30,0.33,0.36,0.40,0.43,0.47,0.51,0.55} and fit p=a/L+b and p=a/L (R2 each).
Also report the cosine-Rayleigh model's OWN exponent p_UB(rho) (purely analytic, no eig solve) to show
the variational upper bound REPRODUCES p(rho) by itself.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational6.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 5000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def Lof(rho): return 2.0 * erfinv(rho) ** 2


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.exp(-0.5 * ((TGRID[None, :] - mus[:, None]) / SIGMA) ** 2)
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def gram(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    return np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])


def tangent_lammin(G):
    K = G.shape[0]
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def cheb(K, deg):
    i = np.arange(K)
    v = np.cos(deg * math.pi * (i + 0.5) / K)
    v = v - v.mean(); return v / np.linalg.norm(v)


def R_cheb(K, rho):
    G = gram(K, rho); v = cheb(K, K - 1)
    return float(v @ G @ v)


rhos = [0.30, 0.33, 0.36, 0.40, 0.43, 0.47, 0.51, 0.55]

print("=" * 100)
print("p(rho): true lam_min exponent vs cosine-Rayleigh UPPER-BOUND exponent (both over K=2..10)")
print("=" * 100)
Ls, p_true, p_ub = [], [], []
Ks = np.arange(2, 11)
for rho in rhos:
    L = Lof(rho)
    lmt = np.array([tangent_lammin(gram(K, rho)) for K in Ks])
    lub = np.array([R_cheb(K, rho) for K in Ks])
    pt = -np.polyfit(np.log(Ks - 1)[1:], np.log(lmt[1:] / lmt[0]), 1)[0]
    pu = -np.polyfit(np.log(Ks - 1)[1:], np.log(lub[1:] / lub[0]), 1)[0]
    Ls.append(L); p_true.append(pt); p_ub.append(pu)
    print("rho=%.2f L=%.4f  p_true=%.3f  p_upperbound=%.3f  (UB reproduces p? diff=%+.3f)" %
          (rho, L, pt, pu, pu - pt))
Ls = np.array(Ls); p_true = np.array(p_true); p_ub = np.array(p_ub)

print()
print("FITS  (L = 2 erfinv(rho)^2):")
for name, y in [("p_true", p_true), ("p_upperbound", p_ub)]:
    A = np.vstack([1.0 / Ls, np.ones_like(Ls)]).T
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ co) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # drop the saturated rho=0.55 (p<~0.4) where power-law itself breaks
    mask = y > 0.5
    Am = np.vstack([1.0 / Ls[mask], np.ones(mask.sum())]).T
    com, *_ = np.linalg.lstsq(Am, y[mask], rcond=None)
    r2m = 1 - ((y[mask] - Am @ com) ** 2).sum() / ((y[mask] - y[mask].mean()) ** 2).sum()
    print("  %s = %.4f/L + %.4f  R2=%.4f   [drop p<0.5: %.4f/L + %.4f R2=%.4f]  (pi^2/4=%.4f)" %
          (name, co[0], co[1], r2, com[0], com[1], r2m, math.pi ** 2 / 4))

print()
print("=" * 100)
print("FINAL closed-form check: lam_min(K,rho) ~ C(rho) (K-1)^{-p(rho)}, p=a/L+b with a,b from p_true fit.")
print("  Quote per-cell % match over K=2..10, rho in the 5 target values.")
print("=" * 100)
A = np.vstack([1.0 / Ls, np.ones_like(Ls)]).T
co, *_ = np.linalg.lstsq(A, p_true, rcond=None)
a, b = co
targets = [0.30, 0.35, 0.40, 0.47, 0.55]
for rho in targets:
    L = Lof(rho)
    p = a / L + b
    lmt = np.array([tangent_lammin(gram(K, rho)) for K in Ks])
    C = lmt[0]  # anchor at K=2 (where (K-1)^{-p}=1)
    pred = C * (Ks - 1.0) ** (-p)
    res = 100 * (pred - lmt) / lmt
    print("rho=%.2f L=%.4f p=%.3f C=%.4e" % (rho, L, p, C))
    print("   K:        " + " ".join("%9d" % K for K in Ks))
    print("   true:     " + " ".join("%9.3e" % x for x in lmt))
    print("   pred:     " + " ".join("%9.3e" % x for x in pred))
    print("   resid%%:   " + " ".join("%+9.1f" % x for x in res))
    print("   median|resid|=%.1f%%  max|resid|=%.1f%%" % (np.median(np.abs(res)), np.max(np.abs(res))))
    print()
