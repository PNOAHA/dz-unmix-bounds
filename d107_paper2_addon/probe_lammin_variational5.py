# -*- coding: utf-8 -*-
"""
probe_lammin_variational5.py -- matching LOWER bound + final closed form + p(rho) verification.

UPPER bound (variational): R(v_{K-1}) with cosine trial >= lambda_min (Rayleigh). Verified tight (PART B1/D1).

LOWER bound for a Gaussian-Toeplitz G_ij = A beta^{(i-j)^2} on the sum-zero tangent:
  G is a real symmetric Toeplitz PSD matrix whose symbol is
     f(theta) = A * theta_3-type = A * sum_m beta^{m^2} e^{i m theta} = A * Theta(theta),
  with Theta(theta) = vartheta_3(theta/2, beta) > 0 for all theta (PSD, since beta in (0,1)).
  Szego: for a Toeplitz matrix with continuous symbol f, ALL eigenvalues lie in [min f, max f].
  The tangent (sum-zero) subspace removes only the theta=0 (DC) eigenvector, whose Rayleigh ~ f(0)=max f.
  Hence lambda_min(tangent) >= min_theta f(theta) = f(pi) = A * vartheta_4(0, beta)   (LOWER bound).
  And lambda_min(tangent) <= f(theta_max) with theta_max = pi(K-1)/K (UPPER, highest tangent mode).
  Squeeze:  A vartheta_4(0,beta) <= lambda_min <= A * f(pi - pi/K).

The K-DEPENDENCE lives entirely in the gap [theta_max, pi]; using
  f(pi - phi) = A * sum_m (-1)^m beta^{m^2} cos(m phi)   (since cos(m(pi-phi))=(-1)^m cos(m phi))
and small phi=pi/K expansion of the theta_4 series gives the explicit lam_min(K).

Jacobi imaginary transform (Poisson summation) for the FLOOR:
  vartheta_4(0, e^{-L}) = sqrt(pi/L) * sum_n (-1)^n exp(-(pi^2/L)(n+1/2)^2) -- dominant n=0:
     ~ 2 sqrt(pi/L) exp(-pi^2/(4L)).
  => FLOOR(rho) = A * 2 sqrt(pi/L) exp(-pi^2/(4L)),   L = dmu^2/(4 sig^2).
  This is the analytic SATURATION value of lam_min as K->inf.  The (K-1)^{-p} regime is the
  pre-asymptotic descent from f(0)~A*(big) at K=2 down toward the floor.

p(rho): the descent's local log-slope. We derive p ~ (pi^2/(4L)) / <ln(K-1)> over the fit window,
i.e. p is LINEAR in 1/L. Empirically p = a/L + b. We pin a,b and the theory ratio.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational5.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 5000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho):
    return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.exp(-0.5 * ((TGRID[None, :] - mus[:, None]) / SIGMA) ** 2)
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


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
    v = v - v.mean(); return v / np.linalg.norm(v)


def rayleigh(G, v):
    v = v - v.mean(); v = v / np.linalg.norm(v); return float(v @ G @ v)


def theta4(beta, nmax=60):
    """vartheta_4(0,beta) = 1 + 2 sum (-1)^n beta^{n^2}."""
    n = np.arange(1, nmax + 1)
    return 1.0 + 2.0 * np.sum(((-1.0) ** n) * beta ** (n ** 2))


def theta4_imag(L, nmax=20):
    """Jacobi imaginary transform: vartheta_4(0,e^{-L}) = sqrt(pi/L) sum_n (-1)^n? -- use exact dual."""
    # vartheta_4(0,q=e^{-L}) = sqrt(pi/L) * vartheta_2(0, e^{-pi^2/L})
    # vartheta_2(0,Q) = 2 sum_{n>=0} Q^{(n+1/2)^2}
    Q = math.exp(-math.pi ** 2 / L)
    n = np.arange(0, nmax)
    t2 = 2.0 * np.sum(Q ** ((n + 0.5) ** 2))
    return math.sqrt(math.pi / L) * t2


rhos = [0.30, 0.35, 0.40, 0.47, 0.55]

print("=" * 100)
print("PART E1: SQUEEZE  A*theta4(beta) <= lam_min <= R_cheb(true Gram).  A = c(0) of true Gram (K-dep small).")
print("   (lower bound uses the symbol floor; we read A and beta from the GRAM diagonal & adjacent ratio)")
print("=" * 100)
for rho in rhos:
    L = (dmu(rho) / SIGMA) ** 2 / 4.0
    beta = math.exp(-L)
    print("-- rho=%.2f  L=%.4f beta=%.4f  theta4=%.4e  theta4_imag=%.4e" %
          (rho, L, beta, theta4(beta), theta4_imag(L)))
    print("  %2s | %12s %12s %12s | %s" % ("K", "A*theta4(LB)", "lam_true", "R_cheb(UB)", "LB<=true<=UB?"))
    for K in range(2, 11):
        G = gram(K, rho)
        lm, _ = tangent_lammin(G)
        # A = average diagonal of G (the c0 scale); but use the UNIFORM central c0 to be consistent
        A = G[K // 2, K // 2]
        LB = A * theta4(beta)
        UB = rayleigh(G, cheb(K, K - 1))
        ok = "OK" if (LB <= lm * 1.001 and lm <= UB * 1.001) else "VIOL"
        print("  %2d | %12.4e %12.4e %12.4e | %s (LB/t=%.2f UB/t=%.2f)" %
              (K, LB, lm, UB, ok, LB / lm, UB / lm))
    print()

print("=" * 100)
print("PART E2: FINAL p(rho) closed form.   p = a/L + b,  L=(dmu/sigma)^2/4.")
print("=" * 100)
Ls, ps, pw = [], [], []
for rho in rhos:
    L = (dmu(rho) / SIGMA) ** 2 / 4.0
    Ks = np.arange(2, 11)
    lms = np.array([tangent_lammin(gram(K, rho))[0] for K in Ks])
    p_full = -np.polyfit(np.log(Ks - 1)[1:], np.log(lms[1:] / lms[0]), 1)[0]
    m = Ks <= 7
    p_win = -np.polyfit(np.log(Ks[m] - 1)[1:], np.log(lms[m][1:] / lms[0]), 1)[0]
    Ls.append(L); ps.append(p_full); pw.append(p_win)
Ls = np.array(Ls); ps = np.array(ps); pw = np.array(pw)
for fitname, yy in [("full K2-10", ps), ("window K2-7", pw)]:
    A1 = np.vstack([1.0 / Ls, np.ones_like(Ls)]).T
    co, *_ = np.linalg.lstsq(A1, yy, rcond=None)
    pred = A1 @ co
    r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    maxres = np.max(np.abs((pred - yy) / yy)) * 100
    print("p(%s) = %.4f/L + %.4f   R2=%.4f  max|resid|=%.1f%%" % (fitname, co[0], co[1], r2, maxres))
    for rho, L, y in zip(rhos, Ls, yy):
        print("    rho=%.2f L=%.4f  p_meas=%.3f  p_pred=%.3f  resid=%+.1f%%" %
              (rho, L, y, co[0] / L + co[1], 100 * (co[0] / L + co[1] - y) / y))
print()
print("In terms of rho: dmu=2 sqrt2 erfinv(rho) sigma => L=(dmu/sigma)^2/4 = 2 erfinv(rho)^2.")
print("So  p(rho) = a / (2 erfinv(rho)^2) + b.")
for rho in rhos:
    Lr = 2.0 * erfinv(rho) ** 2
    print("    rho=%.2f  2*erfinv^2=%.4f  (==L? %.4f)" % (rho, Lr, (dmu(rho)/SIGMA)**2/4))
