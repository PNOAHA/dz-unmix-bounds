# -*- coding: utf-8 -*-
"""
probe_lammin_variational3.py -- pin the SYMBOL structure and derive p(rho) in closed form.

Findings to verify:
(1) Toeplitz coeffs c(m) ~ A * beta^{m^2},  beta = exp(-dmu^2/(4 sigma^2)).
    (Numerator overlap is EXACTLY Gaussian in m; division by q is a slowly varying envelope.)
(2) Symbol f(theta) = c0 + 2 sum_m c(m) cos(m theta) = theta-Jacobi-theta-like; near theta=pi
    f(pi) = c0 + 2 sum (-1)^m c(m) = sum_m (-1)^m c(|m|)  (full two-sided) = a THETA function.
    Two-sided generating series  sum_{m=-inf}^{inf} beta^{m^2} e^{i m theta} = theta_3(theta/2, beta).
    Near theta=pi this is theta_3 at the "half" point => exponentially small in 1/|ln beta|:
        f(pi) ~ A * sqrt(pi / |ln beta|) * exp( - pi^2 / (4 |ln beta|) )   (Poisson summation / Jacobi imaginary transform)
    And theta_max = pi(K-1)/K = pi - pi/K, so the deviation from pi controls the K-dependence:
        f(theta_max) = f(pi - pi/K).
    Jacobi-theta Gaussian-tail argument near theta=pi gives a Gaussian bump of width ~ sqrt(|ln beta|):
        ln f(theta) ~ ln f(pi) + (theta-pi)^2 / (4 |ln beta|)   [curvature of theta-fn log at theta=pi]
    so  f(theta_max) ~ f(pi) * exp( (pi/K)^2 / (4|ln beta|) ).  For large K this -> f(pi); power law arises
    in INTERMEDIATE K where (theta-pi) is order 1.  The cleaner statement: the symbol IS a Gaussian in
    (theta-pi) on log scale with curvature 1/(4 |ln beta|).  Map (K-1)^{-p} onto this.

Better route for p(rho): the symbol near theta=pi is  g(phi) := f(pi - phi) with
    ln g(phi) = ln g(0) + phi^2/(4 L) + ...,   L = |ln beta| = dmu^2/(4 sigma^2).
With phi = pi/K and (K-1)^{-p} fit, p is an EFFECTIVE local slope. We instead directly fit
   ln lam_min  vs  ln(K-1)  AND  vs the symbol model, and report L=dmu^2/(4 sigma^2) as the controlling scale.

This script: (A) verify c(m) ~ beta^{m^2}; (B) verify f(theta) is a theta-function & its curvature at pi;
(C) the clean closed form for p(rho).
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational3.py
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


def toeplitz_coeffs(rho, Kbig=51):
    h = endmembers(Kbig, rho)
    w = np.ones(Kbig) / Kbig
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    mid = Kbig // 2
    return np.array([TRAPZ(h[mid] * h[mid + m] / q, TGRID) for m in range(0, Kbig - mid)])


rhos = [0.30, 0.35, 0.40, 0.47, 0.55]

print("=" * 100)
print("PART C1: verify c(m) ~ A * beta^{m^2},  L=ln(1/beta)=dmu^2/(4 sig^2)")
print("=" * 100)
for rho in rhos:
    c = toeplitz_coeffs(rho)
    L = (dmu(rho) / SIGMA) ** 2 / 4.0  # = ln(1/beta)
    # fit ln c(m) = ln A - L_eff * m^2   for m=1..6 (where c is reliable)
    m = np.arange(1, 7)
    y = np.log(c[1:7])
    A = np.vstack([np.ones(len(m)), -(m ** 2.0)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    L_eff = coef[1]
    print("rho=%.2f  L=dmu^2/4sig^2=%.4f   L_eff(fit ln c ~ -L m^2)=%.4f   ratio=%.3f" %
          (rho, L, L_eff, L_eff / L))
    # show residual of pure Gaussian-in-m model
    pred = A @ coef
    print("        c(m) m=1..6: %s" % " ".join("%.3e" % x for x in c[1:7]))
    print("        resid (lnc-pred): %s" % " ".join("%+.3f" % r for r in (y - pred)))

print()
print("=" * 100)
print("PART C2: symbol near theta=pi is a THETA-function; curvature of ln f at pi ~ 1/(4L)")
print("    Jacobi: sum beta^{m^2} cos(m theta) ; near pi  ln f(pi-phi) = ln f(pi) + kappa*phi^2 + ...")
print("=" * 100)
for rho in rhos:
    c = toeplitz_coeffs(rho)
    L = (dmu(rho) / SIGMA) ** 2 / 4.0
    def f(theta):
        m = np.arange(1, len(c))
        return c[0] + 2.0 * np.sum(c[m] * np.cos(m * theta))
    # numerical curvature of ln f at theta=pi
    dphi = 0.02
    lp = math.log(f(math.pi - dphi)); l0 = math.log(f(math.pi)); lpp = math.log(f(math.pi + dphi))
    kappa = (lp - 2 * l0 + lpp) / (2 * dphi ** 2)  # d2/dphi2 ln f at pi (phi measured from pi)
    print("rho=%.2f  L=%.4f  1/(4L)=%.4f   kappa(num d2 ln f/dphi2 @pi)=%.4f   ratio=%.3f"
          % (rho, L, 1.0 / (4 * L), kappa, kappa / (1.0 / (4 * L))))

print()
print("=" * 100)
print("PART C3: CLOSED FORM for lambda_min and p(rho)")
print("  Model:  lam_min(K) = f(pi) * exp( kappa * (pi/K)^2 ),  kappa = 1/(4L),  L=dmu^2/(4 sig^2)")
print("    => ln lam_min = ln f(pi) + (pi^2/(4L)) / K^2")
print("  Effective local power p(K,rho) = -d ln lam_min / d ln(K-1).")
print("  Asymptotic large-K:  p -> 0 (saturation).  Intermediate K is what the (K-1)^{-p} fit sees.")
print("=" * 100)
for rho in rhos:
    c = toeplitz_coeffs(rho)
    L = (dmu(rho) / SIGMA) ** 2 / 4.0
    def f(theta):
        m = np.arange(1, len(c))
        return c[0] + 2.0 * np.sum(c[m] * np.cos(m * theta))
    fpi = f(math.pi)
    kappa = 1.0 / (4 * L)
    # model prediction over K=2..10
    Ks = np.arange(2, 11)
    model = np.array([fpi * math.exp(kappa * (math.pi / K) ** 2) for K in Ks])
    # true
    lms = []
    for K in Ks:
        G = gram(K, rho); lm, _ = tangent_lammin(G); lms.append(lm)
    lms = np.array(lms)
    p_true = -np.polyfit(np.log(Ks - 1)[1:], np.log(lms[1:] / lms[0]), 1)[0]
    p_model = -np.polyfit(np.log(Ks - 1)[1:], np.log(model[1:] / model[0]), 1)[0]
    print("rho=%.2f  L=%.4f  p_true=%.3f  p_model(symbol)=%.3f   f(pi)=%.3e" %
          (rho, L, p_true, p_model, fpi))
