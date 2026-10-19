# -*- coding: utf-8 -*-
"""
probe2_omega_mobius.py -- WHY omega_q^2 = omega_inf^2 (K-1)/(K+a): decompose numerator/denominator.

omega_q^2 = (v^T A v)/(v^T G v).  Test the hypothesis that the Mobius (linear-fractional in K)
shape comes from:
   v^T A v  ~  alpha * (K-1)         (curvature energy ~ number of inter-peak gaps = K-1)
   v^T G v  =  lambda_min            (already shown ~ theta descent; near-const? or ~1/(K+a)?)
   ->  omega_q^2 ~ alpha (K-1) / lambda_min.   For Mobius we'd need lambda_min ~ const*(K+a).
   BUT lambda_min FALLS with K (it's (K-1)^{-p}).  So that's not it directly.

Alternative (the REAL structure):  normalize v (|v|=1).  Then
   v^T A v = Rayleigh of A (curvature Gram) at the top mode = f_A(theta_top),
   v^T G v = Rayleigh of G at the top mode = f_G(theta_top) = lambda_min.
   omega_q^2 = f_A(theta_top)/f_G(theta_top).  Both are symbols at theta_top=K pi/(K+1) (Dirichlet).

  Write x := pi - theta_top = pi/(K+1)  (the edge gap).  Near theta=pi:
    f_G(theta) = lambda_min ~ f_G(pi) + (1/2) f_G''(pi) (theta-pi)^2 = g0 + g2 x^2
    f_A(theta) ~ a0 + a2 x^2  (A is curvature: a0 = f_A(pi) is LARGE ~ omega_inf^2*g0)
  omega_q^2 = (a0 + a2 x^2)/(g0 + g2 x^2),  x = pi/(K+1).
  As K->inf, x->0, omega_q^2 -> a0/g0 = omega_inf^2.  Finite-K correction in x^2=pi^2/(K+1)^2.
  Map x^2=pi^2/(K+1)^2 to the empirical (K-1)/(K+a): both are smooth in 1/K, so a 2-term match.

This script:
 1. Confirms v^T A v / |v|^2 (Rayleigh of A) and lambda_min separately, their K-trends.
 2. Tests omega_q^2 = (a0+a2 x^2)/(g0+g2 x^2) with x=pi/(K+1): fit a0,a2,g0,g2 and compare to
    direct symbol second derivatives; check it reproduces (K-1)/(K+a) and gives omega_inf^2=a0/g0.
 3. Reports omega_inf^2=a0/g0 vs 0.92^2 Nyq^2/kappa, and the implied 'a'.

Self-contained. English prints.
Run:  python probe2_omega_mobius.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import curve_fit

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 6000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def dgauss(t, mu, s): return -(t - mu) / s ** 2 * gauss(t, mu, s)


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    nrm = TRAPZ(h, TGRID, axis=1)[:, None]
    hp = np.stack([dgauss(TGRID, m, SIGMA) for m in mus]) / nrm
    return h / nrm, hp, mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def rayleighs(K, rho):
    h, hp, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qp = (w[:, None] * hp).sum(0)
    d = hp - h * (qp / q)[None, :]
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    A = np.array([[TRAPZ(d[i] * d[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    vAv = v @ A @ v            # |v|=1 (eigh normalized)
    omq2 = vAv / lam_min
    return vAv, lam_min, omq2


def model_mobius(K, ominf2, a):
    return ominf2 * (K - 1) / (K + a)
def model_x2(K, a0, a2, g0, g2):
    x2 = (math.pi / (K + 1)) ** 2
    return (a0 + a2 * x2) / (g0 + g2 * x2)


print("=" * 95)
print("Decompose omega_q^2 = vAv/lam_min ; test x^2=(pi/(K+1))^2 Dirichlet-edge expansion")
print("=" * 95)
for rho in (0.47, 0.40, 0.35, 0.30):
    kap = 2 * erfinv(rho) ** 2; nyq = math.pi / dmu(rho)
    Ks = np.arange(2, 14)
    vAv = []; lm = []; omq2 = []
    for K in Ks:
        a_, l_, o_ = rayleighs(int(K), rho)
        vAv.append(a_); lm.append(l_); omq2.append(o_)
    vAv = np.array(vAv); lm = np.array(lm); omq2 = np.array(omq2)
    # trends
    pA = np.polyfit(np.log(Ks - 1), np.log(vAv), 1)[0]
    pL = np.polyfit(np.log(Ks - 1)[1:], np.log(lm[1:] / lm[0]), 1)[0]
    # mobius fit
    pm, _ = curve_fit(model_mobius, Ks, omq2, p0=[omq2[-1] * 1.3, 1.0], maxfev=40000)
    r2m = 1 - ((omq2 - model_mobius(Ks, *pm)) ** 2).sum() / ((omq2 - omq2.mean()) ** 2).sum()
    # x^2 edge-expansion fit
    px, _ = curve_fit(model_x2, Ks, omq2, p0=[omq2[-1], 0.0, lm[-1], 0.0], maxfev=80000)
    r2x = 1 - ((omq2 - model_x2(Ks, *px)) ** 2).sum() / ((omq2 - omq2.mean()) ** 2).sum()
    ominf2_x = px[0] / px[2]
    print(f"\n--- rho={rho}  kappa={kap:.4f} ---")
    print("  vAv ~ (K-1)^%.3f   lam_min ~ (K-1)^%.3f" % (pA, pL))
    print("  MOBIUS  omega_q^2=%.4e*(K-1)/(K+%.3f)  R2=%.5f   omega_inf=%.5f"
          % (pm[0], pm[1], r2m, math.sqrt(pm[0])))
    print("  X2-EDGE omega_q^2=(a0+a2 x^2)/(g0+g2 x^2), x=pi/(K+1)  R2=%.5f  -> omega_inf^2=a0/g0=%.4e (omega_inf=%.5f)"
          % (r2x, ominf2_x, math.sqrt(max(ominf2_x, 0))))
    print("    a0=%.4e a2=%.4e g0=%.4e g2=%.4e ; a2/a0=%.3f g2/g0=%.3f"
          % (px[0], px[1], px[2], px[3], px[1] / px[0], px[3] / px[2]))
    print("  CHECK omega_inf(mobius)=%.5f vs 0.92*Nyq/sqrt(kappa)=%.5f  ratio=%.3f"
          % (math.sqrt(pm[0]), 0.92 * nyq / math.sqrt(kap),
             math.sqrt(pm[0]) / (0.92 * nyq / math.sqrt(kap))))
    print("  CHECK a(mobius)=%.3f vs 2.4/kappa^2=%.3f ratio=%.3f ; vs 2.0/kappa^2=%.3f ratio=%.3f"
          % (pm[1], 2.4 / kap ** 2, pm[1] / (2.4 / kap ** 2), 2.0 / kap ** 2, pm[1] / (2.0 / kap ** 2)))
