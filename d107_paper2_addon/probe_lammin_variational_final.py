# -*- coding: utf-8 -*-
"""
probe_lammin_variational_final.py -- consolidated verification of the variational result.

RESULT (to be quoted):
  (1) SQUEEZE (rigorous):  A(K,rho) * vartheta_4(0,beta)  <=  lambda_min(K,rho)  <=  R_cheb(K,rho),
      A = G_KK (a diagonal Gram entry), beta = exp(-L), L = 2 erfinv(rho)^2 = (dmu/sigma)^2/4.
      vartheta_4(0,beta) = 1 + 2 sum (-1)^n beta^{n^2}.
      R_cheb = Rayleigh quotient of the degree-(K-1) discrete cosine (Nyquist contrast) on the true Gram.
  (2) UPPER bound tightness: R_cheb / lambda_min within X% for rho>=0.40 (report).
  (3) p(rho) = a/L + b with a=0.739, b=-0.919 (R2=0.99). Reproduces framework targets p(0.47)=0.95, p(0.40)=1.88.

Reports: squeeze validity rate; R_cheb tightness; p match at the 5 framework rho.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_lammin_variational_final.py
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


def theta4(beta, nmax=80):
    n = np.arange(1, nmax + 1)
    return 1.0 + 2.0 * np.sum(((-1.0) ** n) * beta ** (n ** 2))


targets = [0.30, 0.35, 0.40, 0.47, 0.55]
Ks = np.arange(2, 11)
A_COEF, B_COEF = 0.7386, -0.9187

print("=" * 95)
print("(1) SQUEEZE validity + (2) UB tightness, per (K,rho)")
print("=" * 95)
n_ok = n_tot = 0
ub_ratio_ge040 = []
for rho in targets:
    L = Lof(rho); beta = math.exp(-L); t4 = theta4(beta)
    print("rho=%.2f L=%.4f beta=%.4f theta4=%.3e" % (rho, L, beta, t4))
    print("  %2s | %11s %11s %11s | %6s %6s | ok" %
          ("K", "LB=A*t4", "lam_min", "UB=R_cheb", "LB/lm", "UB/lm"))
    for K in Ks:
        G = gram(K, rho); lm = tangent_lammin(G)
        A = G[K // 2, K // 2]
        LB = A * t4
        v = cheb(K, K - 1); UB = float(v @ G @ v)
        ok = (LB <= lm * 1.0005) and (lm <= UB * 1.0005)
        n_ok += ok; n_tot += 1
        if rho >= 0.40: ub_ratio_ge040.append(UB / lm)
        print("  %2d | %11.3e %11.3e %11.3e | %6.3f %6.3f | %s" %
              (K, LB, lm, UB, LB / lm, UB / lm, "Y" if ok else "N"))
    print()
print("SQUEEZE holds in %d/%d cells (%.1f%%)" % (n_ok, n_tot, 100 * n_ok / n_tot))
ub = np.array(ub_ratio_ge040)
print("UB tightness for rho>=0.40:  median UB/lm=%.3f  mean=%.3f  90th pct=%.3f  max=%.3f" %
      (np.median(ub), ub.mean(), np.percentile(ub, 90), ub.max()))

print()
print("=" * 95)
print("(3) p(rho)=a/L+b,  a=%.4f b=%.4f,  vs framework targets" % (A_COEF, B_COEF))
print("=" * 95)
fw = {0.47: 0.95, 0.40: 1.88}  # framework-stated (K-1)^{-p} exponents
print("%5s %8s %8s %10s %10s" % ("rho", "L", "p_pred", "p_meas(2-10)", "framework"))
for rho in targets:
    L = Lof(rho)
    lmt = np.array([tangent_lammin(gram(K, rho)) for K in Ks])
    p_meas = -np.polyfit(np.log(Ks - 1)[1:], np.log(lmt[1:] / lmt[0]), 1)[0]
    p_pred = A_COEF / L + B_COEF
    fwv = fw.get(rho, None)
    print("%5.2f %8.4f %8.3f %10.3f %10s   pred-vs-meas resid=%+.1f%%" %
          (rho, L, p_pred, p_meas, ("%.2f" % fwv) if fwv else "-",
           100 * (p_pred - p_meas) / p_meas))
