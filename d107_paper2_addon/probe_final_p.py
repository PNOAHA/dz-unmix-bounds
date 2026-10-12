# -*- coding: utf-8 -*-
"""
probe_final_p.py -- final closed form for p(rho) and a full lammin(G,K,rho) predictor.

ESTABLISHED:
  b = 2 erfinv(rho)^2 ;  Toeplitz overlap symbol f_M(theta)=c sqrt(pi/b) Sum_n exp(-(theta-2pi n)^2/4b)
  worst discrete tangent mode at theta_K ~ pi(1-1/K)
  ln lammin(M,K) ~ const - theta_K^2/(4b)  => p_M = alpha * pi^2/(4b), alpha~0.257 (rho-indep slope of (1-1/K)^2 in ln(K-1))
  G reweight subtracts an offset beta:  p_G = p_M - beta.

This script:
  (1) fit p_G(rho) = A/erfinv(rho)^2 + C  across MANY rho, get A,C (closed form).
  (2) compare predicted lammin(G,K,rho)= lammin(M,K,rho)*<1/q> reconstruction using the TRUE worst eigvec.
  (3) report residuals vs the true table over K=2..10, rho in {0.47,0.40,0.35,0.30}.
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def b_of_rho(rho): return 2.0 * erfinv(rho) ** 2
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def tangent_lammin(A, K):
    P = np.eye(K) - np.ones((K, K)) / K
    At = P @ A @ P
    lam, V = np.linalg.eigh(At)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]], V[:, tang[0]]


def true_pG(rho):
    Ks = np.arange(2, 11); lgs = []
    for K in range(2, 11):
        h = endmembers(K, rho); q = np.maximum((endpoint_w(K)[:, None] * h).sum(0), 1e-300)
        G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
        lg, _ = tangent_lammin(G, K)
        lgs.append(lg)
    lgs = np.array(lgs)
    return -np.polyfit(np.log(Ks - 1), np.log(lgs / lgs[0]), 1)[0], lgs


# (1) fit p_G(rho) = A/erfinv(rho)^2 + C
rhos = [0.50, 0.47, 0.45, 0.40, 0.38, 0.35, 0.32, 0.30]
pGs, invs = [], []
print("rho   erfinv   1/erfinv^2   p_M(theory)=alpha*pi^2/4b   true p_G")
alpha = 0.257
for rho in rhos:
    b = b_of_rho(rho)
    pM = alpha * math.pi ** 2 / (4 * b)
    pg, _ = true_pG(rho)
    pGs.append(pg); invs.append(1.0 / erfinv(rho) ** 2)
    print("  %.2f  %.4f   %7.3f      %7.3f                 %7.3f" % (rho, erfinv(rho), 1/erfinv(rho)**2, pM, pg))

invs = np.array(invs); pGs = np.array(pGs)
A, C = np.polyfit(invs, pGs, 1)
pred = A * invs + C
r2 = 1 - ((pGs - pred) ** 2).sum() / ((pGs - pGs.mean()) ** 2).sum()
print("\nFIT  p_G(rho) = A/erfinv(rho)^2 + C :  A=%.4f  C=%.4f  R2=%.4f" % (A, C, r2))
print("   theory A = alpha*pi^2/8 = %.4f  (alpha=0.257)" % (alpha * math.pi ** 2 / 8))
print("   so p_G ~ %.3f/erfinv(rho)^2 %+.3f" % (A, C))
print("   residuals (true - pred):", np.round(pGs - pred, 3))

# (2) full lammin reconstruction with TRUE eigvec Rayleigh on M and <1/q>
print("\nReconstruction lammin(G) = lammin(M)*<1/q>_{vg}  (vg=TRUE G worst eigvec): residuals")
for rho in (0.47, 0.40, 0.35):
    print("  === rho=%.2f ===" % rho)
    errs = []
    for K in range(2, 11):
        h = endmembers(K, rho); q = np.maximum((endpoint_w(K)[:, None] * h).sum(0), 1e-300)
        G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
        M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
        lg, vg = tangent_lammin(G, K)
        # Rayleigh: vg^T G vg / vg^T M vg = <1/q>, and vg^T M vg is M-Rayleigh at vg
        gnum = (vg[:, None] * h).sum(0)
        inv_q = TRAPZ(gnum ** 2 / q, TGRID) / TRAPZ(gnum ** 2, TGRID)
        M_rayl = vg @ M @ vg
        recon = M_rayl * inv_q       # = exactly vg^T G vg = lg (identity check)
        errs.append((recon - lg) / lg * 100)
    print("    recon err %%: ", np.round(errs, 2), " (identity check, should be ~0)")
