# -*- coding: utf-8 -*-
"""
probe_closedform_lammin.py -- a FULLY closed-form predictor of lammin(G,K,rho) and its residuals
vs the true table. No fitting per (K,rho) -- only 2 global constants (alpha not used; we use the
EXACT symbol f_M, plus the reweight power-law <1/q> ~ kappa*(K-1)^beta with 2 global constants).

PREDICTOR:
  lammin_hat(K,rho) = f_M(theta_K, rho) * R(K)
    f_M(theta)=c sqrt(pi/b) Sum_n exp(-(theta-2pi n)^2/4b),  b=2 erfinv(rho)^2, c=1/(2 sig sqrt pi)
    theta_K = pi*(K-1)/K
    R(K)    = reweight = kappa*(K-1)^beta  (the <1/q> Rayleigh enhancement; from endpoint-w plateau ~ 1/K avg)
  Calibrate kappa,beta ONCE (global, rho-independent) and report residuals across rho & K.

Also give the symbol near theta=pi in transparent closed form:
  f_M(pi - x) ~ c sqrt(pi/b) [ exp(-(pi-x)^2/4b) + exp(-(pi+x)^2/4b) ],  x=pi/K
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def b_of_rho(rho): return 2.0 * erfinv(rho) ** 2
def c_const(): return 1.0 / (2 * SIGMA * math.sqrt(math.pi))
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def fM(theta, rho, nmax=6):
    b = b_of_rho(rho)
    s = sum(math.exp(-((theta - 2 * math.pi * n) ** 2) / (4 * b)) for n in range(-nmax, nmax + 1))
    return c_const() * math.sqrt(math.pi / b) * s


def fM_2img(theta, rho):
    # transparent 2-image form
    b = b_of_rho(rho)
    return c_const() * math.sqrt(math.pi / b) * (
        math.exp(-((math.pi - (math.pi - theta)) ** 0 * 0 + (theta) ** 2) * 0)  # placeholder
    )


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


def true_lammin(K, rho):
    h = endmembers(K, rho); q = np.maximum((endpoint_w(K)[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    return tangent_lammin(G, K)[0]


# --- gather true reweight R(K)=lammin(G)/f_M(theta_K) and fit global power law ---
rhos = [0.47, 0.40, 0.35, 0.30]
allK, allLnR = [], []
data = {}
for rho in rhos:
    for K in range(2, 11):
        lg = true_lammin(K, rho)
        fm = fM(math.pi * (K - 1) / K, rho)
        R = lg / fm
        data[(rho, K)] = (lg, fm, R)
        allK.append(K); allLnR.append(math.log(R))
allK = np.array(allK, float); allLnR = np.array(allLnR)
beta, lnkappa = np.polyfit(np.log(allK - 1), allLnR, 1)
kappa = math.exp(lnkappa)
print("Global reweight fit  R(K)=lammin(G)/f_M(theta_K) = kappa*(K-1)^beta")
print("  kappa=%.2f  beta=%.4f  (R^2 of lnR vs ln(K-1) = %.4f)"
      % (kappa, beta,
         1 - ((allLnR - (beta*np.log(allK-1)+lnkappa))**2).sum() / ((allLnR-allLnR.mean())**2).sum()))

print("\nCLOSED-FORM predictor lammin_hat = f_M(pi(K-1)/K) * kappa*(K-1)^beta  -- residuals vs true:")
for rho in rhos:
    print("  === rho=%.2f  (b=%.4f) ===" % (rho, b_of_rho(rho)))
    print("   %2s | %11s %11s %8s" % ("K", "true", "pred", "err%"))
    es = []
    for K in range(2, 11):
        lg, fm, R = data[(rho, K)]
        pred = fm * kappa * (K - 1) ** beta
        e = (pred - lg) / lg * 100
        es.append(e)
        print("   %2d | %11.4e %11.4e %7.1f" % (K, lg, pred, e))
    es = np.array(es)
    print("   --> mean|err|=%.1f%%  max|err|=%.1f%%" % (np.abs(es).mean(), np.abs(es).max()))

# transparent symbol-near-pi check (2-image)
print("\n2-image symbol  f_M(pi-x)~c sqrt(pi/b)[exp(-(pi-x)^2/4b)+exp(-(pi+x)^2/4b)], x=pi/K  vs full:")
for rho in (0.47, 0.40):
    b = b_of_rho(rho)
    for K in (3, 6, 10):
        x = math.pi / K
        two = c_const()*math.sqrt(math.pi/b)*(math.exp(-(math.pi-x)**2/(4*b))+math.exp(-(math.pi+x)**2/(4*b)))
        full = fM(math.pi - x, rho)
        print("   rho=%.2f K=%2d  2img=%.4e full=%.4e ratio=%.4f" % (rho, K, two, full, two/full))
