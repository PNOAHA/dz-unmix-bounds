# -*- coding: utf-8 -*-
"""
probe_ptrue_map.py -- map the TRUE local exponent p(rho) of lambda_min ~ (K-1)^{-p}
over K=2..10 (the measured window), directly from the real Fisher operator
G_ij = int h_i h_j / q (endpoint w).  Then fit p(rho) to candidate closed forms.

Candidates (s = -ln r = 2 erfinv(rho)^2, and z=-1/ln r = 1/s):
  (C1) p = a + b/s          (p ~ 1/s, diverges as rho->0)
  (C2) p = a + b*z, z=1/s   (same)
  (C3) p = a*s^{-c}
  (C4) p = a + b*(-ln(1-rho))   etc.
Report R2 for each. Also report the directly-usable interpolation table.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_ptrue_map.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 4000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def lammin(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


rhos = [0.50, 0.47, 0.45, 0.42, 0.40, 0.37, 0.35, 0.32, 0.30, 0.27, 0.25]
ps, ss, zs = [], [], []
print("%6s %8s %8s %8s" % ("rho", "s=-lnr", "1/s", "p_true(2-10)"))
for rho in rhos:
    r = math.exp(-dmu(rho) ** 2 / (4 * SIGMA ** 2)); s = -math.log(r)
    Ks = np.arange(2, 11)
    lm = np.array([lammin(K, rho) for K in Ks])
    p = -np.polyfit(np.log(Ks - 1)[1:], np.log(lm[1:] / lm[0]), 1)[0]
    ps.append(p); ss.append(s); zs.append(1.0 / s)
    print("%6.2f %8.4f %8.4f %8.3f" % (rho, s, 1.0 / s, p))

ps = np.array(ps); ss = np.array(ss); zs = np.array(zs)


def fit(X, name):
    coef, *_ = np.linalg.lstsq(X, ps, rcond=None)
    pred = X @ coef
    r2 = 1 - ((ps - pred) ** 2).sum() / ((ps - ps.mean()) ** 2).sum()
    print("  %-22s coef=%s  R2=%.5f" % (name, np.round(coef, 4), r2))
    return coef, r2


print("\nFits of p_true(rho):")
fit(np.vstack([np.ones_like(zs), zs]).T, "p = a + b/s")
fit(np.vstack([np.ones_like(ss), ss]).T, "p = a + b*s")
# p = a*s^{-c} -> ln p = ln a - c ln s
ca = np.polyfit(np.log(ss), np.log(ps), 1)
predlp = np.polyval(ca, np.log(ss))
r2 = 1 - ((np.log(ps) - predlp) ** 2).sum() / ((np.log(ps) - np.log(ps).mean()) ** 2).sum()
print("  p = a*s^{-c}           a=%.4f c=%.4f  R2(loglog)=%.5f" % (math.exp(ca[1]), -ca[0], r2))
# the cleanest physical: 1/s = 1/(2 erfinv(rho)^2).  Report p*s (should be ~const if p~1/s)
print("\n  p*s (const if p~1/s):", np.round(ps * ss, 3))
print("  p*sqrt(s):", np.round(ps * np.sqrt(ss), 3))
