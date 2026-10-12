# -*- coding: utf-8 -*-
"""
probe_stretched.py -- the REAL functional form of vMv(K): test
    ln vMv(K) = A - c * (K-1)^alpha          (stretched exponential in K)
vs the local-power illusion.  If alpha~1 it's geometric; the local power p over a
window [Klo,Khi] of a stretched-exp is  p_local = c*alpha*(K-1)^alpha / ln(K-1)-ish,
which DRIFTS -> explains why p is not constant and why p~1/s emerges.

Then relate c, alpha to s=-ln r.  Goal: a closed ln-lambda(K;rho) that reproduces the
table to a few %, from which p(rho) over any window follows analytically.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_stretched.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import curve_fit

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 5000.0 + 1e-9, 0.5)
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


print("Fit ln lambda(K) = A - c*(K-1)^alpha for each rho (K=2..12).")
print("%6s %8s | %8s %8s %8s %8s" % ("rho", "s", "A", "c", "alpha", "c/s"))
Ks = np.arange(2, 13)
x = (Ks - 1).astype(float)
for rho in (0.50, 0.47, 0.45, 0.40, 0.37, 0.35, 0.32, 0.30, 0.27, 0.25):
    r = math.exp(-dmu(rho) ** 2 / (4 * SIGMA ** 2)); s = -math.log(r)
    lm = np.array([lammin(K, rho) for K in Ks])
    y = np.log(lm)

    def f(xx, A, c, al): return A - c * xx ** al
    try:
        popt, _ = curve_fit(f, x, y, p0=[y[0], 1.0, 1.0], maxfev=20000)
        A, c, al = popt
        resid = y - f(x, *popt)
        rms = math.sqrt((resid ** 2).mean())
    except Exception as e:
        A, c, al, rms = float('nan'), float('nan'), float('nan'), float('nan')
    print("%6.2f %8.4f | %8.3f %8.4f %8.4f %8.4f  (rms_lnlam=%.3f)" %
          (rho, s, A, c, al, c / s, rms))
