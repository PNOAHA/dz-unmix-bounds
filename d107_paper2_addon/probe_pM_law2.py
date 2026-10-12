# -*- coding: utf-8 -*-
"""
probe_pM_law2.py -- p_M increases as r->1 (s->0).  Test p_M ~ c/s and log-log p_M vs s.
Also: is the decay of vMv(K) actually GEOMETRIC (vMv ~ rho_g^{2K}) rather than power?
A geometric decay vMv ~ A * g^{K} masquerades as a LOCAL power (K-1)^{-p} whose fitted
p depends on the K-window and on g.  Test that: fit vMv to g^K (geometric) and to (K-1)^{-p}.

The Gaussian-Gram smallest eigenvalue for a banded/totally-positive kernel is known to
decay GEOMETRICALLY: lam_min(K) ~ C * t^{2(K-1)} with t set by r.  Identify t(r).

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_pM_law2.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA = 25.0
m0 = 1.0 / (2 * math.sqrt(math.pi) * SIGMA)


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def vMv_min(K, r):
    M = np.array([[m0 * r ** ((i - j) ** 2) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Mt = P @ M @ P
    lam, V = np.linalg.eigh(Mt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


print("Test geometric decay vMv(K) ~ C * g^(K-1).  g := vMv(K)/vMv(K-1) ratio.")
for rho in (0.47, 0.40, 0.35, 0.30):
    d = dmu(rho); r = math.exp(-d ** 2 / (4 * SIGMA ** 2)); s = -math.log(r)
    print(f"\n--- rho={rho}  r={r:.5f}  s=-ln r={s:.4f}  r^2={r*r:.5f} ---")
    print("%2s | %12s | %8s | %10s" % ("K", "vMv", "ratio g", "g vs r^2"))
    prev = None
    gs = []
    for K in range(2, 16):
        v = vMv_min(K, r)
        g = v / prev if prev else float('nan')
        if prev: gs.append(g)
        print("%2d | %12.5e | %8.5f | %10.5f" % (K, v, g, (g / (r * r)) if prev else float('nan')))
        prev = v
    gs = np.array(gs)
    # asymptotic g (last few)
    g_inf = gs[-4:].mean()
    print("  asymptotic ratio g_inf=%.5f   r^2=%.5f   g_inf/r^2=%.4f   sqrt(g_inf)=%.5f"
          % (g_inf, r * r, g_inf / (r * r), math.sqrt(g_inf)))

print("\n\n=== map g_inf(r): is g_inf = r^2 (i.e. vMv ~ (r^2)^K) ? ===")
print("%6s %8s %10s %10s %10s" % ("rho", "r", "g_inf", "r^2", "g_inf/r^2"))
for rho in (0.50, 0.47, 0.45, 0.40, 0.37, 0.35, 0.32, 0.30, 0.27, 0.25):
    d = dmu(rho); r = math.exp(-d ** 2 / (4 * SIGMA ** 2))
    vs = [vMv_min(K, r) for K in range(10, 18)]
    gs = [vs[i + 1] / vs[i] for i in range(len(vs) - 1)]
    g_inf = np.mean(gs[-4:])
    print("%6.2f %8.4f %10.5f %10.5f %10.4f" % (rho, r, g_inf, r * r, g_inf / (r * r)))
