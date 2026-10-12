# -*- coding: utf-8 -*-
"""
probe2_pR_wdep.py -- is p_R a worst-mode DEFLATION of the 1/K background dilution, and
where does 3/8 enter?

Two clean experiments:
 (E1) FLAT weights w_k=1/K (uniform).  Then q is a periodic comb (away from edges) and the
      1/K background dilution is pure: beta0 = int h^2/q ~ exactly the comb 1/q.  Measure
      p_R for flat w and compare to endpoint w.  Isolates GEOMETRY from weight-imbalance.
 (E2) DEFLATION ratio:  the prompt says naive=~1 (1/K plateau in K-1 ... actually 1/K in K),
      capped to 3/8.  Test the hypothesis  p_R = (3/8) * p_beta0   OR  p_R/p_flat = const.
      Report p(<1/q>_g)/p(beta0) and p(<1/q>_g)/p(flat <1/q>) for each rho -- is the DEFLATION
      RATIO the rho-invariant 3/8 (not p_R itself)?
 (E3) the 3/8 from pairwise overlap: the worst mode is a contrast; its self term (diagonal,
      weight W0=c, K-indep) sees beta0 (~K-diluted), but the off-diagonal pair terms see beta1
      with weight -2 c m1/m0 per neighbor.  Effective fraction of beta0 surviving =
      W0/(W0 + 2|W1|... )?  Compute the "surviving diagonal fraction" and see if ->3/8.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_pR_wdep.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 6000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def w_endpoint(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()
def w_flat(K):
    return np.ones(K) / K


def metrics(K, rho, wfun):
    h, mus = endmembers(K, rho); w = wfun(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    iqg = (v @ G @ v) / (v @ M @ v)
    beta0 = np.diag(G).mean()           # per-peak self-bg
    # surviving-diagonal fraction of the worst mode: W0/(sum_k |W_k|)
    W = {}
    for k in range(K):
        ii = np.arange(K - k); jj = ii + k
        W[k] = (v[ii] * v[jj] * M[ii, jj]).sum() * (1 if k == 0 else 2)
    surv = abs(W[0]) / sum(abs(W[k]) for k in range(K))
    return iqg, beta0, surv


def expo(Ks, ys):
    Ks = np.array(Ks, float); ys = np.array(ys)
    return (np.polyfit(np.log(Ks - 1)[1:], np.log(ys[1:] / ys[0]), 1)[0],
            np.polyfit(np.log(Ks)[1:], np.log(ys[1:] / ys[0]), 1)[0])


print("=" * 95)
print("E1/E2/E3:  p_R deflation vs background dilution; surviving-diagonal fraction -> 3/8 ?")
print("=" * 95)
Krange = list(range(2, 13))
for rho in (0.47, 0.40, 0.35, 0.30):
    rowE = {K: metrics(K, rho, w_endpoint) for K in Krange}
    rowF = {K: metrics(K, rho, w_flat) for K in Krange}
    iqgE = [rowE[K][0] for K in Krange]; b0E = [rowE[K][1] for K in Krange]; survE = [rowE[K][2] for K in Krange]
    iqgF = [rowF[K][0] for K in Krange]; b0F = [rowF[K][1] for K in Krange]
    pIQE = expo(Krange, iqgE); pB0E = expo(Krange, b0E)
    pIQF = expo(Krange, iqgF); pB0F = expo(Krange, b0F)
    print(f"\n--- rho={rho} ---")
    print("  ENDPOINT w:  p(<1/q>_g)[K-1]=%.3f  p(beta0)[K-1]=%.3f  ratio=%.3f  | [K]: %.3f / %.3f ratio=%.3f"
          % (pIQE[0], pB0E[0], pIQE[0] / pB0E[0], pIQE[1], pB0E[1], pIQE[1] / pB0E[1]))
    print("  FLAT     w:  p(<1/q>_g)[K-1]=%.3f  p(beta0)[K-1]=%.3f  ratio=%.3f  | [K]: %.3f / %.3f ratio=%.3f"
          % (pIQF[0], pB0F[0], pIQF[0] / pB0F[0], pIQF[1], pB0F[1], pIQF[1] / pB0F[1]))
    print("  surviving-diagonal frac W0/sum|Wk|: K=2:%.3f K=5:%.3f K=8:%.3f K=12:%.3f"
          % (survE[0], survE[3], survE[6], survE[-1]))
