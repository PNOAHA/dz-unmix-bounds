# -*- coding: utf-8 -*-
"""
probe_decomp_q.py -- exact decomposition of true lambda_min into
   lambda_min  ~  M_min(K,r) / <q>_eff
and identification of <q>_eff via the worst-mode-weighted background.

Better model than crude mean(q): the operator is  R[c] = int (delta_q)^2 / q.
Write  q = qbar * (1 + ...).  The right "effective q" for mode v is the
delta_q^2-WEIGHTED harmonic mean of q:
   1/lambda  ... no; lambda = int dq^2/q.  If q~Qeff const where dq^2 lives,
   lambda = (int dq^2)/Qeff = (c^T M c)/Qeff.
So  Qeff(K) := (v^T M v) / lambda_true(v)   where v = TRUE worst mode, M = overlap Gram.
We test whether Qeff(K) is ~ K-INDEPENDENT (then p_true = p_M) or grows/shrinks with K.

Also: M-eigenvalue worst mode vs q-operator worst mode overlap.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_decomp_q.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3500.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
m0 = 1.0 / (2 * math.sqrt(math.pi) * SIGMA)


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def analyze(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]; lam_true = lam[tang[0]]
    vMv = v @ M @ v
    Qeff = vMv / lam_true
    # delta_q^2-weighted harmonic mean of q (the exact Qeff if mode fixed):
    dq = (v[:, None] * h).sum(0)
    num = TRAPZ(dq * dq, TGRID)
    den = TRAPZ(dq * dq / q, TGRID)
    Qeff_hm = num / den
    return lam_true, vMv, Qeff, Qeff_hm, mus


print("Qeff(K) := (v^T M v)/lambda_true ;  lambda_true = (v^T M v)/Qeff")
print("If Qeff ~ K-indep -> p_true = p_M (steep). It is NOT (true p flatter).")
for rho in (0.47, 0.40, 0.35, 0.30):
    print(f"\n--- rho={rho} ---")
    print("%2s | %11s %11s | %10s %10s | %8s" %
          ("K", "lam_true", "vMv", "Qeff", "Qeff_hm", "Qeff/Q2"))
    Q2 = None
    Ks, Qs, vMvs, lams = [], [], [], []
    for K in range(2, 11):
        lam_true, vMv, Qeff, Qeff_hm, mus = analyze(K, rho)
        if K == 2: Q2 = Qeff
        Ks.append(K); Qs.append(Qeff); vMvs.append(vMv); lams.append(lam_true)
        print("%2d | %11.4e %11.4e | %10.4e %10.4e | %8.4f" %
              (K, lam_true, vMv, Qeff, Qeff_hm, Qeff / Q2))
    Ks = np.array(Ks)
    pQ = np.polyfit(np.log(Ks - 1)[1:], np.log(np.array(Qs)[1:] / Q2), 1)[0]
    pvMv = np.polyfit(np.log(Ks - 1)[1:], np.log(np.array(vMvs)[1:] / vMvs[0]), 1)[0]
    pl = np.polyfit(np.log(Ks - 1)[1:], np.log(np.array(lams)[1:] / lams[0]), 1)[0]
    print("  vMv ~ (K-1)^%.3f ;  Qeff ~ (K-1)^%.3f ;  lam_true ~ (K-1)^%.3f"
          % (pvMv, pQ, pl))
    print("  check: p_lam = p_vMv - p_Qeff = %.3f  (vs fit %.3f)" % (pvMv - pQ, pl))
