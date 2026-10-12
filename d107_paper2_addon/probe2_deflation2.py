# -*- coding: utf-8 -*-
"""
probe2_deflation2.py -- exact slope identity for b and the TRUE meaning of 0.679.

The LS slope of exc ~ a + b ln(K-1) over a window equals the L-WEIGHTED covariance form, but
to leading order b ~ mean of d exc/dL (L=ln(K-1)).  Exactly:
   d exc/dL = exc*(p/2 - s_S(K)),  s_S = -dlnS/dL  (S's local log-slope).
So  b_inst := <d exc/dL>_window = (p/2)<exc> - <exc*s_S>.
The prompt writes b = K0*(p/2)<exc> with K0=0.679.  Thus
   K0 = 1 - <exc*s_S>/((p/2)<exc>) = 1 - (<exc*s_S>/<exc>)/(p/2).
Define the exc-weighted S-slope  qS_w := <exc*s_S>/<exc>.  Then K0 = 1 - qS_w/(p/2) = (p/2 - qS_w)/(p/2).
This is the prompt's form with q_S := qS_w (exc-weighted, not plain mean).

BUT the consolidated 0.679 (probe_reconcile3) is a GLOBAL regression slope of b_fit vs (p/2)<exc>
ACROSS rho, i.e. K0 = sum(b*x)/sum(x^2), x=(p/2)<exc>.  That is dominated by small-rho (large b).
So two distinct '0.679-type' numbers exist:
  (i)  per-rho K0(rho) = b/((p/2)<exc>)   -- NOT constant (we saw ~0.39-0.58).
  (ii) global LS coeff across rho          -- = 0.679 (the published one).

This script computes BOTH and the exact b_inst identity, and reports the honest finding:
the 0.679 is a GLOBAL fit coefficient (regression-leverage weighted), q_S~0.24 is the (real,
rho-flat) S log-slope, but (p/2-q_S)/(p/2) is NOT a clean per-rho law because S's slope q_S and
CR's slope p/2 are different functions of rho (q_S flat, p/2 steepens) -> the ratio cannot be const.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_deflation2.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f


def crs(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    g = (v[:, None] * h).sum(0) / q
    gs = smooth(g)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    var_g = Eq(g * g) - Eq(g) ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / var_g)
    return lam_min, S


KS = np.arange(2, 8)
RHOS = [0.30, 0.35, 0.40, 0.47, 0.55]
rows = []
for rho in RHOS:
    lm = np.array([crs(int(K), rho)[0] for K in KS])
    S = np.array([crs(int(K), rho)[1] for K in KS])
    CR = np.sqrt(lm[0] / lm); exc = CR * S
    L = np.log(KS - 1)
    A = np.vstack([np.ones(len(L)), L]).T
    c, *_ = np.linalg.lstsq(A, exc, rcond=None); b = c[1]
    p = 2 * np.polyfit(L[1:], np.log(CR[1:]), 1)[0]
    lnS = np.log(S); sS = -np.gradient(lnS, L)
    excm = exc.mean()
    qS_plain = sS.mean()
    qS_w = (exc * sS).sum() / exc.sum()
    # exact instantaneous-slope identity for b:
    b_inst = (exc * (p / 2 - sS)).mean()
    rows.append(dict(rho=rho, b=b, p=p, excm=excm, qS_plain=qS_plain, qS_w=qS_w,
                     b_inst=b_inst, x=(p / 2) * excm))

print("=" * 100)
print("Exact slope identity  b_inst = <exc*(p/2 - s_S)>  vs LS-fit b ; q_S forms")
print("=" * 100)
print("%5s | %7s %7s | %6s %6s | %7s %7s | %6s %6s"
      % ("rho", "b_LS", "b_inst", "p", "<exc>", "qS_pl", "qS_w", "K0=b/x", "(p/2-qSw)/(p/2)"))
for r in rows:
    K0 = r["b"] / r["x"]
    defl = (r["p"] / 2 - r["qS_w"]) / (r["p"] / 2)
    print("%5.2f | %7.3f %7.3f | %6.3f %6.3f | %7.3f %7.3f | %6.3f %6.3f"
          % (r["rho"], r["b"], r["b_inst"], r["p"], r["excm"], r["qS_plain"], r["qS_w"], K0, defl))

# global regression coefficient (the published 0.679):
b = np.array([r["b"] for r in rows]); x = np.array([r["x"] for r in rows])
K0_global = (b * x).sum() / (x * x).sum()
r2 = 1 - ((b - K0_global * x) ** 2).sum() / ((b - b.mean()) ** 2).sum()
print("\nGLOBAL regression  b = K0 * (p/2)<exc>  ->  K0 = %.4f  (R2=%.4f)  [the published 0.679]"
      % (K0_global, r2))
qS_all = np.array([r["qS_w"] for r in rows])
print("exc-weighted q_S: mean=%.3f std=%.3f  (claim ~0.24)" % (qS_all.mean(), qS_all.std()))
print("plain      q_S: mean=%.3f std=%.3f" % (np.array([r['qS_plain'] for r in rows]).mean(),
                                              np.array([r['qS_plain'] for r in rows]).std()))
# Does the global K0 equal 1 - <qS_w>/<p/2> using leverage (x)-weighted means?
pov2 = np.array([r["p"] / 2 for r in rows]); excm = np.array([r["excm"] for r in rows])
# leverage-weighted (weight x^2... actually b*x/x^2). Use the identity sum(b x)/sum(x^2) with b=x - <exc sS>:
num_def = (x * (excm * qS_all)).sum()   # sum x * <exc*sS>  approx; check
print("\nDecomp of global K0: K0 = 1 - [sum_rho x*(<exc*sS>)]/[sum_rho x^2]")
ssX = np.array([(r["qS_w"] * r["excm"]) for r in rows])   # <exc*sS> = qS_w*<exc>
K0_decomp = 1 - (x * ssX).sum() / (x * x).sum()
print("   = 1 - %.4f = %.4f   (vs global %.4f)" % ((x * ssX).sum() / (x * x).sum(), K0_decomp, K0_global))
