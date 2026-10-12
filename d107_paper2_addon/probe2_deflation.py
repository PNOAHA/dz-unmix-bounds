# -*- coding: utf-8 -*-
"""
probe2_deflation.py -- FIRST-PRINCIPLES the 0.679 deflation: b = 0.679*(p/2)*<excess>, claim
0.679 = (p/2 - q_S)/(p/2) with q_S ~ 0.24.

LOGIC.  excess(K)=CR(K)*S(K), CR=(K-1)^{p/2} (power law, exponent p/2 in (K-1)),
S=stretched-exp suppression (decreasing).  Fit excess ~ a + b ln(K-1) over K=2..7.
The LS slope b ~ window-mean of d(excess)/d ln(K-1):
   d excess/d ln(K-1) = excess * [ d ln CR/d ln(K-1) + d ln S/d ln(K-1) ]
                      = excess * [ p/2 + dlnS/dln(K-1) ].
Define the LOCAL log-slope of S:   s_S(K) := -d ln S / d ln(K-1)  (positive, S falls).
Then  d excess/dln(K-1) = excess*(p/2 - s_S),  so
   b ~ <excess*(p/2 - s_S)> ~ (p/2)<excess> * <1 - s_S/(p/2)>  ~ (p/2)<excess>*(1 - q_S/(p/2)),
with  q_S := window-mean of s_S (= -dlnS/dln(K-1)).  Then the deflation coeff:
   0.679 = 1 - q_S/(p/2) = (p/2 - q_S)/(p/2).

This script:
 1. Computes CR, S, excess on K=2..7 for several rho; gets the EXACT LS slope b.
 2. Computes q_S = window-mean of -dlnS/dln(K-1) (S's effective log-slope) for each rho;
    tests whether it's rho-flat ~0.24.
 3. Computes the predicted deflation (p/2 - q_S)/(p/2) per rho and compares to 0.679 and to
    the directly-fit ratio  b / ((p/2)<excess>).
 4. Reports whether 0.24 is the right q_S and whether the deflation is truly rho-flat.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_deflation.py
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


print("=" * 95)
print("0.679 deflation:  b = (p/2)<exc> * (1 - q_S/(p/2)),  q_S = <-dlnS/dln(K-1)>_{2..7}")
print("=" * 95)
KS = np.arange(2, 8)            # K=2..7 window
rows = []
for rho in (0.30, 0.35, 0.40, 0.47, 0.55):
    lm = np.array([crs(int(K), rho)[0] for K in KS])
    S = np.array([crs(int(K), rho)[1] for K in KS])
    CR = np.sqrt(lm[0] / lm)
    exc = CR * S
    L = np.log(KS - 1)
    # exact LS slope b of exc ~ a + b L
    A = np.vstack([np.ones(len(L)), L]).T
    c, *_ = np.linalg.lstsq(A, exc, rcond=None); b = c[1]
    # p (CR exponent): CR=(K-1)^{p/2} -> ln CR = (p/2) L ; fit slope, p = 2*slope
    p = 2 * np.polyfit(L[1:], np.log(CR[1:]), 1)[0]
    # q_S: local log-slope of S over the window, s_S(K) = -dlnS/dL ; window-mean
    lnS = np.log(S)
    dlnS = np.gradient(lnS, L)
    qS = -dlnS.mean()
    # also a single global slope of S: -d lnS/dL fit
    qS_fit = -np.polyfit(L[1:], lnS[1:], 1)[0]
    excm = exc.mean()
    defl_direct = b / ((p / 2) * excm)       # = the 0.679 (measured)
    defl_pred = (p / 2 - qS_fit) / (p / 2)   # prediction from q_S
    rows.append((rho, b, p, excm, qS, qS_fit, defl_direct, defl_pred))
    print("rho=%.2f | b=%.3f p=%.3f <exc>=%.3f | q_S(grad)=%.3f q_S(fit)=%.3f | "
          "defl_direct=%.3f defl_pred=%.3f"
          % (rho, b, p, excm, qS, qS_fit, defl_direct, defl_pred))

print("\nSUMMARY:")
qSfits = np.array([r[5] for r in rows])
defl_d = np.array([r[6] for r in rows])
defl_p = np.array([r[7] for r in rows])
print("  q_S(fit) across rho: ", " ".join("%.3f" % x for x in qSfits),
      " mean=%.3f std=%.3f" % (qSfits.mean(), qSfits.std()))
print("  defl_direct (the '0.679'): ", " ".join("%.3f" % x for x in defl_d),
      " mean=%.3f std=%.3f" % (defl_d.mean(), defl_d.std()))
print("  defl_pred (p/2-q_S)/(p/2): ", " ".join("%.3f" % x for x in defl_p),
      " mean=%.3f std=%.3f" % (defl_p.mean(), defl_p.std()))
# best single q_S that makes (p/2-q_S)/(p/2) constant? solve q_S = (p/2)(1-defl) per rho already=qS.
print("\n  implied q_S from 0.679 anchor: q_S = (p/2)*(1-0.679) per rho:")
for r in rows:
    print("    rho=%.2f  (p/2)*(1-0.679)=%.3f  (vs q_S fit %.3f)" % (r[0], (r[2]/2)*(1-0.679), r[5]))
