# -*- coding: utf-8 -*-
"""
compute_sandwich.py -- FOUNDATION for the excess(K) analytic derivation.
Tests whether the M-estimator sandwich reproduces the MEASURED unpinned excess(K),
which (if it does) reduces the whole problem to the asymptotics of two analytic factors.

Model: data grains ~ q(t)=sum_k w*_k h_k(t), h_k=N(mu_k,sigma) area-normalized (raw dictionary).
KDE estimate X(t) = (1/n) sum_g K_bw(t-age_g), K_bw=N(0,bw).  Supervised KL-NMF with fixed raw H.

Sandwich:  Cov(w_hat) = G^{-1} Sigma G^{-1},
  G_ij    = ∫ h_i h_j / q                              (signal Fisher = KL Hessian)
  Sigma_ij= (1/n) Cov_{a~q}[ phi_i(a), phi_j(a) ],  phi_i = (h_i/q) ⋆ K_bw   (KDE-smoothed score kernel)
  -> sigma_err^2 = lambda_max( G^{-1} Sigma G^{-1} ) on the sum-zero tangent.

Limits:
  bw->0 : phi_i = h_i/q, Sigma|tangent = G/n  => sigma_err^2 = 1/(n*lambda_min(G))  (Cramer-Rao; ~ sqrt(K-1) excess)
  bw>0  : phi_i smoothed -> oscillatory worst mode's score variance suppressed -> excess < sqrt(K-1) -> toward ln(K-1)

excess(K) = sigma_err(K)/sigma_err(K=2); n cancels (set n=1). Compare to MEASURED (real NMF):
  rho=0.47 K=2..7 = 1.04 1.43 1.62 1.84 1.92 1.94
  rho=0.40 K=2..7 = 1.01 1.56 1.99 2.15 2.40 2.19
English prints. Run:  & "C:/Users/Administrator/miniconda3/python.exe" compute_sandwich.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW = 25.0, 200.0, 12.0
DT = 1.0
TGRID = np.arange(50.0, 2600.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
MEAS = {0.47: [1.04, 1.43, 1.62, 1.84, 1.92, 1.94],
        0.40: [1.01, 1.56, 1.99, 2.15, 2.40, 2.19]}


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean(); B = []
    for d in range(1, K):
        v = x ** d; v -= v.mean()
        for b in B: v = v - (v @ b) * b
        if np.linalg.norm(v) > 1e-12: B.append(v / np.linalg.norm(v))
    return B


# Gaussian smoothing kernel matrix (rows=a, cols=s): Kmat[a,s]=K_bw(a-s)*DT
_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT


def smooth(f):                       # (f ⋆ K_bw)(a) = sum_s f(s) K_bw(a-s) DT
    return _KS @ f


def sandwich(K, rho, bw_on=True):
    h, _ = endmembers(K, rho)
    w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    hq = h / q                                                   # h_i/q  (K,T)
    phi = np.array([smooth(hq[i]) for i in range(K)]) if bw_on else hq
    Eq = lambda f: TRAPZ(f * q, TGRID)
    m = np.array([Eq(phi[i]) for i in range(K)])
    Sig = np.array([[Eq(phi[i] * phi[j]) - m[i] * m[j] for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P; St = P @ Sig @ P
    # pseudo-inverse on tangent
    Gpi = np.linalg.pinv(Gt, rcond=1e-10)
    Cov = Gpi @ St @ Gpi
    Cov = P @ Cov @ P
    lam, V = np.linalg.eigh(Cov)
    se2 = max(lam[-1], 0.0)
    top = V[:, -1]
    B = contrast_polys(K)
    dom = int(np.argmax([abs(top @ b) for b in B])) + 1
    # also lambda_min(G) for the CR / no-bw reference
    lamG = np.linalg.eigvalsh(Gt)
    lamG = sorted(x for x in lamG if x > 1e-9 * lamG.max())
    return math.sqrt(se2), dom, lamG[0]


for rho in (0.47, 0.40):
    print(f"\n=== rho={rho} ===  (predicted excess vs MEASURED real-NMF)")
    print("%2s | %10s %6s %4s | %10s %6s | %8s %6s"
          % ("K", "se(bw)", "exc", "deg", "se(noBW)", "excCR", "MEAS", "d%"))
    se0 = se0cr = None
    for i, K in enumerate(range(2, 8)):
        se, dom, lamGmin = sandwich(K, rho, bw_on=True)
        secr, _, _ = sandwich(K, rho, bw_on=False)
        if K == 2: se0, se0cr = se, secr
        exc, excCR = se / se0, secr / se0cr
        meas = MEAS[rho][i]
        print("%2d | %10.4e %6.2f %4d | %10.4e %6.2f | %8.2f %+5.0f%%"
              % (K, se, exc, dom, secr, excCR, meas, 100 * (exc - meas) / meas))
