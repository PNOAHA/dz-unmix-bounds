# -*- coding: utf-8 -*-
"""
probe_vAv_law.py -- the rho-dependence of p(rho) traced to vAv = v^T A v (worst-mode curvature Fisher).

Established exactly:  p(rho) = -dln lam_min/dln(K-1) = p_om2(rho) - p_vAv(rho),
  lam_min ~ (K-1)^{-p},  omega_q^2 ~ (K-1)^{p_om2},  vAv ~ (K-1)^{p_vAv}.

We have a SEPARATE clean route to lam_min itself (the CR factor's own derivation), so the real
deliverable here is: confirm S(K)'s suppression exponent.  S = exp(-bw^2 omega_q^2/2), and over the
fitted window the *effective* power-law S ~ (K-1)^{-q_S} has
  ln S = -bw^2 omega_q^2/2 = -bw^2 C2 (K-1)^{p_om2}/2   (NOT a pure power law -- it's a stretched exp).
So S is fundamentally a STRETCHED EXPONENTIAL in (K-1), not a power law; any (K-1)^{-q} fit is local.

This script:
 (1) Quantify: fit S(K) ~ (K-1)^{-q_S} locally (K=2..7) and report q_S(rho); show the residual to a
     stretched-exp ln S = -alpha (K-1)^beta is far better (this is the honest functional form).
 (2) Give the bw-dependence of q_S: since lnS ~ -bw^2(...), q_S grows with bw^2. Tabulate.
 (3) Tie S into the total excess: excess=CR*S, CR~(K-1)^{p/2}. Reproduce the measured excess
     ln-fit (a+b ln(K-1), R2=0.994 at rho .47) from CR*S closed forms.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_vAv_law.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import curve_fit

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 4000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def worst_pack(K, rho, bw=BW):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    g = (v[:, None] * h).sum(0) / q
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, bw) * DT
    gs = KS @ g
    Eq = lambda f: TRAPZ(f * q, TGRID)
    den = Eq(g * g) - Eq(g) ** 2
    S = math.sqrt(max(Eq(gs * gs) - Eq(gs) ** 2, 0) / den)
    fp = np.gradient(g - Eq(g), TGRID)
    om2 = Eq(fp * fp) / den
    return lam_min, S, om2


print("=" * 100)
print("(1) S(K): local power-law vs stretched-exponential ln S = -alpha (K-1)^beta")
print("=" * 100)
for rho in (0.47, 0.40):
    Ks = np.arange(2, 11, dtype=float)
    pk = [worst_pack(int(K), rho) for K in Ks]
    S = np.array([p[1] for p in pk])
    om2 = np.array([p[2] for p in pk])
    # local power-law fit K=2..7
    m = Ks <= 7
    qS = -np.polyfit(np.log(Ks[m] - 1)[1:], np.log(S[m])[1:], 1)[0]
    # stretched exp: -lnS = alpha (K-1)^beta
    y = -np.log(S)

    def se(K, alpha, beta): return alpha * (K - 1) ** beta
    popt, _ = curve_fit(se, Ks, y, p0=[0.05, 1.0], maxfev=20000)
    pred = se(Ks, *popt)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # also: since lnS=-bw^2 om2/2, and om2~(K-1)^p_om2, beta should equal p_om2
    p_om2 = np.polyfit(np.log(Ks - 1)[1:], np.log(om2[1:]), 1)[0]
    print("rho=%.2f:  S~(K-1)^-%.3f (local K=2..7) ;  -lnS=alpha(K-1)^beta: alpha=%.4f beta=%.3f R2=%.4f"
          % (rho, qS, popt[0], popt[1], r2))
    print("          beta(stretched) vs p_om2(=dln om^2/dln(K-1)) : %.3f vs %.3f  (should match)"
          % (popt[1], p_om2))
    print("          alpha vs bw^2 C2/2 :  alpha=%.4f ,  predicted=%.4f  (C2=om2(K)*K/(K-1) avg)"
          % (popt[0], 0.5 * BW ** 2 * np.mean(om2 * Ks / (Ks - 1))))

print("\n" + "=" * 100)
print("(2) bw-dependence of the local S-exponent q_S (S~(K-1)^-q_S, K=2..7)")
print("=" * 100)
print("%6s | %10s %10s" % ("bw", "q_S@rho.47", "q_S@rho.40"))
for bw in (6, 8, 12, 16, 20):
    row = []
    for rho in (0.47, 0.40):
        Ks = np.arange(2, 8, dtype=float)
        S = np.array([worst_pack(int(K), rho, bw)[1] for K in Ks])
        q = -np.polyfit(np.log(Ks - 1)[1:], np.log(S)[1:], 1)[0]
        row.append(q)
    print("%6.1f | %10.3f %10.3f" % (bw, *row))

print("\n" + "=" * 100)
print("(3) total excess = CR*S reproduces measured a+b ln(K-1) fit?")
print("=" * 100)
for rho in (0.47, 0.40):
    Ks = np.arange(2, 11, dtype=float)
    pk = [worst_pack(int(K), rho) for K in Ks]
    lam = np.array([p[0] for p in pk]); S = np.array([p[1] for p in pk])
    CR = np.sqrt(lam[0] / lam)
    exc = CR * S
    m = Ks <= 7
    A = np.vstack([np.ones(m.sum()), np.log(Ks[m] - 1)]).T
    coef, *_ = np.linalg.lstsq(A, exc[m], rcond=None)
    pred = A @ coef
    r2 = 1 - ((exc[m] - pred) ** 2).sum() / ((exc[m] - exc[m].mean()) ** 2).sum()
    print("rho=%.2f:  excess=CR*S  ->  a+b ln(K-1):  a=%.3f b=%.3f R2=%.4f   (excess K=2..10: %s)"
          % (rho, coef[0], coef[1], r2, " ".join("%.2f" % e for e in exc)))
