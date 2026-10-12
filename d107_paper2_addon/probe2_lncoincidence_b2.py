# -*- coding: utf-8 -*-
"""
probe2_lncoincidence_b2.py -- DEEPER adversarial follow-up on claim (b).

probe2_lncoincidence_b.py found that, fit affinely over rho in [0.28,0.60], the simple
1/rho beats 1/erfinv^2 for the FULL p_G(rho).  But the rigorous part of the claim is at
the SYMBOL level: p_M (the pure OVERLAP-matrix exponent) ~ pi^2/(4b), b = 2 erfinv^2,
i.e. p_M ∝ 1/erfinv^2 EXACTLY (no rho-dependent reweighting). p_G = p_M - p_R adds the
empirical <1/q> reweight p_R which is NOT claimed to be 1/erfinv^2.

So the fair adversarial decomposition:
  (1) Does p_M ∝ 1/erfinv^2 (the RIGOROUS claim) hold? Test p_M * erfinv^2 = flat, and
      affine-fit p_M against the 4 candidates. THIS is what claim (b) actually asserts.
  (2) How much does the reweighting p_R = p_M - p_G distort it, and is the 1/rho win on
      p_G just because 1/rho happens to interpolate (theta-descent slope) - (smooth p_R)
      over this narrow rho window?
  (3) Sensitivity: is the 1/rho win robust if we DON'T give every candidate a free
      intercept? Pure 1-param proportionality p = b*X (no intercept) is the literal
      reading of "p ∝ X". Compare R2 of pure-proportional fits.

This separates "the rigorous symbol claim" from "the full empirical p_G".
Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_lncoincidence_b2.py
"""
import math
import numpy as np
from scipy.special import erfinv

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


def tangent_lammin(A, K):
    P = np.eye(K) - np.ones((K, K)) / K
    At = P @ A @ P
    lam, V = np.linalg.eigh(At)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def p_M_and_G(rho, Kmax):
    """Return (p_M, p_G): exponents of lammin(M) and lammin(G) over K=2..Kmax."""
    Ks = np.arange(2, Kmax + 1)
    lM, lG = [], []
    for K in Ks:
        h = endmembers(int(K), rho); w = endpoint_w(int(K))
        q = np.maximum((w[:, None] * h).sum(0), 1e-300)
        M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
        G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
        lM.append(tangent_lammin(M, K)); lG.append(tangent_lammin(G, K))
    lM = np.array(lM); lG = np.array(lG)
    pM = -np.polyfit(np.log(Ks - 1)[1:], np.log(lM[1:] / lM[0]), 1)[0]
    pG = -np.polyfit(np.log(Ks - 1)[1:], np.log(lG[1:] / lG[0]), 1)[0]
    return pM, pG


def affine_fit(X, y):
    A = np.vstack([np.ones_like(X), X]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2, math.sqrt(((y - pred) ** 2).mean())


def prop_fit(X, y):
    """pure proportional y = b X, no intercept -> literal 'p ∝ X'."""
    b = (X @ y) / (X @ X)
    pred = b * X
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2, math.sqrt(((y - pred) ** 2).mean())


RHOS = np.array([0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45,
                 0.47, 0.50, 0.53, 0.56, 0.60])

for Kmax in (7, 10):
    print("\n================  K=2..%d  ================" % Kmax)
    erf = np.array([erfinv(r) for r in RHOS])
    res = np.array([p_M_and_G(rho, Kmax) for rho in RHOS])
    pM, pG = res[:, 0], res[:, 1]
    pR = pM - pG

    print("  rho   erfinv   p_M     p_G     p_R(=pM-pG)  pM*erfinv^2  pG*erfinv^2")
    for i, rho in enumerate(RHOS):
        print("  %.2f  %.4f  %6.3f  %6.3f   %7.3f      %7.3f      %7.3f"
              % (rho, erf[i], pM[i], pG[i], pR[i], pM[i]*erf[i]**2, pG[i]*erf[i]**2))

    cands = {
        "E2 1/erfinv^2": 1.0 / erf ** 2,
        "R2 1/rho^2   ": 1.0 / RHOS ** 2,
        "R1 1/rho     ": 1.0 / RHOS,
        "NL -1/ln rho ": -1.0 / np.log(RHOS),
    }

    for tag, y in (("p_M (RIGOROUS symbol claim)", pM), ("p_G (full empirical)", pG)):
        print("\n  --- fit %s ---" % tag)
        print("    candidate        affine_R2   affine_RMSE  prop_R2(no-intc)")
        rank = []
        for name, X in cands.items():
            _, r2a, ra = affine_fit(X, y)
            _, r2p, rp = prop_fit(X, y)
            rank.append((name, r2a, r2p))
            print("    %-14s  %.5f    %.4f      %.5f" % (name, r2a, ra, r2p))
        best_aff = max(rank, key=lambda t: t[1])[0].strip()
        best_prop = max(rank, key=lambda t: t[2])[0].strip()
        print("    => best affine: %s ;  best PURE-proportional: %s" % (best_aff, best_prop))
        pe = y * erf ** 2
        print("    %s * erfinv^2: mean=%.3f CV=%.1f%%  (flat <=> ∝ 1/erfinv^2)"
              % (tag.split()[0], pe.mean(), 100 * pe.std() / abs(pe.mean())))

print("\nVERDICT NOTE: claim (b)'s RIGOROUS content is p_M ∝ 1/erfinv^2 (symbol-level).")
print("If p_M*erfinv^2 is flat & E2 wins the PURE-proportional fit on p_M, the rigorous core")
print("survives; the 1/rho win on p_G is then an artifact of the empirical reweight p_R over a")
print("narrow rho window. If even p_M prefers 1/rho proportionally, claim (b) NEEDS-REVISION.")
