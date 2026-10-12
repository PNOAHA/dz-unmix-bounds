# -*- coding: utf-8 -*-
"""
probe2_pR_cap.py -- FIRST-PRINCIPLES attempt for p_R, the exponent of <1/q>_g growth.

CORRECT object (Phase 1 verified): work in the KxK Gram space, NOT the raw grid (raw g^2/q
blows up in the q->0 tails: g=(sum v_i h_i)/q so g^2/q ~ (...)^2/q^3 -> 1e297; the Gram form
h_i h_j/q is self-regularized because the numerator vanishes faster than q in the tails).

  G_ij = int h_i h_j / q ,  M_ij = int h_i h_j .
  lambda_min(G) = lambda_min(M) * <1/q>_g  with  <1/q>_g := (v^T G v)/(v^T M v),  v = worst mode.
  Qeff := 1/<1/q>_g = (v^T M v)/lambda_true ~ (K-1)^{-p_R}.

PROMPT HYPOTHESIS: <1/q>_g would grow ~K (the "1/K plateau" -> Qeff~1/K -> p_R~1 in (K-1)),
but is CAPPED near 3/8 because the worst mode g_v has its weight in PAIRWISE OVERLAPS, not
spread over all K peaks.

This script tests the cap mechanism three ways:
 (A) BACKGROUND-SCALING law: q is built from w*h with sum w =1, so as K grows the SAME unit
     mass is split over K peaks -> per-peak weight w_k ~ 1/K -> background q at any peak ~ 1/K
     -> 1/q ~ K  (the naive plateau). Test: does the *unweighted* <1/q> (over q's own support)
     grow ~K?  vs the *worst-mode-weighted* <1/q>_g.
 (B) WORST-MODE LOCALIZATION: the degree-(K-1) contrast v concentrates G-mass in NEAREST-
     NEIGHBOR pairs (M is banded Toeplitz m_k=c exp(-b k^2), so v^T G v is dominated by the
     k=0,1 bands). The contrast "sees" only the local pair geometry, NOT the global 1/K
     dilution -> caps the K-growth. Quantify: fraction of v^T G v from |i-j|<=1 bands.
 (C) DIRECT Qeff from Gram (regularized) and its exponent; compare to the cap 3/8.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_pR_cap.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 4000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def grams(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
    return G, M, q, h, w, mus


def worst_mode(G, K):
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return V[:, tang[0]], lam[tang[0]]


def diag(K, rho):
    G, M, q, h, w, mus = grams(K, rho)
    v, lam = worst_mode(G, K)
    vMv = v @ M @ v
    inv_q_g = (v @ G @ v) / vMv          # <1/q>_g
    Qeff = 1.0 / inv_q_g

    # (B) banded contribution: fraction of v^T G v from |i-j|<=1
    Gv = G * v[None, :] * v[:, None]
    iu, ju = np.indices((K, K))
    band01 = (np.abs(iu - ju) <= 1)
    frac_band01 = Gv[band01].sum() / Gv.sum()
    frac_diag = np.diag(Gv).sum() / Gv.sum()

    # (A) naive background dilution: average of 1/q over the *whole peak region*
    #     weighted by the local signal sum (unweighted by worst mode), should grow ~K.
    sig = (w[:, None] * h).sum(0)        # = q itself; <1/q>_sig weighted by sig = grid length /...
    # use the SUM of unit endmembers (mode-agnostic) weighted 1/q to expose the 1/K dilution:
    hh = h.sum(0)
    inv_q_flat = TRAPZ(hh * hh / q, TGRID) / TRAPZ(hh * hh, TGRID)
    return dict(K=K, rho=rho, inv_q_g=inv_q_g, Qeff=Qeff, vMv=vMv,
                frac_band01=frac_band01, frac_diag=frac_diag, inv_q_flat=inv_q_flat)


print("=" * 105)
print("p_R from Gram (regularized).  <1/q>_g=(vGv)/(vMv);  Qeff=1/<1/q>_g ~ (K-1)^{-p_R}")
print("frac_band01 = fraction of v^T G v from nearest-neighbor bands |i-j|<=1 (locality/cap test)")
print("inv_q_flat = mode-agnostic <1/q> (exposes 1/K background dilution -> the UNCAPPED ~K growth)")
print("=" * 105)
out = {}
for rho in (0.47, 0.40, 0.35, 0.30):
    print(f"\n--- rho={rho} ---")
    print("%2s | %10s %8s | %9s %9s | %10s | %9s"
          % ("K", "<1/q>_g", "Qeff", "fband01", "fdiag", "inv_q_flat", "vMv"))
    rows = []
    for K in range(2, 11):
        d = diag(K, rho)
        rows.append(d)
        print("%2d | %10.4e %8.4f | %9.4f %9.4f | %10.4e | %9.3e"
              % (K, d["inv_q_g"], d["Qeff"], d["frac_band01"], d["frac_diag"],
                 d["inv_q_flat"], d["vMv"]))
    Ks = np.array([r["K"] for r in rows], float)
    iq = np.array([r["inv_q_g"] for r in rows])
    pR = np.polyfit(np.log(Ks - 1)[1:], np.log(iq[1:] / iq[0]), 1)[0]
    iqf = np.array([r["inv_q_flat"] for r in rows])
    pRf = np.polyfit(np.log(Ks - 1)[1:], np.log(iqf[1:] / iqf[0]), 1)[0]
    # also fit vs ln K (the "1/K plateau" is in K, not K-1)
    pR_K = np.polyfit(np.log(Ks)[1:], np.log(iq[1:] / iq[0]), 1)[0]
    pRf_K = np.polyfit(np.log(Ks)[1:], np.log(iqf[1:] / iqf[0]), 1)[0]
    out[rho] = (pR, pRf)
    print("  p_R(<1/q>_g) = %.3f [in K-1] / %.3f [in K] ;  flat <1/q> exponent = %.3f [K-1] / %.3f [K]"
          % (pR, pR_K, pRf, pRf_K))
print("\nSUMMARY p_R(rho):", {r: round(out[r][0], 3) for r in out})
