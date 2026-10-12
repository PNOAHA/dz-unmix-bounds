# -*- coding: utf-8 -*-
"""
probe2_pR_symbol.py -- p_R from the Toeplitz-symbol viewpoint (first-principles attempt).

Established exactly (Phase 1):
  M_ij = c * exp(-b (i-j)^2),  c = 1/(2 sigma sqrt(pi)),  b = 2 erfinv(rho)^2 = kappa.
  G_ij = int h_i h_j / q .  worst mode v = degree-(K-1) contrast.
  lambda_min(G) = lambda_min(M) * <1/q>_g,   Qeff = (v^T M v)/(v^T G v) = 1/<1/q>_g.

We want p_R := exponent of <1/q>_g ~ (K-1)^{p_R}, i.e. Qeff ~ (K-1)^{-p_R}.

KEY structural facts to exploit:
 1. v^T M v = lambda_min(M): the worst mode of G is ~ the worst mode of M (theta-function descent),
    so v^T M v ~ lambda_min(M)(K) ~ exp(-pi^2/(4b)) * polynomial(K).  We have it numerically.
 2. v^T G v = lambda_min(G) = lambda_min(M)*<1/q>_g, with <1/q>_g the worst-mode-weighted bg.
 3. The OBSTRUCTION the prompt names: naive 1/K plateau -> <1/q>_g ~ K (p_R~1 in K) BUT capped
    near 3/8 by pairwise overlap.

THIS SCRIPT tests a concrete cap derivation:
  <1/q>_g = (v^T G v)/(v^T M v).  Write the *background factor* per Toeplitz band:
     G_ij / M_ij = int h_i h_j/q  /  int h_i h_j  =: beta_{|i-j|}(K)   (band-resolved bg enhancement)
  Then v^T G v = sum_{i,j} v_i v_j M_ij beta_{|i-j|},  v^T M v = sum v_i v_j M_ij.
  So  <1/q>_g = weighted average of beta_k with weights  W_k = (sum_{|i-j|=k} v_i v_j M_ij).
  The diagonal band beta_0 = int h_k^2/q (per-peak self-overlap bg) is the LARGEST and grows ~K
  (1/K dilution).  The off-diagonal bands beta_1, beta_2,... are SMALLER (pair overlaps fall in
  the inter-peak gap where q is *higher* relative to product? test) AND carry NEGATIVE weight for
  a contrast (alternating v).  The cap = how much the (negative) off-diagonal bands cancel the
  diagonal ~K growth.

Outputs:
  beta_0(K), beta_1(K) and their K-exponents; the band weights W_0,W_1; and the resulting
  <1/q>_g as a 2-band model  <1/q>_g ~ (W0 beta0 + 2 W1 beta1)/(W0 + 2 W1).  Test the cap.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_pR_symbol.py
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


def band(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
    # worst mode
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    inv_q_g = (v @ G @ v) / (v @ M @ v)
    # band-resolved bg enhancement beta_k = mean over a band of G_ij/M_ij
    betas = {}
    weights = {}
    for k in range(K):
        ii = np.arange(K - k); jj = ii + k
        ratio = G[ii, jj] / M[ii, jj]
        betas[k] = ratio.mean()
        # weight contributed by this band to v^T M v (both +k and -k, symmetric)
        wt = (v[ii] * v[jj] * M[ii, jj]).sum() * (1 if k == 0 else 2)
        weights[k] = wt
    # 2-band reconstruction of <1/q>_g
    vMv = v @ M @ v
    rec2 = sum(betas[k] * weights[k] for k in (0, 1)) / sum(weights[k] for k in (0, 1)) \
        if abs(sum(weights[k] for k in (0, 1))) > 1e-30 else float('nan')
    rec_all = sum(betas[k] * weights[k] for k in range(K)) / vMv
    return inv_q_g, betas, weights, vMv, rec2, rec_all


print("=" * 100)
print("Band decomposition of <1/q>_g = sum_k beta_k W_k / vMv ;  beta_k = mean(G_ij/M_ij) on band k")
print("W_k = band-k contribution to v^T M v (alternating sign for contrast)")
print("=" * 100)
for rho in (0.47, 0.40, 0.35, 0.30):
    print(f"\n--- rho={rho}  kappa=2erfinv^2={2*erfinv(rho)**2:.4f} ---")
    print("%2s | %9s | %8s %8s %8s | %9s %9s %9s | %9s"
          % ("K", "<1/q>_g", "beta0", "beta1", "beta2", "W0", "W1", "W2", "rec_all"))
    b0s, Ks = [], []
    for K in range(2, 11):
        iqg, betas, weights, vMv, rec2, rec_all = band(K, rho)
        b0s.append(betas[0]); Ks.append(K)
        print("%2d | %9.3f | %8.2f %8.2f %8s | %9.2e %9.2e %9s | %9.3f"
              % (K, iqg, betas[0], betas.get(1, float('nan')),
                 ("%8.2f" % betas[2]) if 2 in betas else "  -  ",
                 weights[0], weights.get(1, float('nan')),
                 ("%9.2e" % weights[2]) if 2 in weights else "    -    ", rec_all))
    Ks = np.array(Ks, float); b0 = np.array(b0s)
    pb0 = np.polyfit(np.log(Ks - 1)[1:], np.log(b0[1:] / b0[0]), 1)[0]
    pb0K = np.polyfit(np.log(Ks)[1:], np.log(b0[1:] / b0[0]), 1)[0]
    print("  beta0 (per-peak self-bg) ~ (K-1)^%.3f / K^%.3f   [the UNCAPPED 1/K dilution]"
          % (pb0, pb0K))
