# -*- coding: utf-8 -*-
"""
probe_S_spectral3.py -- nail the S(K) closed form.

FINDINGS so far:
 - Var_q[g] = lambda_min(G,K)  (g is unit worst eigvec).  Decreases as (K-1)^{-p}.
 - Cov_q[g, g''] is ~K-INDEPENDENT (plateau).  -> the suppression numerator is a FIXED local quantity.
 - Taylor S^2 = 1 + bw^2 Cov_q[g,g'']/Var_q[g] works for small K but the full Gaussian smoothing is
   needed at large K (strong smoothing).

The CORRECT exact statement: S^2 = Var_q[K_bw * g] / Var_q[g].
Write the smoothing in the SPECTRAL domain of g UNDER LEBESGUE (since K_bw acts by convolution = mult
in Fourier), but the variance is under q. The clean way:

  Var_q[g] = int g^2 q - (int g q)^2 .  With u(t) := g(t) sqrt(q(t)) we'd mix things. Instead use that
  g = m/q with m = sum v_i h_i (signed, int m=0).  Then:
     Var_q[g]      = int m^2 / q                              = v^T G v = lambda_min
     Var_q[K_bw*g] = int (K_bw*(m/q))^2 q   - (int (K_bw*(m/q)) q)^2
  This is NOT a simple multiplier on m.

EXACT SPECTRAL MODEL (the right one):
  The smoothing multiplies the Fourier transform of g by exp(-bw^2 omega^2/2).
  S^2 = [ int |ghat(omega)|^2 e^{-bw^2 omega^2} W(omega) ... ]  -- but variance is in t under q, not flat.

So the operative quantity is the spectrum of g WEIGHTED so that it integrates to Var_q. The natural object:
  define the "q-spectral density" via the smoothing-response curve:
     R(s) := Var_q[ G_s * g ] / Var_q[g]   as a function of smoothing width s,
  R(0)=1, R is a completely monotone-ish decay. S = sqrt(R(bw)).
  The INITIAL decay rate R'(0) gives the curvature term; the FULL R(bw) needs the whole response.

This script:
 (A) confirms Cov_q[g,g''] plateau and identifies its value c_inf(rho).
 (B) measures the EFFECTIVE smoothing response R(s) for several s, extracts an effective single scale.
 (C) Tests the model  S^2(K) = Var_q[K_bw*g]/Var_q[g] with the decomposition
        Var_q[K_bw*g] = Var_q[g] - D(K),   where D(K) = "smoothing loss".
     Claim: D(K) -> D_inf (K-independent plateau, the loss from smoothing the FIXED cusp structure),
     while Var_q[g]=lambda_min decreases. Then  S^2 = 1 - D(K)/lambda_min(K).
     If D(K)~const and lambda_min~(K-1)^{-p}, then 1-S^2 = D_inf*(K-1)^p / lambda_min(2)... TEST THIS.
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


def smoother(s):
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, s) * DT
    return lambda f: KS @ f


smooth = smoother(BW)
Eq = lambda f, q: TRAPZ(f * q, TGRID)


def worst(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    return h, mus, w, q, v, g, lam[tang[0]]


def varq(f, q):
    return Eq(f * f, q) - Eq(f, q) ** 2


print("=" * 95)
print("(C) MODEL:  S^2 = 1 - D(K)/lambda_min(K),  D(K)=Var_q[g]-Var_q[K_bw*g] = smoothing loss")
print("    test whether D(K) plateaus (K-independent) -> then 1-S^2 = D_inf/lambda_min(K)")
print("=" * 95)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %8s | %10s %10s %10s | %8s %10s"
          % ("K", "S_true", "lam_min", "D(K)=loss", "D/lam_2", "1-S^2", "D(K)/D(2)"))
    D2 = None
    for K in range(2, 11):
        h, mus, w, q, v, g, lam = worst(K, rho)
        vg = varq(g, q)
        vgs = varq(smooth(g), q)
        D = vg - vgs
        if K == 2: D2 = D
        S2 = vgs / vg
        print("%2d | %8.4f | %10.4e %10.4e %10.4e | %8.4f %10.4f"
              % (K, math.sqrt(max(S2, 0)), lam, D, D, 1 - S2, D / D2))

print()
print("=" * 95)
print("(A) Per-cusp localization: D(K) should = (#internal-cusps-weighted) * (per-cusp loss).")
print("    Check D(K) per adjacent-pair.  Number of adjacent pairs = K-1.")
print("    But weights v_i taper, so effective count != K-1. Measure D(K)/(K-1).")
print("=" * 95)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    D2 = None
    for K in range(2, 11):
        h, mus, w, q, v, g, lam = worst(K, rho)
        vg = varq(g, q); vgs = varq(smooth(g), q); D = vg - vgs
        print("K=%2d  D=%.4e  D/(K-1)=%.4e  lam_min=%.4e  lam*(K-1)=%.4e"
              % (K, D, D / (K - 1), lam, lam * (K - 1)))
