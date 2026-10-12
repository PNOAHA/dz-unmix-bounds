# -*- coding: utf-8 -*-
"""
probe_S_powerspec.py -- THE proper resolution of the puzzle via the q-weighted power spectrum.

S^2 = Var_q[K_bw * g] / Var_q[g].

The puzzle was: omega_eff (rms freq of g's UNWEIGHTED FFT) is flat in K, yet S drops. Resolution must
be that the q-WEIGHTED spectrum (the spectrum of the part of g that actually carries Var_q) shifts to
higher freq / has different transfer than the global FFT suggests, OR the transfer is not a simple
single-frequency exp.

Correct object: define u(t) = (g(t)-mean_q g) * sqrt(q(t)).  Then Var_q[g] = int u^2 = ||u||^2.
But smoothing acts on g, not u:  Var_q[smooth g] = int (K_bw*g - mean)^2 q.  Not ||smooth u||.

So decompose in the basis where BOTH the q-weighting and the convolution are diagonal is impossible
(they don't commute). Instead, the HONEST spectral statement:

  Var_q[g]      = (1/2pi) int |ghat(w)|^2 [q-spectral measure]  -- messy.

Cleaner: use the smoothing-RESPONSE function. Define for smoothing width s:
   R(s) = Var_q[G_s * g] / Var_q[g].
R(0)=1. The transfer for a pure freq w is exp(-s^2 w^2). If g had a single q-effective frequency W,
R(s)=exp(-s^2 W^2). Measure R(s) at several s; if it's a single exp, log R(s) is linear in s^2 and
W = W(K) is the q-EFFECTIVE frequency (DIFFERENT from omega_eff!). THIS W should grow with K.

This script:
 (1) measures R(s) for s in a grid, extracts the q-effective frequency W_q(K) = sqrt(-d ln R/d(s^2))|_0
     AND the full-curve effective frequency W_full(K) := sqrt(-ln R(bw))/bw  (the value that makes
     exp(-bw^2 W^2)=S^2 exactly). Show W_q and W_full GROW with K (unlike omega_eff).
 (2) explains WHY: the unweighted FFT omega_eff is dominated by the TAILS of g (where m/q blows up but
     q~0), which carry no Var_q. The q-effective frequency is the right one and it grows ~ as the
     contrast degree (K-1).
 (3) gives S(K) = exp(-bw^2 W_q(K)^2 /2-ish); fit W_q(K).
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
Eqf = lambda f, q: TRAPZ(f * q, TGRID)


def worst(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    return q, g, mus


def varq(f, q):
    m = Eqf(f, q)
    return Eqf((f - m) ** 2, q)


for rho in (0.47, 0.40):
    print(f"\n========== rho={rho}  (dmu={dmu(rho):.1f}) ==========")
    print("  W_q = q-effective frequency from small-s slope of -ln R(s)/s^2")
    print("  W_full = sqrt(-ln(S^2))/bw   (the freq that EXACTLY explains S via exp(-bw^2 W^2))")
    print("  omega_eff = unweighted FFT rms freq (the misleading one)")
    print("  %2s | %8s | %9s %9s %9s | %10s"
          % ("K", "S_true", "W_q", "W_full", "omega_eff", "W_full*(t)"))
    for K in range(2, 11):
        q, g, mus = worst(K, rho)
        vg = varq(g, q)
        # response at small s for slope
        ss = np.array([2.0, 3.0, 4.0])
        lnR = []
        for s in ss:
            gs = smoother(s)(g)
            lnR.append(math.log(varq(gs, q) / vg))
        # -ln R = W^2 s^2  => slope of -lnR vs s^2
        slope = np.polyfit(ss ** 2, -np.array(lnR), 1)[0]
        W_q = math.sqrt(max(slope, 0))
        # full
        gs = smoother(BW)(g)
        S2 = varq(gs, q) / vg
        S = math.sqrt(max(S2, 0))
        W_full = math.sqrt(-math.log(S2)) / BW
        # unweighted FFT rms
        G_hat = np.fft.rfft(g - g.mean())
        freq = np.fft.rfftfreq(len(g), d=DT) * 2 * math.pi
        P_hat = np.abs(G_hat) ** 2
        omega_eff = math.sqrt((freq ** 2 * P_hat).sum() / P_hat.sum())
        print("  %2d | %8.4f | %9.5f %9.5f %9.5f | predicts S=%.4f"
              % (K, S, W_q, W_full, omega_eff, math.exp(-0.5 * BW ** 2 * W_full ** 2)))
