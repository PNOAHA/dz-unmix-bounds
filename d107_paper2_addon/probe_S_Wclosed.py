# -*- coding: utf-8 -*-
"""
probe_S_Wclosed.py -- closed form for the q-effective frequency W(K), hence S(K)=exp(-bw^2 W(K)^2/2).

RESOLUTION of puzzle (established): the suppression is governed by the q-WEIGHTED effective frequency
W(K), which GROWS with K (0.023->0.084), NOT the unweighted omega_eff (flat ~0.06). The unweighted FFT
is dominated by g's tail blow-up (m/q where q~0) which carries no Var_q.

Now: small-s expansion gives the EXACT q-effective frequency:
   W_q(K)^2 = -d/d(s^2) ln R(s)|_{s=0}  =  - Cov_q[g, g''] / Var_q[g]   (since G_s*g ~ g + (s^2/2) g'')
   Indeed Var_q[g+(s^2/2)g''] = Var_q[g] + s^2 Cov_q[g,g''] + O(s^4)
       => R(s) = 1 + s^2 Cov_q[g,g'']/Var_q[g] + ... => -ln R ~ -s^2 Cov/Var
       => W_q^2 = -Cov_q[g,g'']/Var_q[g].
So the EXACT leading q-effective frequency is
        W_q(K)^2 = -Cov_q[g, g''] / Var_q[g] = (-Cov_q[g,g'']) / lambda_min(K).
We showed -Cov_q[g,g''] is ~K-INDEPENDENT (a fixed local "bending energy" of the worst mode in the
q-bulk)!  Call it  E_bend(rho) := -Cov_q[g, g''] (plateau value). Then
        W_q(K)^2 = E_bend(rho) / lambda_min(K).
Since lambda_min(K) decreases with K, W_q GROWS -- exactly the resolution.

S via the SMALL-s (curvature) freq would be S_curv = exp(-bw^2 W_q^2/2). But bw=12 is not small, so the
quartic+ terms matter -> S_curv overshoots. The EXACT S uses W_full. Model both; report which matches.

closed form chain:
   S(K) = exp( -(bw^2/2) * E_bend(rho) / lambda_min(K) )       [curvature/Gaussian-tail model]
With lambda_min(K) = lambda_min(2) (K-1)^{-p(rho)} (from CR derivation):
   ln S(K) = -(bw^2 E_bend / 2 lambda_min(2)) (K-1)^{p(rho)}.
   => -ln S ~ (K-1)^{p}.   TEST the exponent and prefactor.

This script:
 (1) measures E_bend(rho)=-Cov_q[g,g''] plateau, confirms K-independence.
 (2) builds S_curv(K)=exp(-bw^2 E_bend/(2 lambda_min(K))) vs S_true. Quantify residual.
 (3) builds the EXACT-W model S=exp(-bw^2 W_full^2/2) (tautological check).
 (4) tests -ln S ~ (K-1)^p law and the prefactor bw^2 E_bend/(2 lambda_min(2)).
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
Eqf = lambda f, q: TRAPZ(f * q, TGRID)


def deriv2(f):
    d2 = np.empty_like(f)
    d2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / DT ** 2
    d2[0] = d2[1]; d2[-1] = d2[-2]
    return d2


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
    return q, g, lam[tang[0]]


def varq(f, q):
    m = Eqf(f, q); return Eqf((f - m) ** 2, q)


for rho in (0.47, 0.40):
    print(f"\n========== rho={rho} ==========")
    Ks = np.arange(2, 11)
    lam = np.zeros(len(Ks)); Ebend = np.zeros(len(Ks)); Strue = np.zeros(len(Ks))
    for i, K in enumerate(Ks):
        q, g, lm = worst(K, rho)
        lam[i] = lm
        mg = Eqf(g, q)
        gpp = deriv2(g)
        Ebend[i] = -(Eqf((g - mg) * (gpp - Eqf(gpp, q)), q))   # -Cov_q[g,g'']
        Strue[i] = math.sqrt(max(varq(smooth(g), q) / varq(g, q), 0))
    # plateau E_bend: average over K>=4
    E_inf = Ebend[Ks >= 4].mean()
    p = -np.polyfit(np.log(Ks[1:] - 1), np.log(lam[1:] / lam[0]), 1)[0]
    print("  E_bend(rho) plateau (K>=4 mean) = %.4e   (per-K: %s)"
          % (E_inf, np.array2string(Ebend, precision=2)))
    print("  p(rho)=%.3f   lambda_min(2)=%.4e" % (p, lam[0]))
    # curvature model
    Scurv = np.exp(-0.5 * BW ** 2 * Ebend / lam)          # uses true Ebend(K), true lam(K)
    Scurv_inf = np.exp(-0.5 * BW ** 2 * E_inf / lam)      # uses plateau Ebend, true lam
    # power-law for -ln S
    y = -np.log(Strue)
    pl = np.polyfit(np.log(Ks - 1 + 1e-9), np.log(y), 1)
    print("  -ln S ~ %.4e*(K-1)^%.3f" % (math.exp(pl[1]), pl[0]))
    print("  %2s | %8s | %8s %6s | %8s %6s"
          % ("K", "S_true", "S_curv", "err%", "S_curvInf", "err%"))
    for i, K in enumerate(Ks):
        print("  %2d | %8.4f | %8.4f %6.2f | %8.4f %6.2f"
              % (K, Strue[i], Scurv[i], 100 * (Scurv[i] - Strue[i]) / Strue[i],
                 Scurv_inf[i], 100 * (Scurv_inf[i] - Strue[i]) / Strue[i]))
