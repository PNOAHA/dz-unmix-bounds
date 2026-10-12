# -*- coding: utf-8 -*-
"""
probe_S_qspectrum.py -- the EXACT q-weighted spectral density of g_v, the object that explains S(K)
to high accuracy including the bw-not-small regime.

Var_q[g] is a quadratic form. Write the q-weighted autocorrelation of g:
   C(tau) := E_q[ (g(t)-mg)(g(t+tau)-mg) ]  with a q-stationary approximation? No -- q is not stationary.
But we can define the EXACT spectral object that gives S:
   Var_q[K_s * g] = int int g(t) g(t') [ (K_s*K_s applied)... ] -- the smoothing is symmetric, and
   Var_q[K_s*g] = int (K_s*g)(t)^2 q(t) dt - (...)^2.

A fully rigorous spectral statement that holds exactly:
   Define the operator T_q[f](t) = q(t) f(t) (mult by q) and C_s = convolution with G_s.
   Var_q[C_s g] = < C_s g, T_q C_s g > - (<C_s g, q>)^2/?  -- this is <g, C_s T_q C_s g> on the
   centered part. The eigen-decomposition of the pencil (C_s T_q C_s, T_q) is K-dependent; not closed.

PRACTICAL high-accuracy closed form: model the smoothing response as a SINGLE q-effective frequency
W(K) BUT computed from the EXACT response at the operating bw (W_full), and give W_full(K) a closed form.
We have W_full(K)^2 = -ln(S^2)/bw^2. From probe: W_full grows and saturates.

Test the hypothesis W_full(K)^2 = E_bend(K)/lambda_min(K) * c_corr  -- i.e. the curvature freq times a
bw-correction factor c_corr<1 (since strong smoothing weights LOWER freqs of the spectrum more).
Measure c_corr(K) = W_full^2/W_curv^2 = W_full^2 * lambda_min / E_bend; see if ~constant.

Then S(K) = exp(-bw^2/2 * c_corr * E_bend(K)/lambda_min(K)).
Better still: express E_bend(K)/lambda_min(K) in CLOSED form.
   E_bend(K) = -Cov_q[g,g''] = int (g')^2 q  (integrate by parts, boundary ~0 since q->0 at ends!)
   Indeed -int g g'' q = int g' (g q)' = int (g')^2 q + int g g' q'.  Hmm not clean. Measure both:
   compare E_bend to int (g')^2 q  (the q-weighted gradient/Dirichlet energy).
And lambda_min = int (g)^2 q (centered) = Var_q[g].
So  W_curv^2 = E_bend/lambda_min ~ [q-weighted Dirichlet energy]/[q-weighted L2 energy] = a RAYLEIGH
   QUOTIENT = the q-weighted mean-square frequency of g. THIS is the rigorous "effective frequency".
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


def deriv1(f):
    d = np.empty_like(f)
    d[1:-1] = (f[2:] - f[:-2]) / (2 * DT)
    d[0] = d[1]; d[-1] = d[-2]
    return d


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
    print("  W_R^2 := q-weighted Rayleigh quotient = int(g')^2 q / Var_q[g]  (rigorous eff. freq^2)")
    print("  %2s | %8s | %10s %10s | %8s %8s | %8s %6s"
          % ("K", "S_true", "Dir=int g'^2q", "lam=Var_q", "W_R", "W_full", "c_corr", "S_R"))
    WR2 = []; lam_a = []; Strue_a = []
    for K in Ks:
        q, g, lm = worst(K, rho)
        gp = deriv1(g)
        Dir = Eqf(gp * gp, q)            # q-weighted Dirichlet (gradient) energy
        vg = varq(g, q)
        WR2v = Dir / vg
        gs = smooth(g)
        S2 = varq(gs, q) / vg
        S = math.sqrt(max(S2, 0))
        W_full = math.sqrt(-math.log(S2)) / BW
        c_corr = (W_full ** 2) / WR2v
        S_R = math.exp(-0.5 * BW ** 2 * WR2v)
        WR2.append(WR2v); lam_a.append(lm); Strue_a.append(S)
        print("  %2d | %8.4f | %10.4e %10.4e | %8.5f %8.5f | %8.4f %6.4f"
              % (K, S, Dir, lm, math.sqrt(WR2v), W_full, c_corr, S_R))
    WR2 = np.array(WR2); lam_a = np.array(lam_a)
    # c_corr near-constant? report
    # Now express WR2 in closed form: WR2 = Dir/lam. Both are quadratic forms of v.
    # Fit S = exp(-0.5 bw^2 c_corr_const WR2). pick c_corr_const = median
    print("  median c_corr=%.4f" % np.median((np.array([ -math.log(s**2) for s in Strue_a]))/(BW**2*WR2)))
