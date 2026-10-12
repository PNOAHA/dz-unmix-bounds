# -*- coding: utf-8 -*-
"""
probe_S_model.py -- build the WORKING closed form for S(K) from the EXACT identity
    S^2 = 1 - D(K)/lambda_min(K),   D = Var_q[g]-Var_q[K_bw*g] (smoothing loss), lambda_min=Var_q[g].

Strategy: the cleanest route that matches numbers. Two sub-models:

 (M1) lambda_min(K): the smallest tangent eigenvalue of the q-Gram G. This is the CR factor's job
      (separate derivation). Here we just need a closed form. The Gram is the q-weighted Gaussian
      overlap matrix with adjacent overlap r=exp(-2 erfinv(rho)^2) (theta-function off-diagonals,
      VERIFIED above: G_n/G_0 ~ r^{n^2}). For a tridiagonal-dominant Gram T=tridiag(a,1,a) (a=r/(1+...)),
      the smallest eigenvalue of the degree-(K-1) DST mode is  1 - 2 a cos(pi/(K+1)) ... but boundary &
      higher bands shift it. We FIT lambda_min ~ A*(1 - B cos(pi/(K+1)))? Actually simpler &
      validated by the prompt: lambda_min/lambda_min(2) ~ (K-1)^{-p(rho)}. We DERIVE p(rho) elsewhere.
      For S we only need the RATIO D/lambda_min.

 (M2) D(K): smoothing loss. KEY EMPIRICAL FACT (probe_S_spectral3): D(K) is a SLOWLY varying, near-
      plateau quantity. Let me characterize it precisely:
        - It is the worst-mode quadratic form of the loss kernel L: D = v^T L v.
        - v is unit-norm; L has bounded entries. As K grows v spreads over more components but L is local
          (banded), so v^T L v approaches a fixed "bulk per-mode loss" value.
      Model: D(K) = D_inf * f(K) where f->1. Try D(K) = D_inf + c/(K-1) (1/K correction from edges).

 Then  S^2 = 1 - D(K)/lambda_min(K).  Express lambda_min(K)=lambda_min(2)*(K-1)^{-p}. So
        1 - S^2 = [D(K)/lambda_min(2)] * (K-1)^p.
 Test this composite against the true S table.

 Also: directly test the BEST one-parameter description -- is S^2 = 1 - alpha*(K-1)^p (a power law in
 the SAME exponent p as lambda_min)? Since D~const-ish, 1-S^2 ~ const*(K-1)^p. CHECK fit quality.
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
Eqf = lambda f: TRAPZ(f, TGRID)


def get(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    psi = h / q
    G = np.array([[Eqf(h[i] * psi[j]) for j in range(K)] for i in range(K)])
    sp = np.stack([smooth(psi[i]) for i in range(K)])
    Bm = np.array([[Eqf(sp[i] * sp[j] * q) for j in range(K)] for i in range(K)])
    L = G - Bm
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    return lam[tang[0]], float(v @ L @ v)


for rho in (0.47, 0.40):
    print(f"\n========== rho={rho} ==========")
    Ks = np.arange(2, 11)
    lam = np.array([get(K, rho)[0] for K in Ks])
    D = np.array([get(K, rho)[1] for K in Ks])
    S_true = np.sqrt(1 - D / lam)
    # fit lambda_min ~ lam2*(K-1)^{-p}
    p = -np.polyfit(np.log(Ks[1:] - 1), np.log(lam[1:] / lam[0]), 1)[0]
    # fit D(K) = D_inf + c/(K-1)   over K>=3
    Acol = np.vstack([np.ones(len(Ks) - 1), 1.0 / (Ks[1:] - 1)]).T
    cD, *_ = np.linalg.lstsq(Acol, D[1:], rcond=None)
    D_inf, cd = cD
    # composite S model
    lam_model = lam[0] * (Ks - 1.0) ** (-p)
    D_model = D_inf + cd / (Ks - 1.0)
    S_model = np.sqrt(np.clip(1 - D_model / lam_model, 0, 1))
    print("  p(rho)=%.3f   D_inf=%.4e  c_D=%.4e" % (p, D_inf, cd))
    print("  %2s | %9s %9s %9s | %9s | %8s %8s %6s"
          % ("K", "lam_true", "lam_mod", "D_true", "D_mod", "S_true", "S_model", "err%"))
    for i, K in enumerate(Ks):
        print("  %2d | %9.3e %9.3e %9.3e | %9.3e | %8.4f %8.4f %6.2f"
              % (K, lam[i], lam_model[i], D[i], D_model[i], S_true[i], S_model[i],
                 100 * (S_model[i] - S_true[i]) / S_true[i]))
    # also pure power-law fit of 1-S^2
    y = 1 - S_true ** 2
    pl = np.polyfit(np.log(Ks - 1 + 1e-9), np.log(y), 1)
    print("  1-S^2 ~ %.4f*(K-1)^%.3f   (exponent vs p=%.3f)" % (math.exp(pl[1]), pl[0], p))
    Spow = np.sqrt(np.clip(1 - math.exp(pl[1]) * (Ks - 1.0) ** pl[0], 0, 1))
    print("  pure-power S model max err%%: %.2f"
          % (100 * np.max(np.abs(Spow - S_true) / S_true)))
