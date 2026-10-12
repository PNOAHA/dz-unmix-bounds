# -*- coding: utf-8 -*-
"""
probe2_pR_cap_final.py -- the 3/8 cap, made precise, and an honest closed form for p_R.

From probe2_pR_wdep:  the worst mode's surviving-diagonal fraction  D(K) := W0/sum_k|W_k|
  W0 = c  (EXACT, K-indep: sum_i v_i^2 M_ii = c),  W_k = 2 sum_i v_i v_{i+k} M_{|k|}.
  D(K) -> ~0.37 at K~12 for rho~0.47, ~0.31 for rho~0.40.  This is the 'cap' the prompt names.

The mechanism: <1/q>_g = (sum_k beta_k W_k)/(sum_k W_k), beta_0 is the K-diluted self-bg (~K^{p0}),
beta_{k>=1} grow slower; the worst mode puts fraction D on beta_0.  As K grows, D shrinks toward a
LIMIT D_inf, so the effective exponent of <1/q>_g = D-weighted blend of (steep beta0)+(flat betak).

THIS SCRIPT pins:
 1. D(K) and its large-K limit D_inf(rho); test D_inf ~ 3/8.
 2. The EXACT closed form for W_k of the degree-(K-1) discrete-contrast worst mode is hard, but
    for the TOP Dirichlet-sine mode v_n=sin(n pi K/(K+1)) on a Gaussian-banded Toeplitz,
    W0/sum|Wk| has a clean continuum limit.  Compute the sine-mode D and compare to the true D.
 3. Honest p_R: report p_R(rho) (window K=2..7), the formula p_R = D_inf * p0 (cap*dilution),
    and the residual.  State clearly p_R is window-local (lambda_min is theta, not power law).

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_pR_cap_final.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 7000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
C0 = 1.0 / (2 * SIGMA * math.sqrt(math.pi))


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def Dfrac_and_iqg(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    M = np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    iqg = (v @ G @ v) / (v @ M @ v)
    beta0 = np.diag(G).mean()
    Wabs = 0.0; W0 = 0.0
    for k in range(K):
        ii = np.arange(K - k); jj = ii + k
        Wk = (v[ii] * v[jj] * M[ii, jj]).sum() * (1 if k == 0 else 2)
        if k == 0: W0 = Wk
        Wabs += abs(Wk)
    return W0 / Wabs, iqg, beta0


def sine_D(K, rho):
    """Dirichlet top-sine mode v_n=sin(n theta), theta=K pi/(K+1); D for Gaussian-banded m_k."""
    kap = 2 * erfinv(rho) ** 2
    n = np.arange(1, K + 1)
    theta = K * math.pi / (K + 1)
    v = np.sin(n * theta); v = v / np.linalg.norm(v)
    Wabs = 0.0; W0 = 0.0
    for k in range(K):
        mk = C0 * math.exp(-kap * k * k)
        ii = np.arange(K - k); jj = ii + k
        Wk = (v[ii] * v[jj] * mk).sum() * (1 if k == 0 else 2)
        if k == 0: W0 = Wk
        Wabs += abs(Wk)
    return W0 / Wabs


print("=" * 95)
print("3/8 cap:  D(K)=W0/sum|Wk| (surviving diagonal fraction of worst mode).  Limit -> 3/8 ?")
print("=" * 95)
for rho in (0.47, 0.40, 0.35, 0.30):
    print(f"\n--- rho={rho}  kappa={2*erfinv(rho)**2:.4f} ---")
    print("%2s | %7s %7s | %9s %9s" % ("K", "D_true", "D_sine", "<1/q>_g", "beta0"))
    Ds, iqs, b0s, Ks = [], [], [], []
    for K in range(2, 17):
        D, iqg, b0 = Dfrac_and_iqg(K, rho)
        Ds_ = sine_D(K, rho)
        Ds.append(D); iqs.append(iqg); b0s.append(b0); Ks.append(K)
        print("%2d | %7.4f %7.4f | %9.3f %9.3f" % (K, D, Ds_, iqg, b0))
    Ks = np.array(Ks, float)
    # large-K limit of D: fit D = Dinf + c/K
    Dn = np.array(Ds)
    Afit = np.vstack([np.ones(len(Ks)), 1.0 / Ks]).T
    cf, *_ = np.linalg.lstsq(Afit, Dn, rcond=None)
    Dinf = cf[0]
    iq = np.array(iqs); b0 = np.array(b0s)
    pR7 = np.polyfit(np.log(Ks[Ks <= 7] - 1)[1:], np.log((iq[Ks <= 7] / iq[0])[1:]), 1)[0]
    p0_7 = np.polyfit(np.log(Ks[Ks <= 7] - 1)[1:], np.log((b0[Ks <= 7] / b0[0])[1:]), 1)[0]
    print("  D_inf (1/K extrap) = %.4f  (3/8=0.375)" % Dinf)
    print("  p_R(K<=7 window) = %.3f ; p0(beta0,K<=7) = %.3f ; D_inf*p0 = %.3f (vs p_R %.3f)"
          % (pR7, p0_7, Dinf * p0_7, pR7))
