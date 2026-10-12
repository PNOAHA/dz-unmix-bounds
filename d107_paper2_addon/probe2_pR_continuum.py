# -*- coding: utf-8 -*-
"""
probe2_pR_continuum.py -- p_R via the symbol-ratio (continuum worst-mode) model + the 3/8 cap.

CLEANEST formulation. <1/q>_g = (v^T G v)/(v^T M v).  Both M and the band-enhanced
G' := G (with G_ij = beta_{|i-j|} M_ij, beta_k=mean bg enhancement on band k -- verified slowly
varying) are ~Toeplitz.  For the worst mode (top frequency theta ~ pi(K-1)/K -> pi as K->inf),
each becomes its SYMBOL evaluated near theta=pi:
   v^T M v / |v|^2  -> f_M(theta) = sum_k m_k e^{i k theta} = c sum_k e^{-b k^2} cos(k theta)
   v^T G v / |v|^2  -> f_G(theta) = sum_k beta_k m_k cos(k theta)
So  <1/q>_g(theta) = f_G(theta)/f_M(theta) = (sum_k beta_k m_k cos k theta)/(sum_k m_k cos k theta).
At theta=pi: cos(k pi)=(-1)^k, so
   <1/q>_g = (beta0 m0 - 2 beta1 m1 + 2 beta2 m2 - ...)/(m0 - 2 m1 + 2 m2 - ...).
m_k = c e^{-b k^2} with b=kappa=2erfinv(rho)^2.

THE CAP: the per-peak self-bg beta0 ~ 1/(w_k h_peak) GROWS ~K (1/K dilution).  But beta1,beta2
(pair overlaps) grow SLOWER and enter with NEGATIVE signs at theta=pi, so the numerator's K-growth
is partially cancelled.  As K->inf the *exponent* of the ratio is set by how the betas' growth
rates differ across bands -- the prompt's claim is the asymptotic exponent caps near 3/8 in K.

This script:
 (1) measures beta_k(K,rho) (band bg enhancement) and m_k (exact),
 (2) builds the theta=pi alternating-sum model and checks it reproduces <1/q>_g,
 (3) extracts the asymptotic exponent of <1/q>_g (large K) and tests cap ~ 3/8,
 (4) derives beta0's growth: beta0 = int h^2/q.  With q = sum_j w_j h_j and at peak k center
     the neighbors add, model beta0 ~ 1/(w_k_typ) up to a neighbor-overlap factor -> the 1/K.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_pR_continuum.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 6000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
C0 = 1.0 / (2 * SIGMA * math.sqrt(math.pi))     # m_0 = c


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def measure(K, rho):
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
    # band bg enhancement (use central bands, average to reduce edge effects)
    beta = []
    mk = []
    for k in range(min(K, 6)):
        ii = np.arange(K - k); jj = ii + k
        beta.append((G[ii, jj] / M[ii, jj]).mean())
        mk.append(M[ii, jj].mean())
    return iqg, np.array(beta), np.array(mk), v


def theta_pi_model(beta, mk):
    K = len(beta)
    signs = np.array([1.0] + [2.0 * (-1) ** k for k in range(1, K)])
    num = (signs * beta * mk).sum()
    den = (signs * mk).sum()
    return num / den


print("=" * 95)
print("theta=pi alternating-band model:  <1/q>_g = (sum_k s_k beta_k m_k)/(sum_k s_k m_k), s=(-1)^k")
print("m_k = c exp(-kappa k^2) [exact];  beta_k = band bg enhancement [measured]")
print("=" * 95)
for rho in (0.47, 0.40, 0.35, 0.30):
    kap = 2 * erfinv(rho) ** 2
    print(f"\n--- rho={rho}  kappa={kap:.4f} ---")
    print("%2s | %9s %9s | %8s | beta_k (bg enh per band)" % ("K", "<1/q>_g", "model", "ratio"))
    iqs, Ks = [], []
    for K in range(2, 13):
        iqg, beta, mk, v = measure(K, rho)
        mod = theta_pi_model(beta, mk)
        iqs.append(iqg); Ks.append(K)
        bstr = " ".join("%7.1f" % b for b in beta[:5])
        print("%2d | %9.3f %9.3f | %8.3f | %s" % (K, iqg, mod, mod / iqg, bstr))
    Ks = np.array(Ks, float); iq = np.array(iqs)
    # asymptotic exponent: use the LAST few K only (large-K slope)
    pR_all = np.polyfit(np.log(Ks - 1)[1:], np.log(iq[1:] / iq[0]), 1)[0]
    tail = Ks >= 7
    pR_tail = np.polyfit(np.log((Ks - 1)[tail]), np.log(iq[tail]), 1)[0]
    pR_tailK = np.polyfit(np.log(Ks[tail]), np.log(iq[tail]), 1)[0]
    print("  p_R full[K-1]=%.3f ; TAIL(K>=7) p_R[K-1]=%.3f / [K]=%.3f   (cap target ~0.375)"
          % (pR_all, pR_tail, pR_tailK))
