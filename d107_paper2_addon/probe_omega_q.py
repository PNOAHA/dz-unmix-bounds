# -*- coding: utf-8 -*-
"""
probe_omega_q.py -- derive the q-weighted Rayleigh frequency omega_q(K,rho) that drives S.

Established (probe_Sdirect): S(K) = exp(-bw^2 omega_q^2 / 2) to ~4-8%, with
   omega_q^2 = Eq[(g')^2] / Eq[g^2],   g = g_v - mean_q(g_v),  g_v = (sum v_i h_i)/q.

This script:
  1. Confirms the Rayleigh identity is exact for the smoothing ratio in the bw->0 expansion:
        Var_q[g*K_bw] = Var_q[g] - bw^2 Eq[(g')^2] + O(bw^4)
     so S^2 = 1 - bw^2 omega_q^2 + O(bw^4), and exp(-bw^2 omega_q^2/2)=1-bw^2 omega_q^2/2+...
     (this is why S~exp form works; check the leading 1-x vs measured).
  2. Decomposes omega_q^2 = Eq[(g')^2]/Eq[g^2] using g = sum v_i h_i / q.
     Near peak k, q ~ w_k h_k (dominant), so g ~ (sum_i v_i h_i)/(sum_j w_j h_j).
     Test the LOCAL approximation: in the cell around peak k, g(t) ~ v_k/w_k + corrections,
     and g' is dominated by the boundary regions between peaks where the ratio swings.
  3. Build the analytic worst-mode v = degree-(K-1) discrete contrast (Gram-Schmidt of x^deg),
     and check omega_q computed from it (no eigensolve) matches the eigvec-based omega_q.
  4. Fit omega_q(K) scaling and connect to dmu/sigma.

Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_omega_q.py
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


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean()
    B = []
    for deg in range(1, K):
        v = x ** deg; v = v - v.mean()
        for b in B:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            B.append(v / nv)
    return B


def gram_eigvec(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    return V[:, kmin], lam[kmin], h, q, w, mus


def omega_q_from_v(v, h, q):
    g = (v[:, None] * h).sum(0) / q
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID)
    num = Eq(fp * fp); den = Eq(f * f)
    return math.sqrt(num / den), den, num


def measure(K, rho):
    v, lam, h, q, w, mus = gram_eigvec(K, rho)
    g = (v[:, None] * h).sum(0) / q
    gs = smooth(g)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g)
    var_g = Eq(g * g) - mg ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / var_g)
    om_q, den, num = omega_q_from_v(v, h, q)
    S_exp = math.exp(-0.5 * BW ** 2 * om_q ** 2)
    S_lin = math.sqrt(max(1 - BW ** 2 * om_q ** 2, 0))   # leading 1-bw^2 om^2 form

    # analytic contrast worst mode (highest degree K-1)
    B = contrast_polys(K)
    v_an = B[-1]
    # sign/normalization don't matter for omega_q (scale-invariant Rayleigh)
    om_q_an, _, _ = omega_q_from_v(v_an, h, q)
    ov = abs(float(v @ v_an))
    return dict(K=K, rho=rho, S=S, om_q=om_q, S_exp=S_exp, S_lin=S_lin,
                om_q_an=om_q_an, ov=ov)


print("=" * 100)
print("omega_q(K,rho) and S forms; analytic degree-(K-1) contrast reproduction")
print("=" * 100)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %8s | %9s %8s %8s | %9s %6s | %6s"
          % ("K", "S(meas)", "om_q", "S_exp", "S_lin", "om_q_an", "v.an", "ratio"))
    oms = []
    for K in range(2, 11):
        d = measure(K, rho)
        oms.append((K, d["om_q"]))
        print("%2d | %8.4f | %9.5f %8.4f %8.4f | %9.5f %6.3f | %6.3f"
              % (K, d["S"], d["om_q"], d["S_exp"], d["S_lin"], d["om_q_an"],
                 d["ov"], d["om_q_an"] / d["om_q"]))
    # how does omega_q scale with K?  fit log om_q vs log(K-1) and vs log K
    Ks = np.array([k for k, _ in oms], float)
    omv = np.array([o for _, o in oms])
    p1 = np.polyfit(np.log(Ks - 1)[1:], np.log(omv[1:]), 1)[0]
    p2 = np.polyfit(np.log(Ks)[1:], np.log(omv[1:]), 1)[0]
    # also test asymptote: om_q -> const? fit om_q = a - b/(K-1)
    print("  omega_q ~ (K-1)^%.3f ;  ~ K^%.3f ;  om_q(10)/om_q(2)=%.3f" %
          (p1, p2, omv[-1] / omv[0]))
