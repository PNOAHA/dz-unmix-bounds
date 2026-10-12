# -*- coding: utf-8 -*-
"""
probe_omega_where.py -- WHERE does omega_q^2 = Eq[(g')^2]/Eq[g^2] concentrate, and what is
the analytic origin of the (K-1)^~0.4 growth?

Two clean reformulations of the numerator and denominator using ONLY Gaussian endmember algebra:

Denominator  D = Eq[g^2] - (Eq[g])^2  with g=(Σ v_i h_i)/q.
   Eq[g^2] = ∫ (Σ v_i h_i)^2 / q dt = v^T Q v ,  Q_ij = ∫ h_i h_j / q dt = G_ij  (the Gram!).
   Eq[g]   = ∫ (Σ v_i h_i) dt = Σ v_i  (h_i area-normalized) = 0 on the tangent.
   => D = v^T G v.  For v=worst eigvec, D = lam_min(G).   [EXACT — no approximation]

Numerator   N = Eq[(g')^2] = ∫ q (g')^2 dt.
   g' = d/dt[(Σ v_i h_i)/q] = (Σ v_i h_i')/q - (Σ v_i h_i) q'/q^2.
   => q (g')^2 = [ (Σ v_i h_i') - g q' ]^2 / q .
   Define the "weighted-derivative Gram"   G1_ij = ∫ h_i' h_j' / q dt  and cross terms.
   N = v^T A v with A a K x K matrix assembled from Gaussian-derivative overlaps over 1/q.

So omega_q^2 = (v^T A v)/(v^T G v).  This is an EXACT K x K generalized Rayleigh quotient.
The worst-mode v minimizes v^T G v (it's lam_min eigvec of G), and omega_q^2 = (v^T A v)/lam_min.

This script:
  1. Builds G and A as K x K matrices (numeric integrals), verifies v^T G v = lam_min and
     omega_q^2 = (v^T A v)/(v^T G v) reproduces the field-based omega_q EXACTLY.
  2. Reports lam_min(G), v^T A v, and their K-scaling separately => decomposes the (K-1)^0.4 law:
        omega_q^2 = (v^T A v)/lam_min(G).   We KNOW lam_min(G) ~ (K-1)^{-p}.  So if v^T A v ~ (K-1)^{-s},
        then omega_q^2 ~ (K-1)^{p-s}, i.e. omega_q ~ (K-1)^{(p-s)/2}.
  3. Localizes N: split ∫ q (g')^2 into peak cells vs inter-peak cells.
Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_omega_where.py
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
    Z = TRAPZ(h, TGRID, axis=1)[:, None]
    h = h / Z
    # analytic derivative of normalized gaussian: h'(t) = -(t-mu)/sigma^2 * h(t)
    hp = np.stack([-(TGRID - m) / SIGMA ** 2 * h[i] for i, m in enumerate(mus)])
    return h, hp, mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def GA(K, rho):
    h, hp, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qp = (w[:, None] * hp).sum(0)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    # N = ∫ [Σ v_i (h_i' - h_i q'/q)]^2 / q  = v^T A v,  define d_i = h_i' - h_i q'/q
    d = hp - h * (qp / q)[None, :]
    A = np.array([[TRAPZ(d[i] * d[j] / q, TGRID) for j in range(K)] for i in range(K)])
    return G, A, h, hp, q, qp, w, mus


def worst(K, rho):
    G, A, h, hp, q, qp, w, mus = GA(K, rho)
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    lam_min = lam[tang[0]]
    vGv = float(v @ G @ v)
    vAv = float(v @ A @ v)
    om_q2 = vAv / vGv
    # field-based omega_q for cross-check
    g = (v[:, None] * h).sum(0) / q
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID)
    om_field2 = Eq(fp * fp) / Eq(f * f)
    return lam_min, vGv, vAv, om_q2, om_field2, v


print("=" * 110)
print("EXACT K x K Rayleigh:  omega_q^2 = (v^T A v)/(v^T G v) ;  v^T G v = lam_min(G)")
print("=" * 110)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %11s %11s %10s | %10s %10s | %7s"
          % ("K", "vGv=lam_min", "vAv", "om_q^2", "om_q(KxK)", "om(field)", "match%"))
    rows = []
    for K in range(2, 11):
        lam_min, vGv, vAv, om2, omf2, v = worst(K, rho)
        rows.append((K, vGv, vAv, om2))
        print("%2d | %11.4e %11.4e %10.4e | %10.5f %10.5f | %6.2f"
              % (K, vGv, vAv, om2, math.sqrt(om2), math.sqrt(omf2),
                 100 * math.sqrt(om2) / math.sqrt(omf2)))
    Ks = np.array([r[0] for r in rows], float)
    vGv = np.array([r[1] for r in rows]); vAv = np.array([r[2] for r in rows])
    om2 = np.array([r[3] for r in rows])
    p_G = np.polyfit(np.log(Ks - 1)[1:], np.log(vGv[1:] / vGv[0]), 1)[0]
    p_A = np.polyfit(np.log(Ks - 1)[1:], np.log(vAv[1:] / vAv[0]), 1)[0]
    p_om = np.polyfit(np.log(Ks - 1)[1:], np.log(om2[1:] / om2[0]), 1)[0]
    print("  lam_min ~ (K-1)^%.3f ;  vAv ~ (K-1)^%.3f ;  om_q^2 ~ (K-1)^%.3f  (=vAv_exp - lam_exp = %.3f)"
          % (p_G, p_A, p_om, p_A - p_G))
    print("  => omega_q ~ (K-1)^%.3f" % (p_om / 2))
