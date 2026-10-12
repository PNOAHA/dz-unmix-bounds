# -*- coding: utf-8 -*-
"""
probe_omega_local.py -- LOCAL (per-peak) closed form for omega_q^2 = Eq[(g')^2]/Eq[g^2].

Structure of g_v near peak k:
  q(t) ~ sum_j w_j h_j(t).  Near peak k (|t-mu_k| <~ sigma) the dominant terms are h_{k-1},h_k,h_{k+1}.
  g_v = (sum_i v_i h_i)/q.  Define the LOCAL log-spacing field.  Because endmembers are Gaussian,
  h_j(t)/h_k(t) = exp( -[(t-mu_j)^2-(t-mu_k)^2]/(2 sigma^2) )
               = exp( (mu_j-mu_k)(2t-mu_j-mu_k)/(2 sigma^2) ).
  At the peak mu_k these neighbor ratios are exp(-dmu^2/2sigma^2) (=ratio r) etc.

KEY analytic object: the "peak values" of g_v.  Right AT t=mu_k:
   g_v(mu_k) = (sum_i v_i h_i(mu_k)) / (sum_j w_j h_j(mu_k)).
The slope g_v'(mu_k) and curvature drive Eq[(g')^2].  Because q is heavily peaked (sigma small vs.
range), Eq[f] ~ sum_k w_k * (local average of f over the peak-k Gaussian).

So define discrete sequences:
   a_k = g_v(mu_k)                              (peak value of the worst-mode field)
   and approximate omega_q^2 ~ <(g')^2>_q / <g^2>_q with the q mass = sum_k w_k delta-ish at mu_k.

This script tests TWO local models:
 (M1) PURE-PEAK (nearest-neighbor) ratio model: keep only h_{k-1},h_k,h_{k+1} near peak k; r=exp(-dmu^2/2sigma^2).
      Predict g_v on a local grid, compute Eq[(g')^2]/Eq[g^2] and compare to full.
 (M2) DISCRETE model: treat g as sampled sequence a_k=g(mu_k); the q-weighted variance ~ sum w_k (a_k-<a>)^2,
      and Eq[(g')^2] ~ (between-peak oscillation). Test omega_q^2 ~ (1/sigma_eff^2)*<a-mean curvature>.

The point: find what feature of the degree-(K-1) contrast makes omega_q grow ~(K-1)^0.4.
Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_omega_local.py
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


def full_omega_q(K, rho, v):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    g = (v[:, None] * h).sum(0) / q
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID)
    return math.sqrt(Eq(fp * fp) / Eq(f * f))


def peak_values(K, rho, v):
    """a_k = g_v(mu_k) computed from the full field (exact peak samples)."""
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    g = (v[:, None] * h).sum(0) / q
    a = np.array([g[np.argmin(np.abs(TGRID - m))] for m in mus])
    return a, mus, w


# ---- DISCRETE model M2 ----
# q mass near peak k ~ w_k.  g is smooth between peaks with characteristic scale = dmu in t.
# Approximate g(t) by linear interpolation of a_k across peaks; then
#   Eq[(g')^2] ~ sum_k w_k * < g'^2 over cell k >.  With g' ~ (a_{k+1}-a_k)/dmu on each side,
#   and Eq[g^2] ~ sum_k w_k (a_k - abar)^2,  abar = sum_k w_k a_k / sum w_k.
# This gives omega_q^2 ~ [sum_k w_k * ((Da)_k/dmu)^2_avg] / [sum_k w_k (a_k-abar)^2].
def discrete_omega_q(a, w, dm):
    wn = w / w.sum()
    abar = (wn * a).sum()
    var = (wn * (a - abar) ** 2).sum()
    # symmetric finite-difference slope at each peak from neighbors
    slope2 = np.zeros_like(a)
    for k in range(len(a)):
        if 0 < k < len(a) - 1:
            s = (a[k + 1] - a[k - 1]) / (2 * dm)
        elif k == 0:
            s = (a[1] - a[0]) / dm
        else:
            s = (a[-1] - a[-2]) / dm
        slope2[k] = s * s
    num = (wn * slope2).sum()
    return math.sqrt(num / var), abar, var


print("=" * 100)
print("LOCAL/DISCRETE model for omega_q:  a_k=g_v(mu_k), discrete Rayleigh with spacing dmu")
print("=" * 100)
for rho in (0.47, 0.40):
    dm = dmu(rho)
    print(f"\n--- rho={rho}  dmu={dm:.3f} ---")
    print("%2s | %9s %9s %8s | peak-values a_k (degree-(K-1) contrast field)"
          % ("K", "om_full", "om_disc", "ratio"))
    for K in range(2, 11):
        v = contrast_polys(K)[-1]
        omf = full_omega_q(K, rho, v)
        a, mus, w = peak_values(K, rho, v)
        omd, abar, var = discrete_omega_q(a, w, dm)
        astr = " ".join("%+.2f" % x for x in (a / np.abs(a).max()))
        print("%2d | %9.5f %9.5f %8.3f | %s" % (K, omf, omd, omd / omf, astr))
