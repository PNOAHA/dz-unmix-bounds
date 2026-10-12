# -*- coding: utf-8 -*-
"""
probe_S_exact.py -- EXACT S as a ratio of quadratic forms (Gram with a doubly-smoothed kernel),
and the correct omega_q(K) saturation law.

EXACT S:  S^2 = Var_q[g*K_bw]/Var_q[g],  g=(Σ v_i h_i)/q.
  Var_q[g] = v^T G v,  G_ij=∫ h_i h_j/q.
  Var_q[g*K_bw] = ∫ q (g*K_bw)^2 - (∫ q (g*K_bw))^2.
  Note g*K_bw = ((Σ v_i h_i)/q) * K_bw  (smoothing the RATIO, not the numerator). So define
     phi_i = (h_i/q) * K_bw   (KDE-smoothed score kernel, exactly the Sigma-side phi in the framework!)
  Then g*K_bw = Σ v_i phi_i, and
     Var_q[g*K_bw] = v^T Gs v - (v^T b)^2,   Gs_ij=∫ q phi_i phi_j,  b_i=∫ q phi_i.
  So S^2 = [v^T Gs v - (v^T b)^2] / (v^T G v).   EXACT, K x K.   <-- this is the bridge to Sigma.

This shows S is literally the bw-attenuation of the worst mode measured through the SAME phi kernel
that defines Sigma in the sandwich. Verify S_exact == S_field.

omega_q SATURATION: measured omega_q(K) rises then saturates toward a rho-set ceiling.
Fit omega_q(K) = omega_inf * sqrt(1 - c/K) or omega_inf*(1-a/K). Find omega_inf(rho) and relate to
the band-limited Nyquist:  the TRUE ceiling is NOT pi/dmu but the q-weighted/band-limited value.
Report omega_inf and the practical S(K) closed form using the saturating omega_q.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_S_exact.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 4000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    h = h / TRAPZ(h, TGRID, axis=1)[:, None]
    return h, mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def smoother(bw):
    return gauss(TGRID[:, None] - TGRID[None, :], 0.0, bw) * DT


def worst_v(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return V[:, tang[0]], h, q, G


def S_exact_and_field(K, rho, bw=BW):
    v, h, q, G = worst_v(K, rho)
    KS = smoother(bw)
    phi = np.array([KS @ (h[i] / q) for i in range(len(h))])     # phi_i = (h_i/q) * K_bw
    Gs = np.array([[TRAPZ(q * phi[i] * phi[j], TGRID) for j in range(len(h))] for i in range(len(h))])
    b = np.array([TRAPZ(q * phi[i], TGRID) for i in range(len(h))])
    vGv = float(v @ G @ v)
    num = float(v @ Gs @ v) - float(v @ b) ** 2
    S_exact = math.sqrt(max(num, 0) / vGv)
    # field value
    g = (v[:, None] * h).sum(0) / q
    gs = KS @ g
    Eq = lambda f: TRAPZ(f * q, TGRID)
    var_g = Eq(g * g) - Eq(g) ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S_field = math.sqrt(max(var_gs, 0) / var_g)
    fp = np.gradient(g - Eq(g), TGRID)
    om_q = math.sqrt(Eq(fp * fp) / (Eq(g * g) - Eq(g) ** 2))
    return S_exact, S_field, om_q


print("=" * 100)
print("EXACT S^2 = [v^T Gs v - (v^T b)^2]/(v^T G v),  Gs from phi_i=(h_i/q)*K_bw (the Sigma kernel)")
print("=" * 100)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %9s %9s | match%% | %8s" % ("K", "S_exact", "S_field", "om_q"))
    oms = []
    for K in range(2, 11):
        Se, Sf, om = S_exact_and_field(K, rho)
        oms.append((K, om))
        print("%2d | %9.5f %9.5f | %6.2f | %8.5f" % (K, Se, Sf, 100 * Se / Sf, om))
    # saturation fit: om_q^2 = om_inf^2 (1 - c/K)
    Ks = np.array([k for k, _ in oms], float)
    om = np.array([o for _, o in oms])
    # linearize: om^2 = om_inf^2 - om_inf^2 c / K  -> y=om^2, x=1/K linear
    A = np.vstack([np.ones_like(Ks), 1.0 / Ks]).T
    coef, *_ = np.linalg.lstsq(A, om ** 2, rcond=None)
    om_inf2, slope = coef
    cfit = -slope / om_inf2
    pred = A @ coef
    r2 = 1 - ((om ** 2 - pred) ** 2).sum() / ((om ** 2 - (om ** 2).mean()) ** 2).sum()
    print("  om_q^2 = om_inf^2 (1 - c/K):  om_inf=%.5f  c=%.3f  R2=%.4f   (Nyq pi/dmu=%.5f, ratio om_inf/Nyq=%.3f)"
          % (math.sqrt(om_inf2), cfit, r2, math.pi / dmu(rho), math.sqrt(om_inf2) / (math.pi / dmu(rho))))
    print("  om_inf*sigma=%.4f   (1/sigma=%.5f; om_inf vs 1/sigma ratio=%.3f)"
          % (math.sqrt(om_inf2) * SIGMA, 1.0 / SIGMA, math.sqrt(om_inf2) * SIGMA))
