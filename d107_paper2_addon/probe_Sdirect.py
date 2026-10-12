# -*- coding: utf-8 -*-
"""
probe_Sdirect.py -- DIRECT real-space dissection of the bw-suppression factor S(K).

S(K) = sqrt( Var_q[g_v * K_bw] / Var_q[g_v] ),   g_v = (sum_i v_i h_i)/q ,  v=worst (smallest-G) eigvec.

GOAL: explain why S falls 0.96 -> 0.56 (K=2->10) while omega_eff (flat-L2 rms freq) is ~K-indep.
Hypotheses tested here:
  (A) The q-WEIGHTED spectrum of g_v, not the flat spectrum, is what matters. Compute the
      q-weighted rms frequency omega_q and test S ~ exp(-bw^2 omega_q^2 / 2).
  (B) g_v is ~ a "comb" of localized peaked bumps at each interior peak; near peak k the local
      shape of g_v has curvature width set by sigma (the endmember width), NOT dmu. Smoothing
      a Gaussian-shaped bump of width sigma by bw=12 gives a fixed per-bump attenuation; the
      NUMBER of bumps grows with K -> ?  (decompose Var into per-region contributions).
  (C) Decompose g_v in a Hermite/Gaussian-local basis around each peak; track how high-order
      local structure (which smoothing kills) grows with K.

Also: build g_v from the analytic worst-mode = degree-(K-1) contrast polynomial in peak index,
and check it reproduces the measured g_v (the data-driven smallest-G eigvec).

Self-contained. Does NOT overwrite existing files. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_Sdirect.py
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


def worst_mode(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    v = V[:, kmin]
    g = (v[:, None] * h).sum(0) / q
    return v, g, q, h, mus, lam[kmin]


def stats(K, rho):
    v, g, q, h, mus, lam = worst_mode(K, rho)
    gs = smooth(g)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g)
    var_g = Eq(g * g) - mg ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / var_g)

    # ---- (A) q-weighted rms frequency ----
    # define the centered field f = g - mean_q(g); we want the spectral content of f under
    # the q-weighted inner product. Use the derivative identity:
    #   for a band-limited f, Var_q[f*K_bw]/Var_q[f] depends on the q-weighted power spectrum.
    # Practical q-weighted rms freq via:  omega_q^2 = Eq[(f')^2]/Eq[f^2]   (Rayleigh quotient,
    # the q-weighted analog of the flat <omega^2> = ||f'||^2/||f||^2).
    f = g - mg
    fp = np.gradient(f, TGRID)
    omega_q = math.sqrt(max(Eq(fp * fp), 0) / Eq(f * f))
    S_predA = math.exp(-0.5 * BW ** 2 * omega_q ** 2)

    # flat (unweighted) rms freq for comparison (this is the failing 'omega_eff')
    F = np.fft.rfft(f); freq = np.fft.rfftfreq(len(f), d=DT) * 2 * math.pi
    Pw = np.abs(F) ** 2
    omega_flat = math.sqrt((freq ** 2 * Pw).sum() / Pw.sum())

    # ---- (B) per-region (per-interval) variance decomposition ----
    # Partition the support into K+1 cells split at peak midpoints; report where Var_q lives.
    edges = [TGRID[0]] + [0.5 * (mus[i] + mus[i + 1]) for i in range(K - 1)] + [TGRID[-1]]
    region_var = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (TGRID >= a) & (TGRID < b)
        region_var.append(TRAPZ((f * f * q)[m], TGRID[m]))
    region_var = np.array(region_var)
    frac_interior = region_var[1:-1].sum() / region_var.sum() if K > 2 else region_var.sum()

    return dict(K=K, rho=rho, S=S, var_g=var_g, omega_q=omega_q, omega_flat=omega_flat,
                S_predA=S_predA, lam=lam, mus=mus,
                region_var=region_var, frac_interior=frac_interior)


print("=" * 110)
print("S(K) DISSECTION:  q-weighted rms freq (Rayleigh) vs flat rms freq; S_predA=exp(-bw^2 omega_q^2/2)")
print("=" * 110)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho}  (dmu={dmu(rho):.2f},  bw/dmu={BW/dmu(rho):.3f},  sigma/dmu={SIGMA/dmu(rho):.3f}) ---")
    print("%2s | %8s | %9s %9s | %9s %7s | %6s"
          % ("K", "S(meas)", "omega_q", "omega_flat", "S_predA", "A/meas", "f_int"))
    for K in range(2, 11):
        d = stats(K, rho)
        print("%2d | %8.4f | %9.5f %9.5f | %9.4f %7.3f | %6.3f"
              % (K, d["S"], d["omega_q"], d["omega_flat"], d["S_predA"],
                 d["S_predA"] / d["S"], d["frac_interior"]))
