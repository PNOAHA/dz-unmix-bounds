# -*- coding: utf-8 -*-
"""
probe_S_final.py -- consolidated, fully-closed S(K) and end-to-end verification.

ESTABLISHED:
  S^2(K) = Var_q[g*K_bw]/Var_q[g] = < exp(-bw^2 w^2) >_{rho_K}(spectral avg over q-weighted power
  spectrum rho_K of f=g-<g>_q).
  Monochromatic collapse: S = exp(-bw^2 omega_q^2/2), with the RIGOROUS q-effective frequency
     omega_q^2(K) = Eq[(g')^2] / Eq[(g-<g>)^2]  = (q-weighted Dirichlet energy)/(q-weighted variance)
                  = a RAYLEIGH QUOTIENT.   <-- THIS resolves the puzzle (grows with K; FFT omega_eff
                  was flat because it is corrupted by g's tail blow-up carrying ~0 Var_q).
  Two-moment (cumulant) correction removes Jensen bias:
     S = exp(-bw^2 omega_q^2/2 + bw^4 Var_q-spec(w^2)/4),  m2=Eq[(g'')^2]/Eq[(g-<g>)^2], Var=m2-m1^2.

  omega_q(K) closed form (saturation toward Nyquist of the K-1 contrast comb):
     omega_q(K) = omega_inf * sqrt((K-1)/(K+a)).   (R^2>0.997)

THIS SCRIPT:
 (1) FINAL S model = two-moment with omega_q from the CLOSED saturation law (fit omega_inf,a per rho).
     Report % match vs true S table, K=2..10, rho=0.47 & 0.40.
 (2) the FULLY-analytic-frequency variant: omega_q from the Rayleigh quotient computed on the
     analytic worst mode (no per-point fit) -- already have it; report.
 (3) rho-dependence of omega_inf and a: relate omega_inf to Nyquist pi/dmu(rho).
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import curve_fit

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 4000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


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
    return g, q


def measure(K, rho):
    g, q = worst(K, rho)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID); fpp = np.gradient(fp, TGRID)
    den = Eq(f * f)
    m1 = Eq(fp * fp) / den
    m2 = Eq(fpp * fpp) / den
    varw = max(m2 - m1 * m1, 0.0)
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
    gs = KS @ g
    S_true = math.sqrt(max(Eq(gs * gs) - Eq(gs) ** 2, 0) / den)
    return S_true, math.sqrt(m1), varw


print("FINAL CLOSED-FORM S(K) VERIFICATION")
print("=" * 100)
allres = {}
for rho in (0.47, 0.40):
    Ks = np.arange(2, 11, dtype=float)
    Strue = np.zeros(len(Ks)); omq = np.zeros(len(Ks)); varw = np.zeros(len(Ks))
    for i, K in enumerate(Ks):
        Strue[i], omq[i], varw[i] = measure(int(K), rho)
    # closed-form omega_q via saturation law
    def f_sat(K, oinf, a): return oinf * np.sqrt((K - 1) / (K + a))
    popt, _ = curve_fit(f_sat, Ks, omq, p0=[0.13, 5.0], maxfev=40000)
    om_closed = f_sat(Ks, *popt)
    nyq = math.pi / dmu(rho)
    # also closed form for varw: it is ~ (const)*m1^2 ? measure ratio
    rvar = varw / omq ** 4   # varw / m1^2
    # Models:
    S_mono_meas = np.exp(-0.5 * BW ** 2 * omq ** 2)                      # uses measured omega_q
    S_2mom_meas = np.exp(-0.5 * BW ** 2 * omq ** 2 + 0.25 * BW ** 4 * varw)
    S_mono_clsd = np.exp(-0.5 * BW ** 2 * om_closed ** 2)                # uses CLOSED omega_q
    # closed varw: varw ~ c_v * omq^4 ; use median c_v
    cv = np.median(rvar)
    S_2mom_clsd = np.exp(-0.5 * BW ** 2 * om_closed ** 2 + 0.25 * BW ** 4 * cv * om_closed ** 4)
    print(f"\n--- rho={rho}   omega_inf=%.4f (=%.3f*Nyq, Nyq=%.4f)  a=%.3f  c_v(varw/m1^2)=%.3f ---"
          % (popt[0], popt[0] / nyq, nyq, popt[1], cv))
    print("  %2s | %8s | %8s %5s | %8s %5s | %8s %5s | %8s %5s"
          % ("K", "S_true", "mono_m", "e%", "2mom_m", "e%", "mono_c", "e%", "2mom_c", "e%"))
    for i, K in enumerate(Ks):
        print("  %2d | %8.4f | %8.4f %+5.1f | %8.4f %+5.1f | %8.4f %+5.1f | %8.4f %+5.1f"
              % (K, Strue[i],
                 S_mono_meas[i], 100 * (S_mono_meas[i] - Strue[i]) / Strue[i],
                 S_2mom_meas[i], 100 * (S_2mom_meas[i] - Strue[i]) / Strue[i],
                 S_mono_clsd[i], 100 * (S_mono_clsd[i] - Strue[i]) / Strue[i],
                 S_2mom_clsd[i], 100 * (S_2mom_clsd[i] - Strue[i]) / Strue[i]))
    def rms(m): return math.sqrt(np.mean((100 * (m - Strue) / Strue) ** 2))
    def mx(m): return np.max(np.abs(100 * (m - Strue) / Strue))
    print("  RMS%%  mono_meas=%.2f 2mom_meas=%.2f mono_clsd=%.2f 2mom_clsd=%.2f | MAX 2mom_clsd=%.2f"
          % (rms(S_mono_meas), rms(S_2mom_meas), rms(S_mono_clsd), rms(S_2mom_clsd), mx(S_2mom_clsd)))
    allres[rho] = (popt, nyq, cv)

print("\nrho-dependence summary:")
for rho in (0.47, 0.40):
    popt, nyq, cv = allres[rho]
    print("  rho=%.2f: omega_inf=%.4f  omega_inf/Nyq=%.3f  a=%.2f  c_v=%.3f"
          % (rho, popt[0], popt[0] / nyq, popt[1], cv))
