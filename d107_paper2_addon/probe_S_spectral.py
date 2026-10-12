# -*- coding: utf-8 -*-
"""
probe_S_spectral.py -- the TWO-MOMENT spectral closed form for S, and a tight omega_q(K) law.

S^2 = Var_q[g*K_bw]/Var_q[g] = < exp(-bw^2 omega^2) >_rho   (rho = q-weighted power spectrum of f=g-<g>_q).
Monochromatic: S=exp(-bw^2 omega_q^2/2),  omega_q^2 = m1 = Eq[f'^2]/Eq[f^2]  (Jensen: UNDERestimates S).
Two-moment cumulant: S = exp(-bw^2 m1/2 + bw^4 Var(omega^2)/4),  m2=Eq[f''^2]/Eq[f^2], Var=m2-m1^2.

Plus tight omega_q(K) saturation-law fits.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_S_spectral.py
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


def moments_and_S(K, rho, bw=BW):
    g, q = worst(K, rho)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID)
    fpp = np.gradient(fp, TGRID)
    den = Eq(f * f)
    m1 = Eq(fp * fp) / den
    m2 = Eq(fpp * fpp) / den
    varw = max(m2 - m1 * m1, 0.0)
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, bw) * DT
    gs = KS @ g
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / den)
    S_mono = math.exp(-0.5 * bw ** 2 * m1)
    S_2mom = math.exp(-0.5 * bw ** 2 * m1 + 0.25 * bw ** 4 * varw)
    return S, S_mono, S_2mom, math.sqrt(m1), varw


print("=" * 110)
print("TWO-MOMENT spectral S = exp(-bw^2 m1/2 + bw^4 Var(w^2)/4);  m1=<w^2>=omega_q^2")
print("=" * 110)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %8s | %8s %7s | %8s %7s | %8s" %
          ("K", "S(meas)", "S_mono", "err%", "S_2mom", "err%", "om_q"))
    em = []; e2 = []
    for K in range(2, 11):
        S, Sm, S2, omq, varw = moments_and_S(K, rho)
        em.append(100 * (Sm - S) / S); e2.append(100 * (S2 - S) / S)
        print("%2d | %8.4f | %8.4f %+6.2f | %8.4f %+6.2f | %8.5f" %
              (K, S, Sm, 100 * (Sm - S) / S, S2, 100 * (S2 - S) / S, omq))
    print("  RMS err  mono=%.2f%%  2-moment=%.2f%%" %
          (math.sqrt(np.mean(np.array(em) ** 2)), math.sqrt(np.mean(np.array(e2) ** 2))))

print("\n" + "=" * 110)
print("omega_q(K) saturation law fits")
print("=" * 110)
for rho in (0.47, 0.40):
    Ks = np.arange(2, 13, dtype=float)
    omq = np.array([moments_and_S(int(K), rho)[3] for K in Ks])
    nyq = math.pi / dmu(rho)

    def f_s1(K, oinf): return oinf * np.sqrt(1 - 1.0 / K ** 2)
    def f_s2(K, oinf, a): return oinf * np.sqrt((K - 1) / (K + a))
    def f_s3(K, oinf, b): return oinf * (K - 1) / (K - 1 + b)
    out = {}
    for name, fn, p0 in [("S1 oinf*sqrt(1-1/K^2)", f_s1, [0.1]),
                         ("S2 oinf*sqrt((K-1)/(K+a))", f_s2, [0.1, 1.0]),
                         ("S3 oinf*(K-1)/(K-1+b)", f_s3, [0.1, 1.0])]:
        try:
            popt, _ = curve_fit(fn, Ks, omq, p0=p0, maxfev=20000)
            pred = fn(Ks, *popt)
            r2 = 1 - ((omq - pred) ** 2).sum() / ((omq - omq.mean()) ** 2).sum()
            out[name] = (popt, r2)
        except Exception:
            out[name] = (None, float('nan'))
    print(f"--- rho={rho}  Nyq=pi/dmu={nyq:.5f} ---")
    for name, (popt, r2) in out.items():
        ps = " ".join("%.4f" % x for x in popt) if popt is not None else "FAIL"
        oinf = popt[0] if popt is not None else float('nan')
        print("  %-30s params=[%s]  R2=%.4f  oinf/Nyq=%.3f" % (name, ps, r2, oinf / nyq))
