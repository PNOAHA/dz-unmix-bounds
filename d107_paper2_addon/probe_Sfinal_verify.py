# -*- coding: utf-8 -*-
"""
probe_Sfinal_verify.py -- consolidated verification of the S(K,rho,bw) closed form.

CLAIM (S_direct):
  S(K,rho,bw) = exp( -1/2 * bw^2 * omega_q(K,rho)^2 )   [monochromatic; ~4-5% over the window]
  with the EXACT identity omega_q^2 = (v^T A v)/lambda_min(G),  and the practical law
      omega_q(K,rho)^2 = (kappa(rho)/sigma^2) * (K-1)/K        (affine-in-degree / chain top mode)
  kappa(rho) a pure number; equivalently omega_q^2 = C2(rho)*(K-1)/K with C2 in 1/Ma^2.
  Two-moment refinement: S = exp(-bw^2 m1/2 + bw^4 (m2-m1^2)/4).

This script:
  (1) fits omega_q^2 = C2(rho)*(K-1)/K  and reports C2, R2, and kappa=C2*sigma^2.
  (2) reports S(meas) vs S_mono(closed omega_q-fit) vs S_2mom over K=2..10, both rho => % match.
  (3) bw-scaling: at fixed (K,rho) sweep bw and confirm S=exp(-bw^2 om^2/2) shape AND check the
      claimed sigma_err ~ bw^{-0.5..-0.6} comes from the FULL excess (CR*S) + the Sigma normalization,
      not from S alone (S DECREASES with bw, lowering excess; the bw^-0.5 in sigma_err is the
      Sigma/sqrt(n) prefactor). Report d ln S / d ln bw to show S contributes a NEGATIVE log-slope.
  (4) rho-scaling of C2: C2(rho) = kappa/sigma^2 with kappa ~ (pi^2/?)*(sigma/dmu)^2 form? test
      C2(rho)*dmu(rho)^2 = const  (i.e. omega_q ~ 1/dmu, the comb-spacing law).
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_Sfinal_verify.py
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
    return g, q, v


def measure(K, rho, bw=BW):
    g, q, v = worst(K, rho)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID); fpp = np.gradient(fp, TGRID)
    den = Eq(f * f)
    m1 = Eq(fp * fp) / den
    m2 = Eq(fpp * fpp) / den
    varw = max(m2 - m1 * m1, 0.0)
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, bw) * DT
    gs = KS @ g
    S = math.sqrt(max(Eq(gs * gs) - Eq(gs) ** 2, 0) / den)
    return S, m1, varw


print("=" * 110)
print("(1)+(2) omega_q^2 = C2(rho)*(K-1)/K ;  S forms vs measured  (bw=%.0f, sigma=%.0f)" % (BW, SIGMA))
print("=" * 110)
C2_store = {}
for rho in (0.47, 0.40):
    Ks = np.arange(2, 11, dtype=float)
    rows = [measure(int(K), rho) for K in Ks]
    om2 = np.array([r[1] for r in rows])
    x = (Ks - 1) / Ks
    C2 = float(np.dot(x, om2) / np.dot(x, x))     # least-squares through origin
    pred_om2 = C2 * x
    r2 = 1 - ((om2 - pred_om2) ** 2).sum() / ((om2 - om2.mean()) ** 2).sum()
    C2_store[rho] = C2
    kappa = C2 * SIGMA ** 2
    print(f"\n--- rho={rho}: omega_q^2 = C2*(K-1)/K,  C2={C2:.6e} 1/Ma^2,  kappa=C2*sigma^2={kappa:.4f},  R2={r2:.4f}"
          f"   C2*dmu^2={C2*dmu(rho)**2:.4f} ---")
    print("%2s | %8s | %8s %6s | %8s %6s | (closed omega_q from fit)" %
          ("K", "S(meas)", "S_mono*", "err%", "S_2mom", "err%"))
    em = []; e2 = []
    for i, K in enumerate(Ks):
        S, m1, varw = rows[i]
        om2c = C2 * (K - 1) / K                    # CLOSED-FORM omega_q^2 (not measured)
        S_mono = math.exp(-0.5 * BW ** 2 * om2c)
        S_2mom = math.exp(-0.5 * BW ** 2 * m1 + 0.25 * BW ** 4 * varw)
        em.append(100 * (S_mono - S) / S); e2.append(100 * (S_2mom - S) / S)
        print("%2d | %8.4f | %8.4f %+5.1f | %8.4f %+5.1f |" %
              (int(K), S, S_mono, 100 * (S_mono - S) / S, S_2mom, 100 * (S_2mom - S) / S))
    print("  RMS  S_mono(closed)=%.2f%%  S_2mom(measured-moments)=%.2f%%" %
          (math.sqrt(np.mean(np.array(em) ** 2)), math.sqrt(np.mean(np.array(e2) ** 2))))

print("\n" + "=" * 110)
print("(4) rho-scaling: is omega_q ~ 1/dmu  (C2*dmu^2 = const across rho)?  fixed K=8")
print("=" * 110)
K = 8
print("%6s | %8s | %10s %12s %10s" % ("rho", "dmu", "om_q^2", "C2(=om2*K/(K-1))", "C2*dmu^2"))
vals = []
for rho in (0.30, 0.35, 0.40, 0.47, 0.55, 0.65):
    S, m1, varw = measure(K, rho)
    C2 = m1 * K / (K - 1)
    vals.append(C2 * dmu(rho) ** 2)
    print("%6.2f | %8.3f | %10.5e %12.5e %10.4f" % (rho, dmu(rho), m1, C2, C2 * dmu(rho) ** 2))
print("  C2*dmu^2 mean=%.4f  std=%.4f  (const => omega_q ~ 1/dmu => omega_q*sigma ~ 1/(2sqrt2 erfinv(rho)))"
      % (np.mean(vals), np.std(vals)))

print("\n" + "=" * 110)
print("(3) bw-scaling of S (fixed K=5,rho=0.47): S=exp(-bw^2 om^2/2) and d ln S/d ln bw")
print("=" * 110)
K, rho = 5, 0.47
_, om2_0, _ = measure(K, rho, bw=12.0)
prev = None
print("%6s | %8s %8s | %8s" % ("bw", "S(meas)", "S_mono", "dlnS/dlnbw"))
for bw in (4, 6, 8, 12, 16, 20, 24):
    S, m1, varw = measure(K, rho, bw=bw)
    Sm = math.exp(-0.5 * bw ** 2 * om2_0)
    slope = ""
    if prev is not None:
        slope = "%+.3f" % ((math.log(S) - math.log(prev[1])) / (math.log(bw) - math.log(prev[0])))
    print("%6.1f | %8.4f %8.4f | %8s" % (bw, S, Sm, slope))
    prev = (bw, S)
