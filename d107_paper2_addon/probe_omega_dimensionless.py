# -*- coding: utf-8 -*-
"""
probe_omega_dimensionless.py -- omega_q in the dimensionless variable s := sigma/dmu = 1/(2 sqrt2 erfinv(rho)).

We seek omega_q*sigma = F(K, s) where s=sigma/dmu controls comb-overlap.
Two regimes:
  * dilute comb (large dmu, small s): omega_q -> array Nyquist pi/dmu => omega_q*sigma -> pi*s.
  * dense comb (small dmu, large s): Gaussian band-limit caps omega_q ~ O(1/sigma) => omega_q*sigma -> O(1).

Test the SINGLE-Gaussian-band-limited Nyquist model:
   omega_q*sigma = pi*s * g(K) * BL(s)
where g(K) in [0,1] is the finite-K fraction of Nyquist reached (-> 1 as K->inf),
and BL(s) is the Gaussian band-limit factor (-> 1 for small s, saturating for large s).

Empirically fit:
   (a) omega_q*sigma vs s at fixed large K -> get the s-law (regime crossover).
   (b) omega_q(K)/omega_q(inf) vs K -> the universal g(K).
Also connect to lambda_min exponent p(rho): since omega_q^2 = vAv/lam_min and vAv is mild,
   p(rho) = -d ln lam_min/d ln(K-1) is set mostly by d ln vAv - d ln omega_q^2.
Report all. Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_omega_dimensionless.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, DT = 25.0, 200.0, 0.5
TGRID = np.arange(50.0, 6000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def s_of_rho(rho): return SIGMA / dmu(rho)
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def omega_q(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    lam, V = np.linalg.eigh(P @ G @ P)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g); f = g - mg
    fp = np.gradient(f, TGRID)
    return math.sqrt(Eq(fp * fp) / Eq(f * f)), lam[tang[0]]


print("=" * 100)
print("(a) omega_q*sigma vs s=sigma/dmu at fixed large K (=14, near-converged)")
print("=" * 100)
Kbig = 14
print("%6s | %8s %8s | %12s %12s %12s" %
      ("rho", "s", "pi*s", "om_q*sigma", "/(pi*s)", "om_q*dmu"))
for rho in (0.25, 0.30, 0.35, 0.40, 0.47, 0.55, 0.65, 0.75):
    s = s_of_rho(rho)
    om, _ = omega_q(Kbig, rho)
    print("%6.2f | %8.4f %8.4f | %12.4f %12.4f %12.4f" %
          (rho, s, math.pi * s, om * SIGMA, om * SIGMA / (math.pi * s), om * dmu(rho)))

print("\n" + "=" * 100)
print("(b) universal g(K)=omega_q(K)/omega_q(Kbig) — K-shape independent of rho?")
print("=" * 100)
Kref = 16
print("%2s | %10s %10s | %10s %10s" % ("K", "g(K)@.47", "g(K)@.40", "g(K)@.30", "g(K)@.60"))
ref = {rho: omega_q(Kref, rho)[0] for rho in (0.47, 0.40, 0.30, 0.60)}
for K in range(2, 13):
    row = [omega_q(K, rho)[0] / ref[rho] for rho in (0.47, 0.40, 0.30, 0.60)]
    print("%2d | %10.4f %10.4f | %10.4f %10.4f" % (K, *row))

print("\n" + "=" * 100)
print("(c) exact link: p(rho)=-dln lam_min/dln(K-1);  omega_q^2=vAv/lam_min;  decompose")
print("=" * 100)
for rho in (0.47, 0.40):
    Ks = np.arange(2, 11, dtype=float)
    oms = []; lams = []
    for K in Ks:
        om, lm = omega_q(int(K), rho)
        oms.append(om); lams.append(lm)
    oms = np.array(oms); lams = np.array(lams)
    vAv = oms ** 2 * lams      # since omega_q^2 = vAv/lam_min
    p_lam = -np.polyfit(np.log(Ks - 1)[1:], np.log(lams[1:]), 1)[0]
    p_om2 = np.polyfit(np.log(Ks - 1)[1:], np.log((oms ** 2)[1:]), 1)[0]
    p_vAv = np.polyfit(np.log(Ks - 1)[1:], np.log(vAv[1:]), 1)[0]
    print("rho=%.2f:  lam_min~(K-1)^%.3f ;  omega_q^2~(K-1)^%.3f ;  vAv~(K-1)^%.3f"
          % (rho, -p_lam, p_om2, p_vAv))
    print("          check: p_lam = p_om2 - p_vAv  =>  %.3f = %.3f - %.3f = %.3f"
          % (p_lam, p_om2, p_vAv, p_om2 - p_vAv))
