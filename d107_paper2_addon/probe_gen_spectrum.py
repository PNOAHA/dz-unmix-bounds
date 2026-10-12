# -*- coding: utf-8 -*-
"""
probe_gen_spectrum.py -- the generalized pencil (A,G) spectrum = squared local frequencies of
each contrast mode. omega_q^2 = mu_max(A,G) (the degree-(K-1), worst-Fisher, most-oscillatory mode).

Establish the closed form for omega_q^2*sigma^2 = sigma^2 * mu_max(A,G) as a function of (K,rho).

TESTS:
 (1) Per-degree generalized eigenvalues mu_d (d=1..K-1): is mu_d * sigma^2 ~ linear in d?
     i.e. omega_q^2 ~ (a + b*(K-1))/sigma^2  ->  omega_q^2*sigma^2 affine in (K-1)?
 (2) The continuum limit: for an INFINITE equally spaced Gaussian comb with spacing dmu and
     constant weights, the worst contrast is the Nyquist mode cos(pi k) (alternating +/-).
     Its real-space field g(t)=cos(pi*(t-START)/dmu)-like => local frequency omega = pi/dmu.
     Check omega_q -> pi/dmu as K->inf? (=> S floor). Compare omega_q to pi/dmu.
 (3) rho dependence of the prefactor: omega_q^2*sigma^2 vs rho at fixed K.
     dmu=2 sqrt2 erfinv(rho) sigma, so pi/dmu*sigma = pi/(2 sqrt2 erfinv(rho)). Check the
     Nyquist prediction omega_q*sigma -> pi*sigma/dmu = pi/(2 sqrt2 erfinv(rho)).
Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_gen_spectrum.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.linalg import eigh as geigh

SIGMA, START, DT = 25.0, 200.0, 0.5
TGRID = np.arange(50.0, 4000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    h = h / TRAPZ(h, TGRID, axis=1)[:, None]
    hp = np.stack([-(TGRID - m) / SIGMA ** 2 * h[i] for i, m in enumerate(mus)])
    return h, hp, mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def GA(K, rho, w=None):
    h, hp, mus = endmembers(K, rho)
    if w is None: w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qp = (w[:, None] * hp).sum(0)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    d = hp - h * (qp / q)[None, :]
    A = np.array([[TRAPZ(d[i] * d[j] / q, TGRID) for j in range(K)] for i in range(K)])
    return G, A


def gen_spectrum(K, rho, w=None):
    G, A = GA(K, rho, w)
    P = np.eye(K) - np.ones((K, K)) / K
    _, U = np.linalg.eigh(P)
    Tb = U[:, 1:]
    At = Tb.T @ A @ Tb; Gt = Tb.T @ G @ Tb
    mu = np.sort(np.real(geigh(At, Gt, eigvals_only=True)))
    return mu


print("=" * 100)
print("(1) per-degree gen-eigs mu_d*sigma^2  (d=1..K-1); is the spectrum affine in degree?")
print("=" * 100)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho}  dmu={dmu(rho):.3f}  pi/dmu*sigma={math.pi*SIGMA/dmu(rho):.4f} "
          f"(Nyquist omega*sigma); (pi/dmu*sigma)^2={(math.pi*SIGMA/dmu(rho))**2:.4f} ---")
    print("%2s | mu_d * sigma^2 (ascending, d=1..K-1)" % "K")
    for K in (3, 5, 7, 10):
        mu = gen_spectrum(K, rho)
        print("%2d | %s" % (K, " ".join("%.4f" % (m * SIGMA ** 2) for m in mu)))
    # the top eig = omega_q^2; how does omega_q*sigma approach pi/dmu*sigma?
    print("  omega_q*sigma (=sqrt(mu_max)*sigma) vs K  ->  Nyquist pi*sigma/dmu=%.4f:"
          % (math.pi * SIGMA / dmu(rho)))
    for K in range(2, 13):
        mu = gen_spectrum(K, rho)
        print("    K=%2d  omega_q*sigma=%.4f   ratio to Nyquist=%.3f"
              % (K, math.sqrt(mu[-1]) * SIGMA, math.sqrt(mu[-1]) * SIGMA / (math.pi * SIGMA / dmu(rho))))

print("\n" + "=" * 100)
print("(3) rho dependence of omega_q*sigma at fixed large K, vs Nyquist pi*sigma/dmu = pi/(2 sqrt2 erfinv(rho))")
print("=" * 100)
K = 10
print("%6s | %8s | %12s %12s | %8s" % ("rho", "dmu", "om_q*sigma", "Nyq=pi*s/dmu", "ratio"))
for rho in (0.30, 0.35, 0.40, 0.47, 0.55, 0.65):
    mu = gen_spectrum(K, rho)
    nyq = math.pi * SIGMA / dmu(rho)
    print("%6.2f | %8.3f | %12.4f %12.4f | %8.3f"
          % (rho, dmu(rho), math.sqrt(mu[-1]) * SIGMA, nyq, math.sqrt(mu[-1]) * SIGMA / nyq))
