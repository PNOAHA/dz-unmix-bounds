# -*- coding: utf-8 -*-
"""
probe_closedform.py -- FINAL closed form for S(K,rho,bw) via omega_q -> Nyquist of the Gaussian comb.

PHYSICAL PICTURE (established by probe_gen_spectrum):
  * worst (smallest-Fisher) mode = degree-(K-1) contrast = the Nyquist standing wave of the K-point
    Gaussian array (alternating sign). Its real-space "local frequency" is omega_q.
  * For an INFINITE array (K->inf), the alternating mode has wavevector k=pi/dmu; the field
    g(t)=(Σ (-1)^i h_i)/q is a band-limited cosine at omega_Nyq=pi/dmu (verified ratio->1).
  * For FINITE K, only K points sample [0, pi/dmu]; the top discrete mode sits at
       omega_q = (pi/dmu) * sin( pi/2 * (K-1)/K )                (standing-wave-on-a-string analog)
    This is the standard 1D-chain dispersion: a chain of N=K sites, free ends, the highest normal
    mode has wavenumber k_max = pi*(K-1)/K /a (a=dmu), but the *effective continuous frequency* of
    a discrete cosine cos(k_max * j) sampled and band-interpolated is omega = (2/a) sin(k_max a/2).
    Test BOTH:
       (P1) omega_q = (pi/dmu) * (K-1)/K               [linear-in-degree fraction]
       (P2) omega_q = (2/dmu) * sin( pi/2 * (K-1)/K )  [chain dispersion]
    against measured omega_q.

  * Gaussian band-limit: each h has spectrum exp(-sigma^2 omega^2/2); the ratio g=num/q does not
    simply inherit this, but the SMOOTHING acts as exp(-bw^2 omega_q^2/2). The relevant scale is the
    competition bw vs dmu (strong smoothing when bw ~ dmu).

FINAL:  S(K,rho,bw) = exp( -bw^2 omega_q(K,rho)^2 / 2 ),
        omega_q(K,rho) = (2/dmu(rho)) * sin( (pi/2)*(K-1)/K ),  dmu=2 sqrt2 erfinv(rho) sigma.
Verify against measured S (the field/eigvec value) and against compute_factors S.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_closedform.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.linalg import eigh as geigh

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
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


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f


def measured(K, rho, bw=BW):
    h, hp, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, bw) * DT
    gs = KS @ g
    Eq = lambda f: TRAPZ(f * q, TGRID)
    mg = Eq(g)
    var_g = Eq(g * g) - mg ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / var_g)
    fp = np.gradient(g - mg, TGRID)
    om_q = math.sqrt(Eq(fp * fp) / Eq((g - mg) ** 2))
    return S, om_q


print("=" * 110)
print("CLOSED FORM:  omega_q ~ (2/dmu) sin( (pi/2)(K-1)/K ) ;  S = exp(-bw^2 omega_q^2/2)")
print("=" * 110)
for rho in (0.47, 0.40):
    dm = dmu(rho)
    print(f"\n--- rho={rho}  dmu={dm:.3f} ---")
    print("%2s | %8s %8s %8s | %8s | %8s %8s %8s | err%%(S2) err%%(S1)"
          % ("K", "om(meas)", "om_P1", "om_P2", "S(meas)", "S_P2", "S_P1", "S_exp_om"))
    e2 = []; e1 = []
    for K in range(2, 11):
        S, om = measured(K, rho)
        omP1 = (math.pi / dm) * (K - 1) / K
        omP2 = (2.0 / dm) * math.sin((math.pi / 2) * (K - 1) / K)
        S_P1 = math.exp(-0.5 * BW ** 2 * omP1 ** 2)
        S_P2 = math.exp(-0.5 * BW ** 2 * omP2 ** 2)
        S_exp = math.exp(-0.5 * BW ** 2 * om ** 2)
        e2.append(100 * (S_P2 - S) / S); e1.append(100 * (S_P1 - S) / S)
        print("%2d | %8.5f %8.5f %8.5f | %8.4f | %8.4f %8.4f %8.4f | %+7.2f %+7.2f"
              % (K, om, omP1, omP2, S, S_P2, S_P1, S_exp, 100 * (S_P2 - S) / S, 100 * (S_P1 - S) / S))
    print("  RMS err: S_P2(chain)=%.2f%%   S_P1(linear)=%.2f%%"
          % (math.sqrt(np.mean(np.array(e2) ** 2)), math.sqrt(np.mean(np.array(e1) ** 2))))

# bw scaling check: S(bw) = exp(-bw^2 omega_q^2/2), fixed K=5 rho=0.47
print("\n" + "=" * 110)
print("bw-scaling of S at fixed K=5,rho=0.47:  S(bw)=exp(-bw^2 omega_q^2/2)?")
print("=" * 110)
K, rho = 5, 0.47
_, om0 = measured(K, rho, bw=12.0)   # omega_q is bw-independent (signal geometry)
print("%6s | %8s %8s | err%%" % ("bw", "S(meas)", "S_pred"))
for bw in (4, 6, 8, 12, 16, 20, 24):
    S, om = measured(K, rho, bw=bw)
    Spred = math.exp(-0.5 * bw ** 2 * om0 ** 2)
    print("%6.1f | %8.4f %8.4f | %+6.2f" % (bw, S, Spred, 100 * (Spred - S) / S))
