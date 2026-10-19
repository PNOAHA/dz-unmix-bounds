# -*- coding: utf-8 -*-
"""
probe2_omega_edge.py -- FIRST-PRINCIPLES omega_q(K,rho) from the banded-Toeplitz edge eigenproblem.

TARGET (to derive):  omega_q^2 = omega_inf^2 * (K-1)/(K+a),
  omega_inf(rho) ~ 0.92 * Nyq / sqrt(kappa),   a(rho) ~ 2.4/kappa^2,   kappa = 2 erfinv(rho)^2,
  Nyq = pi/dmu  (Nyquist of the peak lattice, dmu = sigma * 2 sqrt2 erfinv(rho)).

DERIVATION LOGIC:
  omega_q^2 = (v^T A v)/(v^T G v) = (v^T A v)/lambda_min,  A_ij = int d_i d_j/q, d_i=h_i'-h_i q'/q.
  Both A and G are (approx) banded Toeplitz in the peak index.  The worst mode v is the HIGHEST
  eigenmode.  For an open-boundary (Dirichlet) banded Toeplitz of size K, the eigenmodes are
  v^{(j)}_n = sin(n theta_j),  theta_j = j pi/(K+1),  and the TOP mode (j=K-1... actually j=K)
  has theta_top = K pi/(K+1) = pi (1 - 1/(K+1)) -> pi.  Plugging into the Toeplitz symbols:
     v^T A v / |v|^2 -> f_A(theta_top),   v^T G v/|v|^2 -> f_G(theta_top)=lambda_min.
  So omega_q^2 = f_A(theta_top)/f_G(theta_top).

  Near theta=pi both symbols are dominated by their high-freq behavior.  Because A involves
  DERIVATIVES (d_i ~ h_i'), f_A picks up an extra factor ~ (Nyquist freq)^2 ~ (1/sigma)^2 relative
  to f_G; and the (K-1)/(K+a) shape comes from theta_top^2-like edge correction:
     theta_top = pi K/(K+1).  Expand structures in (K - boundary offset).
  Test whether the empirical (K-1)/(K+a) is just theta_top^2 / pi^2 rescaled, i.e.
     (K-1)/(K+a) ~ a Mobius (linear-fractional) function of K = same family as (K/(K+1))^2 linearized.

This script:
 1. Builds A,G Toeplitz symbols numerically (band-averaged) and evaluates the ratio at theta_top.
 2. Fits omega_q^2(K) to omega_inf^2 (K-1)/(K+a); reports omega_inf, a; checks 0.92 Nyq/sqrt(kappa)
    and 2.4/kappa^2.
 3. Tests the Dirichlet top-mode theta_top=K pi/(K+1) prediction of omega_q vs measured.

Self-contained. English prints.
Run:  python probe2_omega_edge.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import curve_fit

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 6000.0 + 1e-9, 0.25)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def dgauss(t, mu, s): return -(t - mu) / s ** 2 * gauss(t, mu, s)


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    nrm = TRAPZ(h, TGRID, axis=1)[:, None]
    hp = np.stack([dgauss(TGRID, m, SIGMA) for m in mus]) / nrm
    return h / nrm, hp, mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def AG(K, rho):
    h, hp, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qp = (w[:, None] * hp).sum(0)
    d = hp - h * (qp / q)[None, :]
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    A = np.array([[TRAPZ(d[i] * d[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    omq2 = (v @ A @ v) / lam_min

    # Dirichlet top-mode prediction: theta_top = K pi/(K+1); evaluate symbol ratio there
    # band symbols (band-averaged Toeplitz coefficients)
    def sym(MAT, theta):
        s = 0.0
        for k in range(K):
            ii = np.arange(K - k); jj = ii + k
            ck = MAT[ii, jj].mean()
            s += ck * math.cos(k * theta) * (1 if k == 0 else 2)
        return s
    theta_top = K * math.pi / (K + 1)
    omq2_sym = sym(A, theta_top) / sym(G, theta_top)
    return omq2, omq2_sym, lam_min


print("=" * 95)
print("omega_q^2 = (vAv)/lam_min ; fit to omega_inf^2 (K-1)/(K+a). Nyq=pi/dmu, kappa=2erfinv^2")
print("=" * 95)


def model(K, ominf2, a):
    return ominf2 * (K - 1) / (K + a)


for rho in (0.47, 0.40, 0.35, 0.30):
    kap = 2 * erfinv(rho) ** 2
    nyq = math.pi / dmu(rho)
    Ks = np.arange(2, 13)
    omq2 = []; omq2_sym = []
    for K in Ks:
        a_, b_, lm = AG(int(K), rho)
        omq2.append(a_); omq2_sym.append(b_)
    omq2 = np.array(omq2)
    try:
        popt, _ = curve_fit(model, Ks, omq2, p0=[omq2[-1] * 1.3, 1.0], maxfev=20000)
        ominf2, a = popt
        pred = model(Ks, *popt)
        r2 = 1 - ((omq2 - pred) ** 2).sum() / ((omq2 - omq2.mean()) ** 2).sum()
    except Exception as e:
        ominf2, a, r2 = float('nan'), float('nan'), float('nan')
    ominf = math.sqrt(max(ominf2, 0))
    ominf_pred = 0.92 * nyq / math.sqrt(kap)
    a_pred = 2.4 / kap ** 2
    print(f"\n--- rho={rho}  kappa={kap:.4f}  Nyq=pi/dmu={nyq:.5f} ---")
    print("  FIT: omega_inf^2=%.5e (omega_inf=%.5f) ; a=%.3f ; R2=%.5f"
          % (ominf2, ominf, a, r2))
    print("  PRED omega_inf=0.92*Nyq/sqrt(kappa)=%.5f  (ratio fit/pred=%.3f)"
          % (ominf_pred, ominf / ominf_pred if ominf_pred else float('nan')))
    print("  PRED a=2.4/kappa^2=%.3f  (fit a=%.3f, ratio=%.3f)"
          % (a_pred, a, a / a_pred if a_pred else float('nan')))
    print("  omega_q^2 measured vs Dirichlet-symbol(theta=K pi/(K+1)):")
    for i, K in enumerate(Ks):
        print("    K=%2d  meas=%.5e  sym=%.5e  ratio=%.3f"
              % (K, omq2[i], omq2_sym[i], omq2_sym[i] / omq2[i] if omq2[i] else float('nan')))
