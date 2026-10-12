# -*- coding: utf-8 -*-
"""
probe_A_vs_G.py -- relate the derivative-Gram A to the Gram G to get omega_q^2 in closed form.

omega_q^2 = (v^T A v)/(v^T G v),   v = lam_min eigvec of G (tangent).
A_ij = ∫ d_i d_j / q,  d_i = h_i' - h_i q'/q.   G_ij = ∫ h_i h_j / q.

IDEA 1 (Gaussian curvature identity):  For a single Gaussian, h_i'(t) = -(t-mu_i)/sigma^2 h_i.
  In the near-peak region the dominant content of d_i is the "local frequency" 1/sigma.
  Conjecture A ~ G / sigma_eff^2 with sigma_eff a fixed local scale => omega_q^2 = lam_min(A-part)/lam_min(G).
  Test: is A approx (c/sigma^2) G in the worst-mode direction?  Compute (vAv)/(vGv)*sigma^2.

IDEA 2 (operator form): omega_q^2 is the Rayleigh quotient of the pencil (A,G) at the G-smallest vector.
  Compute the FULL generalized spectrum of (A,G) on the tangent: gen-eigs mu of A v = mu G v.
  The worst-G mode need NOT be the largest-mu mode; check where it sits. If the worst-G mode
  has near-MAX generalized eig, then omega_q^2 ~ mu_max(A,G) = top "local curvature frequency^2".

IDEA 3 (bw scaling sanity): S = exp(-bw^2 omega_q^2/2) predicts log S linear in bw^2; and the
  reported sigma_err ~ bw^{-0.5..-0.6} is a SEPARATE (CR/Sigma) effect; but within S, the bw
  dependence is exactly exp(-bw^2 omega_q^2/2). Verify by sweeping bw in {6,12,18,24} for fixed K.
Self-contained. English prints.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_A_vs_G.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.linalg import eigh as geigh

SIGMA, START, DT = 25.0, 200.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
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


def GA(K, rho):
    h, hp, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qp = (w[:, None] * hp).sum(0)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    d = hp - h * (qp / q)[None, :]
    A = np.array([[TRAPZ(d[i] * d[j] / q, TGRID) for j in range(K)] for i in range(K)])
    return G, A


def worst_v(G):
    K = G.shape[0]
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return V[:, tang[0]], lam[tang[0]]


print("=" * 100)
print("A vs G: is omega_q^2 ~ const/sigma^2 ?   sigma^2=%.0f" % SIGMA ** 2)
print("Also: generalized pencil (A,G) — where does the worst-G mode sit in the A/G spectrum?")
print("=" * 100)
for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %10s | %12s | %10s %10s %10s | %s"
          % ("K", "om_q^2*s^2", "om_q^2", "genmu_min", "genmu_at_v", "genmu_max", "rank_of_v(0=smallest)"))
    for K in range(2, 11):
        G, A = GA(K, rho)
        v, lam_min = worst_v(G)
        om2 = float(v @ A @ v) / float(v @ G @ v)
        # generalized eig on the tangent: project both to tangent basis
        P = np.eye(K) - np.ones((K, K)) / K
        # orthonormal tangent basis (K-1 vectors)
        _, U = np.linalg.eigh(P)
        Tb = U[:, 1:]            # columns span tangent (drop the ~0 = ones direction)
        At = Tb.T @ A @ Tb; Gt = Tb.T @ G @ Tb
        gmu = np.sort(np.real(geigh(At, Gt, eigvals_only=True)))
        # which gen-eig corresponds to v? v in tangent coords:
        vc = Tb.T @ v
        gval, gvec = geigh(At, Gt)
        # project vc onto gen-eigvecs, find dominant
        ov = np.abs(gvec.T @ vc) / (np.linalg.norm(gvec, axis=0) * np.linalg.norm(vc) + 1e-30)
        idx = int(np.argmax(ov))
        order = np.argsort(np.real(gval))
        rank = int(np.where(order == idx)[0][0])
        print("%2d | %10.5f | %12.4e | %10.5f %10.5f %10.5f | %d/%d  ov=%.2f"
              % (K, om2 * SIGMA ** 2, om2, gmu[0], np.real(gval[idx]), gmu[-1],
                 rank, K - 1, ov[idx]))
