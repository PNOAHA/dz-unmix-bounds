# -*- coding: utf-8 -*-
"""
analytic_worst_mode.py -- ANALYTIC cause of the degree-(K-1) worst mode (closes the §0.5a (II)
"解析成因 open" item for the worst-mode SHAPE).

Claim: the measured worst tangent mode (degree-(K-1) weight-contrast) is the SMALLEST-eigenvalue
eigenvector of the CONTINUOUS supervised-weight Fisher information restricted to the sum-zero tangent:
    I_ij = ∫ h_i(t) h_j(t) / q_w(t) dt ,   q_w = Σ_k w_k h_k ,   h_k = area-normalized endmember.
The least-Fisher contrast is the one whose density perturbation Σ c_k h_k has the most cancellation
(most oscillatory sign pattern) = the highest-degree (K-1) discrete contrast polynomial.

This is the SIGNAL geometry (raw Gaussians, NO KDE, NO noise) -> sets the worst-mode SHAPE.
(The excess MAGNITUDE / ln(K-1) law involves noise propagation and stays open.)

Pure linear algebra; matches the dz-unmix-cg geometry (sigma=25, equal-spacing rho, endpoint w).
English prints. Run:   python analytic_worst_mode.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 2500.0 + 1e-9, 1.0)          # fine grid for the integral
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu_of_rho(rho):
    return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def endmembers(K, rho):
    mus = np.array([START + i * dmu_of_rho(rho) for i in range(K)])
    h = np.exp(-0.5 * ((TGRID[None, :] - mus[:, None]) / SIGMA) ** 2)
    return h / TRAPZ(h, TGRID, axis=1)[:, None]       # (K,T) area-normalized


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K)
    return w / w.sum()


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean()
    B = []
    for deg in range(1, K):
        v = x ** deg; v = v - v.mean()
        for b in B:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            B.append(v / nv)
    return B


def continuous_fisher(K, rho, w):
    h = endmembers(K, rho)
    q = (w[:, None] * h).sum(0)
    q = np.maximum(q, 1e-30)
    I = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            I[i, j] = TRAPZ(h[i] * h[j] / q, TGRID)
    return I


print("Continuous tangent-Fisher smallest-eigenvector vs degree-(K-1) contrast")
print("(sigma=25, endpoint w; I_ij=∫ h_i h_j / q)")
print("%2s %5s | %9s %9s | %8s | %s"
      % ("K", "rho", "lam_min", "lam_2nd", "|.deg(K-1)|", "dom_deg(=K-1?)"))
for rho in (0.47, 0.35):
    for K in range(2, 8):
        w = endpoint_w(K)
        I = continuous_fisher(K, rho, w)
        P = np.eye(K) - np.ones((K, K)) / K            # sum-zero projector
        M = P @ I @ P
        lam, V = np.linalg.eigh(M)                      # ascending; lam[0]~0 (the 1-direction)
        # tangent eigenpairs = drop the ~0 eigenvalue belonging to the all-ones direction
        # smallest tangent eigenvalue is the 2nd smallest of M (1st is the projected-out 1)
        order = np.argsort(lam)
        # identify the all-ones direction (overlap with normalized ones)
        ones = np.ones(K) / math.sqrt(K)
        tang = [k for k in order if abs(V[:, k] @ ones) < 0.5]
        kmin = tang[0]
        vmin = V[:, kmin]
        B = contrast_polys(K)
        ov = [abs(float(vmin @ b)) for b in B]
        dom = int(np.argmax(ov)) + 1
        lam_t = sorted(lam[k] for k in tang)
        print("%2d %5.2f | %9.3e %9.3e | %8.3f | deg=%d %s"
              % (K, rho, lam_t[0], lam_t[1] if len(lam_t) > 1 else float('nan'),
                 ov[-1], dom, "YES" if dom == K - 1 else "NO -> "
                 + " ".join(f"{o:.2f}" for o in ov)))
    print()
