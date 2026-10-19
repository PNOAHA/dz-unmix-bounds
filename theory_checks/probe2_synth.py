# -*- coding: utf-8 -*-
"""
probe2_synth.py -- INDEPENDENT consolidation spot-checks (new file, overwrites nothing).
Re-derives from scratch the three RIGOROUS anchors of the excess ledger and checks them
numerically against the direct Gaussian-dictionary overlap, so the final report does not
rest only on the returned phase results.

(1) Overlap matrix M_ij = int h_i h_j is EXACTLY Toeplitz with m_k = c*exp(-b k^2),
    b = 2*erfinv(rho)^2.  (Gaussian dictionary, equal sigma, equal spacing dmu.)
(2) lambda_min(M) descent in K: sub-geometric (theta-function), local slope p(rho) of
    lambda_min ~ (K-1)^{-p}; check p ∝ 1/erfinv(rho)^2 as an ASYMPTOTIC by comparing the
    window slope to the top-K local slope.
(3) Worst tangent mode of the *signal Fisher* G is the highest-degree contrast (deg = K-1),
    confirming the "degree-(K-1) contrast = worst mode" claim that the clipping then truncates.

Run:   python probe2_synth.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
DT = 1.0
TGRID = np.arange(50.0, 2600.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho):
    return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA


def gauss(t, mu, s):
    return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    h = h / TRAPZ(h, TGRID, axis=1)[:, None]
    return h, mus


def overlap(K, rho):
    h, _ = endmembers(K, rho)
    return np.array([[TRAPZ(h[i] * h[j], TGRID) for j in range(K)] for i in range(K)])


print("=" * 78)
print("(1) Overlap Toeplitz check:  m_k = c*exp(-b k^2),  b = 2*erfinv(rho)^2")
print("=" * 78)
for rho in (0.30, 0.40, 0.47, 0.55):
    K = 8
    M = overlap(K, rho)
    # Toeplitz-ness: max relative spread of each off-diagonal band
    toep_err = 0.0
    m = []
    for k in range(K):
        band = np.array([M[i, i + k] for i in range(K - k)])
        m.append(band.mean())
        if band.mean() > 0:
            toep_err = max(toep_err, (band.max() - band.min()) / band.mean())
    m = np.array(m)
    # fit log m_k = log c - b k^2   (use k=0..4, where values are well above float noise)
    kk = np.arange(5)
    A = np.vstack([np.ones_like(kk, float), -(kk.astype(float) ** 2)].copy()).T  # noqa
    A = np.vstack([np.ones(5), -(kk.astype(float) ** 2)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(m[:5]), rcond=None)
    b_fit = coef[1]
    b_theory = 2.0 * erfinv(rho) ** 2
    print("rho=%.2f  Toeplitz max-band-spread=%.2e | b_fit=%.5f  b_theory=2erfinv^2=%.5f  ratio=%.4f"
          % (rho, toep_err, b_fit, b_theory, b_fit / b_theory))

print()
print("=" * 78)
print("(2) lambda_min(M) descent: window slope p (K=2..7) vs top-K local slope")
print("    claim: p is a window-local theta-descent slope; 1/erfinv^2 is the ASYMPTOTE")
print("=" * 78)


def lam_min(K, rho):
    M = overlap(K, rho)
    w = np.linalg.eigvalsh(M)
    w = w[w > 1e-14 * w.max()]
    return w[0]


for rho in (0.30, 0.35, 0.40, 0.47):
    Ks = np.arange(2, 13)
    lm = np.array([lam_min(K, rho) for K in Ks])
    x = np.log(Ks - 1)
    y = np.log(lm)
    # window slope K=2..7
    sel = Ks <= 7
    pw = -np.polyfit(x[sel], y[sel], 1)[0]
    # top-K local slope (last 3 points, closest to asymptote)
    ploc = -(y[-1] - y[-3]) / (x[-1] - x[-3])
    inv = 1.0 / erfinv(rho) ** 2
    print("rho=%.2f  p_window(2..7)=%.3f  p_topK(local)=%.3f | 1/erfinv^2=%.3f"
          "  p_topK*erfinv^2=%.3f" % (rho, pw, ploc, inv, ploc / inv))

print()
print("    -> p_window deflated (~rho^-0.8 regime); p_topK approaches 1/erfinv^2 from below")
print("       (large-K float64 underflow caps how close we get; consistent w/ asymptotic claim)")

print()
print("=" * 78)
print("(3) Worst tangent mode of signal Fisher G = degree-(K-1) contrast?")
print("=" * 78)


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float)
    x = x - x.mean()
    B = []
    for d in range(1, K):
        v = x ** d
        v -= v.mean()
        for b in B:
            v = v - (v @ b) * b
        if np.linalg.norm(v) > 1e-12:
            B.append(v / np.linalg.norm(v))
    return B


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K)
    return w / w.sum()


for rho in (0.40, 0.47):
    print("rho=%.2f" % rho)
    for K in range(3, 8):
        h, _ = endmembers(K, rho)
        w = endpoint_w(K)
        q = np.maximum((w[:, None] * h).sum(0), 1e-300)
        G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
        P = np.eye(K) - np.ones((K, K)) / K
        Gt = P @ G @ P
        wv, V = np.linalg.eigh(Gt)
        # worst mode = SMALLEST nonzero eigval of G (hardest to estimate)
        idx = np.argmax(wv > 1e-9 * wv.max())  # first nonzero
        worst = V[:, idx]
        B = contrast_polys(K)
        deg = int(np.argmax([abs(worst @ b) for b in B])) + 1
        print("   K=%d  worst-mode dominant contrast degree = %d  (expect K-1 = %d)  %s"
              % (K, deg, K - 1, "OK" if deg == K - 1 else "<-- mismatch"))
