# -*- coding: utf-8 -*-
"""
probe_S_spectral2.py -- understand the STRUCTURE of g_v and why smoothing suppresses it more at large K.

Key realization: Var_q[g] = lambda_min(G) (g is the unit-norm worst eigvec, so the quadratic form = its
eigenvalue). The smoothed Var_q[smooth(g)] is a DIFFERENT quadratic form. We want the ratio.

The smoothing operator in t-domain: (K_bw * g)(t). Var_q is the q-weighted L2 inner product.
Insight to test: g_v = (sum_i v_i h_i)/q. The numerator m(t)=sum_i v_i h_i(t) is a "signed mixture"
(a sum of Gaussians with contrast weights v). Then g = m/q. Var_q[g] = int (m/q)^2 q - (int m)^2
  = int m^2/q  - 0   (since int m = sum_i v_i = 0 on the tangent).
So Var_q[g] = int m(t)^2 / q(t) dt  ==  v^T G v  = lambda_min.  Good, consistent.

For the SMOOTHED version: gs = K_bw * (m/q).  This does NOT equal (K_bw*m)/q. The smoothing is applied
to the RATIO g=m/q, not to m. So the structure of g (the ratio) matters.

Let me look at g directly and at smooth(g), and decompose the suppression.
We dump g_v on a window around the peaks for K=2,4,6,8,10 and inspect the oscillation amplitude & local
scale. Hypothesis: g_v develops SHARP localized features (cusps between adjacent peaks) whose LOCAL width
shrinks / amplitude pattern changes with K -- a local-curvature effect, not a global-frequency effect.
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f
Eq = lambda f, q: TRAPZ(f * q, TGRID)


def worst(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]
    g = (v[:, None] * h).sum(0) / q
    return h, mus, w, q, v, g


print("=" * 90)
print("STEP 2: g_v in the q-bulk. Show g sampled at the endmember means and at the midpoints.")
print("Hypothesis: between adjacent peaks g_v oscillates; in the bulk it is bounded; in the TAILS")
print("(outside the support) m/q blows up but q~0 so contributes nothing to Var_q.")
print("=" * 90)
rho = 0.47
for K in (2, 4, 6, 8, 10):
    h, mus, w, q, v, g = worst(K, rho)
    # sample g at means
    g_at_mean = [g[np.argmin(np.abs(TGRID - m))] for m in mus]
    # midpoints
    mids = (mus[:-1] + mus[1:]) / 2
    g_at_mid = [g[np.argmin(np.abs(TGRID - md))] for md in mids]
    print("\nK=%d  v=%s" % (K, np.array2string(v, precision=3, suppress_small=True)))
    print("  dmu=%.1f  (peak spacing in t)" % dmu(rho))
    print("  g at means:     %s" % np.array2string(np.array(g_at_mean), precision=3))
    print("  g at midpoints: %s" % np.array2string(np.array(g_at_mid), precision=3))

print()
print("=" * 90)
print("STEP 3: the GROUND TRUTH model. Apply K_bw to g and recompute Var_q exactly, but ALSO test")
print("the 'Taylor/curvature' model: smooth(g) ~ g + (bw^2/2) g''. Then")
print("  Var_q[smooth g] ~ Var_q[g] + bw^2 Cov_q[g, g''] + (bw^4/4)Var_q[g'']")
print("  S^2 ~ 1 + bw^2 Cov_q[g,g'']/Var_q[g] + ...   (leading: 1 + bw^2 <g g''>_q / <g^2>_q )")
print("Check if leading curvature term -bw^2 * (q-weighted mean curvature ratio) reproduces 1-S^2.")
print("=" * 90)

def deriv2(f):
    d2 = np.empty_like(f)
    d2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / DT ** 2
    d2[0] = d2[1]; d2[-1] = d2[-2]
    return d2

for rho in (0.47, 0.40):
    print(f"\n--- rho={rho} ---")
    print("%2s | %8s %8s | %10s %10s | %8s %8s"
          % ("K", "S_true", "S^2", "Var_q[g]", "Cov_q[g,g'']", "1+bw2*r", "sqrt"))
    for K in range(2, 11):
        h, mus, w, q, v, g = worst(K, rho)
        gs = smooth(g)
        mg = Eq(g, q)
        vg = Eq((g - mg) ** 2, q)
        vgs = Eq((gs - Eq(gs, q)) ** 2, q)
        S2 = vgs / vg
        gpp = deriv2(g)
        cov_g_gpp = Eq((g - mg) * (gpp - Eq(gpp, q)), q)
        r = cov_g_gpp / vg
        S2_taylor = 1.0 + BW ** 2 * r
        print("%2d | %8.4f %8.4f | %10.4e %10.4e | %8.4f %8.4f"
              % (K, math.sqrt(max(S2, 0)), S2, vg, cov_g_gpp, S2_taylor,
                 math.sqrt(max(S2_taylor, 0))))
