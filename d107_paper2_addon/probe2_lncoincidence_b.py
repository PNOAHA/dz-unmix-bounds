# -*- coding: utf-8 -*-
"""
probe2_lncoincidence_b.py -- ADVERSARIAL stress of claim (b):
  "the rho-dependence of the local exponent p is  p ∝ 1/erfinv(rho)^2."

We map the TRUE local exponent p(rho) of  lambda_min(G) ~ (K-1)^{-p}  directly from the
real signal-Fisher operator G_ij = int h_i h_j / q (endpoint w), over a DENSE rho grid in
[0.28, 0.60].  Then we adversarially pit  1/erfinv(rho)^2  against natural alternatives:

  (cand)  basis  X(rho)               rationale
  --------------------------------------------------------------
  E2      1/erfinv(rho)^2             the claimed law (s=2 erfinv^2 is the Toeplitz decay const)
  R2      1/rho^2                     naive "similarity^-2"
  R1      1/rho                       naive "similarity^-1"
  NL      -1/ln(rho)                  log-overlap heuristic
  DM2     (dmu/sigma)^-2 = 1/(8 erfinv^2)  the (signal-to-noise spacing)^-2  [= E2 rescaled!]

Each candidate is fit as  p(rho) = a + b * X(rho)  (affine, fair: every law gets an intercept
and a slope), R2 in p-space.  We ALSO test on TWO windows (p fit over K=2..7 and K=2..10),
because p is a window-local slope (theta-function descent, not a true power law) -- if the
WINNING functional form is the SAME on both windows, the 1/erfinv^2 claim is robust to that
caveat.

Note DM2 = (2 sqrt2 erfinv)^-2 = 1/(8 erfinv^2) is COLLINEAR with E2 (differ by constant
factor) -> identical affine fit. We report it to make explicit that "(dmu/sigma)^-2" and
"1/erfinv^2" are the SAME hypothesis, not a competitor. The real competitors are R2, R1, NL.

Deterministic linear algebra. Reuses probe_ptrue_map machinery. English prints.
Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_lncoincidence_b.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 5000.0 + 1e-9, 0.5)   # extended top so K=10 peaks at small rho stay inside
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


def lammin(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    return lam[tang[0]]


def p_true(rho, Kmax):
    Ks = np.arange(2, Kmax + 1)
    lm = np.array([lammin(int(K), rho) for K in Ks])
    return -np.polyfit(np.log(Ks - 1)[1:], np.log(lm[1:] / lm[0]), 1)[0]


def affine_fit(X, y):
    A = np.vstack([np.ones_like(X), X]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse = math.sqrt(((y - pred) ** 2).mean())
    return coef, r2, rmse


# top-peak guard: for K=10 at rho=0.28, top mu must stay < grid_top - 3 sigma
RHOS = np.array([0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45,
                 0.47, 0.50, 0.53, 0.56, 0.60])

for Kmax in (7, 10):
    print("\n================  p(rho) fit over K=2..%d  ================" % Kmax)
    # guard the grid: ensure top peak inside for the largest K at smallest rho
    rmax = SIGMA * 2 * math.sqrt(2) * erfinv(RHOS.max()) * (Kmax - 1) + START
    if rmax >= TGRID[-1] - 3 * SIGMA:
        print("  WARN: top peak %.0f near grid top %.0f" % (rmax, TGRID[-1]))
    ps = np.array([p_true(rho, Kmax) for rho in RHOS])

    erf = np.array([erfinv(r) for r in RHOS])
    cands = {
        "E2  1/erfinv^2 ": 1.0 / erf ** 2,
        "DM2 (dmu/sig)^-2": 1.0 / (8.0 * erf ** 2),     # collinear with E2
        "R2  1/rho^2    ": 1.0 / RHOS ** 2,
        "R1  1/rho      ": 1.0 / RHOS,
        "NL  -1/ln rho  ": -1.0 / np.log(RHOS),
    }

    print("  rho   erfinv  p_true   1/erfinv^2   1/rho^2   1/rho   -1/lnrho")
    for i, rho in enumerate(RHOS):
        print("  %.2f  %.4f  %6.3f   %8.3f  %8.3f %7.3f  %7.3f"
              % (rho, erf[i], ps[i], 1/erf[i]**2, 1/rho**2, 1/rho, -1/math.log(rho)))

    print("\n  candidate            a        b        R2       RMSE")
    results = {}
    for name, X in cands.items():
        coef, r2, rmse = affine_fit(X, ps)
        results[name] = (r2, rmse)
        print("  %-18s %7.3f  %7.3f  %.5f  %.4f" % (name, coef[0], coef[1], r2, rmse))

    # ranking (exclude DM2 which is identical to E2 by construction)
    comp = {k: v for k, v in results.items() if not k.startswith("DM2")}
    ranked = sorted(comp.items(), key=lambda kv: -kv[1][0])
    print("\n  RANK by R2 (E2 = the claim; competitors R2/R1/NL):")
    for k, (r2, rmse) in ranked:
        print("     %-18s R2=%.5f" % (k, r2))
    win = ranked[0][0].strip()
    print("  --> best functional form: %s" % win)

    # also: pure proportionality test p ~ b/erfinv^2 with NO intercept vs WITH intercept,
    # and the constant p*erfinv^2 (should be ~flat if p ∝ 1/erfinv^2)
    print("\n  p * erfinv^2 (flat => p ∝ 1/erfinv^2; trend => needs intercept/curvature):")
    print("    ", np.round(ps * erf ** 2, 3))
    print("    mean=%.3f  CV=%.1f%%" % ((ps * erf ** 2).mean(),
          100 * (ps * erf ** 2).std() / (ps * erf ** 2).mean()))

print("\nVERDICT NOTE: if E2 (1/erfinv^2) has the highest R2 on BOTH windows and beats")
print("R2/R1/NL, claim (b) SURVIVES as the controlling rho-invariant (with the honest caveat")
print("that an intercept is needed because p is a window-local theta-descent slope, not exact).")
