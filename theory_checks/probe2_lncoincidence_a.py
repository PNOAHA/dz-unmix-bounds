# -*- coding: utf-8 -*-
"""
probe2_lncoincidence_a.py -- ADVERSARIAL stress of claim (a):
  "a + b*ln(K-1) is a WINDOW COINCIDENCE of (power-law CR) x (stretched-exp S),
   NOT a fundamental log law."

A genuine log law excess(K) = a + b ln(K-1) would have window-STABLE coefficients
(a, b) and excellent R2 on ANY K-window. A window coincidence will:
  (i)  drift its (a, b) as the fit window changes (2..5, 2..7, 2..10, 3..10),
  (ii) and the *individual* factors CR(K), S(K) will NOT themselves be logarithmic
       (CR is a power law (K-1)^{p/2}; S is a stretched exponential exp(-a (K-1)^beta)).

We quantify drift by:
  - the spread of (a, b) across windows (max-min, % drift),
  - comparing ln-fit R2 to a power-law fit R2 excess ~ A (K-1)^B on the SAME window
    (if ln is fundamental, ln should win everywhere; if it's a coincidence, power law
     fits comparably or better and the ln coefficients wander),
  - direct tests that CR is power-law (not log) and S is stretched-exp (not log):
      ln(CR) vs ln(K-1) should be LINEAR (R2~1, power law) but CR vs ln(K-1) curved;
      -ln(S) vs (K-1) should be near-linear/stretched (R2 of ln(-ln S) ~ ln(K-1) gives beta).

Deterministic linear algebra (no Monte Carlo). Reuses the exact worst-mode machinery
from compute_factors.py. English prints.
Run:   python probe2_lncoincidence_a.py
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
    return h / TRAPZ(h, TGRID, axis=1)[:, None]


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f


def worst_mode(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    v = V[:, kmin]; lam_min = lam[kmin]
    g = (v[:, None] * h).sum(0) / q
    gs = smooth(g)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    var_g = Eq(g * g) - Eq(g) ** 2
    var_gs = Eq(gs * gs) - Eq(gs) ** 2
    S = math.sqrt(max(var_gs, 0) / var_g)
    return lam_min, S


def r2_of(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def lnfit(Ks, y):
    A = np.vstack([np.ones(len(Ks)), np.log(Ks - 1)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1], r2_of(y, A @ coef)


def powfit(Ks, y):
    # y = A*(K-1)^B  -> ln y = ln A + B ln(K-1); report R2 in linear space
    ca = np.polyfit(np.log(Ks - 1), np.log(y), 1)
    pred = np.exp(np.polyval(ca, np.log(Ks - 1)))
    return math.exp(ca[1]), ca[0], r2_of(y, pred)


WINDOWS = [(2, 5), (2, 7), (2, 10), (3, 10)]

for rho in (0.47, 0.40, 0.35):
    print("\n================  rho=%.2f  ================" % rho)
    Kall = np.arange(2, 11)
    lm = np.zeros(len(Kall)); Sv = np.zeros(len(Kall))
    for i, K in enumerate(Kall):
        lm[i], Sv[i] = worst_mode(int(K), rho)
    CR = np.sqrt(lm[0] / lm)             # CR(K)=sqrt(lammin(2)/lammin(K))
    exc = CR * Sv

    print(" K :  CR(K)    S(K)    excess")
    for i, K in enumerate(Kall):
        print("%2d : %7.4f %7.4f %8.4f" % (K, CR[i], Sv[i], exc[i]))

    # ---- (i) window-stability of the ln-fit (THE headline adversarial test) ----
    print("\n  [a-i] ln-fit excess ~ a + b ln(K-1) across windows  (drift => coincidence):")
    print("   window     a       b      R2_ln  |  powfit A    B     R2_pow")
    a_list, b_list = [], []
    for (lo, hi) in WINDOWS:
        m = (Kall >= lo) & (Kall <= hi)
        Ks = Kall[m].astype(float); y = exc[m]
        a, b, r2l = lnfit(Ks, y)
        Ap, Bp, r2p = powfit(Ks, y)
        a_list.append(a); b_list.append(b)
        print("   %2d..%-2d  %7.3f %7.3f  %.4f  | %7.3f %6.3f  %.4f"
              % (lo, hi, a, b, r2l, Ap, Bp, r2p))
    a_arr = np.array(a_list); b_arr = np.array(b_list)
    print("   --> a drift: min=%.3f max=%.3f  span=%.3f  (%.0f%% of mean)"
          % (a_arr.min(), a_arr.max(), a_arr.max() - a_arr.min(),
             100 * (a_arr.max() - a_arr.min()) / abs(a_arr.mean())))
    print("   --> b drift: min=%.3f max=%.3f  span=%.3f  (%.0f%% of mean)"
          % (b_arr.min(), b_arr.max(), b_arr.max() - b_arr.min(),
             100 * (b_arr.max() - b_arr.min()) / abs(b_arr.mean())))

    # ---- (ii) CR is power-law NOT log; S is stretched-exp NOT log ----
    Kf = Kall.astype(float)
    # CR: power-law fit (ln CR linear in ln(K-1)) vs log fit (CR linear in ln(K-1))
    _, Bcr, r2_cr_pow = powfit(Kf, CR)
    _, _, r2_cr_log = lnfit(Kf, CR)
    # residual curvature of ln(CR) vs ln(K-1): quadratic coeff (0 => pure power law)
    q2_cr = np.polyfit(np.log(Kf - 1), np.log(CR), 2)[0]
    print("\n  [a-ii] CR(K): power-law (K-1)^%.3f  R2_pow=%.5f   vs   log-fit R2=%.5f"
          % (Bcr, r2_cr_pow, r2_cr_log))
    print("         curvature of ln CR vs ln(K-1) (quad coef, ~0 => pure power law): %.2e" % q2_cr)

    # S: stretched-exp -ln S = a (K-1)^beta -> ln(-ln S) linear in ln(K-1), slope=beta
    mlnS = -np.log(np.maximum(Sv, 1e-12))
    good = mlnS > 1e-6
    cb = np.polyfit(np.log(Kf[good] - 1 + 1e-12)[1:], np.log(mlnS[good][1:]), 1)
    beta = cb[0]
    pred_lnS = np.polyval(cb, np.log(Kf - 1 + 1e-12))
    r2_S_stretch = r2_of(np.log(mlnS[1:]), pred_lnS[1:])
    _, _, r2_S_log = lnfit(Kf, Sv)
    print("  [a-ii] S(K): stretched-exp exp(-a (K-1)^beta), beta=%.3f  R2(ln(-lnS))=%.4f"
          % (beta, r2_S_stretch))
    print("         (beta!=0 and S not linear in ln(K-1): log-fit of S R2=%.4f)" % r2_S_log)

print("\nVERDICT NOTE: window-stable (a,b) + ln winning on every window would SUPPORT a true log law;")
print("drifting (a,b) + power-law CR + stretched-exp S => 'ln(K-1)' is a fitted-window coincidence.")
