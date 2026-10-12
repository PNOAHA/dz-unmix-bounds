# -*- coding: utf-8 -*-
"""
probe_reconcile.py -- adjudicate the five derivation attempts.

Builds the REAL G, A (Dirichlet/curvature), Gs (smoothed Sigma-kernel) operators on the
same grid/setup as compute_factors.py and tests every proposed closed form numerically.

Outputs:
 (1) lambda_min(G,K,rho): which p(rho) law is right. Tests
       p_A: 0.288/erfinv(rho)^2 - 0.438      (attempt 1, Toeplitz symbol)
       p_B: 0.305/erfinv(rho)^2 - 0.42       (attempt 2, global LS 0.61/s, s=2 erfinv^2)
       p_C: 0.739/(2 erfinv(rho)^2) - 0.919  (attempt 3, theta-fn squeeze)
     against the DIRECT log-log fit of lambda_min over K=2..10 (and K=2..7) per rho.
 (2) S(K,rho,bw): the EXACT KxK quadratic form, the omega_q Rayleigh quotient identity,
     and the monochromatic / two-moment closed forms. Resolves omega_eff puzzle.
 (3) excess = CR*S; fit a + b ln(K-1); express b vs p(rho).

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_reconcile.py
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


def dgauss(t, mu, s):
    # derivative of area-normalized gaussian
    return -(t - mu) / s**2 * gauss(t, mu, s)


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f


def operators(K, rho):
    """Return G, worst eigvec v, lam_min, and all the derived quantities."""
    h, mus = endmembers(K, rho)
    w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qp = (w[:, None] * np.stack([dgauss(TGRID, m, SIGMA) for m in mus])).sum(0)  # q'

    # signal Fisher G_ij = int h_i h_j / q
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])

    # tangent worst mode
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    kmin = tang[0]
    v = V[:, kmin]; lam_min = lam[kmin]

    # --- S exact (KxK quadratic form via smoothed phi = (h_i/q) * K_bw ) ---
    phi = np.stack([smooth(h[i] / q) for i in range(K)])
    # Var_q[g*K_bw] = v^T Gs v - (v^T b)^2 ;  Gs_ij = int q phi_i phi_j, b_i = int q phi_i
    Gs = np.array([[TRAPZ(q * phi[i] * phi[j], TGRID) for j in range(K)] for i in range(K)])
    bvec = np.array([TRAPZ(q * phi[i], TGRID) for i in range(K)])
    var_gs_exact = v @ Gs @ v - (v @ bvec)**2
    var_g = lam_min  # = v^T G v on tangent (int g^2 q with g=sum v_i h_i/q, mean-removed gives lam_min)
    # field value for cross-check
    g = (v[:, None] * h).sum(0) / q
    gs = smooth(g)
    Eq = lambda f: TRAPZ(f * q, TGRID)
    var_g_field = Eq(g * g) - Eq(g)**2
    var_gs_field = Eq(gs * gs) - Eq(gs)**2
    S_exact = math.sqrt(max(var_gs_exact, 0) / var_g)
    S_field = math.sqrt(max(var_gs_field, 0) / var_g_field)

    # --- omega_q Rayleigh quotient: omega_q^2 = Eq[g'^2]/Eq[(g-<g>)^2] ---
    # g' = (sum v_i h_i')/q - g q'/q ; d_i = h_i' - h_i q'/q
    hp = np.stack([dgauss(TGRID, m, SIGMA) for m in mus])
    d = hp - h * (qp / q)[None, :]
    A = np.array([[TRAPZ(d[i] * d[j] / q, TGRID) for j in range(K)] for i in range(K)])
    vAv = v @ A @ v
    omega_q2 = vAv / lam_min
    # omega_eff (unweighted FFT rms freq)
    G_hat = np.fft.rfft(g - g.mean())
    freq = np.fft.rfftfreq(len(g), d=DT) * 2 * math.pi
    P_hat = np.abs(G_hat)**2
    omega_eff = math.sqrt((freq**2 * P_hat).sum() / P_hat.sum())
    # second moment m2 = Eq[g''^2]/Eq[(g-<g>)^2] for two-moment correction
    gpp = np.gradient(np.gradient(g, TGRID), TGRID)
    m2 = (Eq(gpp*gpp) - 0) / var_g  # approx Eq[g''^2]/var
    return dict(G=G, v=v, lam_min=lam_min, S_exact=S_exact, S_field=S_field,
                omega_q2=omega_q2, omega_eff=omega_eff, vAv=vAv, m2=m2,
                var_gs_exact=var_gs_exact, var_gs_field=var_gs_field)


RHOS = [0.30, 0.35, 0.40, 0.47, 0.55]
KS = list(range(2, 11))

# collect
data = {}
for rho in RHOS:
    data[rho] = {K: operators(K, rho) for K in KS}

print("="*78)
print("PART 0: sanity -- S_exact vs S_field (KxK quadratic form vs direct field)")
print("="*78)
maxerr = 0.0
for rho in [0.47, 0.40]:
    for K in KS:
        d = data[rho][K]
        e = abs(d['S_exact'] - d['S_field']) / d['S_field'] * 100
        maxerr = max(maxerr, e)
print(f"max |S_exact - S_field|/S_field over rho in (0.47,0.40), K=2..10 = {maxerr:.4f}%")

print("\n" + "="*78)
print("PART 1: lambda_min p(rho). Direct fits vs the three closed forms.")
print("="*78)
print("%5s | %8s %8s | %8s %8s %8s" %
      ("rho", "p(2..10)", "p(2..7)", "pA", "pB", "pC"))
def pA(rho): return 0.288/erfinv(rho)**2 - 0.438
def pB(rho): return 0.305/erfinv(rho)**2 - 0.42
def pC(rho): return 0.739/(2*erfinv(rho)**2) - 0.919
rows_p = []
for rho in RHOS:
    lm = np.array([data[rho][K]['lam_min'] for K in KS])
    Karr = np.array(KS, dtype=float)
    # direct fit over full window
    p_full = -np.polyfit(np.log(Karr - 1), np.log(lm / lm[0]), 1)[0]
    m7 = Karr <= 7
    p_7 = -np.polyfit(np.log(Karr[m7] - 1), np.log((lm/lm[0])[m7]), 1)[0]
    rows_p.append((rho, p_full, p_7))
    print("%5.2f | %8.3f %8.3f | %8.3f %8.3f %8.3f" %
          (rho, p_full, p_7, pA(rho), pB(rho), pC(rho)))

# residual rms of each closed form vs direct (both windows)
print("\nResiduals (closed form - direct):")
for name, fn in [("pA", pA), ("pB", pB), ("pC", pC)]:
    r_full = np.array([fn(rho) - pf for rho, pf, p7 in rows_p])
    r_7 = np.array([fn(rho) - p7 for rho, pf, p7 in rows_p])
    print("  %s: vs p(2..10) rms=%.3f max=%.3f | vs p(2..7) rms=%.3f max=%.3f" %
          (name, np.sqrt((r_full**2).mean()), np.abs(r_full).max(),
           np.sqrt((r_7**2).mean()), np.abs(r_7).max()))

# Fit our OWN A/erfinv^2 + C to the direct exponents (both windows) to get the honest law
print("\nOur own LS fit p = A/erfinv(rho)^2 + C:")
x = np.array([1/erfinv(rho)**2 for rho, _, _ in rows_p])
for label, idx in [("p(2..10)", 1), ("p(2..7)", 2)]:
    y = np.array([r[idx] for r in rows_p])
    Amat = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
    pred = Amat @ coef
    ss = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
    print("  %s:  p = %.4f/erfinv^2 + %.4f   R2=%.4f" % (label, coef[0], coef[1], ss))

print("\n" + "="*78)
print("PART 2: S(K) -- omega_eff puzzle and omega_q resolution")
print("="*78)
print("%5s %2s | %8s %8s | %8s %8s %8s | %8s" %
      ("rho","K","omega_eff","omega_q","S_exact","S_mono","S_2mom","resid%"))
for rho in [0.47, 0.40]:
    for K in KS:
        d = data[rho][K]
        omq = math.sqrt(d['omega_q2'])
        S_mono = math.exp(-0.5*BW**2*d['omega_q2'])
        # two-moment cumulant: S = exp(-bw^2 m1/2 + bw^4 (m2-m1^2)/4)
        m1 = d['omega_q2']; m2 = d['m2']
        S_2 = math.exp(-0.5*BW**2*m1 + 0.25*BW**4*max(m2-m1**2,0))
        resid = (S_mono - d['S_exact'])/d['S_exact']*100
        print("%5.2f %2d | %8.4f %8.4f | %8.4f %8.4f %8.4f | %+7.1f" %
              (rho, K, d['omega_eff'], omq, d['S_exact'], S_mono, S_2, resid))
# is omega_eff K-flat while omega_q grows?
for rho in [0.47, 0.40]:
    oe = np.array([data[rho][K]['omega_eff'] for K in KS])
    oq = np.array([math.sqrt(data[rho][K]['omega_q2']) for K in KS])
    print(f"rho={rho}: omega_eff range [{oe.min():.4f},{oe.max():.4f}] (CV {oe.std()/oe.mean()*100:.1f}%); "
          f"omega_q range [{oq.min():.4f},{oq.max():.4f}] grows {oq[-1]/oq[0]:.2f}x")

# S_mono accuracy
print("\nS_mono RMS error vs S_exact:")
for rho in RHOS:
    errs = []
    for K in KS:
        d = data[rho][K]
        S_mono = math.exp(-0.5*BW**2*d['omega_q2'])
        errs.append((S_mono - d['S_exact'])/d['S_exact'])
    errs = np.array(errs)
    print("  rho=%.2f: RMS=%.2f%% (signed mean %+.2f%%)" % (rho, np.sqrt((errs**2).mean())*100, errs.mean()*100))

print("\n" + "="*78)
print("PART 3: excess = CR*S; fit a + b ln(K-1); b vs p(rho)")
print("="*78)
print("%5s | %8s %8s %8s | %8s %8s" %
      ("rho","a","b","R2","p(2..7)","b_pred=p/2*<...>"))
rows_b = []
for rho in RHOS:
    lm = np.array([data[rho][K]['lam_min'] for K in KS])
    S = np.array([data[rho][K]['S_exact'] for K in KS])
    CR = np.sqrt(lm[0]/lm)
    exc = CR * S
    Karr = np.array(KS, dtype=float)
    m7 = Karr <= 7
    Amat = np.vstack([np.ones(m7.sum()), np.log(Karr[m7]-1)]).T
    coef, *_ = np.linalg.lstsq(Amat, exc[m7], rcond=None)
    pred = Amat @ coef
    ss = 1 - ((exc[m7]-pred)**2).sum()/((exc[m7]-exc[m7].mean())**2).sum()
    p7 = [r[2] for r in rows_p if r[0]==rho][0]
    rows_b.append((rho, coef[0], coef[1], ss, p7))
    print("%5.2f | %8.3f %8.3f %8.3f | %8.3f" % (rho, coef[0], coef[1], ss, p7))

# relate b to p: excess ~ (K-1)^{p/2} * S. Over K=2..7, the local slope of ln(excess) vs ln(K-1)
# at the window is approx p/2 + d ln S / d ln(K-1). Express b as linear in p.
print("\nb vs p regression:  b = c1*p + c0")
pp = np.array([r[4] for r in rows_b]); bb = np.array([r[2] for r in rows_b])
Amat = np.vstack([pp, np.ones_like(pp)]).T
coef, *_ = np.linalg.lstsq(Amat, bb, rcond=None)
pred = Amat @ coef
ss = 1 - ((bb-pred)**2).sum()/((bb-bb.mean())**2).sum()
print("  c1=%.4f c0=%.4f R2=%.4f" % (coef[0], coef[1], ss))
print("  check: b(0.47)=%.3f pred=%.3f ; b(0.40)=%.3f pred=%.3f" %
      (bb[RHOS.index(0.47)], (Amat@coef)[RHOS.index(0.47)],
       bb[RHOS.index(0.40)], (Amat@coef)[RHOS.index(0.40)]))

# saturation check: where does excess turn over (unconstrained)?
print("\nExcess monotone? (unconstrained -- turnover is from clipping, not here)")
for rho in [0.47, 0.40]:
    lm = np.array([data[rho][K]['lam_min'] for K in KS])
    S = np.array([data[rho][K]['S_exact'] for K in KS])
    exc = np.sqrt(lm[0]/lm)*S
    dexc = np.diff(exc)
    turn = np.where(dexc < 0)[0]
    print("  rho=%.2f: excess K=2..10 = %s ; turns down at K=%s" %
          (rho, " ".join("%.2f"%e for e in exc), (KS[turn[0]+1] if len(turn) else "none")))
