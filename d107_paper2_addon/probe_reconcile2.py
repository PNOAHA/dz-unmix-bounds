# -*- coding: utf-8 -*-
"""
probe_reconcile2.py -- second-pass adjudication.
 1. clean p(rho): use the K=2..7 window (the one the prompt's targets 0.95/1.88 came from)
    AND K=2..10; report which closed form best matches each. Settle on consolidated law.
 2. clean two-moment S: compute m2 = Eq[g''^2]/Eq[(g-<g>)^2] analytically (no np.gradient),
    via second derivative of the gaussians. Report S_2mom accuracy.
 3. b(p): the excess local log-slope. Derive b analytically:
       excess(K) = CR(K)*S(K),  CR=(K-1)^{p/2} approx, S=stretched-exp.
    The a+b ln(K-1) fit's b is the AVERAGE slope of excess (LINEAR in excess), not of ln(excess).
    So b = mean over window of  d(excess)/d ln(K-1). Decompose.
 4. turnover: confirm unconstrained excess saturates/turns only at large K (rho-dependent),
    and that the K~7 empirical turnover must come from elsewhere (clipping).
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

def dmu(rho): return 2.0*math.sqrt(2.0)*erfinv(rho)*SIGMA
def gauss(t, mu, s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))
def dgauss(t, mu, s): return -(t-mu)/s**2*gauss(t, mu, s)
def ddgauss(t, mu, s):
    z = (t-mu)/s
    return (z**2 - 1.0)/s**2 * gauss(t, mu, s)

def endmembers(K, rho):
    mus = np.array([START+i*dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h/TRAPZ(h, TGRID, axis=1)[:, None], mus

def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w/w.sum()

_KS = gauss(TGRID[:, None]-TGRID[None, :], 0.0, BW)*DT
def smooth(f): return _KS @ f

def core(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None]*h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i]*h[j]/q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K)-np.ones((K, K))/K
    Gt = P@G@P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K)/math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k]@ones) < 0.5]
    kmin = tang[0]; v = V[:, kmin]; lam_min = lam[kmin]
    # S exact
    phi = np.stack([smooth(h[i]/q) for i in range(K)])
    Gs = np.array([[TRAPZ(q*phi[i]*phi[j], TGRID) for j in range(K)] for i in range(K)])
    bvec = np.array([TRAPZ(q*phi[i], TGRID) for i in range(K)])
    var_gs = v@Gs@v - (v@bvec)**2
    S = math.sqrt(max(var_gs, 0)/lam_min)
    # omega_q^2 and m2 analytic
    qp = (w[:, None]*np.stack([dgauss(TGRID, m, SIGMA) for m in mus])).sum(0)
    qpp = (w[:, None]*np.stack([ddgauss(TGRID, m, SIGMA) for m in mus])).sum(0)
    hp = np.stack([dgauss(TGRID, m, SIGMA) for m in mus])
    hpp = np.stack([ddgauss(TGRID, m, SIGMA) for m in mus])
    g = (v[:, None]*h).sum(0)/q
    # g' = (sum v_i h_i')/q - g q'/q
    num = (v[:, None]*hp).sum(0)
    gprime = num/q - g*qp/q
    # g'' = (sum v_i h_i'')/q - 2 g' q'/q - g q''/q  (quotient rule twice)
    num2 = (v[:, None]*hpp).sum(0)
    gpp = num2/q - 2*gprime*qp/q - g*qpp/q
    Eq = lambda f: TRAPZ(f*q, TGRID)
    var_g = Eq(g*g) - Eq(g)**2
    m1 = Eq(gprime*gprime)/var_g
    m2 = Eq(gpp*gpp)/var_g
    return dict(lam_min=lam_min, S=S, m1=m1, m2=m2, var_g=var_g)

RHOS = [0.30, 0.35, 0.40, 0.47, 0.55]
KS = list(range(2, 11))
D = {rho: {K: core(K, rho) for K in KS} for rho in RHOS}

print("="*78)
print("PART 1 (refined): p(rho) -- consolidated law")
print("="*78)
def pA(rho): return 0.288/erfinv(rho)**2 - 0.438   # attempt 1
def pC(rho): return 0.739/(2*erfinv(rho)**2) - 0.919  # attempt 3
print("%5s | %8s %8s | %8s %8s | best-window match" %
      ("rho","p(2..7)","p(2..10)","pA","pC"))
rp = []
for rho in RHOS:
    lm = np.array([D[rho][K]['lam_min'] for K in KS]); Ka = np.array(KS, float)
    m7 = Ka <= 7
    p7 = -np.polyfit(np.log(Ka[m7]-1), np.log((lm/lm[0])[m7]), 1)[0]
    pf = -np.polyfit(np.log(Ka-1), np.log(lm/lm[0]), 1)[0]
    rp.append((rho, p7, pf))
    print("%5.2f | %8.3f %8.3f | %8.3f %8.3f" % (rho, p7, pf, pA(rho), pC(rho)))
# best LS fit to BOTH windows
for lbl, idx in [("p(2..7)",1), ("p(2..10)",2)]:
    x = np.array([1/erfinv(r[0])**2 for r in rp]); y = np.array([r[idx] for r in rp])
    Am = np.vstack([x, np.ones_like(x)]).T; c, *_ = np.linalg.lstsq(Am, y, rcond=None)
    ss = 1-((y-Am@c)**2).sum()/((y-y.mean())**2).sum()
    print("  consolidated %s: p = %.3f/erfinv^2 %+.3f  R2=%.4f" % (lbl, c[0], c[1], ss))

print("\n" + "="*78)
print("PART 2 (refined): S two-moment with ANALYTIC m2")
print("="*78)
print("%5s %2s | %8s %8s %8s | mono%% 2mom%%" % ("rho","K","S_exact","S_mono","S_2mom"))
for rho in [0.47, 0.40]:
    for K in KS:
        d = D[rho][K]
        S_mono = math.exp(-0.5*BW**2*d['m1'])
        S_2 = math.exp(-0.5*BW**2*d['m1'] + 0.25*BW**4*max(d['m2']-d['m1']**2, 0))
        e1 = (S_mono-d['S'])/d['S']*100; e2 = (S_2-d['S'])/d['S']*100
        print("%5.2f %2d | %8.4f %8.4f %8.4f | %+5.1f %+5.1f" %
              (rho, K, d['S'], S_mono, S_2, e1, e2))
print("\n2-moment RMS:")
for rho in RHOS:
    e2 = []
    for K in KS:
        d = D[rho][K]
        S_2 = math.exp(-0.5*BW**2*d['m1'] + 0.25*BW**4*max(d['m2']-d['m1']**2, 0))
        e2.append((S_2-d['S'])/d['S'])
    e2 = np.array(e2)
    print("  rho=%.2f 2mom RMS=%.2f%% mean=%+.2f%%" % (rho, np.sqrt((e2**2).mean())*100, e2.mean()*100))

print("\n" + "="*78)
print("PART 3 (refined): b(p). b = average d(excess)/d ln(K-1) over K=2..7.")
print("="*78)
# Analytic: excess = CR*S. d excess/d ln(K-1) = excess*(d ln CR/d ln(K-1) + d ln S/d ln(K-1))
#   d ln CR/d ln(K-1) = p/2 (CR~(K-1)^{p/2}).
#   d ln S/d ln(K-1) = -q_S (S stretched, local slope).
# The lstsq b is ~ <excess * (p/2 - q_S)> over the window. Show b tracks p/2 scaled by <excess>.
print("%5s | %6s %6s %6s %6s %6s | %6s" %
      ("rho","b_fit","p7","p7/2","<exc>","qS","p/2*<exc>"))
rb = []
for rho in RHOS:
    lm = np.array([D[rho][K]['lam_min'] for K in KS]); S = np.array([D[rho][K]['S'] for K in KS])
    CR = np.sqrt(lm[0]/lm); exc = CR*S; Ka = np.array(KS, float); m7 = Ka <= 7
    Am = np.vstack([np.ones(m7.sum()), np.log(Ka[m7]-1)]).T
    c, *_ = np.linalg.lstsq(Am, exc[m7], rcond=None)
    bfit = c[1]
    p7 = [r[1] for r in rp if r[0]==rho][0]
    # local S slope qS over window
    qS = -np.polyfit(np.log(Ka[m7]-1+1e-9)[1:], np.log(S[m7][1:]/S[m7][0]), 1)[0] if m7.sum()>1 else 0
    exc_mean = exc[m7].mean()
    rb.append((rho, bfit, p7, qS, exc_mean))
    print("%5.2f | %6.3f %6.3f %6.3f %6.3f %6.3f | %6.3f" %
          (rho, bfit, p7, p7/2, exc_mean, qS, p7/2*exc_mean))
# best single-parameter model b = alpha*(p/2)*<exc>  vs  b = beta*p
print("\nModels for b:")
pp = np.array([r[2] for r in rb]); bb = np.array([r[1] for r in rb])
ee = np.array([r[4] for r in rb])
# model A: b ~ k*p
kA = (bb*pp).sum()/(pp*pp).sum(); rA = 1-((bb-kA*pp)**2).sum()/((bb-bb.mean())**2).sum()
print("  b = %.3f * p           R2=%.4f" % (kA, rA))
# model B: b ~ k*(p/2)*<exc>
xB = (pp/2)*ee; kB = (bb*xB).sum()/(xB*xB).sum(); rB = 1-((bb-kB*xB)**2).sum()/((bb-bb.mean())**2).sum()
print("  b = %.3f * (p/2)*<exc> R2=%.4f" % (kB, rB))
# model C: b linear in p with intercept
Am = np.vstack([pp, np.ones_like(pp)]).T; c, *_ = np.linalg.lstsq(Am, bb, rcond=None)
rC = 1-((bb-Am@c)**2).sum()/((bb-bb.mean())**2).sum()
print("  b = %.3f*p %+.3f       R2=%.4f" % (c[0], c[1], rC))
print("  ANCHORS: b(0.47)=%.3f, b(0.40)=%.3f" % (bb[RHOS.index(0.47)], bb[RHOS.index(0.40)]))

print("\n" + "="*78)
print("PART 4: turnover (unconstrained excess vs K, extended to K=14)")
print("="*78)
KS2 = list(range(2, 15))
for rho in [0.55, 0.47, 0.40, 0.35]:
    lm = np.array([core(K, rho)['lam_min'] for K in KS2])
    S = np.array([core(K, rho)['S'] for K in KS2])
    exc = np.sqrt(lm[0]/lm)*S
    d = np.diff(exc); turn = np.where(d < 0)[0]
    print("rho=%.2f peak@K=%s : %s" %
          (rho, (KS2[turn[0]] if len(turn) else ">14"),
           " ".join("%.2f"%e for e in exc)))
