# -*- coding: utf-8 -*-
"""
probe2_deflation3.py -- pin the EXACT 0.679 vs 0.562 discrepancy and report the honest constant.

reconcile3 got K0=0.679 using: full() S via exact KxK form, p7 = -slope of ln(lm/lm0) over K<=7,
<exc> = mean over K=2..7, and b = LS slope of (a + b ln(K-1)).  My probe2_deflation2 got 0.562.
Reproduce reconcile3 EXACTLY here (copy full()) so the number is unambiguous, then decompose.
Also include rho=0.55 vs not, since the global coeff is leverage-sensitive.
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
def dmu(rho): return 2.0*math.sqrt(2.0)*erfinv(rho)*SIGMA
def gauss(t, mu, s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))
def endmembers(K, rho):
    mus = np.array([START+i*dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h/TRAPZ(h, TGRID, axis=1)[:, None], mus
def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w/w.sum()
_KS = gauss(TGRID[:, None]-TGRID[None, :], 0.0, BW)*DT
def smooth(f): return _KS @ f

def full(K, rho):
    """EXACT copy of reconcile3.full (KxK S form)."""
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None]*h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i]*h[j]/q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K)-np.ones((K, K))/K
    Gt = P@G@P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K)/math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k]@ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    phi = np.stack([smooth(h[i]/q) for i in range(K)])
    Gs = np.array([[TRAPZ(q*phi[i]*phi[j], TGRID) for j in range(K)] for i in range(K)])
    bvec = np.array([TRAPZ(q*phi[i], TGRID) for i in range(K)])
    S = math.sqrt(max(v@Gs@v - (v@bvec)**2, 0)/lam_min)
    return lam_min, S

KS = list(range(2, 11))
def consolidated(RHOS):
    rows = []
    for rho in RHOS:
        lm = np.array([full(K, rho)[0] for K in KS]); S = np.array([full(K, rho)[1] for K in KS])
        CR = np.sqrt(lm[0]/lm); exc = CR*S; Ka = np.array(KS, float); m7 = Ka <= 7
        Am = np.vstack([np.ones(m7.sum()), np.log(Ka[m7]-1)]).T
        c, *_ = np.linalg.lstsq(Am, exc[m7], rcond=None); bfit = c[1]
        p7 = -np.polyfit(np.log(Ka[m7]-1), np.log((lm/lm[0])[m7]), 1)[0]
        excm = exc[m7].mean()
        rows.append((rho, bfit, p7, excm))
    pp = np.array([r[2] for r in rows]); bb = np.array([r[1] for r in rows]); ee = np.array([r[3] for r in rows])
    x = (pp/2)*ee; k = (bb*x).sum()/(x*x).sum(); ss = 1-((bb-k*x)**2).sum()/((bb-bb.mean())**2).sum()
    return k, ss, rows

for label, RHOS in [("WITH 0.55 (reconcile3 set)", [0.30, 0.35, 0.40, 0.47, 0.55]),
                    ("WITHOUT 0.55", [0.30, 0.35, 0.40, 0.47]),
                    ("narrow 0.40-0.47", [0.40, 0.47])]:
    k, ss, rows = consolidated(RHOS)
    print("\n%s:  K0=%.4f  R2=%.4f" % (label, k, ss))
    for (rho, bfit, p7, excm) in rows:
        print("   rho=%.2f b=%.3f p7=%.3f <exc>=%.3f  per-rho K0=b/((p/2)<exc>)=%.3f"
              % (rho, bfit, p7, excm, bfit/((p7/2)*excm)))

# honest summary: per-rho K0 over the core rho band, and its mean
k, ss, rows = consolidated([0.30, 0.35, 0.40, 0.47, 0.55])
perrho = np.array([r[1]/((r[2]/2)*r[3]) for r in rows])
print("\nper-rho K0 (b/((p/2)<exc>)): ", " ".join("%.3f" % x for x in perrho),
      "mean=%.3f" % perrho.mean())
print("GLOBAL leverage-weighted K0 = %.4f  (this is the published 0.679-type number)" % k)
