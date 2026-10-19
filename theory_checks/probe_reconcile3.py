# -*- coding: utf-8 -*-
"""
probe_reconcile3.py -- final closure checks.
 1. omega_q EXACT identity: omega_q^2 = (v^T A v)/lam_min reproduces the FFT-independent
    Rayleigh value; confirm the worst-G mode = max generalized eigvec of pencil (A,G).
 2. b = 0.679*(p/2)*<exc>: this says b_fit (slope of LINEAR a+b ln(K-1) fit) equals
    the window-mean of d(excess)/dln(K-1). Show the prompt anchors fall out, and
    derive WHY the coefficient is ~0.68 (= ratio of mean-slope to endpoint p/2*<exc>).
 3. excess = CR*S exact reproduction of the prompt's per-K excess (the compute_factors targets).
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.linalg import eigh as geigh

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
def dmu(rho): return 2.0*math.sqrt(2.0)*erfinv(rho)*SIGMA
def gauss(t, mu, s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))
def dgauss(t, mu, s): return -(t-mu)/s**2*gauss(t, mu, s)
def endmembers(K, rho):
    mus = np.array([START+i*dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h/TRAPZ(h, TGRID, axis=1)[:, None], mus
def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w/w.sum()
_KS = gauss(TGRID[:, None]-TGRID[None, :], 0.0, BW)*DT
def smooth(f): return _KS @ f

def full(K, rho):
    h, mus = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None]*h).sum(0), 1e-300)
    qp = (w[:, None]*np.stack([dgauss(TGRID, m, SIGMA) for m in mus])).sum(0)
    G = np.array([[TRAPZ(h[i]*h[j]/q, TGRID) for j in range(K)] for i in range(K)])
    hp = np.stack([dgauss(TGRID, m, SIGMA) for m in mus])
    d = hp - h*(qp/q)[None, :]
    A = np.array([[TRAPZ(d[i]*d[j]/q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K)-np.ones((K, K))/K
    Gt = P@G@P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K)/math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k]@ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    omega_q2_form = (v@A@v)/lam_min
    # generalized eig of (A,G) on tangent: max gen eigval, its eigvec overlap with v
    At = P@A@P
    ge, gv = geigh(At + 1e-12*np.eye(K), Gt + 1e-9*np.eye(K))
    # max gen-eigval eigvec (drop near-singular DC)
    idx = np.argsort(ge)
    # pick the one with largest finite ge whose vec is tangent
    cand = [k for k in idx[::-1] if abs((gv[:, k]/np.linalg.norm(gv[:, k]))@ones) < 0.5]
    gvec = gv[:, cand[0]]; gvec = gvec/np.linalg.norm(gvec)
    overlap = abs(gvec @ v)
    # S exact
    phi = np.stack([smooth(h[i]/q) for i in range(K)])
    Gs = np.array([[TRAPZ(q*phi[i]*phi[j], TGRID) for j in range(K)] for i in range(K)])
    bvec = np.array([TRAPZ(q*phi[i], TGRID) for i in range(K)])
    S = math.sqrt(max(v@Gs@v - (v@bvec)**2, 0)/lam_min)
    return lam_min, S, omega_q2_form, overlap

print("="*78)
print("PART 1: omega_q exact identity + worst-mode = max gen-eig(A,G)")
print("="*78)
print("%5s %2s | omega_q2_form  overlap(genvec,v)" % ("rho","K"))
for rho in [0.47, 0.40]:
    for K in [3, 6, 10]:
        lm, S, oq2, ov = full(K, rho)
        print("%5.2f %2d | %12.5e   %.4f" % (rho, K, oq2, ov))

print("\n" + "="*78)
print("PART 2: per-K excess reproduction (vs prompt targets) + b decomposition")
print("="*78)
KS = list(range(2, 11))
# prompt targets
TGT = {
 0.47: [0.963,1.224,1.437,1.583,1.672,1.711,1.720,1.705,1.679],  # from compute_factors run
 0.40: [0.969,1.392,1.854,2.291,2.683,2.992,3.229,3.387,3.488],
}
for rho in [0.47, 0.40]:
    lm = np.array([full(K, rho)[0] for K in KS])
    S = np.array([full(K, rho)[1] for K in KS])
    exc = np.sqrt(lm[0]/lm)*S
    errs = [(exc[i]-TGT[rho][i])/TGT[rho][i]*100 for i in range(len(KS))]
    print("rho=%.2f excess: %s" % (rho, " ".join("%.3f"%e for e in exc)))
    print("        vs tgt max|err|=%.2f%%" % max(abs(e) for e in errs))

print("\n" + "="*78)
print("PART 3: CONSOLIDATED b(p) law and anchor reproduction")
print("="*78)
RHOS = [0.30,0.35,0.40,0.47,0.55]
rows = []
for rho in RHOS:
    lm = np.array([full(K, rho)[0] for K in KS]); S = np.array([full(K, rho)[1] for K in KS])
    CR = np.sqrt(lm[0]/lm); exc = CR*S; Ka = np.array(KS, float); m7 = Ka <= 7
    Am = np.vstack([np.ones(m7.sum()), np.log(Ka[m7]-1)]).T
    c, *_ = np.linalg.lstsq(Am, exc[m7], rcond=None); bfit = c[1]
    p7 = -np.polyfit(np.log(Ka[m7]-1), np.log((lm/lm[0])[m7]), 1)[0]
    excm = exc[m7].mean()
    rows.append((rho, bfit, p7, excm))
# fit b = k*(p/2)*<exc>
pp = np.array([r[2] for r in rows]); bb = np.array([r[1] for r in rows]); ee = np.array([r[3] for r in rows])
x = (pp/2)*ee; k = (bb*x).sum()/(x*x).sum(); ss = 1-((bb-k*x)**2).sum()/((bb-bb.mean())**2).sum()
print("CONSOLIDATED:  b(rho) = %.3f * (p(rho)/2) * <excess>_{2..7}     R2=%.4f" % (k, ss))
print("  where p(rho) = 0.24/erfinv(rho)^2 - 0.19 (K=2..7 window)")
print("%5s | b_fit  b_pred  p7    <exc>" % "rho")
for (rho, bfit, p7, excm) in rows:
    print("%5.2f | %.3f  %.3f  %.3f  %.3f" % (rho, bfit, k*(p7/2)*excm, p7, excm))
print("\nAnchors: b(0.47)=%.3f (prompt ~0.44), b(0.40)=%.3f (prompt ~1.14)" %
      (bb[RHOS.index(0.47)], bb[RHOS.index(0.40)]))
