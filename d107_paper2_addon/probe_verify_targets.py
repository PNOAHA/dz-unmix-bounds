# -*- coding: utf-8 -*-
"""
probe_verify_targets.py -- final verification of closed forms against the PROMPT's stated targets.

Targets from prompt:
  rho=0.47: lammin(K=2..10)=6.07e-1,3.18e-1,1.99e-1,1.42e-1,1.12e-1,9.57e-2,8.64e-2,8.10e-2,7.82e-2 ; p~0.95
  rho=0.40: lammin(K=2..10)=4.54e-1,1.87e-1,9.10e-2,5.10e-2,3.23e-2,2.26e-2,1.71e-2,1.39e-2,1.18e-2 ; p~1.88

Closed forms to test:
  (1) p_G(rho) = 0.288/erfinv(rho)^2 - 0.438
  (2) lammin shape predictor: lammin_hat(K) = lammin(2) * [f_M(theta_K)/f_M(theta_2)] * [(K-1)^{-p_R}]
      with theta_K=pi(K-1)/K, p_R=0.37, anchored at the TRUE lammin(2).  (1 anchor per rho, else closed.)
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START = 25.0, 200.0
TGRID = np.arange(50.0, 3000.0 + 1e-9, 0.5)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

TGT = {
 0.47: [6.07e-1,3.18e-1,1.99e-1,1.42e-1,1.12e-1,9.57e-2,8.64e-2,8.10e-2,7.82e-2],
 0.40: [4.54e-1,1.87e-1,9.10e-2,5.10e-2,3.23e-2,2.26e-2,1.71e-2,1.39e-2,1.18e-2],
}

def b_of_rho(rho): return 2.0 * erfinv(rho) ** 2
def c_const(): return 1.0 / (2 * SIGMA * math.sqrt(math.pi))
def fM(theta, rho, nmax=6):
    b = b_of_rho(rho)
    s = sum(math.exp(-((theta - 2*math.pi*n)**2)/(4*b)) for n in range(-nmax, nmax+1))
    return c_const()*math.sqrt(math.pi/b)*s

def pG_closed(rho): return 0.288/erfinv(rho)**2 - 0.438

print("(1) p_G closed form vs prompt-stated exponent:")
print("  rho=0.47: pG_closed=%.3f  (prompt table ~0.95)" % pG_closed(0.47))
print("  rho=0.40: pG_closed=%.3f  (prompt table ~1.88)" % pG_closed(0.40))

print("\n(2) lammin shape predictor (anchored at true lammin(K=2), p_R=0.37):")
p_R = 0.37
for rho in (0.47, 0.40):
    print("  === rho=%.2f ===" % rho)
    print("   %2s | %11s %11s %8s" % ("K", "target", "pred", "err%"))
    tgt = TGT[rho]
    lam2 = tgt[0]
    f2 = fM(math.pi*(2-1)/2, rho)
    es = []
    for idx, K in enumerate(range(2, 11)):
        fK = fM(math.pi*(K-1)/K, rho)
        pred = lam2 * (fK/f2) * (K-1)**(-p_R)
        e = (pred - tgt[idx])/tgt[idx]*100
        es.append(e)
        print("   %2d | %11.4e %11.4e %7.1f" % (K, tgt[idx], pred, e))
    es = np.array(es)
    print("   --> mean|err|=%.1f%%  max|err|=%.1f%%" % (np.abs(es).mean(), np.abs(es).max()))
    # also report the fitted exponent of the PREDICTION to compare to target slope
    Ks = np.arange(2, 11)
    preds = np.array([lam2*(fM(math.pi*(K-1)/K, rho)/f2)*(K-1)**(-p_R) for K in Ks])
    pp = -np.polyfit(np.log(Ks-1), np.log(preds/preds[0]), 1)[0]
    pt = -np.polyfit(np.log(Ks-1), np.log(np.array(tgt)/tgt[0]), 1)[0]
    print("   pred exponent=%.3f  target exponent=%.3f" % (pp, pt))
