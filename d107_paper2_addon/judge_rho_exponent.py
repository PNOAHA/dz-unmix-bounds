# -*- coding: utf-8 -*-
"""
judge_rho_exponent.py -- settle part (c): does the LARGE-rho curvature regime obey
sigma_err = C_curv/(sqrt(N)*rho) with C_curv ~ const (claim (c)), OR is C_curv
strongly rho-dependent / sigma_err roughly rho-flat (theory_consistency refutation)?

For a fine rho grid in the unpinned regime at fixed N, measures along the CURVATURE
mode u2=(1,-2,1)/sqrt6 (the worst mode at large rho):
  - sigma_u2 = std along u2
  - C_curv = sigma_u2 * sqrt(N) * rho     (claim's prefactor; const iff rho^-1 law holds)
  - also the empirical local rho-slope of sigma_u2 and of sigma_err(top eig).
Reuses the real dz-unmix-cg core via diag_k3_mode.what_for.
"""
import math
import numpy as np
import diag_k3_mode as D

u1 = D.u1
u2 = D.u2

# fine large-rho grid (unpinned, > rho_c~0.29) at two N to separate rho-dep from N-dep
RHOS = [0.30, 0.32, 0.35, 0.38, 0.41, 0.44, 0.47]
NS = [250, 350, 700, 1500]
print("Part (c) test: C_curv = sigma_u2*sqrt(N)*rho should be CONSTANT if sigma_err~rho^-1")
print("%4s | " % "N" + " ".join("rho=%.2f" % r for r in RHOS) + "   | rho-slope(sig_u2)")
for N in NS:
    Ccurv = []
    sig_u2 = []
    sig_top = []
    for rho in RHOS:
        seed = 20260615 + 3 * 1000 + int(rho * 1e4) + N
        what, w = D.what_for(rho, N, seed)
        dev = what - what.mean(0, keepdims=True)
        cov = dev.T @ dev / what.shape[0]
        lam = np.linalg.eigvalsh(cov)
        var_u2 = float(u2 @ cov @ u2)
        s_u2 = math.sqrt(max(var_u2, 0.0))
        s_top = math.sqrt(max(float(lam[-1]), 0.0))
        sig_u2.append(s_u2)
        sig_top.append(s_top)
        Ccurv.append(s_u2 * math.sqrt(N) * rho)
    lr = np.log(np.array(RHOS))
    slope_u2 = np.polyfit(lr, np.log(np.array(sig_u2)), 1)[0]
    slope_top = np.polyfit(lr, np.log(np.array(sig_top)), 1)[0]
    print("%4d | " % N + " ".join("%6.3f" % c for c in Ccurv)
          + "   | sig_u2 slope=%+.3f  sig_top slope=%+.3f" % (slope_u2, slope_top))
    print("      Ccurv range [%.3f, %.3f] spread %.0f%%; sig_top(rho): %s"
          % (min(Ccurv), max(Ccurv), 100 * (max(Ccurv) - min(Ccurv)) / np.mean(Ccurv),
             " ".join("%.4f" % s for s in sig_top)))
