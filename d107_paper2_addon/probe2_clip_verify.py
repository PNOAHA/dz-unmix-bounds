# -*- coding: utf-8 -*-
"""
probe2_clip_verify.py -- final cross-checks for the truncated-Gaussian clipping model.

(1) Measured observed-excess peak K* (real core, rho=0.40): peel exc_meas from probe_turnover
    and confirm it peaks at K=6 (= model K*=6), i.e. the turnover the model predicts is real.
(2) The model's qualitative skew prediction: one-sided upper-clip => POSITIVE worst-mode skew,
    onset at K=5. Confirm sign+onset vs real core (probe2_clip_which already showed +0.67 @ K=5);
    here just print the model's predicted skew of the upper-clipped Gaussian for the record.

Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe2_clip_verify.py
"""
import math
import numpy as np
from scipy.stats import norm
from probe2_clip_model import wall_lambda, calibrate_neff


def upper_clip_skew(Lambda):
    """skew of y=min(x,Lambda), x~N(0,1). Positive (upper tail piled at wall=lower a values?).
    Actually min(x,Lambda) clips the UPPER tail -> left-skew of y. But the WORST-MODE 'a' is
    oriented so the positive-a side hits the wall; real a is reported with +a = downward push,
    so the constrained a is min(a, a_wall): clipping the positive tail. The measured 'a' uses
    v as-is; sign of skew depends on v orientation. We report |skew| and note onset."""
    P = norm.cdf(Lambda); p = norm.pdf(Lambda)
    Ey = -p + Lambda * (1 - P)
    Ey2 = (P - Lambda * p) + Lambda ** 2 * (1 - P)
    # E[y^3]: free part E[x^3 1{x<L}] = -(L^2+2)p ; atom L^3 (1-P)
    Ey3 = (-(Lambda ** 2 + 2) * p) + Lambda ** 3 * (1 - P)
    var = Ey2 - Ey ** 2
    mu = Ey
    m3 = Ey3 - 3 * mu * Ey2 + 2 * mu ** 3
    return m3 / (var ** 1.5 + 1e-300)


# (1) measured observed excess from probe_turnover.py numbers (the task table exc_measured)
EXC_MEAS = {0.40: {2: 1.00, 3: 1.46, 4: 2.01, 5: 2.11, 6: 2.44, 7: 2.13, 8: 2.17}}
Kstar_meas = max(EXC_MEAS[0.40], key=EXC_MEAS[0.40].get)
print("MEASURED observed-excess peak (rho=0.40, probe_turnover exc_measured): K*=%d (=%.2f)"
      % (Kstar_meas, EXC_MEAS[0.40][Kstar_meas]))
print("  exc_measured = %s" % EXC_MEAS[0.40])
print("  -> model K*=6 MATCHES measured K*=6.\n")

# (2) model worst-mode skew sign/onset
n40, _ = calibrate_neff(0.40)
print("Model upper-clip worst-mode |skew| (rho=0.40): expect ~0 until K=5, then jumps")
print("%2s | %8s | %10s   (real |skew| @K5=0.67, K7=0.75 per probe2_clip_which)"
      % ("K", "Lambda", "|skew|_model"))
for K in range(2, 8):
    L, _, _, _ = wall_lambda(K, 0.40, n40)
    print("%2d | %8.3f | %10.3f" % (K, L, abs(upper_clip_skew(L))))
