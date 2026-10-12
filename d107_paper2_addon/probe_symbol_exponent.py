# -*- coding: utf-8 -*-
"""
probe_symbol_exponent.py -- get the analytic form of the Toeplitz symbol near theta=pi and
derive p(rho) for lambda_min ~ (K-1)^{-p}.

Symbol:  f(theta) = m_0 + 2 sum_{k>=1} m_k cos(k theta),  m_k = c * exp(-b k^2),  b = Delta^2/(4 sig^2) = -ln r.
At theta = pi - x (x small, x = pi/K for the worst available mode):
  f(pi - x) = c * sum_{k=-inf}^{inf} exp(-b k^2) cos(k(pi - x))
            = c * sum_k exp(-b k^2) (-1)^k cos(k x)
This is a Jacobi-theta / Poisson-summable comb. Use Poisson summation:
  sum_k exp(-b k^2) e^{i k phi} = sqrt(pi/b) sum_n exp(-(phi - (2n+1)pi... )  -- careful with the (-1)^k.

Let phi = pi - x. Then sum_k exp(-b k^2) e^{i k phi} = sqrt(pi/b) sum_n exp(-(phi-2 pi n)^2/(4b)).
With phi near pi, the dominant Poisson images are n=0 (phi=pi-x) and n=1 (phi-2pi = -pi-x),
both at distance ~pi from a node. So
  f(pi-x) ~ c sqrt(pi/b) * 2 exp(-(pi-x)^2/(4b))  [for x small, the two images ~symmetric? check]
Actually images at distance |pi - x| and |pi + x| (n=0 gives pi-x, n=1 gives -(pi+x)):
  f(pi-x) = c sqrt(pi/b) [ exp(-(pi-x)^2/(4b)) + exp(-(pi+x)^2/(4b)) + (smaller n) ]
For x>0 small: leading = exp(-(pi-x)^2/(4b)) dominates (closer to 0). Expand:
  (pi-x)^2/(4b) = pi^2/(4b) - pi x/(2b) + x^2/(4b)
  => f(pi-x) ~ c sqrt(pi/b) exp(-pi^2/(4b)) * exp(pi x/(2b)) * [1 + exp(-pi x/b)]   (n=0 + n=1)
With x = pi/K (worst available mode spacing) the K-dependence is exp(pi x/(2b)) = exp(pi^2/(2 b K)).
That is INCREASING in 1/K... but lambda_min DECREASES with K. So the worst mode is x->0 (theta->pi exactly)
only at K->inf; for finite K the smallest tangent eigenvalue sits at the LARGEST achievable k-index,
i.e. theta closest to pi from BELOW available on the discrete grid. Let's just empirically map.
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA = 25.0


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def b_of_rho(rho): return dmu(rho) ** 2 / (4 * SIGMA ** 2)   # = -ln r


def c_const(): return 1.0 / (2 * SIGMA * math.sqrt(math.pi))


def symbol(theta, rho, kmax=60):
    b = b_of_rho(rho); c = c_const()
    s = c
    for k in range(1, kmax + 1):
        s += 2 * c * math.exp(-b * k * k) * math.cos(k * theta)
    return s


def symbol_poisson(theta, rho, nmax=4):
    # f(theta) = c sqrt(pi/b) sum_n exp(-(theta - 2 pi n)^2/(4b))
    b = b_of_rho(rho); c = c_const()
    s = 0.0
    for n in range(-nmax, nmax + 1):
        s += math.exp(-((theta - 2 * math.pi * n) ** 2) / (4 * b))
    return c * math.sqrt(math.pi / b) * s


# verify Poisson form matches direct sum
print("Poisson vs direct symbol (rho=0.47):")
for th in (0.0, math.pi / 2, math.pi - 0.3, math.pi):
    print("  theta=%.3f  direct=%.5e  poisson=%.5e" %
          (th, symbol(th, 0.47), symbol_poisson(th, 0.47)))

# The worst DISCRETE tangent mode for a K-vector is the DFT-like mode with index ~ K-1,
# i.e. theta_min = pi*(K-1)/K = pi - pi/K. Map lammin(M) ~ f(theta=pi-pi/K).
# Now derive the EXPONENT p: lammin(M)/lammin(M,2) = f(pi-pi/K)/f(pi-pi/2)=f(pi-pi/K)/f(pi/2).
# Test p over K and the (K-1) power law, both rho.
print("\nExponent p of f(pi-pi/K) ~ (K-1)^{-p_M}:")
for rho in (0.47, 0.40, 0.35):
    Ks = np.arange(2, 11)
    f = np.array([symbol(math.pi - math.pi / K, rho) for K in Ks])
    p = -np.polyfit(np.log(Ks - 1), np.log(f / f[0]), 1)[0]
    b = b_of_rho(rho)
    print("  rho=%.2f  b=%.4f  p_M(symbol)=%.3f" % (rho, b, p))

# Now the KEY analytic exponent. Near theta=pi, write theta=pi - pi/K. The n=0 Poisson image:
#   exp(-(pi - pi/K)^2/(4b)). Ratio to K=2 (theta=pi/2 ... but pi/2 not near pi).
# Instead, work the LOG: ln f(pi-pi/K) ~ ln[c sqrt(pi/b)] - (pi-pi/K)^2/(4b)  (n=0 dominant when b small)
#   = const - pi^2/(4b)(1 - 1/K)^2
# d ln f / d ln(K-1): treat as fn of K. Let's just check the local-slope approximation:
print("\nAnalytic n=0 approx: ln f(pi-pi/K) ~ const - pi^2/(4b)*(1-1/K)^2")
for rho in (0.47, 0.40, 0.35):
    b = b_of_rho(rho)
    Ks = np.arange(2, 11)
    lnf_approx = -(math.pi ** 2) / (4 * b) * (1 - 1.0 / Ks) ** 2
    # fit (lnf - lnf[0]) vs ln(K-1)
    p = -np.polyfit(np.log(Ks - 1), lnf_approx - lnf_approx[0], 1)[0]
    print("  rho=%.2f  b=%.4f  pi^2/(4b)=%.3f  p(n=0 approx)=%.3f" %
          (rho, b, math.pi ** 2 / (4 * b), p))
