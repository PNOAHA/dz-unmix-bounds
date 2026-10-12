# Closed-form rho_c from the deficit criterion, and its small-rho leading order.
# Criterion (exact, in physical mu units):
#   pin  <=>  bw^2 + Var_w*(mu) > (mu_K - mubar*)(mubar* - mu_1)
# With equally spaced centers mu_k = mu_1 + (k-1) dmu and index moments
#   kbar in [0,K-1], Vk = index variance of w*, define
#   Var_w*(mu) = dmu^2 * Vk
#   (mu_K-mubar*)(mubar*-mu_1) = dmu^2 * (K-1-kbar)(kbar) =: dmu^2 * C
# So criterion:  bw^2 > dmu^2 (C - Vk).
# Define D := C - Vk = (K-1-kbar)*kbar - Vk  (the index "variance deficit coefficient").
# Pinning holds iff  dmu^2 < bw^2 / D   (requires D>0), i.e.
#   dmu_c = bw / sqrt(D),  and rho_c = erf( dmu_c / (2 sqrt2 sigma) ) inverting dmu=2 sqrt2 sigma erfinv(rho).
# Small-rho: dmu ~ sigma sqrt(2 pi) rho  =>  rho_c ~ bw / (sigma sqrt(2 pi D)).

import numpy as np
from math import sqrt, pi, erf
from scipy.special import erfinv

SIGMA=25.0
def dmu_of_rho(r): return SIGMA*2*sqrt(2)*erfinv(r)
def rho_of_dmu(d): return erf(d/(2*sqrt(2)*SIGMA))

# index moments from N2 predictions
PARAM = {  # K: (kbar, Vk)
  3:(1.0833333333333333, 0.5763888888888891),
  4:(1.625, 1.234375),
  5:(2.125, 1.859375),
}
def D_of(K):
    kbar,Vk = PARAM[K]; C=(K-1-kbar)*kbar; return C-Vk, C

print(f"{'K':>2} {'C':>10} {'Vk':>10} {'D=C-Vk':>10}")
for K in (3,4,5):
    D,C = D_of(K); kbar,Vk=PARAM[K]
    print(f"{K:>2} {C:>10.5f} {Vk:>10.5f} {D:>10.5f}")

print()
print(f"{'K':>2} {'bw':>5} {'rho_c(exact)':>13} {'rho_c(smallrho)':>16} {'rho*_obs':>10}")
N2obs = {(3,6.):0.1559375,(3,12.):0.265,(3,24.):0.4203125,
         (4,6.):0.093125,(4,12.):0.179375,(4,24.):0.28875,
         (5,6.):0.066875,(5,12.):0.123125,(5,24.):0.20875}
for K in (3,4,5):
    D,_=D_of(K)
    for bw in (6.,12.,24.):
        dmu_c = bw/sqrt(D)
        rc_exact = rho_of_dmu(dmu_c)
        rc_small = bw/(SIGMA*sqrt(2*pi*D))
        print(f"{K:>2} {bw:>5.1f} {rc_exact:>13.6f} {rc_small:>16.6f} {N2obs[(K,bw)]:>10.6f}")

print()
print("leading-order law:  rho_c ~ bw / (sigma * sqrt(2 pi * D)),  D=(K-1-kbar)kbar - Vk")
print("=> rho_c proportional to bw/sigma, and to 1/sqrt(D) (D grows ~ (K-1)^2/4 for centered w*)")
