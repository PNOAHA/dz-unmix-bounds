"""
D1-KKT leading-order constant for the MIDDLE slack, symmetric K=3.
Test the closed form derived analytically:

  eta_mid (= 1 - I_2 at the symmetric 2-endmember q0, a=1/2) should, to leading order in dmu,
  equal a positive multiple of  (bw^2 + Var_w*(mu) - (mu_K-mubar)(mubar-mu_1)) / sigma_eff^2-type scale.

For K=3 symmetric (a=1/2, mu=(-d,0,+d) recentered, d=dmu), q0 = (h(-d)+h(+d))/2.
We test the *small-dmu* prediction:

  Define VARIANCE DEFICIT  D := Var(p) - Var(q0).
    Var(p)  = sigma^2 + bw^2 + Var_w*(mu)              (data)
    Var(q0) = sigma^2 + (mu_K-mubar)(mubar-mu_1) at a=1/2 symmetric = sigma^2 + d^2
  so D = bw^2 + Var_w*(mu) - d^2 = bw^2 + (2c)d^2 - d^2 = bw^2 - (1-2c) d^2   [Var_w*=2c*d^2 here? check]

Actually with mu=(-d,0,d): Var_w*(mu)=c*d^2+0+c*d^2 = 2c d^2 ; (mu_K-mubar)(mubar-mu_1)=(d)(d)=d^2.
So moment release happens when d^2 > bw^2 + 2c d^2 - ... wait recompute:
  criterion pinned  <=>  (mu_K-mubar)(mubar-mu_1) < bw^2 + Var_w*(mu)
                    <=>  d^2 < bw^2 + 2c d^2  <=> d^2 (1-2c) < bw^2.
So C-coefficient = (1-2c). For c=1/3: (1-2/3)=1/3, matches n2 moment_pred C.

Leading-order CLAIM to test numerically:
  eta_mid  ~  kappa(sigma_eff) * (bw^2 - (1-2c) d^2) * d^2 / sigma^4   (a product: deficit * curvature weight)
We don't fix kappa a priori; we test that eta_mid is (i) positive iff deficit>0, (ii) ~ d^2 * deficit
to leading order, i.e. eta_mid/(d^2 * deficit) -> const as d->0.
"""
import numpy as np, math, os
from scipy.special import erfinv as _erfinv

SIGMA=25.0; START=200.0
TGRID=np.arange(50.0,2500.0+1e-9,2.0)   # finer grid for accuracy
TRAPZ=getattr(np,'trapezoid',None) or np.trapz
SQRT2=math.sqrt(2.0)

def hk(mu,s=SIGMA):
    g=np.exp(-0.5*((TGRID-mu)/s)**2); return g/TRAPZ(g,TGRID)
def peaks(rho):
    dmu=SIGMA*2*SQRT2*float(_erfinv(rho)); return dmu

def eta_mid_symmetric(c, rho, bw):
    """K=3 symmetric weights (c,1-2c,c). a=1/2 by symmetry. return eta_mid, d, deficit."""
    d=peaks(rho)
    mus=np.array([START-d,START,START+d])
    seff=math.sqrt(SIGMA**2+bw**2)
    w=np.array([c,1-2*c,c])
    comp=np.array([hk(m,seff) for m in mus])
    p=(w[:,None]*comp).sum(0); p/=TRAPZ(p,TGRID)
    h1=hk(mus[0]); h2=hk(mus[1]); h3=hk(mus[2])
    a=0.5
    q0=a*h1+(1-a)*h3+1e-300
    I2=TRAPZ(p/q0*h2,TGRID)/TRAPZ(h2,TGRID)
    eta=1.0-I2
    Varw=2*c*d**2
    deficit=bw**2+Varw-d**2     # = bw^2 -(1-2c)d^2
    return eta,d,deficit

print("symmetric K=3, test eta_mid ~ d^2*deficit/sigma^4 scaling")
print("%-4s %-6s %-10s %-12s %-12s %-14s %-14s"%("c","bw","rho","d","deficit","eta_mid","eta/(d2*def/s4)"))
for c in (1/3, 0.25, 0.40):
    for bw in (12.0,):
        for rho in (0.01,0.02,0.04,0.08):
            eta,d,deff=eta_mid_symmetric(c,rho,bw)
            scale=d**2*deff/SIGMA**4
            print("%-4.2f %-6.0f %-10.3f %-12.5f %-12.4f %-14.6e %-14.6f"%
                  (c,bw,rho,d,deff,eta,eta/scale if scale!=0 else float('nan')))
    print()

# Compare the limiting constant across c: theory says the curvature weight should be
# a pure-Gaussian factor independent of c at leading order (the deficit carries the c-dependence).
print("limiting eta/(d^2*deficit/sigma^4) as rho->0 (smallest rho per c):")
for c in (1/3,0.25,0.40):
    eta,d,deff=eta_mid_symmetric(c,0.005,12.0)
    print("  c=%.3f  const=%.6f"%(c, eta/(d**2*deff/SIGMA**4)))
