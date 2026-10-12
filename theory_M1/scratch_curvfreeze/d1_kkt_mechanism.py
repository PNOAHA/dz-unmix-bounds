"""
Confirm the analytic mechanism behind eta_mid = (1/2) d^2 deficit / sigma^4.

Derivation skeleton (symmetric K=3, recenter mu=(-d,0,d)):
  I_2 = integral p(t) h2(t)/q0(t) dt   (h2 area-normalized; integral h2 = 1)
  with p,q0,h2 all gaussians.  Write everything as gaussian-ratio.
  h2(t) = N(t;0,sigma).  q0 = 1/2[N(t;-d,sigma)+N(t;d,sigma)].
  At d->0, q0 -> N(t;0,sigma)*(1 + (d^2/2)(t^2/sigma^4 - 1/sigma^2) + O(d^4))   (even expansion)
  so h2/q0 -> 1 - (d^2/2)(t^2/sigma^4 - 1/sigma^2) + O(d^4).
  p = sum w_k N(t; mu_k, seff). Its even part around 0 to O(d^2):
    p -> N(t;0,seff)*(1 + (Var_w*mu /2)(t^2/seff^4 -1/seff^2)+...) BUT seff != sigma.
  The leading 'deficit' enters via integral p*(t^2)/sigma^4 etc.
  We just CONFIRM numerically the decomposition I_2 = 1 - (d^2/2)/sigma^4 * (E_p[t^2 weighted] - ...) = 1 - eta.

Concretely test the identity:
  eta_mid ?= (d^2/(2 sigma^4)) * ( Var(p) - Var(q0_raw) )       [leading order]
  where Var(q0_raw)= sigma^2 + d^2 (the a=1/2 symmetric two-endmember variance),
        Var(p)     = sigma^2 + bw^2 + Var_w*(mu).
  => Var(p)-Var(q0) = bw^2 + Var_w*(mu) - d^2 = deficit.  So eta = d^2 deficit/(2 sigma^4).
"""
import numpy as np, math
from scipy.special import erfinv as _erfinv
SIGMA=25.0; START=200.0
TGRID=np.arange(-1500.0,1500.0+1e-9,1.0)   # centered, fine
TRAPZ=getattr(np,'trapezoid',None) or np.trapz
SQRT2=math.sqrt(2.0)
def N(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))
def peaks(rho): return SIGMA*2*SQRT2*float(_erfinv(rho))

print("Test identity  eta_mid = d^2 * deficit / (2 sigma^4)  (centered, fine grid)")
print("%-5s %-6s %-8s %-12s %-14s %-14s %-10s"%("c","bw","rho","d","eta_exact","eta_predict","ratio"))
for c in (1/3,0.25):
    for bw in (12.0,24.0):
        for rho in (0.005,0.01,0.02,0.04):
            d=peaks(rho); seff=math.sqrt(SIGMA**2+bw**2)
            w=np.array([c,1-2*c,c]); mus=np.array([-d,0.0,d])
            p=(w[:,None]*np.array([N(TGRID,m,seff) for m in mus])).sum(0)
            p/=TRAPZ(p,TGRID)
            h2=N(TGRID,0.0,SIGMA)
            q0=0.5*N(TGRID,-d,SIGMA)+0.5*N(TGRID,d,SIGMA)+1e-300
            I2=TRAPZ(p*h2/q0,TGRID)/TRAPZ(h2,TGRID)
            eta=1.0-I2
            Varw=2*c*d**2
            deficit=bw**2+Varw-d**2
            pred=d**2*deficit/(2*SIGMA**4)
            print("%-5.3f %-6.0f %-8.3f %-12.5f %-14.6e %-14.6e %-10.5f"%
                  (c,bw,rho,d,eta,pred,eta/pred if pred!=0 else float('nan')))
    print()

# Also verify the GENERAL (asymmetric) a-fixing first-order condition reduces to mean matching:
# I_1 = I_K  <=>  a -> (mu_K - mubar*)/(mu_K - mu_1) to leading order. Already shown a~0.4583 in n2
# for w*_middle=(0.25,0.4167,0.3333): mubar index = 0*0.25+1*0.4167+2*0.3333=1.0833,
# a_pred=(2-1.0833)/(2-0)=0.4583. matches n2 abar_pred. Confirm:
wm=np.array([0.25,0.4166666666666667,0.3333333333333333])
k=np.array([0,1,2.0]); mbar=(wm*k).sum()
print("asym mean-match a_pred=(2-mbar)/2 = %.6f  (n2 abar_pred=0.458333)"%((2-mbar)/2))
