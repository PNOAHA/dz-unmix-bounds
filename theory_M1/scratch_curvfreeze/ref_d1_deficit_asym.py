"""
Referee check #3: the asymmetric deficit identity and the variance bookkeeping the
author uses in the general criterion. ASCII-only, new file.

Author's general criterion:
  PINNED  <=>  (muK - mubar*)(mubar* - mu1) < bw^2 + Var_w*(mu).
and Var(q0) at the mean-matched 2-endmember a equals (muK-mubar)(mubar-mu1).

CHECK A: algebraic identity. For a 2-point distribution on {mu1, muK} with mean mubar,
   weight a on mu1: a = (muK-mubar)/(muK-mu1). Its variance is a(1-a)(muK-mu1)^2.
   Claim: a(1-a)(muK-mu1)^2 == (muK-mubar)(mubar-mu1).  Verify symbolically/numerically.

CHECK B: the leading constant in the ASYMMETRIC eta_mid closed form. The author isolated
   the constant 1/2 only in the SYMMETRIC case and admits the asymmetric leading constant
   "was not independently isolated". We isolate it here: for K=3 'middle' (asymmetric a),
   define the general prediction
       eta_mid_pred = (d^2 / (2 s^4)) * deficit,   deficit = Var(p) - Var(q0_meanmatched)
   with Var(q0_meanmatched) = (muK-mubar)(mubar-mu1) (a NOT 1/2). Compare eta_mid_pred to the
   exact GH eta_mid as rho->0 and Richardson-extrapolate the ratio. If it -> 1, the asymmetric
   closed form (with constant 1/2) is validated; if it -> something else, the author overclaimed.

   NOTE on a subtlety the author GLOSSED: in the asymmetric case the local curvature weight is
   evaluated at the data mean mubar, and the 'h_mid' atom sits at mu_2 != mubar. So the naive
   transcription of the symmetric formula is NOT obviously correct. This is the real test.
"""
import numpy as np, math, os, json
from scipy.special import erfinv as _erfinv
from numpy.polynomial.hermite_e import hermegauss

SIGMA=25.0; START=200.0; SQRT2=math.sqrt(2.0)
HERE=os.path.dirname(os.path.abspath(__file__))
NQ=220; _xq,_wq=hermegauss(NQ); _wq=_wq/math.sqrt(2*math.pi)
def dmu_of_rho(rho): return SIGMA*2*SQRT2*float(_erfinv(rho))
def gpdf(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))

# CHECK A
print("=== CHECK A: a(1-a)(muK-mu1)^2 == (muK-mubar)(mubar-mu1) ? ===")
rng=np.random.default_rng(0); maxerr=0.0
for _ in range(100000):
    mu1,muK=sorted(rng.uniform(-10,10,2))
    if muK-mu1<1e-3: continue
    mubar=rng.uniform(mu1,muK)
    a=(muK-mubar)/(muK-mu1)
    lhs=a*(1-a)*(muK-mu1)**2; rhs=(muK-mubar)*(mubar-mu1)
    maxerr=max(maxerr,abs(lhs-rhs))
print("max abs error over 1e5 random configs:", maxerr, "(==0 => identity holds)\n")

def wstar(K, kind):
    w=np.linspace(1.0,0.6,K)
    if kind=='middle' and K>=3: w=np.roll(w,K//2)
    return w/w.sum()

def I_vec_asym(a, mus, w, bw):
    seff=math.sqrt(SIGMA**2+bw**2)
    K=len(mus); out=np.zeros(K)
    h1=lambda t: gpdf(t,mus[0],SIGMA); hK=lambda t: gpdf(t,mus[-1],SIGMA)
    for m in range(K):
        nodes=mus[m]+seff*_xq
        q0=a*h1(nodes)+(1-a)*hK(nodes)
        for j in range(K):
            out[j]+=w[m]*np.sum(_wq*(gpdf(nodes,mus[j],SIGMA)/q0))
    return out

def solve_a(mus,w,bw):
    lo,hi=1e-7,1-1e-7
    f=lambda a: (I_vec_asym(a,mus,w,bw)[0]-I_vec_asym(a,mus,w,bw)[-1])
    flo=f(lo)
    for _ in range(120):
        mid=0.5*(lo+hi); fm=f(mid)
        if (flo>0)==(fm>0): lo,flo=mid,fm
        else: hi=mid
    a=0.5*(lo+hi); return a, I_vec_asym(a,mus,w,bw)

print("=== CHECK B: asymmetric K=3 'middle' eta_mid vs predicted (d^2/(2 s^4))*deficit ===")
print("deficit = Var(p) - (muK-mubar)(mubar-mu1);  ratio = eta_exact / eta_pred")
print("%-6s %-10s %-14s %-14s %-12s %-12s"%("bw","rho","eta_exact","eta_pred","ratio","a"))
w=wstar(3,'middle')
for bw in (6.0,12.0):
    ratios=[]; ds=[]
    for rho in (0.04,0.02,0.01,0.005,0.0025):
        d=dmu_of_rho(rho)
        mus=np.array([START,START+d,START+2*d])
        mubar=float((w*mus).sum())
        Varp=SIGMA**2+bw**2+float((w*mus**2).sum()-mubar**2)
        Varq0=(mus[-1]-mubar)*(mubar-mus[0])
        deficit=Varp-Varq0
        a,Iv=solve_a(mus,w,bw)
        eta_exact=1.0-Iv[1]
        eta_pred=(d**2/(2*SIGMA**4))*deficit
        r=eta_exact/eta_pred
        ratios.append(r); ds.append(d)
        if rho in (0.04,0.01,0.0025):
            print("%-6.0f %-10.4f %-14.6e %-14.6e %-12.8f %-12.6f"%(bw,rho,eta_exact,eta_pred,r,a))
    # Richardson on last two
    r1,r2=ratios[-2],ratios[-1]; d1,d2=ds[-2],ds[-1]
    A=(r1-r2)/(d1**2-d2**2); R0=r2-A*d2**2
    print("   -> bw=%.0f Richardson ratio R0 = %.8f  (==1 => asym closed form validated)\n"%(bw,R0))

# CHECK C: uniqueness of the LP maximizer (could a 3-support tie the extreme pair?)
# Among ALL 2-support solutions at fixed mean, is the extreme-pair value strictly larger?
print("=== CHECK C: strictness of LP optimum (extreme pair strictly beats any other pair) ===")
for K in (4,5):
    d=dmu_of_rho(0.05); mus=np.array([START+i*d for i in range(K)])
    w=wstar(K,'middle'); mubar=float((w*mus).sum())
    vals=[]
    for i in range(K):
        for j in range(i+1,K):
            wi=(mubar-mus[j])/(mus[i]-mus[j])
            if wi<0 or wi>1: continue
            val=wi*mus[i]**2+(1-wi)*mus[j]**2
            vals.append((val,(i,j)))
    vals.sort(reverse=True)
    gap=vals[0][0]-vals[1][0]
    print("  K=%d top pair %s val=%.4f ; 2nd %s val=%.4f ; gap=%.4f (>0 => strict)"%
          (K,vals[0][1],vals[0][0],vals[1][1],vals[1][0],gap))
