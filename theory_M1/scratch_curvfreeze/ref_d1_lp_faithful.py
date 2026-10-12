"""
Referee check #5: is the fixed-mean KL really minimized at extreme support (LP faithfulness),
and is 'KL decreasing in model variance toward data var' actually monotone (not just at endpoints)?
Also sanity-check dmu ~ sigma*sqrt(2pi)*rho small-rho relation. ASCII-only, new file.
"""
import numpy as np, math, os
from scipy.special import erfinv as _erfinv
SIGMA=25.0; START=200.0; SQRT2=math.sqrt(2.0)
TRAPZ=getattr(np,'trapezoid',None) or np.trapz
def dmu_of_rho(rho): return SIGMA*2*SQRT2*float(_erfinv(rho))
def gpdf(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))
def wstar(K,kind):
    w=np.linspace(1.0,0.6,K)
    if kind=='middle' and K>=3: w=np.roll(w,K//2)
    return w/w.sum()

# small-rho relation: dmu = sigma*2*sqrt2*erfinv(rho) ~ sigma*sqrt(2pi)*rho ?
print("=== dmu small-rho relation: ratio dmu/(sigma*sqrt(2pi)*rho) -> 1 ? ===")
for rho in (0.2,0.1,0.05,0.01,0.001):
    d=dmu_of_rho(rho); approx=SIGMA*math.sqrt(2*math.pi)*rho
    print("  rho=%.4f dmu=%.5f approx=%.5f ratio=%.6f"%(rho,d,approx,d/approx))
print()

# LP faithfulness: scan the fixed-mean family, report KL vs model variance, monotonicity.
print("=== fixed-mean KL vs model variance (K=3 middle, bw=12), is KL min at extreme + monotone? ===")
K=3; bw=12.0; rho=0.05
w=wstar(3,'middle'); d=dmu_of_rho(rho)
mus=np.array([START+i*d for i in range(K)]); mubar=float((w*mus).sum())
seff=math.sqrt(SIGMA**2+bw**2)
lo=START-6*seff; hi=mus[-1]+6*seff; t=np.linspace(lo,hi,8000)
Hall=np.array([gpdf(t,m,SIGMA) for m in mus]); Hall/=TRAPZ(Hall,t,axis=1)[:,None]
P=np.array([gpdf(t,m,seff) for m in mus]); p=(w[:,None]*P).sum(0); p/=TRAPZ(p,t)
m1,m2,m3=mus
def KL(qw):
    q=qw+1e-300; return float(TRAPZ(p*np.log((p+1e-300)/q),t))
rows=[]
for w2 in np.linspace(0.0,0.95,96):
    A=np.array([[1.0,1.0],[m1,m3]]); b=np.array([1-w2, mubar-w2*m2])
    w1,w3=np.linalg.solve(A,b)
    if w1<-1e-9 or w3<-1e-9: continue
    wv=np.array([max(w1,0),w2,max(w3,0)])
    qw=(wv[:,None]*Hall).sum(0)
    modvar=float((wv*mus**2).sum()-mubar**2)
    rows.append((w2,modvar,KL(qw)))
rows=np.array(rows)
# KL min location and monotonicity of KL vs modvar
imin=int(np.argmin(rows[:,2]))
# sort by modvar; check KL decreasing as modvar increases
order=np.argsort(rows[:,1]); kl_sorted=rows[order,2]
mono = np.all(np.diff(kl_sorted)<=1e-9)
print("  KL-min at w2=%.4f (modvar=%.3f, KL=%.6e); extreme support? %s"%
      (rows[imin,0],rows[imin,1],rows[imin,2], rows[imin,0]<1e-3))
print("  KL strictly decreasing as model-variance increases over whole family? %s"%mono)
print("  data var=%.3f ; max model var (w2=0)=%.3f (deficit=%.3f)"%
      (SIGMA**2+bw**2+float((w*mus**2).sum()-mubar**2), rows[order][-1,1],
       (SIGMA**2+bw**2+float((w*mus**2).sum()-mubar**2))-rows[order][-1,1]))
