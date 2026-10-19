"""Population (infinite-grain) KL-NMF example: K=3 sources at 150/160/170 Ma (sigma=25, bw=12),
w=(0.25,0.5,0.25). Shows the population-level estimator bias for raw-sigma vs broadened dictionaries
(multiplicative updates and Nelder-Mead simplex check). Requires NumPy>=2.0 (np.trapezoid) and SciPy.
Usage: python analysis/pop_kl_example.py
"""
import numpy as np
from scipy.optimize import minimize
T=np.arange(50.0,2500.0+1e-9,5.0); trap=np.trapezoid
def g(mu,s):
    h=np.exp(-0.5*((T-mu)/s)**2); return h/trap(h,T)
mus=[150,160,170]; s=25.; bw=12.; se=np.hypot(s,bw); w=np.array([.25,.5,.25])
X=sum(wi*g(m,se) for wi,m in zip(w,mus))
def fit(Hs):
    H=np.array(Hs)
    # multiplicative updates, many iters (same rule as implementation)
    W=np.full(3,1/3); Hsum=H.sum(1)
    for _ in range(200000):
        WH=W@H+1e-300; W=W*((X/WH)@H.T)/Hsum
    W/=W.sum()
    # also direct simplex optimisation as check
    def f(z):
        ww=np.exp(z)/np.exp(z).sum(); m=ww@H
        return trap(X*np.log((X+1e-300)/(m+1e-300)),T)
    best=min((minimize(f,z0,method='Nelder-Mead',options=dict(xatol=1e-12,fatol=1e-16,maxiter=20000)) for z0 in [np.zeros(3),np.array([1.,-3,1]),np.array([-1.,2,-1])]),key=lambda r:r.fun)
    return W, f(np.log(W+1e-300)), np.exp(best.x)/np.exp(best.x).sum(), best.fun
for name,Hs in [('raw-sigma',[g(m,s) for m in mus]),('broadened',[g(m,se) for m in mus])]:
    W,fW,W2,f2=fit(Hs)
    print(name,'MU w=',np.round(W,6),'KL',fW,'| NM w=',np.round(W2,6),'KL',f2,'| L2 bias',np.linalg.norm(W-w),'MAE',np.abs(W-w).mean())
