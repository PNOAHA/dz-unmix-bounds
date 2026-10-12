# -*- coding: utf-8 -*-
# Focused confirmation: top1 endpoint flip occurs at predicted z; (K-1) collapse via rho_1K.
# Small grid, binned-KDE accelerator (validated reldiff 5e-4 by C1). Background-safe incremental JSON.
import numpy as np, math, json, time
_np=np; TRAPZ=getattr(_np,"trapezoid",None) or _np.trapz
from scipy.special import erfinv as _ei, erf as _erf
def erfinv(x): return float(_ei(x))
def erf(x): return float(_erf(x))
SIGMA=25.0; START=200.0; BW=12.0
TGRID=np.arange(50.0,2500.0+1e-9,5.0); T=len(TGRID)
def peaks(K,rho):
    dmu=SIGMA*2*math.sqrt(2)*erfinv(rho); return np.array([START+i*dmu for i in range(K)]),dmu
def H_of(K,rho):
    mus,_=peaks(K,rho); h=np.exp(-0.5*((TGRID[None,:]-mus[:,None])/SIGMA)**2)
    return h/TRAPZ(h,TGRID,axis=1)[:,None]
# binned KDE: bin ages to a fine grid then convolve (fast, validated)
FINE=np.arange(50.0,2500.0+1e-9,0.5)
def kde_batch(ages_list,bw=BW):
    out=np.empty((len(ages_list),T))
    for i,ages in enumerate(ages_list):
        hist,_=np.histogram(ages,bins=np.append(FINE,FINE[-1]+0.5))
        u=(TGRID[:,None]-FINE[None,:])/bw
        d=(np.exp(-0.5*u*u)@hist)/(len(ages)*bw*math.sqrt(2*math.pi))
        out[i]=d/TRAPZ(d,TGRID)
    return out
def gen_batch(w,N,mus,R,rng):
    res=[]
    for _ in range(R):
        c=rng.multinomial(N,w)
        ages=np.concatenate([np.clip(rng.normal(mus[i],SIGMA,c[i]),TGRID[0],TGRID[-1]) for i in range(len(w))])
        res.append(ages)
    return res
def unmix(X,H,n_iter):
    R,K=X.shape[0],H.shape[0]; rng=np.random.default_rng(0)
    W=rng.uniform(0.1,1.0,size=(R,K)); eps=1e-10; Hsum=H.sum(1)[None,:]
    for _ in range(n_iter):
        WH=W@H+eps; W=W*((X/WH)@H.T)/(Hsum+eps)
    return W/W.sum(1,keepdims=True)
def endpoint_w(K): w=np.linspace(1.0,0.6,K); return w/w.sum()

C=0.48  # measured universal constant
out=[]; t0=time.time()
# sweep N down until endpoint top1 leaves 1.0; K=3,4,5; rho=0.08,0.12
for K in [3,4,5]:
    w=endpoint_w(K); 
    for rho in [0.08,0.12]:
        mus,dmu=peaks(K,rho); H=H_of(K,rho); rho1K=erf((K-1)*dmu/(2*math.sqrt(2)*SIGMA))
        # a_inf margin via reduced projection (approx mean-matching for endpoint)
        for N in [300,600,1200,2500,5000,10000]:
            rng=np.random.default_rng(4242+K*97+int(rho*1000)+N)
            R=400
            ages=gen_batch(w,N,mus,R,rng); X=kde_batch(ages)
            W=unmix(X,H,n_iter=15000)
            kstar=int(np.argmax(w))
            top1=float((W.argmax(1)==kstar).mean())
            a=W[:,0]; Dw=a.mean()-0.5; stda=a.std()
            z=(math.sqrt(2)/C)*math.sqrt(N)*rho1K*Dw   # predicted z
            top1_pred=0.5*(1+math.erf(z/math.sqrt(2)))
            rec=dict(K=K,rho=rho,N=N,rho1K=rho1K,top1=top1,a_mean=float(a.mean()),Dw=float(Dw),
                     std_a=float(stda),z_pred=z,top1_pred=top1_pred,
                     u=math.sqrt(N)*(K-1)*rho*Dw, u_extreme=math.sqrt(N)*rho1K*Dw)
            out.append(rec)
            print(f"K={K} rho={rho} N={N:>6} top1={top1:.3f} pred={top1_pred:.3f} Dw={Dw:.3f} z={z:.2f} u_ext={rec['u_extreme']:.3f}",flush=True)
            json.dump(out,open("top1_confirm.json","w"),indent=1)
print(f"done {time.time()-t0:.0f}s")
