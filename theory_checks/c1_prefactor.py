import numpy as np, math, json, os, time
try:
    from scipy.special import erfinv as _erfinv
    def erfinv(x): return float(_erfinv(x))
except Exception:
    def erfinv(x):
        a=0.147; ln=math.log(1-x*x); t=2/(math.pi*a)+ln/2
        return math.copysign(math.sqrt(math.sqrt(t*t-ln/a)-t), x)
try:
    from scipy.special import erf as _erf
    def erf(x): return float(_erf(x))
except Exception:
    def erf(x):
        # Abramowitz-Stegun 7.1.26
        t=1.0/(1.0+0.3275911*abs(x))
        y=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*math.exp(-x*x)
        return math.copysign(y,x)

SIGMA=25.0; START=200.0; BW=12.0
TGRID=np.arange(50.0,2500.0+1e-9,5.0); TRAPZ=getattr(np,'trapezoid',None) or np.trapz

def peaks(K,rho):
    dmu=SIGMA*2.0*math.sqrt(2.0)*erfinv(rho); return np.array([START+i*dmu for i in range(K)]), dmu
def H_of(K,rho):
    mus,_=peaks(K,rho); h=np.exp(-0.5*((TGRID[None,:]-mus[:,None])/SIGMA)**2)
    return h/TRAPZ(h,TGRID,axis=1)[:,None]
def gen(w,N,mus,rng):
    counts=rng.multinomial(N,w)
    return np.concatenate([np.clip(rng.normal(mus[i],SIGMA,c),TGRID[0],TGRID[-1]) for i,c in enumerate(counts)])
def kde(ages,bw=BW):
    d=np.zeros_like(TGRID)
    for s in range(0,len(ages),20000):
        u=(TGRID[:,None]-ages[None,s:s+20000])/bw; d+=np.exp(-0.5*u*u).sum(1)
    d/=(len(ages)*bw*math.sqrt(2*math.pi)); return d/TRAPZ(d,TGRID)
def kde_binned(ages,bw=BW,refine=10):
    # histogram-based KDE: faithful, ~20-35x faster than per-sample exact KDE.
    lo,hi=TGRID[0],TGRID[-1]
    step=(TGRID[1]-TGRID[0])/refine
    nb=int(round((hi-lo)/step))+1
    edges=np.linspace(lo-step/2,hi+step/2,nb+1)
    centers=(edges[:-1]+edges[1:])/2
    hist,_=np.histogram(ages,bins=edges)
    u=(TGRID[:,None]-centers[None,:])/bw
    d=np.exp(-0.5*u*u)@hist
    d/=(len(ages)*bw*math.sqrt(2*math.pi)); return d/TRAPZ(d,TGRID)
def unmix(X,H,n_iter=40000):
    R,K=X.shape[0],H.shape[0]; rng=np.random.default_rng(0)
    W=rng.uniform(0.1,1.0,size=(R,K)); eps=1e-10; Hsum=H.sum(1)[None,:]
    for _ in range(n_iter):
        WH=W@H+eps; W=W*((X/WH)@H.T)/(Hsum+eps)
    return W/W.sum(1,keepdims=True)
def fixed_interior_w(K,kind):
    if kind=='endpoint': w=np.linspace(1.0,0.6,K)        # argmax k*=index 0 (an ENDPOINT)
    elif kind=='middle':
        w=np.linspace(1.0,0.6,K)
        if K>=3: w=np.roll(w,K//2)
    else: raise ValueError(kind)
    return w/w.sum()

# ---- rho_c per K from D=(K-1-kbar)kbar - Vk with endpoint w* ----
# kbar = weighted-mean index (E[k]) under w*, Vk = weighted variance of index.
def index_deficit_D(K,w):
    idx=np.arange(K,dtype=float)
    kbar=float(np.sum(w*idx))
    Vk=float(np.sum(w*(idx-kbar)**2))
    D=(K-1-kbar)*kbar - Vk
    return D,kbar,Vk
def rho_c_of(K,w):
    D,kbar,Vk=index_deficit_D(K,w)
    if D<=0:
        return float('inf'),D,kbar,Vk  # no deficit -> formula degenerate; treat as no pinning bound
    val=BW/(2.0*math.sqrt(2.0)*SIGMA*math.sqrt(D))
    return erf(val),D,kbar,Vk

def summarize_sigma_err(Wh):
    # mirror project's _summarize: largest eigenvalue of K-dim weight covariance
    C=np.cov(Wh.T)  # KxK
    C=np.atleast_2d(C)
    ev=np.linalg.eigvalsh(C)
    lam=float(ev[-1])
    lam=max(lam,0.0)
    return math.sqrt(lam)

def run_cell(K,rho,N,R,n_iter,kind='endpoint',seed=None,use_binned=True,refine=10):
    if seed is None:
        seed=7000 + K*100 + int(rho*1000)
    rng=np.random.default_rng(seed)
    mus,dmu=peaks(K,rho)
    H=H_of(K,rho)
    w=fixed_interior_w(K,kind)
    Wh=np.empty((R,K))
    # generate R samples, KDE each, unmix in a batch
    Xs=np.empty((R,TGRID.shape[0]))
    kde_fn=(lambda a: kde_binned(a,refine=refine)) if use_binned else kde
    for r in range(R):
        ages=gen(w,N,mus,rng)
        Xs[r]=kde_fn(ages)
    Wh=unmix(Xs,H,n_iter=n_iter)
    # pinning check: interior weights are indices 1..K-2
    if K>=3:
        interior=Wh[:,1:K-1]
        max_interior=float(np.max(interior)) if interior.size>0 else 0.0
        mean_interior=float(np.mean(interior)) if interior.size>0 else 0.0
    else:
        max_interior=0.0; mean_interior=0.0
    sigma_err=summarize_sigma_err(Wh)
    a_hat=Wh[:,0]  # surviving extreme weight (endmember 0)
    std_a=float(np.std(a_hat,ddof=1))
    ratio=sigma_err/std_a if std_a>0 else float('nan')
    # extreme-pair KS distance
    rho_1K=erf((K-1)*dmu/(2.0*math.sqrt(2.0)*SIGMA))
    return dict(K=K,rho=rho,N=N,R=R,n_iter=n_iter,kind=kind,seed=seed,
                dmu=float(dmu),
                sigma_err=sigma_err,std_a_hat=std_a,ratio_sigma_over_std=ratio,
                max_interior=max_interior,mean_interior=mean_interior,
                rho_1K=float(rho_1K),
                mean_w0=float(np.mean(a_hat)),mean_wK=float(np.mean(Wh[:,-1])),
                w_star=w.tolist())

def linfit(x,y):
    # least squares y = a + b x ; return b (slope), a (intercept)
    x=np.asarray(x,float); y=np.asarray(y,float)
    A=np.vstack([np.ones_like(x),x]).T
    coef,res,rank,sv=np.linalg.lstsq(A,y,rcond=None)
    a,b=coef[0],coef[1]
    yhat=a+b*x
    ss_res=np.sum((y-yhat)**2); ss_tot=np.sum((y-np.mean(y))**2)
    r2=1-ss_res/ss_tot if ss_tot>0 else float('nan')
    return float(b),float(a),float(r2)

def main():
    out_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c1_prefactor_vs_K.json')
    Ks=[3,4,5,6]
    rhos=[0.04,0.06,0.08,0.11,0.15]
    N=100000; R=600; n_iter=40000
    kind='endpoint'

    results={'meta':dict(SIGMA=SIGMA,START=START,BW=BW,N=N,R=R,n_iter=n_iter,kind=kind,
                         Ks=Ks,rhos=rhos),
             'rho_c':{}, 'cells':[], 'fits':{}, 'N_scan':[]}

    # rho_c per K
    for K in Ks:
        w=fixed_interior_w(K,kind)
        rc,D,kbar,Vk=rho_c_of(K,w)
        results['rho_c'][str(K)]=dict(rho_c=rc,D=D,kbar=kbar,Vk=Vk)
        print(f"K={K}: rho_c={rc:.4f} D={D:.4f} kbar={kbar:.4f} Vk={Vk:.4f}",flush=True)

    def save():
        with open(out_path,'w') as f: json.dump(results,f,indent=2)

    save()

    # main rho sweep
    for K in Ks:
        rc=results['rho_c'][str(K)]['rho_c']
        for rho in rhos:
            if rho>=rc:
                print(f"SKIP K={K} rho={rho} (>= rho_c={rc:.4f})",flush=True)
                continue
            t0=time.time()
            cell=run_cell(K,rho,N,R,n_iter,kind=kind)
            cell['released']= cell['max_interior']>=1e-6
            cell['rho_c']=rc
            results['cells'].append(cell)
            print(f"K={K} rho={rho:.3f} sigma_err={cell['sigma_err']:.4e} "
                  f"std_a={cell['std_a_hat']:.4e} ratio={cell['ratio_sigma_over_std']:.3f} "
                  f"max_int={cell['max_interior']:.2e} rho_1K={cell['rho_1K']:.4f} "
                  f"[{time.time()-t0:.0f}s]",flush=True)
            save()

    # ---- fits per K over pinned+unsaturated cells ----
    for K in Ks:
        cells=[c for c in results['cells'] if c['K']==K and (not c['released']) and c['max_interior']<1e-6]
        if len(cells)<2:
            results['fits'][str(K)]=dict(note='insufficient pinned cells',n=len(cells))
            continue
        rho_arr=np.array([c['rho'] for c in cells])
        se_arr=np.array([c['sigma_err'] for c in cells])
        rho1K_arr=np.array([c['rho_1K'] for c in cells])
        # fit log(se)=log(A)-s*log(rho)  -> slope on log(rho) is -s; report slope directly
        slope,intercept,r2=linfit(np.log(rho_arr),np.log(se_arr))
        s_K=-slope  # exponent in se ~ rho^{slope}; "s_K" defined as -slope so se~A*rho^{-s_K}... use slope as the exponent
        A_K=math.exp(intercept)  # se = A_K * rho^{slope}
        cK=A_K*math.sqrt(N)
        # also fit vs rho_1K
        slope1K,intercept1K,r21K=linfit(np.log(rho1K_arr),np.log(se_arr))
        A_1K=math.exp(intercept1K); c_1K=A_1K*math.sqrt(N)
        results['fits'][str(K)]=dict(
            n=len(cells), rhos_used=rho_arr.tolist(),
            slope_logrho=slope, exponent_s_K=slope, r2=r2,
            A_K=A_K, cK=cK, cK_times_Km1=cK*(K-1),
            slope_logrho1K=slope1K, r2_1K=r21K, A_1K=A_1K, c_1K=c_1K)
        print(f"\nFIT K={K}: slope(log rho)={slope:.3f} (R2={r2:.3f}) A_K={A_K:.4e} "
              f"cK=A_K*sqrt(N)={cK:.4e} cK*(K-1)={cK*(K-1):.4e}",flush=True)
        print(f"        vs rho_1K: slope={slope1K:.3f} (R2={r21K:.3f}) c_1K={c_1K:.4e}",flush=True)
        save()

    # ---- N-scan at fixed rho=0.06 to confirm 1/sqrt(N) and isolate A_K ----
    rho_fix=0.06
    Ns=[20000,100000,400000]
    for K in Ks:
        rc=results['rho_c'][str(K)]['rho_c']
        if rho_fix>=rc:
            print(f"N-scan SKIP K={K} (rho_fix>=rho_c)",flush=True)
            continue
        for Nn in Ns:
            t0=time.time()
            seed=8000 + K*100 + Nn//1000
            cell=run_cell(K,rho_fix,Nn,R,n_iter,kind=kind,seed=seed)
            cell['released']=cell['max_interior']>=1e-6
            results['N_scan'].append(cell)
            print(f"N-scan K={K} N={Nn} sigma_err={cell['sigma_err']:.4e} "
                  f"max_int={cell['max_interior']:.2e} [{time.time()-t0:.0f}s]",flush=True)
            save()
        # fit log(se) vs log(N) -> slope expect -0.5
        cells=[c for c in results['N_scan'] if c['K']==K and (not c['released'])]
        if len(cells)>=2:
            Narr=np.array([c['N'] for c in cells]); sarr=np.array([c['sigma_err'] for c in cells])
            sl,ic,r2=linfit(np.log(Narr),np.log(sarr))
            # cK from N-scan: se*sqrt(N) at rho_fix gives A_K(rho_fix)*sqrt(N)... -> cK_Nscan = mean(se*sqrt(N))*rho_fix^{s}
            # use se = cK/(sqrt(N) rho^{1}) => cK = se*sqrt(N)*rho
            cK_est=float(np.mean(sarr*np.sqrt(Narr)*rho_fix))
            results['fits'].setdefault(str(K),{})
            results['fits'][str(K)]['N_scan_slope']=sl
            results['fits'][str(K)]['N_scan_r2']=r2
            results['fits'][str(K)]['cK_from_Nscan_rho006']=cK_est
            print(f"N-scan FIT K={K}: slope(log N)={sl:.3f} (R2={r2:.3f}) cK_est(rho=0.06)={cK_est:.4e}",flush=True)
            save()

    # ---- summary tables ----
    print("\n===== SUMMARY cK table =====",flush=True)
    cK_by_K={}
    for K in Ks:
        f=results['fits'].get(str(K),{})
        if 'cK' in f:
            cK_by_K[K]=f['cK']
            print(f"K={K}: cK={f['cK']:.4e}  cK*(K-1)={f['cK_times_Km1']:.4e}  c_1K={f.get('c_1K',float('nan')):.4e}",flush=True)
    if 3 in cK_by_K:
        base=cK_by_K[3]
        ratios={K:cK_by_K[K]/base for K in cK_by_K}
        results['summary']=dict(cK_by_K=cK_by_K,
                                cK_ratios_to_K3={str(k):v for k,v in ratios.items()},
                                cK_times_Km1={str(K):cK_by_K[K]*(K-1) for K in cK_by_K})
        print("cK ratios to K=3:",{K:round(v,3) for K,v in ratios.items()},flush=True)
        print("(K-1)*cK:",{K:round(cK_by_K[K]*(K-1),5) for K in cK_by_K},flush=True)
    save()
    print("\nDONE. Saved to",out_path,flush=True)

if __name__=='__main__':
    main()
