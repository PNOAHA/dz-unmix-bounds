import numpy as np, math, json, time, os, sys
from math import erf as _erf

try:
    from scipy.special import erfinv as _erfinv
    def erfinv(x): return float(_erfinv(x))
except Exception:
    def erfinv(x):
        a=0.147; ln=math.log(1-x*x); t=2/(math.pi*a)+ln/2
        return math.copysign(math.sqrt(math.sqrt(t*t-ln/a)-t), x)

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

def unmix(X,H,n_iter=40000):
    R,K=X.shape[0],H.shape[0]; rng=np.random.default_rng(0)
    W=rng.uniform(0.1,1.0,size=(R,K)); eps=1e-10; Hsum=H.sum(1)[None,:]
    for _ in range(n_iter):
        WH=W@H+eps; W=W*((X/WH)@H.T)/(Hsum+eps)
    return W/W.sum(1,keepdims=True)

def fixed_interior_w(K,kind):
    if kind=='endpoint': w=np.linspace(1.0,0.6,K)        # argmax k*=index 0 (an ENDPOINT) - survives pinning
    elif kind=='middle':
        w=np.linspace(1.0,0.6,K)
        if K>=3: w=np.roll(w,K//2)                       # argmax k* INTERIOR - killed by pinning
    else: raise ValueError(kind)
    return w/w.sum()

def make_endpoint_w(K, Delta_w, interior_mass=0.02):
    extreme_total = 1.0 - interior_mass
    w0 = (extreme_total + Delta_w)/2.0
    wK = (extreme_total - Delta_w)/2.0
    w=np.full(K, (interior_mass/max(K-2,1)) if K>2 else 0.0)
    w[0]=w0; w[K-1]=wK
    return w/w.sum()

def rho_c_est(K):
    kbar=(K-1)/2.0
    Vk=(K*K-1)/12.0
    D=(K-1-kbar)*kbar - Vk
    if D<=0: return None
    return _erf(BW/(2.0*math.sqrt(2.0)*SIGMA*math.sqrt(D)))

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c3_top1_split.json')

# -------- parallel KDE worker --------
# Each realization is fully determined by (w, N, mus, seed). Worker returns the KDE row.
def _worker(args):
    w, N, mus, seed = args
    rng=np.random.default_rng(seed)
    ages=gen(w,N,mus,rng)
    return kde(ages)

def build_X(pool, w, N, mus, base_seed, R):
    seeds=[int(base_seed)+r for r in range(R)]
    args=[(w,N,mus,s) for s in seeds]
    rows=pool.map(_worker, args, chunksize=max(1,R//(os.cpu_count()*2)))
    return np.array(rows)

def save(results):
    for attempt in range(3):
        try:
            with open(OUT,'w') as f: json.dump(results,f,indent=2)
            return
        except Exception as e:
            time.sleep(1.0)
            if attempt==2: print("SAVE FAILED:",e, flush=True)

def main():
    import multiprocessing as mp
    results={'meta':{'SIGMA':SIGMA,'START':START,'BW':BW,
                     'note':'C3 top1 endpoint vs interior split + endpoint pairwise threshold',
                     'rho_c_est':{str(K):rho_c_est(K) for K in [2,3,4,5]}},
             'part1':[], 'part2':[]}
    t0=time.time()
    pool=mp.Pool(processes=min(24, os.cpu_count()))
    try:
        # ============ PART 1 ============
        K=4; N=20000; R=400; n_iter=40000
        rhos=[0.05,0.08,0.11,0.14,0.18,0.22,0.28,0.35]
        rc=rho_c_est(K)
        print("=== PART 1 (K=4) === rho_c_est=%.4f"%rc, flush=True)
        for kind in ['endpoint','middle']:
            wstar=fixed_interior_w(K,kind); kstar=int(np.argmax(wstar))
            print(f"  config={kind} wstar={np.round(wstar,4)} kstar={kstar}", flush=True)
            for rho in rhos:
                mus,dmu=peaks(K,rho); H=H_of(K,rho)
                base_seed=10_000_000 + (1 if kind=='middle' else 0)*1_000_000 + int(rho*1000)*1000
                Xs=build_X(pool,wstar,N,mus,base_seed,R)
                Wh=unmix(Xs,H,n_iter=n_iter)
                top1=float(np.mean(np.argmax(Wh,axis=1)==kstar))
                meanw=Wh.mean(0); wk=float(meanw[kstar])
                rec={'config':kind,'K':K,'rho':rho,'kstar':kstar,'rho_c_est':rc,
                     'top1':top1,'mean_w':[float(x) for x in meanw],
                     'w_kstar_mean':wk,'pinned':bool(rho<rc)}
                results['part1'].append(rec)
                print(f"    rho={rho:.2f} top1={top1:.3f} w_kstar={wk:.4f} meanw={np.round(meanw,3)} t={time.time()-t0:.0f}s", flush=True)
                save(results)

        # ============ PART 2 ============
        Ks=[3,4,5]; Deltas=[0.02,0.05,0.10,0.20]; Ns=[10000,40000,160000]
        rhos2=[0.05,0.08,0.12]; R=400; n_iter=40000
        print("=== PART 2 endpoint pairwise threshold ===", flush=True)
        for K in Ks:
            rc=rho_c_est(K)
            for rho in rhos2:
                mus,dmu=peaks(K,rho); H=H_of(K,rho)
                for Delta in Deltas:
                    wstar=make_endpoint_w(K,Delta); kstar=int(np.argmax(wstar))
                    for N in Ns:
                        base_seed=20_000_000 + K*1_000_000 + int(rho*1000)*10_000 + int(Delta*1000)*100 + (Ns.index(N)+1)
                        Xs=build_X(pool,wstar,N,mus,base_seed,R)
                        Wh=unmix(Xs,H,n_iter=n_iter)
                        top1=float(np.mean(np.argmax(Wh,axis=1)==kstar))
                        s_indep=math.sqrt(N)*rho*Delta
                        s_Km1=math.sqrt(N)*(K-1)*rho*Delta
                        rec={'K':K,'rho':rho,'Delta_w':Delta,'N':N,'kstar':kstar,
                             'top1':top1,'sqrtN_rho_dw':s_indep,'sqrtN_Km1_rho_dw':s_Km1,'rho_c_est':rc}
                        results['part2'].append(rec)
                        print(f"  K={K} rho={rho:.2f} dw={Delta:.2f} N={N:>6d} top1={top1:.3f} "
                              f"sN*r*dw={s_indep:.3f} sN*(K-1)*r*dw={s_Km1:.3f} t={time.time()-t0:.0f}s", flush=True)
                        save(results)
    finally:
        pool.close(); pool.join()
    save(results)
    print("DONE total_sec=%.1f"%(time.time()-t0), flush=True)
    return results

def selftest():
    import multiprocessing as mp
    K=4; rho=0.08; N=20000; R=60
    mus,dmu=peaks(K,rho); H=H_of(K,rho)
    w=fixed_interior_w(K,'endpoint'); kstar=int(np.argmax(w)); base_seed=12345
    t0=time.time(); rows=[]
    for r in range(R):
        rng=np.random.default_rng(base_seed+r); rows.append(kde(gen(w,N,mus,rng)))
    Xs_ser=np.array(rows); ts=time.time()-t0
    pool=mp.Pool(processes=min(24,os.cpu_count()))
    t0=time.time(); Xs_par=build_X(pool,w,N,mus,base_seed,R); tp=time.time()-t0
    pool.close(); pool.join()
    print('max abs diff serial vs parallel:', float(np.max(np.abs(Xs_ser-Xs_par))), flush=True)
    print('serial %.2fs parallel %.2fs speedup %.1fx'%(ts,tp,ts/max(tp,1e-9)), flush=True)

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='selftest':
        selftest()
    else:
        main()
