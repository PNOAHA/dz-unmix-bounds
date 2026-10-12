import numpy as np, math, json, time, os

try:
    from scipy.special import erfinv as _erfinv
    def erfinv(x): return float(_erfinv(x))
except Exception:
    def erfinv(x):
        a=0.147; ln=math.log(1-x*x); t=2/(math.pi*a)+ln/2
        return math.copysign(math.sqrt(math.sqrt(t*t-ln/a)-t), x)

SIGMA=25.0; START=200.0; BW=12.0
TGRID=np.arange(50.0,2500.0+1e-9,5.0).astype(np.float64)
TRAPZ=getattr(np,'trapezoid',None) or np.trapz

def peaks(K,rho):
    dmu=SIGMA*2.0*math.sqrt(2.0)*erfinv(rho)
    return np.array([START+i*dmu for i in range(K)],dtype=np.float64), dmu

def H_of(K,rho):
    mus,_=peaks(K,rho)
    h=np.exp(-0.5*((TGRID[None,:]-mus[:,None])/SIGMA)**2)
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
    if kind=='endpoint': w=np.linspace(1.0,0.6,K)
    elif kind=='middle':
        w=np.linspace(1.0,0.6,K)
        if K>=3: w=np.roll(w,K//2)
    else: raise ValueError(kind)
    return w/w.sum()

# ------------------------------------------------------------------
# C2 specifics
# ------------------------------------------------------------------
OUT="C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c2_raw_vs_kde_fisher.json"

K=3
RHOS=[0.04,0.06,0.08,0.11,0.15]
N=100000
R=600

def population_density(K,rho,wstar):
    """Population mixture density g* on TGRID (area-normalized)."""
    H=H_of(K,rho)
    g=wstar@H
    return g/TRAPZ(g,TGRID)

def converged_population_a_full(K,rho,wstar):
    """Run FULL K-EM unmix on the population density p=g* (no sampling noise).
    Diagnostic only: with raw-sigma dictionary and a raw-sigma population this
    recovers wstar (no pinning, since pinning is a KDE-data phenomenon)."""
    H=H_of(K,rho)
    p=population_density(K,rho,wstar)
    What=unmix(p[None,:],H,n_iter=40000)[0]
    return What

def reduced_population_a(K,rho,wstar):
    """a_inf = KL projection of the POPULATION mixture g* onto the REDUCED
    2-endmember family q(a) = a*h_1 + (1-a)*h_K (extreme pair, raw sigma).
    This is the population fixed point of the pinned estimator. Solve by
    1-D multiplicative-update KL minimization restricted to {h_1,h_K}."""
    H=H_of(K,rho)
    Hred=np.vstack([H[0],H[K-1]])          # (2,T)
    p=population_density(K,rho,wstar)      # target = population mixture
    # KL(p || a h1 + (1-a) hK) minimization over the 2-simplex via MU updates
    W=np.array([[0.5,0.5]],dtype=np.float64); eps=1e-12; Hsum=Hred.sum(1)[None,:]
    for _ in range(200000):
        WH=W@Hred+eps; W=W*((p[None,:]/WH)@Hred.T)/(Hsum+eps)
    W=W/W.sum(1,keepdims=True)
    a_inf=float(W[0,0])                    # fraction on h_1
    return a_inf

def I_red_quad(K,rho,a_inf):
    """Per-grain reduced 1-param Fisher for mixing fraction a on the
    extreme endmember pair {h_1,h_K}, raw sigma:
       I_red = integral (h_1 - h_K)^2 / q0 dt,  q0 = a h_1 + (1-a) h_K."""
    H=H_of(K,rho)
    h1=H[0]; hK=H[K-1]
    q0=a_inf*h1+(1.0-a_inf)*hK
    eps=1e-300
    integrand=(h1-hK)**2/np.maximum(q0,eps)
    # mask where q0 underflows to avoid blowups; q0>0 wherever either gaussian nonzero
    return float(TRAPZ(integrand,TGRID))

def reduced_a_from_W(What,K):
    a=What[0]; aK=What[K-1]; d=a+aK
    return a/d if d>0 else 0.5

def run():
    t0=time.time()
    wstar=fixed_interior_w(K,'endpoint')
    print("wstar(endpoint K=3) =", wstar.tolist())
    results={"meta":{"K":K,"N":N,"R":R,"SIGMA":SIGMA,"BW":BW,"wstar":wstar.tolist()},"rows":[]}
    # load partial if exists
    if os.path.exists(OUT):
        try:
            with open(OUT,"r",encoding="utf-8") as f: prev=json.load(f)
            done={r["rho"] for r in prev.get("rows",[])}
            if prev.get("meta",{}).get("R")==R and prev.get("meta",{}).get("N")==N:
                results=prev
                print("resuming; already have rho:", sorted(done))
        except Exception as e:
            print("could not load partial:",e)
    done_rhos={r["rho"] for r in results["rows"]}

    for rho in RHOS:
        if rho in done_rhos:
            print(f"skip rho={rho} (done)"); continue
        tr=time.time()
        H=H_of(K,rho)
        mus,dmu=peaks(K,rho)
        # population a_inf : KL projection of g* onto reduced {h_1,h_K}
        a_inf=reduced_population_a(K,rho,wstar)
        What_pop=converged_population_a_full(K,rho,wstar)  # diagnostic
        # also mean-match estimate for reference
        mubar_star=float(wstar@mus)
        a_mm=(mus[K-1]-mubar_star)/(mus[K-1]-mus[0])
        # naive Fisher
        I_red=I_red_quad(K,rho,a_inf)
        sigma_naive=math.sqrt(2.0)/math.sqrt(N*I_red)

        # Monte Carlo: R replicates of KDE-data estimator
        rng=np.random.default_rng(12345+int(round(rho*1000)))
        a_hats=np.empty(R,dtype=np.float64)
        X_all=np.empty((R,TGRID.shape[0]),dtype=np.float64)
        # batch the unmix over all R at once for speed (vectorized W update)
        for r in range(R):
            ages=gen(wstar,N,mus,rng)
            X_all[r]=kde(ages)
        What_all=unmix(X_all,H,n_iter=40000)  # (R,K) on simplex
        for r in range(R):
            a_hats[r]=reduced_a_from_W(What_all[r],K)
        interior_mean=float(What_all[:,1:K-1].mean()) if K>2 else 0.0
        interior_max=float(What_all[:,1:K-1].max()) if K>2 else 0.0

        a_mean=float(a_hats.mean())
        var_direct=float(a_hats.var(ddof=1))
        sigma_a_direct=math.sqrt(var_direct)
        # sigma_err_measured = sqrt(2)*std(a_hat)  (extreme-EM diagonal d=(e1-eK)/sqrt2)
        sigma_meas=math.sqrt(2.0)*sigma_a_direct

        # delta-method: Var(a) ~ M Cov(X) M^T
        # M = d a_hat / d X  (reduced sensitivity). Estimate empirically by
        # linear regression of a_hat on (X - Xbar) across the R samples.
        Xbar=X_all.mean(0)
        dX=X_all-Xbar[None,:]            # (R, T)
        da=a_hats-a_mean                 # (R,)
        # Cov(X): empirical (T,T) is huge (490x490) - fine
        CovX=np.cov(X_all.T)             # (T,T)
        # Solve for M via least squares: da ~ dX @ M  -> M = pinv(dX) da
        # but dX is (R,T) with T> ? T=490 < R=600 so overdetermined in T-space is fine
        # Use ridge-stabilized normal equations for the sensitivity vector
        # M minimizes || dX @ M - da ||. M shape (T,)
        # Then delta Var(a) = M^T CovX M (should reproduce var_direct if linear)
        lam=1e-12*np.trace(dX.T@dX)/dX.shape[1]
        A=dX.T@dX + lam*np.eye(dX.shape[1])
        b=dX.T@da
        M=np.linalg.solve(A,b)           # (T,)
        var_delta=float(M@CovX@M)
        ratio_delta=var_delta/var_direct if var_direct>0 else float('nan')

        kappa=sigma_meas/sigma_naive

        row={
            "rho":rho,"dmu":dmu,
            "a_inf":a_inf,"a_mm":float(a_mm),"a_mc_mean":a_mean,
            "What_pop":What_pop.tolist(),
            "I_red":I_red,
            "sigma_naive":sigma_naive,
            "sigma_meas":sigma_meas,
            "kappa":kappa,
            "var_direct":var_direct,"var_delta":var_delta,"ratio_delta_over_direct":ratio_delta,
            "interior_mean":interior_mean,"interior_max":interior_max,
            "mus":mus.tolist(),
        }
        results["rows"].append(row)
        # incremental save
        with open(OUT,"w",encoding="utf-8") as f: json.dump(results,f,indent=2)
        print(f"rho={rho}: a_inf={a_inf:.4f} a_mm={a_mm:.4f} a_mc={a_mean:.4f} "
              f"I_red={I_red:.6e} sig_naive={sigma_naive:.4e} sig_meas={sigma_meas:.4e} "
              f"kappa={kappa:.4f} delta/direct={ratio_delta:.4f} [{time.time()-tr:.1f}s]")

    # ---- fits ----
    rows=sorted(results["rows"],key=lambda r:r["rho"])
    lr=np.log(np.array([r["rho"] for r in rows]))
    def fit_exp(yvals):
        ly=np.log(np.array(yvals)); A=np.vstack([lr,np.ones_like(lr)]).T
        sl,ic=np.linalg.lstsq(A,ly,rcond=None)[0]; return float(sl),float(ic)
    I_slope,I_ic=fit_exp([r["I_red"] for r in rows])
    sn_slope,_=fit_exp([r["sigma_naive"] for r in rows])
    sm_slope,_=fit_exp([r["sigma_meas"] for r in rows])
    kappas=[r["kappa"] for r in rows]
    kappa_mean=float(np.mean(kappas)); kappa_std=float(np.std(kappas,ddof=1))
    kappa_cv=kappa_std/kappa_mean
    results["fits"]={
        "I_red_exponent":I_slope,"I_red_intercept":I_ic,
        "sigma_naive_exponent":sn_slope,"sigma_meas_exponent":sm_slope,
        "kappa_mean":kappa_mean,"kappa_std":kappa_std,"kappa_cv":kappa_cv,
        "kappa_min":float(min(kappas)),"kappa_max":float(max(kappas)),
    }
    with open(OUT,"w",encoding="utf-8") as f: json.dump(results,f,indent=2)

    print("\n==== SUMMARY ====")
    print(f"I_red exponent (vs rho)      : {I_slope:+.4f}  (expect +2)")
    print(f"sigma_naive exponent         : {sn_slope:+.4f}  (expect -1)")
    print(f"sigma_meas  exponent         : {sm_slope:+.4f}  (expect -1)")
    print(f"kappa mean +- std            : {kappa_mean:.4f} +- {kappa_std:.4f}  (CV={kappa_cv:.3f})")
    print(f"kappa range                  : [{min(kappas):.4f}, {max(kappas):.4f}]")
    print(f"delta/direct ratios          : {[round(r['ratio_delta_over_direct'],3) for r in rows]}")
    print(f"total time {time.time()-t0:.1f}s")
    return results

if __name__=="__main__":
    run()
