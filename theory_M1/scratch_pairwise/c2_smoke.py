import numpy as np, math, time
import importlib.util, types
spec=importlib.util.spec_from_file_location("c2","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c2_raw_vs_kde_fisher.py")
src=open(spec.origin,"r",encoding="utf-8").read()
src=src.replace('if __name__=="__main__":\n    run()','')
mod=types.ModuleType("c2"); exec(compile(src,spec.origin,"exec"),mod.__dict__)

K=3
wstar=mod.fixed_interior_w(K,'endpoint')
print("wstar:",wstar.tolist())
t0=time.time()
for rho in [0.04,0.08,0.15]:
    H=mod.H_of(K,rho); mus,dmu=mod.peaks(K,rho)
    a_inf=mod.reduced_population_a(K,rho,wstar)
    Wfull=mod.converged_population_a_full(K,rho,wstar)
    mubar=float(wstar@mus); a_mm=(mus[K-1]-mubar)/(mus[K-1]-mus[0])
    I_red=mod.I_red_quad(K,rho,a_inf)
    sig_naive=math.sqrt(2.0)/math.sqrt(1e5*I_red)
    print(f"rho={rho}: dmu={dmu:.3f} Wfull={[round(x,5) for x in Wfull.tolist()]} "
          f"a_inf_red={a_inf:.4f} a_mm={a_mm:.4f} I_red={I_red:.5e} sig_naive={sig_naive:.4e}")
# quick MC with tiny R to confirm pinning + alignment of a_mc with a_inf_red
rho=0.08; H=mod.H_of(K,rho); mus,_=mod.peaks(K,rho)
rng=np.random.default_rng(7); R=30
X=np.empty((R,mod.TGRID.shape[0]))
for r in range(R):
    ages=mod.gen(wstar,100000,mus,rng); X[r]=mod.kde(ages)
W=mod.unmix(X,H,n_iter=8000)
a=[mod.reduced_a_from_W(W[r],K) for r in range(R)]
print(f"MC rho=0.08 (R=30,n_iter=8000): a_mc mean={np.mean(a):.4f} std={np.std(a,ddof=1):.5f} "
      f"interior W[1] mean={float(W[:,1].mean()):.3e} max={float(W[:,1].max()):.3e}")
print("smoke time",round(time.time()-t0,1),"s")
