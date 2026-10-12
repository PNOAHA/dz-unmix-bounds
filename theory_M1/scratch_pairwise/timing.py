import numpy as np, math, time
import importlib.util, sys
spec=importlib.util.spec_from_file_location("c3","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c3_top1_split.py")
c3=importlib.util.module_from_spec(spec); sys.argv=['t']; spec.loader.exec_module(c3)

K=4; rho=0.08
mus,dmu=c3.peaks(K,rho); H=c3.H_of(K,rho)
w=c3.make_endpoint_w(K,0.10)
for N in [10000,40000,160000]:
    rng=np.random.default_rng(1)
    R=10
    t0=time.time()
    Xs=np.empty((R,len(c3.TGRID)))
    for r in range(R):
        ages=c3.gen(w,N,mus,rng); Xs[r]=c3.kde(ages)
    dt=time.time()-t0
    print(f'N={N:>7d} gen+kde per realization = {dt/R*1000:.1f} ms')

# unmix timing at full n_iter for R=400
R=400; n_iter=40000
Xs=np.tile(c3.kde(c3.gen(w,40000,mus,np.random.default_rng(2))),(R,1))
t0=time.time(); c3.unmix(Xs,H,n_iter=n_iter); print(f'unmix R={R} n_iter={n_iter}: {time.time()-t0:.1f} s')
