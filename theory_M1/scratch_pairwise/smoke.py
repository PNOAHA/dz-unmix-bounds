import numpy as np, math, time
import importlib.util
spec=importlib.util.spec_from_file_location("c3","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c3_top1_split.py")
c3=importlib.util.module_from_spec(spec)
# avoid running __main__
import sys
sys.argv=['smoke']
# load functions only by executing module body (it won't run __main__ because name != __main__)
spec.loader.exec_module(c3)

# quick timing of one unmix cell at small R/n_iter
K=4; N=20000; R=20; n_iter=5000
for kind in ['endpoint','middle']:
    w=c3.fixed_interior_w(K,kind); print(kind,'wstar',np.round(w,4),'kstar',int(np.argmax(w)))
print('rho_c_est K=4',c3.rho_c_est(4),'K=3',c3.rho_c_est(3),'K=5',c3.rho_c_est(5))

rho=0.08
mus,dmu=c3.peaks(K,rho); H=c3.H_of(K,rho)
w=c3.fixed_interior_w(K,'endpoint'); kstar=int(np.argmax(w))
rng=np.random.default_rng(1)
t0=time.time()
Xs=np.empty((R,len(c3.TGRID)))
for r in range(R):
    ages=c3.gen(w,N,mus,rng); Xs[r]=c3.kde(ages)
t1=time.time()
Wh=c3.unmix(Xs,H,n_iter=n_iter)
t2=time.time()
print('endpoint rho=0.08 meanw',np.round(Wh.mean(0),3),'top1',float(np.mean(np.argmax(Wh,1)==kstar)))
print('gen+kde sec/cell(R=%d)'%R,round(t1-t0,2),'unmix sec(n_iter=%d,R=%d)'%(n_iter,R),round(t2-t1,2))

# interior
w=c3.fixed_interior_w(K,'middle'); kstar=int(np.argmax(w))
rng=np.random.default_rng(1)
Xs=np.empty((R,len(c3.TGRID)))
for r in range(R):
    ages=c3.gen(w,N,mus,rng); Xs[r]=c3.kde(ages)
Wh=c3.unmix(Xs,H,n_iter=n_iter)
print('middle rho=0.08 meanw',np.round(Wh.mean(0),3),'top1',float(np.mean(np.argmax(Wh,1)==kstar)),'kstar',kstar,'w_kstar',round(float(Wh.mean(0)[kstar]),4))
