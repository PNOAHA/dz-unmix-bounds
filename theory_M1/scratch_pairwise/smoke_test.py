import time
import importlib.util
spec=importlib.util.spec_from_file_location("c1","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c1_prefactor.py")
c1=importlib.util.module_from_spec(spec)
import numpy as np, math
spec.loader.exec_module(c1)

for K in [3,4,5,6]:
    w=c1.fixed_interior_w(K,'endpoint')
    rc,D,kbar,Vk=c1.rho_c_of(K,w)
    print(f"K={K} w*={np.round(w,4).tolist()} rho_c={rc:.4f} D={D:.4f} kbar={kbar:.4f} Vk={Vk:.4f}")

t0=time.time()
cell=c1.run_cell(3,0.06,5000,30,2000,kind='endpoint')
print("smoke cell K=3 rho=0.06:",{k:round(cell[k],6) for k in ['sigma_err','std_a_hat','ratio_sigma_over_std','max_interior','rho_1K','mean_w0','mean_wK']})
print("time",round(time.time()-t0,2))

H=c1.H_of(3,0.06)
ages=c1.gen(c1.fixed_interior_w(3,'endpoint'),100000,c1.peaks(3,0.06)[0],np.random.default_rng(1))
X=np.tile(c1.kde(ages),(50,1))
t1=time.time(); Wh=c1.unmix(X,H,n_iter=40000); print("unmix R=50 n_iter=40000 time",round(time.time()-t1,2))
print("kde single time test...")
t2=time.time()
for _ in range(10):
    c1.kde(c1.gen(c1.fixed_interior_w(3,'endpoint'),100000,c1.peaks(3,0.06)[0],np.random.default_rng(2)))
print("10x (gen+kde) N=1e5 time",round(time.time()-t2,2))
