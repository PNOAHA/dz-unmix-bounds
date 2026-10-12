import time
import importlib.util
spec=importlib.util.spec_from_file_location("c1","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c1_prefactor.py")
c1=importlib.util.module_from_spec(spec)
import numpy as np, math
spec.loader.exec_module(c1)

for K in [3,6]:
    for rho in [0.06,0.11]:
        rc=c1.rho_c_of(K,c1.fixed_interior_w(K,'endpoint'))[0]
        if rho>=rc:
            print(f"K={K} rho={rho} skip (>=rho_c={rc:.3f})"); continue
        t0=time.time(); ce=c1.run_cell(K,rho,100000,80,40000,seed=12345,use_binned=False); te=time.time()-t0
        t1=time.time(); cb=c1.run_cell(K,rho,100000,80,40000,seed=12345,use_binned=True,refine=10); tb=time.time()-t1
        print(f"K={K} rho={rho}: EXACT se={ce['sigma_err']:.5e} std_a={ce['std_a_hat']:.5e} ({te:.0f}s) | "
              f"BIN se={cb['sigma_err']:.5e} std_a={cb['std_a_hat']:.5e} ({tb:.0f}s) | "
              f"rel_se_diff={abs(ce['sigma_err']-cb['sigma_err'])/ce['sigma_err']:.3e} "
              f"max_int_bin={cb['max_interior']:.2e}")
