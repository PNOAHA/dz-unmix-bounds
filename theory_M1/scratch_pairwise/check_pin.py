import time
import importlib.util
spec=importlib.util.spec_from_file_location("c1","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c1_prefactor.py")
c1=importlib.util.module_from_spec(spec)
import numpy as np, math
spec.loader.exec_module(c1)

# check pinning at N=1e5 for each K at rho=0.06 with modest R
for K in [3,4,5,6]:
    t0=time.time()
    cell=c1.run_cell(K,0.06,100000,40,40000,kind='endpoint')
    print(f"K={K} rho=0.06 N=1e5 R=40: sigma_err={cell['sigma_err']:.4e} "
          f"std_a={cell['std_a_hat']:.4e} ratio={cell['ratio_sigma_over_std']:.3f} "
          f"max_int={cell['max_interior']:.3e} mean_int={cell['mean_interior']:.3e} "
          f"mw0={cell['mean_w0']:.4f} mwK={cell['mean_wK']:.4f} rho_1K={cell['rho_1K']:.4f} "
          f"[{time.time()-t0:.0f}s]")
