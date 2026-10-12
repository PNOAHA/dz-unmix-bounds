import numpy as np, time, os
import importlib.util, sys
spec=importlib.util.spec_from_file_location("c3","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c3_top1_split.py")
c3=importlib.util.module_from_spec(spec); sys.argv=['v']; spec.loader.exec_module(c3)

if __name__=="__main__":
    import multiprocessing as mp
    K=4; rho=0.08; N=20000; R=60
    mus,dmu=c3.peaks(K,rho); H=c3.H_of(K,rho)
    w=c3.fixed_interior_w(K,'endpoint'); kstar=int(np.argmax(w))
    base_seed=12345

    # serial reference using same per-realization seeds
    t0=time.time()
    rows_ser=[]
    for r in range(R):
        rng=np.random.default_rng(base_seed+r)
        ages=c3.gen(w,N,mus,rng); rows_ser.append(c3.kde(ages))
    Xs_ser=np.array(rows_ser); ts=time.time()-t0

    # parallel
    pool=mp.Pool(processes=min(24,os.cpu_count()))
    t0=time.time()
    Xs_par=c3.build_X(pool,w,N,mus,base_seed,R)
    tp=time.time()-t0
    pool.close(); pool.join()

    print('max abs diff serial vs parallel X:', np.max(np.abs(Xs_ser-Xs_par)))
    Wh_s=c3.unmix(Xs_ser,H,n_iter=2000); Wh_p=c3.unmix(Xs_par,H,n_iter=2000)
    print('top1 serial',float(np.mean(np.argmax(Wh_s,1)==kstar)),'parallel',float(np.mean(np.argmax(Wh_p,1)==kstar)))
    print('serial gen+kde %.2fs  parallel %.2fs  speedup %.1fx'%(ts,tp,ts/tp))
