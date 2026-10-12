import time
import importlib.util
spec=importlib.util.spec_from_file_location("c1","C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c1_prefactor.py")
c1=importlib.util.module_from_spec(spec)
import numpy as np, math
spec.loader.exec_module(c1)

TGRID=c1.TGRID; BW=c1.BW; TRAPZ=c1.TRAPZ
# fine binning grid for histogram-based KDE: bin width = grid step/ refine
def kde_binned(ages, bw=BW, refine=5):
    lo,hi=TGRID[0],TGRID[-1]
    step=(TGRID[1]-TGRID[0])/refine
    nb=int(round((hi-lo)/step))+1
    edges=np.linspace(lo-step/2, hi+step/2, nb+1)
    centers=(edges[:-1]+edges[1:])/2
    hist,_=np.histogram(ages,bins=edges)
    # convolve hist (on centers) with Gaussian evaluated at TGRID
    u=(TGRID[:,None]-centers[None,:])/bw
    G=np.exp(-0.5*u*u)
    d=G@hist
    d/=(len(ages)*bw*math.sqrt(2*math.pi))
    return d/TRAPZ(d,TGRID)

rng=np.random.default_rng(3)
mus=c1.peaks(4,0.06)[0]; w=c1.fixed_interior_w(4,'endpoint')
ages=c1.gen(w,100000,mus,rng)

t0=time.time(); d_exact=c1.kde(ages); te=time.time()-t0
for refine in [4,6,10]:
    t1=time.time(); d_bin=kde_binned(ages,refine=refine); tb=time.time()-t1
    maxabs=np.max(np.abs(d_exact-d_bin))
    rel=np.max(np.abs(d_exact-d_bin))/np.max(d_exact)
    print(f"refine={refine}: maxabs={maxabs:.3e} relmax={rel:.3e} L1={np.sum(np.abs(d_exact-d_bin))*5:.3e} time_bin={tb:.3f} time_exact={te:.3f}")

# downstream effect: unmix on exact vs binned and compare w_hat
H=c1.H_of(4,0.06)
Xe=np.tile(d_exact,(20,1)); Xb=np.tile(kde_binned(ages,refine=6),(20,1))
We=c1.unmix(Xe,H,40000); Wb=c1.unmix(Xb,H,40000)
print("w_hat exact[0]:",np.round(We[0],6).tolist())
print("w_hat bin6 [0]:",np.round(Wb[0],6).tolist())
print("max |dw|:",np.max(np.abs(We-Wb)))
