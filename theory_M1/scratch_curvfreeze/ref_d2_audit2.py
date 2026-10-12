# Deeper adversarial probes of D2 Layer A rigor.
# ASCII-only.
import numpy as np
from math import sqrt, pi, erf
from scipy.special import erfinv
from scipy.optimize import minimize, linprog

SIGMA=25.0; START=200.0
def dmu_of_rho(r): return SIGMA*2*sqrt(2)*erfinv(r)
def rho_of_dmu(d): return erf(d/(2*sqrt(2)*SIGMA))
def mus_of(K,rho): return START+dmu_of_rho(rho)*np.arange(K)
def gauss(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*sqrt(2*pi))
def wstar_runner(K,kind='middle'):
    w=np.linspace(1.0,0.6,K)
    if kind=='middle' and K>=3: w=np.roll(w,K//2)
    return w/w.sum()

# PROBE A: is var(p) > sup_w var(q_w) GLOBALLY (over all means), or only locally near mubar*?
# sup_w var(q_w) over the WHOLE simplex (any mean) = sigma^2 + max mixing variance
#   = sigma^2 + (mu_K - m)(m - mu_1) maximized over m -> at m=midpoint = sigma^2 + ((mu_K-mu_1)/2)^2
# The relevant comparison for "p outside achievable SET" is at p's OWN mean (Env(mubar*)),
# since the achievable set is {(m, v): v <= Env(m)} and p=(mubar*, var_p).
print("="*74)
print("PROBE A: data point (mubar*, var_p) vs achievable region {v<=Env(m)}")
print("   pinned <=> var_p > Env(mubar*).  Also report global max Env and where p sits")
print("="*74)
for (K,bw,rho) in [(3,12.0,0.05),(3,12.0,0.20),(3,12.0,0.35),(3,12.0,0.45)]:
    mus=mus_of(K,rho); w=wstar_runner(K)
    mubar=float(w@mus); var_p=(SIGMA**2+bw**2)+float(w@(mus-mubar)**2)
    env_at_mean=SIGMA**2+(mus[-1]-mubar)*(mubar-mus[0])
    env_global=SIGMA**2+((mus[-1]-mus[0])/2)**2
    print(f"  rho={rho:.2f}: var_p={var_p:9.3f} Env(mubar*)={env_at_mean:9.3f} "
          f"deficit(var_p-Env)={var_p-env_at_mean:+9.3f}  Env_global={env_global:9.3f}")

# PROBE B: could the KL projection land on a DIFFERENT 2-atom edge than {1,K}?
# Enumerate all 2-atom faces {i,j}; for each, find best KL on that face; compare to full-simplex min.
print()
print("="*74)
print("PROBE B: KL on every 2-atom face {i,j}; which face wins? (K=4, pinned)")
print("="*74)
def klmin_on_support(p, Hraw, dt, support):
    Hs=Hraw[support]
    def unpack(z):
        e=np.exp(z-z.max()); return e/e.sum()
    def obj(z):
        w=unpack(z); q=w@Hs; q/=(q.sum()*dt)
        m=p>1e-300
        return float(np.sum(p[m]*np.log(p[m]/np.clip(q[m],1e-300,None)))*dt)
    best=None
    for tr in range(6):
        n=len(support); z0=np.random.RandomState(tr).randn(n)*0.7
        r=minimize(obj,z0,method="L-BFGS-B",options={"maxiter":6000,"ftol":1e-16})
        if best is None or r.fun<best.fun: best=r
    return best.fun, unpack(best.x)
K=4; bw=12.0; rho=0.08
mus=mus_of(K,rho); ws=wstar_runner(K); seff=sqrt(SIGMA**2+bw**2)
lo=mus[0]-9*seff; hi=mus[-1]+9*seff; t=np.linspace(lo,hi,8001); dt=t[1]-t[0]
Heff=np.stack([gauss(t,mu,seff) for mu in mus]); Heff/=(Heff.sum(1,keepdims=True)*dt)
Hraw=np.stack([gauss(t,mu,SIGMA) for mu in mus]); Hraw/=(Hraw.sum(1,keepdims=True)*dt)
p=ws@Heff; p/=(p.sum()*dt)
import itertools
faces=list(itertools.combinations(range(K),2))
results=[]
for f in faces:
    kv,w=klmin_on_support(p,Hraw,dt,list(f))
    results.append((kv,f,w))
results.sort()
for kv,f,w in results:
    print(f"  face {f}: KLmin={kv:.6e}  w={np.round(w,4).tolist()}")
print(f"  --> winning face = {results[0][1]} (D2 claims it must be {{0,{K-1}}})")
# also full-simplex
kv_full,w_full=klmin_on_support(p,Hraw,dt,list(range(K)))
print(f"  full-simplex KLmin={kv_full:.6e} w={np.round(w_full,5).tolist()}")

# PROBE C: over/under-prediction sign of rho_c vs rho*_obs across all 9 + endpoint
print()
print("="*74)
print("PROBE C: sign of (rho_c - rho*_obs): does D2's 'systematically OVER-predicts' hold?")
print("="*74)
def Dcoef(w,K):
    idx=np.arange(K,dtype=float); kbar=float((w*idx).sum()); Vk=float((w*idx*idx).sum()-kbar*kbar)
    return (K-1-kbar)*kbar-Vk
N2=[((3,6.0,'middle'),0.145625),((3,12.0,'middle'),0.26875),((3,24.0,'middle'),0.4203125),
    ((4,6.0,'middle'),0.093125),((4,12.0,'middle'),0.179375),((4,24.0,'middle'),0.28875),
    ((5,6.0,'middle'),0.066875),((5,12.0,'middle'),0.123125),((5,24.0,'middle'),0.20875),
    ((3,12.0,'endpoint'),0.29125)]
nover=nunder=0
for (K,bw,kind),robs in N2:
    w=wstar_runner(K,kind); D=Dcoef(w,K)
    rc=rho_of_dmu(bw/sqrt(D))
    sign='OVER' if rc>robs else 'under'
    if rc>robs: nover+=1
    else: nunder+=1
    print(f"  K={K} bw={bw:>5.1f} {kind:>8}: rho_c={rc:.5f} obs={robs:.5f} -> {sign} (relerr={abs(rc-robs)/robs:.4f})")
print(f"  OVER-predict: {nover}/{len(N2)}   under-predict: {nunder}/{len(N2)}")

# PROBE D: tautology check - is rho_c(D2) literally the same formula as N2 rho_pred?
print()
print("="*74)
print("PROBE D: is D2's rho_c the SAME closed form as N2's rho_pred (=> not independent)?")
print("="*74)
print("  N2 moment_pred: C_runner=(K-1-mbar)mbar - Vk ; dmu_thr=bw/sqrt(C_runner); rho_pred=erf(...)")
print("  D2 rho_c:       D=(K-1-kbar)kbar - Vk        ; dmu_c=bw/sqrt(D);          rho_c=erf(...)")
print("  C_runner == D identically => rho_pred == rho_c by construction (tautology).")
print("  The only NON-tautological N2 comparison is vs rho_star_obs (the actual KL release).")
