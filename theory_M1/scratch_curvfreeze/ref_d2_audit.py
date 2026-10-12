# Adversarial audit of D2-moment-geom. Independent recompute.
# ASCII-only. New scratch file only.
import numpy as np
import json, os
from math import erf, sqrt, pi
from scipy.special import erfinv
from scipy.optimize import linprog, minimize, brentq

SIGMA = 25.0
START = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

def dmu_of_rho(rho):
    return SIGMA*2.0*sqrt(2.0)*erfinv(rho)

def rho_of_dmu(d):
    return erf(d/(2.0*sqrt(2.0)*SIGMA))

def mus_of(K, rho):
    return START + dmu_of_rho(rho)*np.arange(K)

# ---- runner's EXACT w_star (linspace then roll), to avoid any reconstruction error ----
def wstar_runner(K, kind='middle'):
    w = np.linspace(1.0, 0.6, K)
    if kind == 'middle' and K >= 3:
        w = np.roll(w, K//2)
    return w/w.sum()

print("="*72)
print("CHECK 0: runner w_star and its index moments (vs D2's reconstructed WSTAR)")
print("="*72)
for K in (3,4,5):
    w = wstar_runner(K)
    idx = np.arange(K, dtype=float)
    kbar = float((w*idx).sum())
    Vk = float((w*idx*idx).sum() - kbar*kbar)
    Cprod = (K-1-kbar)*kbar
    D = Cprod - Vk
    print(f"  K={K}: w={np.round(w,6).tolist()}")
    print(f"        kbar={kbar:.6f} Vk={Vk:.6f}  Cprod=(K-1-kbar)kbar={Cprod:.6f}  D=Cprod-Vk={D:.6f}")

# ---- CHECK 1: LP envelope == (mu_K-m)(m-mu_1), support {0,K-1} ----
print()
print("="*72)
print("CHECK 1: LP max mixing-variance == (mu_K-m)(m-mu_1)")
print("="*72)
def lp_maxvar(mus, m):
    K = len(mus)
    c = -((mus-m)**2)
    A_eq = np.vstack([np.ones(K), mus]); b_eq = np.array([1.0, m])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0,None)]*K, method="highs")
    return -res.fun, res.x
maxabs = 0.0; bad_support = 0
for K in (3,4,5):
    for rho in (0.05,0.15,0.30):
        mus = mus_of(K,rho)
        for frac in (0.1,0.3,0.5,0.7,0.9):
            m = mus[0]+frac*(mus[-1]-mus[0])
            vlp,w = lp_maxvar(mus,m)
            van = (mus[-1]-m)*(m-mus[0])
            maxabs = max(maxabs, abs(vlp-van))
            sup = np.where(w>1e-9)[0].tolist()
            if sup not in ([0,K-1],[0],[K-1]) and not (len(sup)==1):
                bad_support += 1
print(f"  max |LP - (mu_K-m)(m-mu_1)| = {maxabs:.3e}   bad_support_count={bad_support}")

# ---- CHECK 2: the deficit criterion in BOTH forms, and rho_c ----
# Form used by N2 threshold block:  LHS=dmu^2 (K-1-kbar)kbar ,  RHS=bw^2+dmu^2 Vk ; pin iff RHS>LHS? or LHS>RHS?
# D2 statement: pin iff bw^2 + Var_w*(mu) > (mu_K-mubar*)(mubar*-mu_1)
#   i.e. bw^2 + dmu^2 Vk  >  dmu^2 Cprod   <=> bw^2 > dmu^2 (Cprod-Vk) = dmu^2 D
print()
print("="*72)
print("CHECK 2: criterion direction + rho_c (exact) vs N2 rho_pred and rho_star_obs")
print("="*72)
N2 = {
 (3,6.0):(0.145625,0.1474793277733929),
 (3,12.0):(0.26875000000000004,0.2899630026655925),
 (3,24.0):(0.4203125000000001,0.5428892442733334),
 (4,6.0):(0.093125,0.09551685204116767),
 (4,12.0):(0.179375,0.18966974339559167),
 (4,24.0):(0.28875000000000006,0.36877260696755476),
 (5,6.0):(0.066875,0.06560720926886457),
 (5,12.0):(0.123125,0.1307715854848991),
 (5,24.0):(0.20875000000000002,0.2580538872187239),
}
def Dcoef(K):
    w = wstar_runner(K); idx=np.arange(K,dtype=float)
    kbar=float((w*idx).sum()); Vk=float((w*idx*idx).sum()-kbar*kbar)
    return (K-1-kbar)*kbar - Vk
print(f"  {'K':>2}{'bw':>6}{'rho_c_exact':>13}{'rho_pred_N2':>13}{'reldiff':>11}{'rho*_obs':>11}{'relerr_obs':>11}")
for (K,bw),(robs,rpred) in N2.items():
    D = Dcoef(K)
    dmu_c = bw/sqrt(D)
    rc = rho_of_dmu(dmu_c)
    reld = abs(rc-rpred)/rpred
    rele = abs(rc-robs)/robs
    print(f"  {K:>2}{bw:>6.1f}{rc:>13.6f}{rpred:>13.6f}{reld:>11.2e}{robs:>11.6f}{rele:>11.4f}")

# small-rho law
print()
print("  small-rho law rho_c ~ bw/(sigma sqrt(2 pi D)) vs exact:")
for (K,bw) in [(3,6.0),(3,24.0),(5,6.0),(5,24.0)]:
    D=Dcoef(K); rc=rho_of_dmu(bw/sqrt(D)); rsmall=bw/(SIGMA*sqrt(2*pi*D))
    print(f"    K={K} bw={bw:>5.1f}: exact={rc:.6f} small={rsmall:.6f} reldiff={abs(rc-rsmall)/rc:.3e}")

# ---- CHECK 3: population KL minimizer var on envelope (pinned) / below (released) ----
print()
print("="*72)
print("CHECK 3: population KL min var(q_inf) vs envelope @ its own mean")
print("="*72)
def gauss(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*sqrt(2*pi))
def pop_min(K,wstar,bw,rho,ngrid=8001):
    mus=mus_of(K,rho); seff=sqrt(SIGMA**2+bw**2)
    lo=mus[0]-9*seff; hi=mus[-1]+9*seff
    t=np.linspace(lo,hi,ngrid); dt=t[1]-t[0]
    Heff=np.stack([gauss(t,mu,seff) for mu in mus]); Heff/=(Heff.sum(1,keepdims=True)*dt)
    Hraw=np.stack([gauss(t,mu,SIGMA) for mu in mus]); Hraw/=(Hraw.sum(1,keepdims=True)*dt)
    p=wstar@Heff; p/=(p.sum()*dt)
    def unpack(z): e=np.exp(z-z.max()); return e/e.sum()
    def obj(z):
        w=unpack(z); q=w@Hraw; q/=(q.sum()*dt)
        m=p>1e-300
        return float(np.sum(p[m]*np.log(p[m]/np.clip(q[m],1e-300,None)))*dt)
    best=None
    for tr in range(8):
        z0=np.random.RandomState(tr).randn(K)*0.7
        r=minimize(obj,z0,method="L-BFGS-B",options={"maxiter":8000,"ftol":1e-16,"gtol":1e-13})
        if best is None or r.fun<best.fun: best=r
    w=unpack(best.x)
    mq=float(w@mus); vq=SIGMA**2+float(w@(mus-mq)**2)
    env=SIGMA**2+(mus[-1]-mq)*(mq-mus[0])
    mp=float(wstar@mus); vp=(SIGMA**2+bw**2)+float(wstar@(mus-mp)**2)
    return w,mq,vq,env,vp,mus
WS3=wstar_runner(3)
for rho in (0.05,0.10,0.20,0.35):
    w,mq,vq,env,vp,mus=pop_min(3,WS3,12.0,rho)
    sup=np.where(w>1e-6)[0].tolist()
    a_pred=(mus[-1]-(WS3@mus))/(mus[-1]-mus[0])
    print(f"  rho={rho:.2f} sup={sup} var_q={vq:9.4f} env={env:9.4f} (var_q-env)={vq-env:+.3e} var_p={vp:9.4f} a={w[0]:.5f} a_pred={a_pred:.5f}")

# ---- CHECK 4: the fixed-mean leak family KL monotonicity (D2 LayerB) ----
print()
print("="*72)
print("CHECK 4: fixed-mean leak -> KL monotone increasing as variance drops")
print("="*72)
K=3; bw=12.0; rho=0.20; WS=wstar_runner(3)
mus=mus_of(K,rho); seff=sqrt(SIGMA**2+bw**2)
lo=mus[0]-9*seff; hi=mus[-1]+9*seff
t=np.linspace(lo,hi,8001); dt=t[1]-t[0]
Heff=np.stack([gauss(t,mu,seff) for mu in mus]); Heff/=(Heff.sum(1,keepdims=True)*dt)
Hraw=np.stack([gauss(t,mu,SIGMA) for mu in mus]); Hraw/=(Hraw.sum(1,keepdims=True)*dt)
p=WS@Heff; p/=(p.sum()*dt)
mubar=float(WS@mus); mu0,mu1,mu2=mus
def klf(q):
    m=p>1e-300; return float(np.sum(p[m]*np.log(p[m]/np.clip(q[m],1e-300,None)))*dt)
prev=None; mono=True
for eps in [0.0,0.02,0.05,0.10,0.20,0.30]:
    rem=1-eps; rhs=mubar-eps*mu1
    w2=(rhs-mu0*rem)/(mu2-mu0); w0=rem-w2
    if w0<0 or w2<0:
        print(f"  eps={eps:.2f} infeasible"); continue
    w=np.array([w0,eps,w2]); q=w@Hraw; q/=(q.sum()*dt)
    mq=float(w@mus); vmix=float(w@(mus-mq)**2); kv=klf(q)
    if prev is not None and kv < prev-1e-12: mono=False
    prev=kv
    print(f"  eps={eps:.2f} vmix={vmix:10.4f} KL={kv:.6e}")
print(f"  monotone increasing in eps: {mono}")

# ---- CHECK 5: is the criterion edge regime (D<=0) reachable for extreme-concentrated w*? ----
print()
print("="*72)
print("CHECK 5: D=(K-1-kbar)kbar - Vk sign for various w* (D2 stated-gap on D>0)")
print("="*72)
K=3
for w in [np.array([0.5,0.0001,0.4999]), np.array([0.45,0.1,0.45]),
          np.array([0.1,0.8,0.1]), np.array([0.33,0.34,0.33])]:
    w=w/w.sum(); idx=np.arange(K,dtype=float)
    kbar=float((w*idx).sum()); Vk=float((w*idx*idx).sum()-kbar*kbar)
    D=(K-1-kbar)*kbar-Vk
    print(f"  w={np.round(w,4).tolist()} kbar={kbar:.4f} Vk={Vk:.4f} D={D:+.5f}")
print("  (D->0 or <0 means rho_c-> inf / criterion always satisfied: pinning for ALL rho)")
