# Final probes: (E) exactness of var(q_w)=sigma^2+Var_mix on the actual finite grid,
# (F) Layer-A relaxation: is the moment-only argument actually a LOWER bound certificate?
#     check that NO interior w can have var(q_w) > Env(mubar*) but the data needs it; and that
#     the moment relaxation's optimal value lower-bounds true KL (it does: KL >=0, only equality at p=q).
# (G) the "exact saturation var(q_inf)=Env" claim - is the residual optimizer tol or a real gap?
#     refine with finer grid + tighter tol and watch the residual.
# ASCII-only.
import numpy as np
from math import sqrt,pi,erf
from scipy.special import erfinv
from scipy.optimize import minimize

SIGMA=25.0; START=200.0
def dmu_of_rho(r): return SIGMA*2*sqrt(2)*erfinv(r)
def mus_of(K,rho): return START+dmu_of_rho(r=rho)*np.arange(K)
def gauss(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*sqrt(2*pi))
def wstar_runner(K,kind='middle'):
    w=np.linspace(1.0,0.6,K)
    if kind=='middle' and K>=3: w=np.roll(w,K//2)
    return w/w.sum()

print("="*74)
print("PROBE E: var(q_w)=sigma^2+Var_mix(w) exact on the finite area-normalized grid?")
print("  (atoms truncated/renormalized -> the law-of-total-variance identity is only")
print("   approximate; quantify the error that propagates into the envelope)")
print("="*74)
for (K,rho,bw) in [(3,0.20,0.0),(3,0.05,0.0),(4,0.08,0.0)]:
    mus=mus_of(K,rho)
    for ngrid,span in [(6001,8),(8001,9),(20001,12)]:
        seff=SIGMA
        lo=mus[0]-span*seff; hi=mus[-1]+span*seff
        t=np.linspace(lo,hi,ngrid); dt=t[1]-t[0]
        Hraw=np.stack([gauss(t,mu,SIGMA) for mu in mus]); Hraw/=(Hraw.sum(1,keepdims=True)*dt)
        w=wstar_runner(K)
        q=w@Hraw; q/=(q.sum()*dt)
        mq_grid=float((t*q).sum()*dt)
        vq_grid=float(((t-mq_grid)**2*q).sum()*dt)
        mq_an=float(w@mus); vq_an=SIGMA**2+float(w@(mus-mq_an)**2)
        print(f"  K={K} rho={rho} grid={ngrid} span={span}: "
              f"var_grid={vq_grid:.4f} var_analytic={vq_an:.4f} diff={vq_grid-vq_an:+.3e}")

print()
print("="*74)
print("PROBE F: Layer-A as a rigorous LOWER bound on the boundary location")
print("  Claim to test: if var_p>Env(m) for ALL m in [mu_1,mu_K] reachable by q_w near mubar*,")
print("  then no q_w matches p in (mean,var); KL>0 strictly; min is at boundary of moment set.")
print("  Test: scan Env(m)-var_p over m; if <0 everywhere q can sit at p's mean band, p is strictly")
print("  above achievable set in a mean-neighborhood -> moment-projection forced to max-var edge.")
print("="*74)
for (K,rho,bw) in [(3,0.05,12.0),(3,0.20,12.0),(3,0.35,12.0)]:
    mus=mus_of(K,rho); w=wstar_runner(K)
    mubar=float(w@mus); var_p=(SIGMA**2+bw**2)+float(w@(mus-mubar)**2)
    ms=np.linspace(mus[0]+1e-6,mus[-1]-1e-6,2001)
    env=SIGMA**2+(mus[-1]-ms)*(ms-mus[0])
    # max envelope over ALL means:
    above_everywhere = np.all(var_p>env)
    above_at_mean = var_p > (SIGMA**2+(mus[-1]-mubar)*(mubar-mus[0]))
    print(f"  rho={rho:.2f}: var_p>Env at p-mean: {above_at_mean};  var_p>Env for ALL m: {above_everywhere}"
          f"  (max Env={env.max():.3f}, var_p={var_p:.3f})")

print()
print("="*74)
print("PROBE G: does var(q_inf) -> Env exactly as grid/tol refine, or floor at a real gap?")
print("="*74)
def pop_min(K,wstar,bw,rho,ngrid,span,ftol):
    mus=mus_of(K,rho); seff=sqrt(SIGMA**2+bw**2)
    lo=mus[0]-span*seff; hi=mus[-1]+span*seff
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
    for tr in range(10):
        z0=np.random.RandomState(tr).randn(K)*0.7
        r=minimize(obj,z0,method="L-BFGS-B",options={"maxiter":20000,"ftol":ftol,"gtol":1e-14})
        if best is None or r.fun<best.fun: best=r
    w=unpack(best.x); mq=float(w@mus); vq=SIGMA**2+float(w@(mus-mq)**2)
    env=SIGMA**2+(mus[-1]-mq)*(mq-mus[0])
    return vq,env,w
W3=wstar_runner(3)
for (ngrid,span,ftol) in [(6001,8,1e-15),(8001,9,1e-16),(16001,11,1e-16)]:
    vq,env,w=pop_min(3,W3,12.0,0.20,ngrid,span,ftol)
    print(f"  grid={ngrid} span={span} ftol={ftol:.0e}: var_q={vq:.6f} env={env:.6f} "
          f"|var_q-env|={abs(vq-env):.3e} w_mid={w[1]:.3e}")
print("  (if |var_q-env| shrinks with refinement -> consistent with EXACT saturation;")
print("   if it floors -> a genuine higher-moment gap, i.e. KL min var strictly below Env)")
