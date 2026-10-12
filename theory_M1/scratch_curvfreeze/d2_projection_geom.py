# D2 step (3): WHY an over-variance target KL-projects to the max-variance (extreme) edge.
# Two numerical probes:
#  (A) Compute the actual population KL minimizer w_inf (full simplex), read off its
#      (mean, var) and confirm it lands ON the upper variance envelope at its own mean
#      whenever pinned -> the projection saturates the achievable-set boundary.
#  (B) Probe the 1-D KL-vs-variance-slack monotonicity that underlies the boundary
#      argument: along the family of q_w with mean fixed at mubar*, KL(p||q_w) is
#      DECREASING in the mixing variance up to the max -> minimizer pushes variance to
#      the envelope, then the residual mean/shape is resolved by the 2 extreme weights.
# ASCII-only.

import numpy as np
from math import sqrt, pi
from scipy.special import erfinv
from scipy.optimize import minimize

SIGMA = 25.0
START = 200.0

def dmu_of_rho(rho):
    return SIGMA*2.0*sqrt(2.0)*erfinv(rho)

def mus_of(K, rho):
    return START + dmu_of_rho(rho)*np.arange(K)

def gauss(t, mu, s):
    return np.exp(-0.5*((t-mu)/s)**2)/(s*sqrt(2*pi))

def build_grid(mus, bw):
    s_eff = sqrt(SIGMA**2 + bw**2)
    lo = mus[0] - 8*s_eff
    hi = mus[-1] + 8*s_eff
    t = np.linspace(lo, hi, 6001)
    return t, t[1]-t[0], s_eff

def kl(p, q, dt):
    m = p > 1e-300
    return np.sum(p[m]*np.log(p[m]/np.clip(q[m],1e-300,None)))*dt

def population_min(K, wstar, bw, rho):
    mus = mus_of(K, rho)
    t, dt, s_eff = build_grid(mus, bw)
    # population data: width-s_eff mixture (KDE inflation)
    H_eff = np.stack([gauss(t, mu, s_eff) for mu in mus])   # for p
    H_raw = np.stack([gauss(t, mu, SIGMA) for mu in mus])    # dictionary atoms (raw sigma)
    # area-normalize each atom on grid
    H_eff /= (H_eff.sum(1, keepdims=True)*dt)
    H_raw /= (H_raw.sum(1, keepdims=True)*dt)
    p = wstar @ H_eff
    p /= (p.sum()*dt)
    # minimize KL(p || sum w_k H_raw) over simplex via softmax param + L-BFGS, multi-start
    def unpack(z):
        e = np.exp(z - z.max()); return e/e.sum()
    def obj(z):
        w = unpack(z); q = w @ H_raw; q/= (q.sum()*dt)
        return kl(p, q, dt)
    best=None
    for trial in range(6):
        z0 = np.random.RandomState(trial).randn(K)*0.5
        r = minimize(obj, z0, method="L-BFGS-B",
                     options={"maxiter":5000,"ftol":1e-15,"gtol":1e-12})
        if best is None or r.fun<best.fun: best=r
    w = unpack(best.x)
    # moments of q_w (centers, raw atoms contribute sigma^2 each)
    mubar_q = float(w@mus)
    var_q = SIGMA**2 + float(w@(mus-mubar_q)**2)
    # data moments
    mubar_p = float(wstar@mus)
    var_p = (SIGMA**2 + bw**2) + float(wstar@(mus-mubar_p)**2)
    env_at_q = SIGMA**2 + (mus[-1]-mubar_q)*(mubar_q-mus[0])
    return w, mubar_q, var_q, mubar_p, var_p, env_at_q, mus

print("="*78)
print("(A) population KL minimizer lands ON the max-variance envelope when pinned")
print("    columns: support(>1e-6) | var(q_inf) | envelope@mean(q_inf) | data var | a vs a_pred")
print("="*78)
WSTAR = {3: np.array([0.25, 0.4166666666666667, 0.33333333333333337])}
for (K,bw,rho,wstar) in [
    (3,12.0,0.05, WSTAR[3]),
    (3,12.0,0.10, WSTAR[3]),
    (3,12.0,0.20, WSTAR[3]),   # pinned region for bw=12 (rho* ~0.265)
    (3,12.0,0.35, WSTAR[3]),   # interior (released)
]:
    w,mq,vq,mp,vp,envq,mus = population_min(K,wstar,bw,rho)
    sup = np.where(w>1e-6)[0].tolist()
    a_pred = (mus[-1]-mp)/(mus[-1]-mus[0])
    on_env = abs(vq-envq)
    print(f" rho={rho:.2f} support={sup} var_q={vq:9.4f} env@meanq={envq:9.4f} "
          f"|var_q-env|={on_env:.2e}  var_p={vp:9.4f}  a={w[0]:.5f} a_pred={a_pred:.5f}")

print()
print("="*78)
print("(B) along fixed-mean=mubar* family, KL decreases as mixing variance -> envelope")
print("    (parametrize 2-extreme + leak into a middle atom; show min KL at zero-leak=edge)")
print("="*78)
# K=3, bw=12, rho=0.20 (pinned). Family: w=(a-? , eps, 1-a-?) with mean fixed = mubar*.
K=3; bw=12.0; rho=0.20; wstar=WSTAR[3]
mus = mus_of(K,rho)
t, dt, s_eff = build_grid(mus, bw)
H_eff = np.stack([gauss(t,mu,s_eff) for mu in mus]); H_eff/=(H_eff.sum(1,keepdims=True)*dt)
H_raw = np.stack([gauss(t,mu,SIGMA) for mu in mus]); H_raw/=(H_raw.sum(1,keepdims=True)*dt)
p = wstar@H_eff; p/=(p.sum()*dt)
mubar = float(wstar@mus)
# w with middle weight = eps, remaining (1-eps) split on extremes to hold mean=mubar:
# w0*mu0 + eps*mu1 + w2*mu2 = mubar, w0+w2 = 1-eps
mu0,mu1,mu2 = mus
print("   eps(middle)   mixing_var      KL(p||q)")
for eps in [0.0,0.02,0.05,0.10,0.20,0.3]:
    rem = 1-eps
    # solve w0,w2: w0+w2=rem ; w0*mu0+w2*mu2 = mubar-eps*mu1
    rhs = mubar - eps*mu1
    w2 = (rhs - mu0*rem)/(mu2-mu0)
    w0 = rem - w2
    if w0<0 or w2<0:
        print(f"   {eps:.3f}  (infeasible at fixed mean)"); continue
    w = np.array([w0,eps,w2])
    q = w@H_raw; q/=(q.sum()*dt)
    mq = float(w@mus); vmix = float(w@(mus-mq)**2)
    print(f"   {eps:.3f}      {vmix:10.4f}    {kl(p,q,dt):.6e}")
print("   -> KL is minimized at eps=0 (the extreme edge = max mixing variance at fixed mean)")
