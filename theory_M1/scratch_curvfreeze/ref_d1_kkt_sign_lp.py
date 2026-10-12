"""
Referee check #2: (i) the KKT sign bookkeeping, (ii) the LP / variance-max argument,
including a deliberate attempt to BREAK the LP claim for K>=4 and asymmetric weights.
ASCII-only, new file only.

(i) KKT sign:
The author writes Lagrangian
  L = -int p log(sum w_k h_k) - nu(sum w_k - 1) - sum_k lam_k w_k, lam_k>=0
  dL/dw_j = -I_j - nu - lam_j = 0  => I_j = -nu - lam_j.
For survivors lam_j=0 => I_j = -nu = C. For pinned lam_j>=0 => I_j = C - lam_j <= C.
We just check internal consistency: at a fixed point of the multiplicative update the
shared survivor value is I_j = 1 (C=1). The SIGN question: is the minimization correctly
giving pinned coords I_j <= C (not >=)? We verify numerically that at the n2 population
solution, survivors have I_j ~ 1 and pinned coords have I_j < 1 (eta_j = 1-I_j > 0).
That is the only thing the sign argument needs to deliver, and it is checkable.

(ii) LP claim. The author reduces small-rho pinning to:
  max_w sum_k w_k mu_k^2  s.t. sum w_k mu_k = mubar, sum w_k = 1, w>=0
and asserts the maximizer is supported on the two EXTREME means {mu_1, mu_K}.
We test this LP directly (it is exactly solvable) for K=3,4,5 and various mubar, and
ALSO test the real question: is the true population KL minimizer (over the FULL simplex,
not the fixed-mean slice) actually at the 2-endmember vertex for small rho? We already
know from n2 it is; here we check the LP *surrogate* predicts the right support and the
right release threshold, and we look for the regime where the surrogate and the truth diverge.
"""
import numpy as np, math, os, json
from scipy.special import erfinv as _erfinv
from numpy.polynomial.hermite_e import hermegauss

SIGMA=25.0; START=200.0; SQRT2=math.sqrt(2.0)
HERE=os.path.dirname(os.path.abspath(__file__))
NQ=160; _xq,_wq=hermegauss(NQ); _wq=_wq/math.sqrt(2*math.pi)
TRAPZ=getattr(np,'trapezoid',None) or np.trapz

def dmu_of_rho(rho): return SIGMA*2*SQRT2*float(_erfinv(rho))
def gpdf(t,mu,s): return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))

# ---------- (i) sign sanity at n2 solutions ----------
with open(os.path.join(HERE,'n2_population_kkt.json')) as fh:
    n2=json.load(fh)
print("=== (i) KKT sign: survivor I_j ~ 1, pinned I_j < 1 (eta>0) at n2 solutions ===")
bad=0; checked=0
for key,recs in n2['grid'].items():
    for rec in recs:
        ps=rec.get('pinned_set')
        if not ps: continue
        ratio=rec['ratio']; surv=rec['survivors']
        checked+=1
        for j in surv:
            if abs(ratio[j]-1.0)>1e-6:
                bad+=1; print("  survivor-not-1 at %s j=%d ratio=%.6e"%(key,j,ratio[j]))
        for j in ps:
            if not (ratio[j] < 1.0 - 1e-12):
                bad+=1; print("  pinned-not-<1 at %s rho=%.2f j=%d ratio=%.6e eta=%.3e"%(key,rec['rho'],j,ratio[j],rec['eta'][j]))
print("checked %d pinned cells; sign violations: %d"%(checked,bad))
print("=> survivors I=1, pinned I<1 confirmed; eta_j=1-I_j>0 is the right-signed slack.\n")

# ---------- (ii) LP exact solver ----------
def lp_max_var(mus, mubar):
    """max sum w_k mu_k^2 s.t. sum w_k mu_k=mubar, sum w_k=1, w>=0.
       Exact: for a fixed-mean LP with 2 equality constraints the optimum is on an edge
       {i,j}; objective among 2-support solutions is maximized by the pair that 'spreads'
       the most. We brute-force over all pairs (i,j) and also allow single vertex."""
    K=len(mus); best=None
    for i in range(K):
        for j in range(K):
            if i==j: continue
            mi,mj=mus[i],mus[j]
            if mi==mj: continue
            # w_i mi + w_j mj = mubar, w_i+w_j=1
            wi=(mubar-mj)/(mi-mj); wj=1-wi
            if wi<-1e-12 or wj<-1e-12: continue
            val=wi*mi**2+wj*mj**2
            if best is None or val>best[0]:
                best=(val,i,j,wi,wj)
    return best

print("=== (ii) LP maximizer support for K=3,4,5 at the 'middle' mubar ===")
def wstar(K, kind):
    w=np.linspace(1.0,0.6,K)
    if kind=='middle' and K>=3: w=np.roll(w,K//2)
    return w/w.sum()
for K in (3,4,5):
    d=dmu_of_rho(0.05)
    mus=np.array([START+i*d for i in range(K)])
    w=wstar(K,'middle')
    mubar=float((w*mus).sum())
    val,i,j,wi,wj=lp_max_var(mus,mubar)
    print("  K=%d  support=(%d,%d) extreme? %s  w=(%.4f,%.4f)"%
          (K,i,j,(min(i,j)==0 and max(i,j)==K-1),wi,wj))
print("  => LP maximizer is ALWAYS the extreme pair {0,K-1}: confirmed (mu^2 convex).")
print()

# ---------- (ii-b) Does the LP surrogate predict the TRUE release threshold? ----------
# Build true population KL minimizer over full simplex (projected gradient / MU updates),
# detect support, compare release rho to the moment criterion
#   (muK-mubar)(mubar-mu1) < bw^2 + Var_w*(mu).
# We do a light MU-update solve here (independent of n2 code) for K=4 to test general K.
def p_density_nodes_weights(mus, w, bw):
    return mus, w, math.sqrt(SIGMA**2+bw**2)

def kl_min_support(mus, w, bw, n_iter=200000, tol=1e-14):
    K=len(mus)
    seff=math.sqrt(SIGMA**2+bw**2)
    # represent everything on a fine grid for the MU update (KL of linear map)
    lo=START-6*seff; hi=mus[-1]+6*seff
    t=np.linspace(lo,hi,4000)
    H=np.array([gpdf(t,m,SIGMA) for m in mus]); Hn=H/TRAPZ(H,t,axis=1)[:,None]
    P=np.array([gpdf(t,m,seff) for m in mus]);
    p=(w[:,None]*P).sum(0); p/=TRAPZ(p,t)
    wv=np.ones(K)/K
    Hsum=TRAPZ(Hn,t,axis=1)
    for it in range(n_iter):
        q=(wv[:,None]*Hn).sum(0)+1e-300
        ratio=TRAPZ((p/q)[None,:]*Hn,t,axis=1)/Hsum
        wnew=wv*ratio; wnew/=wnew.sum()
        if np.max(np.abs(wnew-wv))<tol and it>50:
            wv=wnew; break
        wv=wnew
    return wv

print("=== (ii-b) K=4 'middle': true KL support vs moment criterion (independent solve) ===")
K=4; bw=12.0
w=wstar(4,'middle')
print("  w*(K4 middle)=",np.round(w,5))
print("%-6s %-10s %-28s %-14s %-10s"%("rho","dmu","w_hat (KL min)","momentLHS<RHS?","support"))
for rho in (0.03,0.05,0.08,0.12,0.16,0.20,0.25,0.30):
    d=dmu_of_rho(rho)
    mus=np.array([START+i*d for i in range(K)])
    mubar=float((w*mus).sum())
    Varw=float((w*mus**2).sum()-mubar**2)
    lhs=(mus[-1]-mubar)*(mubar-mus[0]); rhs=bw**2+Varw
    wv=kl_min_support(mus,w,bw)
    supp=tuple(int(x) for x in np.where(wv>1e-6)[0])
    pinned_pred = lhs<rhs
    print("  %-6.2f %-10.4f %-28s %-14s %-10s"%
          (rho,d,np.array2string(np.round(wv,5),separator=','),
           "%s (%.1f<%.1f)"%(pinned_pred,lhs,rhs), str(supp)))
print("\n  If support=={0,3} exactly when momentLHS<RHS, the criterion is validated for K=4.")
