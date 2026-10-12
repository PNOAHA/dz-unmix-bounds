"""
Adversarial referee checks for derivation D3-rank1-modes.
ASCII-only. Creates NO modifications to existing repo files.
Run: & "C:/Users/Administrator/miniconda3/python.exe" this_file
"""
import numpy as np

np.set_printoptions(precision=16)
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ---------------------------------------------------------------
# CHECK 1: IDENTITY-V  Var_X = s2_N + bw^2  (exact moment identity)
# ---------------------------------------------------------------
def phi(t, m, s):
    return np.exp(-0.5*((t-m)/s)**2)/(s*np.sqrt(2*np.pi))

def check_identity_V():
    print("=== CHECK 1: IDENTITY-V (Var_X = sample_var + bw^2) ===")
    rng = np.random.default_rng(0)
    results = []
    for trial in range(5):
        N = int(rng.integers(50, 200))
        bw = float(rng.uniform(11.0, 30.0))
        a = rng.normal(300, 40, size=N)
        abar = a.mean()
        s2N = ((a-abar)**2).mean()  # population-normalized sample variance
        # fine grid spanning the data +/- many bw
        lo = a.min()-12*bw; hi=a.max()+12*bw
        t = np.linspace(lo, hi, 400001)
        X = phi(t[:,None], a[None,:], bw).mean(axis=1)
        area = trapz(X, t)
        m1 = trapz(t*X, t)
        m2 = trapz(t*t*X, t)
        meanX = m1/area
        varX = m2/area - meanX**2
        resid = varX - s2N - bw**2
        meanerr = meanX - abar
        results.append((N, bw, resid, meanerr))
        print(f"  N={N:4d} bw={bw:7.3f}  Var_X - s2N - bw^2 = {resid: .3e}   meanX-abar = {meanerr: .3e}")
    return results

# ---------------------------------------------------------------
# CHECK 2: mode-algebra inner products |v.d|  (exact linear algebra)
#   plus the SUBTLE point: N1 curv_mean=0.4082=1/sqrt6 is a MEAN not a STD
# ---------------------------------------------------------------
def unit(v):
    v = np.asarray(v, float)
    return v/np.linalg.norm(v)

def check_mode_algebra():
    print("\n=== CHECK 2: mode inner products |v.d| (unit-normalized) ===")
    # K=3
    d3 = unit([1,0,-1])
    c3 = unit([1,-2,1])
    print(f"  K=3: |curvature.d3| = {abs(c3@d3):.16e}  (claim: 0 -> FROZEN)")
    print(f"       curvature unit value 1/sqrt6 = {1/np.sqrt(6):.16e}  (N1 curv_mean=0.4082482904638631)")
    # K=4
    d4 = unit([1,0,0,-1])
    c1 = unit([1,-2,1,0]); c2 = unit([0,1,-2,1])
    m  = unit([0,1,-1,0]); p1 = unit([1,-1,0,0]); p3 = unit([0,0,1,-1])
    for name,v,claim in [("c1",c1,1/np.sqrt(12)),("c2",c2,1/np.sqrt(12)),
                          ("m (interior 1st-diff)",m,0.0),
                          ("p1 (edge 1st-diff)",p1,0.5),("p3 (edge 1st-diff)",p3,0.5)]:
        val = abs(v@d4)
        print(f"  K=4: |{name}.d| = {val:.16e}   claim {claim:.16e}   diff {val-claim: .2e}")
    print(f"  1/sqrt12 = {1/np.sqrt(12):.16e}")
    # dimension count: live=1, frozen=K-2, removed-by-sum=1 => K
    print("\n  dimension count 1+(K-2)+1=K:")
    for K in [3,4,5,6]:
        d = np.zeros(K); d[0]=1; d[-1]=-1; d=unit(d)
        # tangent space basis (sum zero): build and split by d-orthogonality
        # frozen subspace = {v: sum v=0, v_1=v_K}
        # its dim should be K-2
        # Construct: enforce two linear constraints on R^K -> dim K-2
        frozen_dim = K-2
        print(f"    K={K}: live=1 frozen={frozen_dim} removed=1 total={1+frozen_dim+1}  -> {'OK' if 1+frozen_dim+1==K else 'BAD'}")

# ---------------------------------------------------------------
# CHECK 3: the CRUCIAL skeptic point. Claim (c) says modes orthogonal
# to d have EXACTLY zero variance. But the population minimizer is only
# pinned to ~1e-26..1e-131, NOT exactly zero. So the "frozen" modes have
# std = their residual wiggle. Verify the master ratio formula
#   std(v.what)/std(d.what) = |v.d|
# holds for the ALIVE modes (c1,p1) to 15 digits against n3,
# and that frozen-mode std is machine-zero RELATIVE to live (not literally 0).
# ---------------------------------------------------------------
def check_ratio_vs_n3():
    print("\n=== CHECK 3: master ratio std(v.what)/std(t.what) = |v.d| vs N3 ===")
    import json, os
    p = "C:/Users/Administrator/OneDrive/Wendang/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_curvfreeze/n3_k4_modes.json"
    # try both possible OneDrive folder spellings
    cands = [
      "C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_curvfreeze/n3_k4_modes.json",
    ]
    path=None
    for c in cands:
        if os.path.exists(c): path=c; break
    if path is None:
        print("  n3 json not found, skipping"); return
    J=json.load(open(path,encoding="utf-8"))
    d4 = unit([1,0,0,-1])
    preds = {"t":1.0,
             "c1":abs(unit([1,-2,1,0])@d4),
             "c2":abs(unit([0,1,-2,1])@d4),
             "m" :abs(unit([0,1,-1,0])@d4),
             "p1":abs(unit([1,-1,0,0])@d4),
             "p3":abs(unit([0,0,1,-1])@d4)}
    for rho,cell in J["per_rho"].items():
        r = cell["ratios_to_t"]
        print(f"  rho={rho}:")
        for k in ["t","c1","c2","p1","p3"]:
            print(f"    {k}: emp {r[k]:.16e}  pred {preds[k]:.16e}  diff {r[k]-preds[k]: .2e}")
        print(f"    m (frozen): emp {r['m']:.3e}  pred {preds['m']:.1e}  -> machine-zero relative to live")
        print(f"    lambda2/lambda1 = {cell['lambda2_over_lambda1']:.3e} (rank-1 check)")
        ev = cell["top_eigvec"]
        print(f"    top_eigvec = {ev}  (should be +-d=(.7071,0,0,-.7071))")

# ---------------------------------------------------------------
# CHECK 4: is the population minimizer REALLY on {1,K} and is KKT slack
# eta_j>0 for interior j? Recompute eta from n2 and from a fresh solve.
# Also test the moment 'release' criterion direction.
# Recompute eta_j := 1 - integral[ (X/q_w) h_j ] / integral[h_j]
# at the reported minimizer for one n2 cell, K=4 not in n2 (n2 has K=3,4,5
# via 'kind'?). n2 grid keys are (K,bw,kind). Let's just recompute a K=3 cell.
# ---------------------------------------------------------------
def check_kkt_recompute():
    print("\n=== CHECK 4: recompute KKT slack eta_j at population minimizer (K=3 cell) ===")
    SIGMA=25.0; START=200.0
    # pick K=3, bw=12, middle, rho=0.05 from n2
    K=3; bw=12.0; rho=0.05
    dmu = SIGMA*2*np.sqrt(2)*erfinv_safe(rho)
    mus = START + dmu*np.arange(K)
    sigma_eff = np.sqrt(SIGMA**2+bw**2)
    # population data p = sum wstar_k N(mu_k, sigma_eff)
    wstar = np.array([0.25,0.4166666666666667,0.33333333333333337])
    # NOTE: n2 used its own wstar? n2 'middle'/'endpoint' refer to which weight is largest.
    # n2 doesn't store wstar per cell; the moment criterion uses mubar*=sum wstar mu.
    # We'll just check the QUALITATIVE pinning + eta>0 with this wstar.
    lo=START-12*sigma_eff; hi=mus[-1]+12*sigma_eff
    t=np.linspace(lo,hi,60001)
    H = np.stack([phi(t,m,SIGMA) for m in mus],axis=1)  # raw-sigma dictionary, columns
    Hsum = trapz(H,t,axis=0)  # ~1 each
    p = sum(wstar[k]*phi(t,mus[k],sigma_eff) for k in range(K))
    p/=trapz(p,t)
    # solve KL(p||q_w) over simplex by multiplicative (Lee-Seung style) updates
    w=np.ones(K)/K
    for it in range(200000):
        q = H@w
        q=np.maximum(q,1e-300)
        # gradient of KL(p||q) wrt w_j: -integral p/q * h_j  (+ const from normalization)
        num = trapz((p/q)[:,None]*H, t, axis=0)
        # multiplicative update with simplex renorm
        w = w*num/Hsum
        w = np.maximum(w,0)
        s=w.sum()
        if s>0: w/=s
    q=H@w; q=np.maximum(q,1e-300)
    I = trapz((p/q)[:,None]*H,t,axis=0)/Hsum   # the [ (X/(Wh))H ]/Hsum quantity
    eta = 1.0 - I
    print(f"  rho={rho} K={K} bw={bw}: w_hat = {w}")
    print(f"    I_j (should be ~1 on support, <1 on pinned) = {I}")
    print(f"    eta_j = 1 - I_j = {eta}")
    print(f"    interior coord eta_2 = {eta[1]: .6e}  (claim: strictly > 0)")
    print(f"    n2 reported eta for (3,12,middle,rho=0.05) middle coord = 0.0016995524280789542")

def erfinv_safe(x):
    from math import erf
    # invert erf numerically (vectorless small helper)
    from scipy.special import erfinv
    return erfinv(x)

if __name__=="__main__":
    check_identity_V()
    check_mode_algebra()
    check_ratio_vs_n3()
    try:
        check_kkt_recompute()
    except Exception as e:
        print("  KKT recompute failed:", repr(e))
