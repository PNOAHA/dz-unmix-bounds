import json, math
import numpy as np
P="C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c1_prefactor_vs_K.json"
d=json.load(open(P))
N0=d['meta']['N']

# rho_1K for each K at rho=0.06 from the cells
def cell(K,rho):
    for c in d['cells']:
        if c['K']==K and abs(c['rho']-rho)<1e-9: return c
    return None

print("=== A. N-scan cK_est (rho=0.06, pure 1/sqrtN regime) ===")
print("cK_est = sigma_err*sqrt(N)*rho  (adjacent-rho normalization)")
ck_ns={}
for K in [3,4,5,6]:
    ck_ns[K]=d['fits'][str(K)]['cK_from_Nscan_rho006']
base=ck_ns[3]
print(f"{'K':>2} {'cK_est':>10} {'cK/cK3':>8} {'2/(K-1)':>8} {'(K-1)*cK':>10}")
for K in [3,4,5,6]:
    print(f"{K:>2} {ck_ns[K]:>10.4e} {ck_ns[K]/base:>8.3f} {2.0/(K-1):>8.3f} {ck_ns[K]*(K-1):>10.4e}")

print("\n=== B. Extreme-pair collapse: c_1K_ns = sigma_err*sqrt(N)*rho_1K (rho=0.06) ===")
print("If sigma_err = C/(sqrt(N)*rho_1K) with K-INDEPENDENT C, this is flat.")
c1k_ns={}
print(f"{'K':>2} {'rho_1K':>8} {'c_1K_ns':>10} {'c_1K/c_1K3':>10}")
for K in [3,4,5,6]:
    c=cell(K,0.06)
    rho1K=c['rho_1K']
    # use N-scan N=1e5 sigma_err (==rho-sweep value) -> but cleanest: use cK_est*(rho_1K/rho)
    c1k_ns[K]=ck_ns[K]*(rho1K/0.06)
base1=c1k_ns[3]
for K in [3,4,5,6]:
    c=cell(K,0.06); rho1K=c['rho_1K']
    print(f"{K:>2} {rho1K:>8.4f} {c1k_ns[K]:>10.4e} {c1k_ns[K]/base1:>10.3f}")

print("\n=== C. Direct ratio test on N-scan sigma_err at each N (rho fixed 0.06) ===")
print("sigma_err(K)/sigma_err(K=3) at matched N, and rho_1K(3)/rho_1K(K) prediction")
rho1K={K:cell(K,0.06)['rho_1K'] for K in [3,4,5,6]}
for Ntest in [20000,100000,400000]:
    se={}
    for c in d['N_scan']:
        if abs(c['rho']-0.06)<1e-9 and c['N']==Ntest:
            se[c['K']]=c['sigma_err']
    if 3 in se:
        print(f" N={Ntest}:")
        for K in [3,4,5,6]:
            if K in se:
                ratio=se[K]/se[3]
                pred_1K = rho1K[3]/rho1K[K]   # if se ~ 1/rho_1K
                pred_inv_Km1 = (3-1)/(K-1)     # if se ~ 1/(K-1) i.e. cK~1/(K-1) adjacent
                print(f"   K={K}: se/se3={ratio:.3f}  pred[1/rho_1K]={pred_1K:.3f}  pred[1/(K-1)]={pred_inv_Km1:.3f}")

print("\n=== D. exponent s_K (adjacent rho fit) and saturation diagnosis ===")
for K in [3,4,5,6]:
    f=d['fits'][str(K)]
    print(f"K={K}: s_K(adjacent)={f['exponent_s_K']:+.3f} (R2={f['r2']:.3f}); s_K(vs rho_1K)={f['slope_logrho1K']:+.3f} (R2={f['r2_1K']:.3f}); N-scan slope={f['N_scan_slope']:+.3f}")

print("\n=== E. ratio sigma_err/std(a_hat) across all pinned cells (PRE3) ===")
ratios=[c['ratio_sigma_over_std'] for c in d['cells'] if c['max_interior']<1e-6]
print(f"all-cell ratio: mean={np.mean(ratios):.4f} min={min(ratios):.4f} max={max(ratios):.4f} (sqrt2={math.sqrt(2):.4f})")

print("\n=== F. Power-law fit of cK_est_Nscan vs (K-1) ===")
Km1=np.array([K-1 for K in [3,4,5,6]],float)
ck=np.array([ck_ns[K] for K in [3,4,5,6]])
A=np.vstack([np.ones_like(np.log(Km1)),np.log(Km1)]).T
coef,_,_,_=np.linalg.lstsq(A,np.log(ck),rcond=None)
print(f"log(cK_est_Nscan) = {coef[0]:.4f} + {coef[1]:.4f}*log(K-1)  => cK_est ~ (K-1)^{coef[1]:.3f}")
# compare to c_1K (extreme-pair): does c_1K ~ const? fit c_1K vs (K-1)
c1k=np.array([d['fits'][str(K)]['c_1K'] for K in [3,4,5,6]])
coef2,_,_,_=np.linalg.lstsq(A,np.log(c1k),rcond=None)
print(f"log(c_1K_rhofit) ~ (K-1)^{coef2[1]:.3f}  (flat => exponent~0)")
