import json, math, numpy as np
P="C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c3_top1_split.json"
d=json.load(open(P))

print("=== rho_c_est ===", d['meta'].get('rho_c_est'))

print("\n=== PART 1: endpoint vs interior split (K=4) ===")
print(f"{'config':9s} {'rho':>5s} {'top1':>6s} {'w_kstar':>8s} {'pinned':>6s}  meanw")
for kind in ['endpoint','middle']:
    for r in [x for x in d['part1'] if x['config']==kind]:
        mw=" ".join(f"{v:.3f}" for v in r['mean_w'])
        print(f"{kind:9s} {r['rho']:5.2f} {r['top1']:6.3f} {r['w_kstar_mean']:8.4f} {str(r['pinned']):>6s}  [{mw}]")

# Part 2 analysis: for each (K, rho), find threshold Delta_w*sqrt(N) at which top1 crosses 0.84.
# We have a grid over Delta and N. Compute proxy variable u = sqrt(N)*Delta (the product that
# should set the margin SNR), and find where top1 crosses 0.84 by interpolation in u.
print("\n=== PART 2: endpoint pairwise threshold ===")
p2=d['part2']
def crossing_u(rows, key_u='u'):
    # rows: list of dicts with 'u' and 'top1', sorted by u. top1 typically increases with u.
    rows=sorted(rows, key=lambda r:r[key_u])
    us=[r[key_u] for r in rows]; t=[r['top1'] for r in rows]
    target=0.84
    # find first bracket where it crosses upward
    for i in range(len(rows)-1):
        if t[i] < target <= t[i+1] or (t[i]<=target<t[i+1]):
            # linear interp
            if t[i+1]==t[i]: return us[i]
            frac=(target-t[i])/(t[i+1]-t[i])
            return us[i]+frac*(us[i+1]-us[i])
    # if all above target -> below smallest u; if all below -> above largest
    if all(x>=target for x in t): return us[0]  # threshold <= min u
    if all(x<target for x in t): return None     # never reached in range
    return None

# For each (K,rho): collect over all (Delta,N) the u=sqrt(N)*Delta and top1.
# The threshold u* = sqrt(N)*Delta at top1=0.84. Then test:
#   K-indep model: u* should scale as 1/rho  => c_indep = u* * rho  constant across K
#   (K-1) model:   u* should scale as 1/((K-1) rho) => c_Km1 = u* * (K-1) * rho constant across K
print(f"{'K':>2s} {'rho':>5s} {'u*=sqrtN*dw@0.84':>16s} {'c_indep=u*·rho':>16s} {'c_Km1=u*·(K-1)·rho':>20s}")
summary=[]
Ks=sorted(set(r['K'] for r in p2)); rhos=sorted(set(r['rho'] for r in p2))
for K in Ks:
    for rho in rhos:
        rows=[{'u':math.sqrt(r['N'])*r['Delta_w'],'top1':r['top1']} for r in p2 if r['K']==K and abs(r['rho']-rho)<1e-9]
        u_star=crossing_u(rows)
        if u_star is None:
            print(f"{K:>2d} {rho:5.2f} {'>max (no cross)':>16s}")
            continue
        c_indep=u_star*rho
        c_Km1=u_star*(K-1)*rho
        summary.append({'K':K,'rho':rho,'u_star':u_star,'c_indep':c_indep,'c_Km1':c_Km1})
        print(f"{K:>2d} {rho:5.2f} {u_star:16.2f} {c_indep:16.3f} {c_Km1:20.3f}")

# Collapse quality: coefficient of variation of c across all (K,rho) cells, for each model.
def cv(vals):
    vals=np.array(vals); return float(np.std(vals)/np.mean(vals)) if len(vals)>1 and np.mean(vals)!=0 else float('nan')
if summary:
    ci=[s['c_indep'] for s in summary]; ck=[s['c_Km1'] for s in summary]
    print(f"\nCollapse CV (lower=cleaner):  c_indep CV={cv(ci):.3f}  (mean={np.mean(ci):.3f})   "
          f"c_Km1 CV={cv(ck):.3f} (mean={np.mean(ck):.3f})")
    # Also test within-K-across-rho 1/rho scaling: for each K, is u*·rho constant?
    print("\nWithin-K (does u*·rho stay constant across rho? tests 1/rho law):")
    for K in Ks:
        sub=[s for s in summary if s['K']==K]
        if len(sub)>=2:
            cr=[s['c_indep'] for s in sub]
            print(f"  K={K}: u*·rho over rho = {[round(s['c_indep'],2) for s in sub]}  CV={cv(cr):.3f}")
    print("\nAcross-K (at matched rho, does the prefactor need (K-1)?):")
    for rho in rhos:
        sub=sorted([s for s in summary if abs(s['rho']-rho)<1e-9], key=lambda s:s['K'])
        if len(sub)>=2:
            print(f"  rho={rho:.2f}: u* by K {[(s['K'],round(s['u_star'],1)) for s in sub]}  "
                  f"c_indep {[round(s['c_indep'],2) for s in sub]}  c_Km1 {[round(s['c_Km1'],2) for s in sub]}")

# Raw top1 table for part2
print("\n=== PART 2 raw top1 grid ===")
print(f"{'K':>2s} {'rho':>5s} {'dw':>5s} {'N':>7s} {'top1':>6s} {'sqrtN*dw':>9s}")
for r in sorted(p2,key=lambda x:(x['K'],x['rho'],x['Delta_w'],x['N'])):
    print(f"{r['K']:>2d} {r['rho']:5.2f} {r['Delta_w']:5.2f} {r['N']:>7d} {r['top1']:6.3f} {math.sqrt(r['N'])*r['Delta_w']:9.1f}")
