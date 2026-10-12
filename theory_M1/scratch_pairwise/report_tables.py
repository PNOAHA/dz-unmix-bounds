import json, math
import numpy as np
P="C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_pairwise/c1_prefactor_vs_K.json"
d=json.load(open(P))
N=d['meta']['N']

print("== rho_c ==")
for K in ['3','4','5','6']:
    rc=d['rho_c'][K]; print(f"K={K}: rho_c={rc['rho_c']:.4f} D={rc['D']:.3f}")

print("\n== cells (pinned only check) ==")
print(f"{'K':>2} {'rho':>6} {'sigma_err':>12} {'std_a':>12} {'ratio':>6} {'max_int':>10} {'rho_1K':>8} {'se*rho':>10} {'se*rho1K':>10}")
for c in d['cells']:
    pin = c['max_interior']<1e-6
    flag='' if pin else ' RELEASED'
    print(f"{c['K']:>2} {c['rho']:>6.3f} {c['sigma_err']:>12.4e} {c['std_a_hat']:>12.4e} "
          f"{c['ratio_sigma_over_std']:>6.3f} {c['max_interior']:>10.1e} {c['rho_1K']:>8.4f} "
          f"{c['sigma_err']*c['rho']:>10.4e} {c['sigma_err']*c['rho_1K']:>10.4e}{flag}")

print("\n== fits per K ==")
for K in ['3','4','5','6']:
    f=d['fits'].get(K,{})
    if 'cK' not in f: print(f"K={K}: {f}"); continue
    print(f"K={K}: slope(log rho)={f['exponent_s_K']:+.3f} R2={f['r2']:.4f} | "
          f"A_K={f['A_K']:.4e} cK={f['cK']:.4e} cK*(K-1)={f['cK_times_Km1']:.4e} | "
          f"vs rho_1K slope={f['slope_logrho1K']:+.3f} R2={f['r2_1K']:.4f} c_1K={f['c_1K']:.4e}"
          + (f" | Nscan slope={f.get('N_scan_slope','?'):.3f} cK_Nscan={f.get('cK_from_Nscan_rho006',float('nan')):.4e}" if 'N_scan_slope' in f else ''))

print("\n== N-scan (rho=0.06) ==")
for c in d['N_scan']:
    print(f"K={c['K']} N={c['N']:>7} sigma_err={c['sigma_err']:.4e} se*sqrt(N)={c['sigma_err']*math.sqrt(c['N']):.4e} max_int={c['max_interior']:.1e}")

print("\n== KEY TEST table ==")
cK={}
for K in [3,4,5,6]:
    f=d['fits'].get(str(K),{})
    if 'cK' in f: cK[K]=f['cK']
if cK:
    base=cK[min(cK)]
    print(f"{'K':>2} {'cK':>12} {'cK/cK3':>8} {'(K-1)*cK':>12} {'1/(K-1)':>8} {'c_1K':>12}")
    for K in sorted(cK):
        f=d['fits'][str(K)]
        print(f"{K:>2} {cK[K]:>12.4e} {cK[K]/base:>8.3f} {cK[K]*(K-1):>12.4e} {1.0/(K-1):>8.3f} {f.get('c_1K',float('nan')):>12.4e}")
    print("\nexpected if cK~1/(K-1): cK/cK3 should be 2/(K-1):",
          {K: round(2.0/(K-1),3) for K in sorted(cK)})
    print("if cK flat: cK/cK3 all ~1.0; if rho_1K-collapse clean: c_1K flat across K")
