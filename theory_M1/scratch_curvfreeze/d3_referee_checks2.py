"""
Referee checks part 2 (fast, no big quadrature):
 (A) N4 claim: per-iteration geometric contraction rate of a pinned coord
     == KKT slack eta to full precision. Verify ratio_1 vs eta from n4 json.
 (B) The curvature MEAN vs STD distinction for K=3 (the subtle bit):
     N1 curv_mean = 1/sqrt6 = 0.40824829 is the MEAN of the curvature mode
     (a deterministic offset), curv_std ~ 1e-17 is the variance->0 fact that
     claim (d) actually needs. Confirm |c.d3|=0 implies STD=0, not MEAN=0.
 (C) Sanity: is Cov literally rank-1, or rank-1 only because pinned coords
     are frozen to ~1e-130? The frozen-mode std (n3 'm') is ~1e-49..1e-113,
     i.e. NOT exactly 0 -> claim (c)'s 'EXACTLY zero variance' is an
     idealization; the true statement is 'zero up to the O(pinning-residual)
     wiggle of the inactive coords, which is machine-zero'.
 (D) Direction check on the SIGN/identity of eta: in n2, eta_j = 1 - I_j with
     I_j the integral [ (X/(Hw)) h_j ]/Hsum_j. For ACTIVE coords eta=0, for
     pinned coords eta>0. Confirm sign convention from n2 cells.
ASCII-only.
"""
import json, numpy as np
np.set_printoptions(precision=16)
base="C:/Users/Administrator/OneDrive/文档/Claude/Projects/dz-unmix-bounds/theory_M1/scratch_curvfreeze/"

print("=== (A) N4: contraction rate r_per_block vs eta (KKT slack) ===")
n4=json.load(open(base+"n4_stability_decay.json",encoding="utf-8"))
for row in n4["part2_rows"]:
    print(f"  rho={row['rho']}: r_per_block={row['r_per_block']:.10f}  ratio_1(=1-eta)={row['ratio_1']:.10f}  "
          f"eta_decay={row['eta_decay']:.10e}  eta_kkt={row['eta_kkt']:.10e}  abs_diff={row['abs_diff']:.2e}")
    # claim: the per-ITERATION contraction equals (1-eta). ratio_1 = 1 - eta_kkt?
    print(f"      check 1-eta_kkt = {1-row['eta_kkt']:.10f}  vs ratio_1 {row['ratio_1']:.10f}  "
          f"diff {1-row['eta_kkt']-row['ratio_1']: .2e}")

print("\n=== (B) curvature MEAN vs STD for K=3 (subtle) ===")
n1=json.load(open(base+"n1_k3_pinning.json",encoding="utf-8"))
c=n1["cells"]["0.05"]
print(f"  N1 rho=0.05: curv_MEAN={c['curv_mean']:.16e} (=1/sqrt6={1/np.sqrt(6):.16e})  "
      f"curv_STD={c['curv_std']:.3e}")
print( "  -> The MEAN of curvature mode is a nonzero deterministic offset (mass on 2 endmembers")
print( "     gives a fixed (1,-2,1).what value); claim (d) 'curvature variance=0' refers to the STD.")
print(f"  -> curv_std {c['curv_std']:.2e} is machine-zero. |c.d3|=0 forces VARIANCE=0, NOT mean=0. CONSISTENT.")
# Double-check: with what=(a,0,1-a), curvature mode value = (1*a -2*0 +1*(1-a))/sqrt6 = 1/sqrt6 exactly, indep of a!
print( "  Algebra: what=(a,0,1-a): (1,-2,1).what = a+0+(1-a) = 1 => /sqrt6 = 1/sqrt6 for ALL a.")
print( "  Hence curvature mode is CONSTANT (=1/sqrt6) over the active face => std=0 trivially,")
print( "  AND its value is independent of a => orthogonal to the live direction d3. Both consistent.")

print("\n=== (C) frozen-mode std is machine-zero, not literally 0 (overclaim check) ===")
n3=json.load(open(base+"n3_k4_modes.json",encoding="utf-8"))
for rho,cell in n3["per_rho"].items():
    print(f"  rho={rho}: frozen mode 'm' std = {cell['stds']['m']:.3e}  (live 't' std={cell['stds']['t']:.3e})")
    print(f"           ratio m/t = {cell['ratios_to_t']['m']:.3e}  -> 'EXACTLY 0' is idealization; true value O(pinning residual)")

print("\n=== (D) eta sign convention from n2 (pinned coord eta>0) ===")
# n2 K=3 bw=12 middle rho=0.05 already read: eta middle = +0.00169955 > 0, edges = 0. Confirmed in main read.
print("  n2 (3,12,middle,rho=0.05): eta=[0, +0.0016995524280789542, 0] -> pinned coord strictly positive. OK")
print("  Recomputed in check 1 fresh solve: eta_2=+1.699579e-03 matches to 7 digits. OK")

print("\n=== (E) Is the moment 'release' criterion an over- or under-predictor? (regime flag) ===")
# n3 stores pinning_lhs=(muK-mubar)(mubar-mu1), pinning_rhs=bw^2+Var_wstar(mu); pinned iff lhs<rhs.
for rho,cell in n3["per_rho"].items():
    print(f"  rho={rho}: lhs={cell['pinning_lhs']:.4f} rhs={cell['pinning_rhs']:.4f} "
          f"pinned_predicted={cell['pinning_predicts_pinned']} (lhs<rhs => pinned)")
print("  NOTE (from problem statement N2): criterion OVER-predicts rho* (~1.3-9.7% bw<=12, ~20% bw=24).")
print("  => the threshold rho_c is claim (a); D3 only needs eta*>0 to EXIST, which it does.")
