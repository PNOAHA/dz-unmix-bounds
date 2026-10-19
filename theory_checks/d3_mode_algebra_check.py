"""
D3 mode-algebra self-check (ASCII-only).
Verifies the closed-form claim: for the rank-1 covariance with range span{d},
d = (e_1 - e_K)/sqrt2, every tangent mode v has
    std(v . w_hat) / std(d . w_hat) = |v.d| / |d|^2  ... NO.
Careful: w_hat - w_inf = beta * d_vec where d_vec = (e_1 - e_K) (UNNORMALIZED choice matters).
We must be consistent: define the live direction as the UNIT vector u = (e_1-e_K)/sqrt2.
Then w_hat - w_inf = s * u, s scalar random. For any vector v,
    v . (w_hat - w_inf) = s (v . u),  so std(v.w_hat) = |v.u| * std(s).
Taking v = u gives std(u.w_hat) = |u.u| std(s) = std(s) (since u unit).
So ratio std(v.w_hat)/std(u.w_hat) = |v.u| / |u.u| = |v.u| (u unit => u.u=1).

But the numerics report ratios_to_t where t is the *unnormalized* pairwise/edge mode.
Need to reproduce EXACTLY what the numerics computed: they project w_hat onto a set of
mode vectors v_i (each NORMALIZED to unit), take std of each projection, and ratio to the
'.t' mode std. We must use the SAME normalization the numerics used.

From n3: t-mode std (rho=0.05) = 0.024198862, ratio t=1.
 c1 ratio 0.288675 = 1/sqrt12, p1 ratio 0.5, m ratio ~0.
The mode vectors used (read off the names) are the NORMALIZED versions:
 t  = (1,0,0,-1)/sqrt2           (the live unit direction u)
 c1 = (1,-2,1,0)/sqrt6
 c2 = (0,1,-2,1)/sqrt6
 m  = (0,1,-1,0)/sqrt2
 p1 = (1,-1,0,0)/sqrt2           edge first-diff
 p3 = (0,0,1,-1)/sqrt2
Ratio for unit-normalized v: std(v.w)/std(u.w) = |v.u| (since w-w_inf = s u, u unit).
Check each.
"""
import os
import numpy as np

s2 = np.sqrt(2.0)
s6 = np.sqrt(6.0)

u  = np.array([1,0,0,-1])/s2          # live unit direction
c1 = np.array([1,-2,1,0])/s6
c2 = np.array([0,1,-2,1])/s6
m  = np.array([0,1,-1,0])/s2
p1 = np.array([1,-1,0,0])/s2
p3 = np.array([0,0,1,-1])/s2

def ratio(v):
    return abs(np.dot(v,u))   # = |v.u| because w-w_inf = s*u, u unit, std(v.w)=|v.u|*std(s)

print("predicted ratios_to_t (unit-normalized modes):")
for name,v in [("t",u),("c1",c1),("c2",c2),("m",m),("p1",p1),("p3",p3)]:
    print(f"  {name}: {ratio(v):.15f}")

print()
print("1/sqrt12 =", 1.0/np.sqrt(12.0))
print("c1.u exact =", np.dot(c1,u), " = (1*1 + 0 + 0 + 0*-1)/(sqrt6*sqrt2) = 1/sqrt12 =", 1/np.sqrt(12))

# Also verify the UNNORMALIZED dot products quoted in the handoff/task:
print()
print("Unnormalized dot products v.(e1-eK), e1-eK = (1,0,0,-1):")
d_un = np.array([1,0,0,-1])
for name,v in [("c1_un (1,-2,1,0)",np.array([1,-2,1,0])),
               ("c2_un (0,1,-2,1)",np.array([0,1,-2,1])),
               ("m_un  (0,1,-1,0)",np.array([0,1,-1,0])),
               ("p1_un (1,-1,0,0)",np.array([1,-1,0,0])),
               ("p3_un (0,0,1,-1)",np.array([0,0,1,-1]))]:
    print(f"  {name}.(e1-eK) = {np.dot(v,d_un)}")

# K=3 curvature check
print()
print("K=3: u3 = (1,0,-1)/sqrt2, curvature c=(1,-2,1)/sqrt6")
u3 = np.array([1,0,-1])/s2
c3 = np.array([1,-2,1])/s6
print("  c3.u3 =", np.dot(c3,u3), "(should be 0 -> frozen, F4)")

# Now cross-check against n3 json reported stds: ratio of raw stds.
import json
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_results', 'n3_k4_modes.json')
with open(p,"r",encoding="utf-8") as f:
    n3 = json.load(f)
for rk in ["0.05","0.10"]:
    st = n3["per_rho"][rk]["stds"]
    print(f"\n n3 rho={rk}: empirical std-ratios to t:")
    for k in ["c1","c2","m","p1","p3"]:
        print(f"   {k}: emp={st[k]/st['t']:.15e}   pred={ratio({'c1':c1,'c2':c2,'m':m,'p1':p1,'p3':p3}[k]):.15f}")

# dimension count sanity: K live/frozen decomposition
for K in [3,4,5,6]:
    live = 1
    frozen = K-2
    sumto1 = 1
    print(f"K={K}: live(d)={live} + frozen(perp d, sum0)={frozen} + removed(sum-to-1)={sumto1} = {live+frozen+sumto1} = K? {live+frozen+sumto1==K}")
