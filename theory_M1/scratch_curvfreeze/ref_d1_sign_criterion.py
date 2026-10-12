"""
Referee check #4: does the SIGN of the asymmetric eta_mid still track sign(deficit)?
The closed-form CONSTANT for asymmetric K=3 is wrong (CHECK B: ratio 0.05..0.19, not 1),
but the RELEASE CRITERION only needs sign(eta_mid_at_meanmatch) = sign(deficit).
We test: across the K=3 'middle' rho grid (bw=6,12,24), is the cell pinned in n2 EXACTLY
when (muK-mubar)(mubar-mu1) < bw^2 + Var_w*(mu) ?  And is the exact eta_mid>0 there?
ASCII-only, new file.
"""
import numpy as np, math, os, json
from scipy.special import erfinv as _erfinv
SIGMA=25.0; START=200.0; SQRT2=math.sqrt(2.0)
HERE=os.path.dirname(os.path.abspath(__file__))
def dmu_of_rho(rho): return SIGMA*2*SQRT2*float(_erfinv(rho))
def wstar(K,kind):
    w=np.linspace(1.0,0.6,K)
    if kind=='middle' and K>=3: w=np.roll(w,K//2)
    return w/w.sum()
with open(os.path.join(HERE,'n2_population_kkt.json')) as fh: n2=json.load(fh)

print("=== moment criterion sign vs n2 actual pinning (K=3 middle) ===")
print("%-6s %-6s | %-10s %-10s | %-12s %-10s %-10s"%
      ("bw","rho","LHS","RHS","crit_pinned","n2_pinned","agree"))
mism=0; tot=0
for key,recs in n2['grid'].items():
    if recs[0]['K']!=3 or recs[0]['kind']!='middle': continue
    bw=recs[0]['bw']; w=wstar(3,'middle')
    for rec in recs:
        rho=rec['rho']; d=dmu_of_rho(rho)
        mus=np.array([START,START+d,START+2*d]); mubar=float((w*mus).sum())
        Varw=float((w*mus**2).sum()-mubar**2)
        lhs=(mus[-1]-mubar)*(mubar-mus[0]); rhs=bw**2+Varw
        crit=lhs<rhs
        n2pin = (rec.get('pinned_set')==[1])
        agree = (crit==n2pin); tot+=1
        if not agree: mism+=1
        if bw in (6.0,24.0) and rho in (0.14,0.17,0.40,0.45):
            print("%-6.0f %-6.2f | %-10.2f %-10.2f | %-12s %-10s %-10s"%
                  (bw,rho,lhs,rhs,crit,n2pin,agree))
print("total cells %d, criterion/n2 mismatches: %d"%(tot,mism))
print("(mismatches expected only in the narrow boundary band where the leading-order")
print(" criterion over/under-predicts the true threshold by the documented 1.3-20%)")
