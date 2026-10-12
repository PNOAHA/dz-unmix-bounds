# -*- coding: utf-8 -*-
# Deterministic c-calibration (no MC): reduced extreme-pair Fisher I_red, KDE correction kappa
# vs C1 measured sigma_err, small-rho closed form, and top1 endpoint threshold prediction.
import numpy as np, math, json
import numpy as _np
trapz=getattr(_np,"trapezoid",None) or _np.trapz
try:
    from scipy.special import erfinv as _erfinv, erf as _erf
    def erfinv(x): return float(_erfinv(x))
    def erf(x): return float(_erf(x))
except Exception:
    raise SystemExit("need scipy")

SIGMA=25.0; START=200.0; BW=12.0
TGRID=np.arange(50.0,2500.0+1e-9,5.0)
def peaks(K,rho):
    dmu=SIGMA*2.0*math.sqrt(2.0)*erfinv(rho); return np.array([START+i*dmu for i in range(K)]), dmu
def gauss(mu,s): 
    h=np.exp(-0.5*((TGRID-mu)/s)**2); return h/trapz(h,TGRID)

def reduced_a_inf(K,rho):
    # KL projection of population p (sigma_eff comps, weights w* endpoint) onto {h_1,h_K} raw-sigma family
    mus,dmu=peaks(K,rho); seff=math.sqrt(SIGMA**2+BW**2)
    w=np.linspace(1.0,0.6,K); w=w/w.sum()                      # endpoint w*
    p=sum(w[i]*gauss(mus[i],seff) for i in range(K)); p=p/trapz(p,TGRID)
    h1=gauss(mus[0],SIGMA); hK=gauss(mus[-1],SIGMA)
    # 1-D KL min over a: minimize -∫ p log(a h1+(1-a)hK); convex, bisect on derivative
    def dKL(a):
        q=a*h1+(1-a)*hK+1e-300
        return -trapz(p*(h1-hK)/q, TGRID)
    lo,hi=1e-9,1-1e-9
    for _ in range(200):
        mid=0.5*(lo+hi)
        if dKL(mid)>0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi), mus, dmu

def I_red(K,rho,a):
    mus,dmu=peaks(K,rho)
    h1=gauss(mus[0],SIGMA); hK=gauss(mus[-1],SIGMA)
    q0=a*h1+(1-a)*hK+1e-300
    return float(trapz((h1-hK)**2/q0, TGRID))

# load C1 measured sigma_err
J=json.load(open("c1_prefactor_vs_K.json")) if False else None
# c1 json structure unknown; reconstruct measured from key_numbers table in the report instead:
# N=1e5 measured sigma_err table (from C1 report):
MEAS={3:{0.04:1.910e-2,0.06:1.190e-2,0.08:9.532e-3,0.11:6.912e-3,0.15:5.236e-3},
      4:{0.04:1.273e-2,0.06:8.576e-3,0.08:6.331e-3,0.11:4.806e-3,0.15:3.873e-3},
      5:{0.04:9.487e-3,0.06:6.656e-3,0.08:5.122e-3,0.11:4.196e-3},
      6:{0.04:7.366e-3,0.06:5.373e-3,0.08:4.420e-3}}
N=1e5
print("K  rho   a_inf   I_red     sig_naive  sig_meas   kappa   rho_1K   C=sig*sqrtN*rho1K")
rows=[]
for K in [3,4,5,6]:
    for rho in sorted(MEAS[K]):
        a,mus,dmu=reduced_a_inf(K,rho)
        Ir=I_red(K,rho,a)
        sig_naive=math.sqrt(2)/math.sqrt(N*Ir)
        sig_meas=MEAS[K][rho]
        kappa=sig_meas/sig_naive
        rho1K=erf((K-1)*dmu/(2*math.sqrt(2)*SIGMA))
        C=sig_meas*math.sqrt(N)*rho1K
        rows.append((K,rho,kappa,C))
        print(f"{K}  {rho:.2f}  {a:.4f}  {Ir:.4e}  {sig_naive:.4e}  {sig_meas:.4e}  {kappa:.4f}  {rho1K:.4f}  {C:.4f}")

ks=[r[2] for r in rows]
Cs=[r[3] for r in rows]
print(f"\nkappa (sig_meas/sig_naive): mean={np.mean(ks):.4f} std={np.std(ks):.4f} CV={np.std(ks)/np.mean(ks):.3f} range=[{min(ks):.3f},{max(ks):.3f}]")
print(f"C = sig_meas*sqrtN*rho_1K:  mean={np.mean(Cs):.4f} std={np.std(Cs):.4f} CV={np.std(Cs)/np.mean(Cs):.3f}")

# small-rho closed form for I_red: h1-hK ~ -(K-1)dmu * h'(t), h'=dh/dmu. For Gaussian, ∫(h')^2/h-ish.
# leading: I_red ~ ((K-1)dmu)^2 * Fisher_loc, Fisher_loc=∫ (dh/dmu)^2 / h dt = 1/sigma^2 (Gaussian location Fisher).
# so I_red ~ ((K-1)dmu)^2 / sigma^2. dmu = sigma*sqrt(2pi)*rho (small). => I_red ~ 2pi ((K-1)rho)^2.
# sig_naive = sqrt2/sqrt(N I_red) = sqrt2/sqrt(N 2pi) /((K-1)rho) = 1/(sqrt(pi N)) /((K-1)rho).
print("\n-- small-rho closed form check (K=3, rho=0.04) --")
K=3; rho=0.04; a,mus,dmu=reduced_a_inf(K,rho)
Ir=I_red(K,rho,a); Ir_pred=2*math.pi*((K-1)*rho)**2
print(f"I_red exact={Ir:.4e}  pred 2pi((K-1)rho)^2={Ir_pred:.4e}  ratio={Ir/Ir_pred:.4f}")
# C_pred from naive: sig_naive*sqrtN*rho_1K, rho_1K~(K-1)rho => C_naive = sqrt2/sqrt(2pi) = 1/sqrt(pi)
print(f"C_naive predicted = 1/sqrt(pi) = {1/math.sqrt(math.pi):.4f}  (times kappa gives measured C)")

# top1 endpoint threshold: top1 correct iff a_hat > 1/2; margin Dw=a_inf-1/2; std(a)=sig_err/sqrt2.
print("\n-- top1 endpoint threshold (the endpoint w* used: a_inf and margin) --")
for K in [3,4,5,6]:
    rho=0.06; a,mus,dmu=reduced_a_inf(K,rho); rho1K=erf((K-1)*dmu/(2*math.sqrt(2)*SIGMA))
    Dw=a-0.5
    sig=MEAS[K][rho]; stda=sig/math.sqrt(2)
    z=Dw/stda
    print(f"K={K} rho={rho}: a_inf={a:.4f} Dw=a-0.5={Dw:.4f} std(a)={stda:.4e} z=Dw/std={z:.2f}  P_fail=Phi(-z)={0.5*(1-math.erf(z/math.sqrt(2))):.2e}")
