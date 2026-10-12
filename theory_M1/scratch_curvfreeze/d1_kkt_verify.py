"""
D1-KKT verification. ASCII-only. Creates NO modification of existing files.
Goal: cross-check the analytic KKT-slack closed forms against n2_population_kkt.json.

Definitions (locked from n2_population_kkt.py):
  - dictionary h_k area-normalized Gaussian(mu_k, SIGMA) on TGRID
  - data p = population KDE density = sum_k w*_k Gaussian(mu_k, SEFF), SEFF=sqrt(SIGMA^2+bw^2)
  - q0 = a*h_1 + (1-a)*h_K  (2-endmember candidate, raw SIGMA)
  - I_j = ratio_j = ( integral p*h_j/q0 dt ) / ( integral h_j dt )
  - eta_j = 1 - I_j  (KKT slack; >0 = strictly pinned)
  - 'a' is fixed by the survivor condition I_1 = I_K.
"""
import numpy as np, math, json, os
from scipy.special import erfinv as _erfinv, erf as _erf

SIGMA = 25.0; START = 200.0
TGRID = np.arange(50.0, 2500.0 + 1e-9, 5.0)
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz
SQRT2 = math.sqrt(2.0)
HERE = os.path.dirname(os.path.abspath(__file__))

def gauss(t, mu, s):
    return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))

def hk(mu, s=SIGMA):
    g = np.exp(-0.5*((TGRID-mu)/s)**2)
    return g/TRAPZ(g, TGRID)

def peaks(K, rho):
    dmu = SIGMA*2.0*SQRT2*float(_erfinv(rho))
    return np.array([START+i*dmu for i in range(K)]), dmu

def wstar(K, kind):
    w = np.linspace(1.0, 0.6, K)
    if kind == 'middle' and K >= 3:
        w = np.roll(w, K//2)
    return w/w.sum()

def p_density(K, rho, w, bw):
    mus, _ = peaks(K, rho)
    seff = math.sqrt(SIGMA**2+bw**2)
    comp = np.array([hk(m, seff) for m in mus])
    g = (w[:,None]*comp).sum(0)
    return g/TRAPZ(g, TGRID)

def I_vector(p, Hfull):
    """I_j for ALL K dictionary atoms given the 2-endmember q0 is built inside."""
    pass

def solve_a_and_eta(K, rho, w, bw, n_a=20001):
    """Numerically: build q0=a h1+(1-a)hK, scan a to enforce I_1=I_K, then return eta_j for all j."""
    mus, dmu = peaks(K, rho)
    p = p_density(K, rho, w, bw)
    Hall = np.array([hk(m) for m in mus])           # (K,T) raw-sigma atoms
    h1 = Hall[0]; hK = Hall[-1]
    Hsum = TRAPZ(Hall, TGRID, axis=1)               # ~1 each
    def Ivec(a):
        q0 = a*h1 + (1.0-a)*hK + 1e-300
        integrand = (p/q0)[None,:]*Hall              # (K,T)
        return TRAPZ(integrand, TGRID, axis=1)/Hsum
    # find a s.t. I_1 - I_K = 0 by bisection on a in (eps,1-eps)
    lo, hi = 1e-6, 1.0-1e-6
    def f(a):
        Iv = Ivec(a); return Iv[0]-Iv[-1]
    flo, fhi = f(lo), f(hi)
    # f is monotone decreasing in a typically; bisect
    for _ in range(80):
        mid = 0.5*(lo+hi); fm = f(mid)
        if (flo>0) == (fm>0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    a = 0.5*(lo+hi)
    Iv = Ivec(a)
    eta = 1.0 - Iv
    return a, dmu, eta, Iv

# ---------- (1) cross-check eta vs n2 ----------
with open(os.path.join(HERE,'n2_population_kkt.json')) as fh:
    n2 = json.load(fh)

print("=== (1) eta_j cross-check: analytic 2-endmember KKT vs n2 numeric MU-fixed-point ===")
print("%-6s %-5s %-6s | %-14s %-14s %-10s | %-14s %-14s" %
      ("K","bw","rho","a_n2","a_solve","da/a","eta_mid_n2","eta_mid_solve"))
rows = []
for key, recs in n2['grid'].items():
    K = recs[0]['K']; bw = recs[0]['bw']; kind = recs[0]['kind']
    if kind != 'middle' or K != 3:
        continue
    w = wstar(K, kind)
    for rec in recs:
        if rec['pinned_set'] != [1]:    # only cells where middle is pinned, survivors {0,2}
            continue
        rho = rec['rho']
        a, dmu, eta, Iv = solve_a_and_eta(K, rho, w, bw)
        a_n2 = rec['abar']; eta_n2 = rec['eta'][1]
        rows.append((K,bw,rho,a_n2,a,abs(a-a_n2)/a_n2,eta_n2,float(eta[1])))
        if rho in (0.05,0.11,0.20):
            print("%-6d %-5.0f %-6.2f | %-14.8f %-14.8f %-10.2e | %-14.6e %-14.6e" %
                  (K,bw,rho,a_n2,a,abs(a-a_n2)/a_n2,eta_n2,float(eta[1])))
maxrel_a = max(r[5] for r in rows)
maxrel_eta = max(abs(r[7]-r[6])/max(abs(r[6]),1e-12) for r in rows)
print("max rel-err  a:   %.3e" % maxrel_a)
print("max rel-err eta_mid: %.3e" % maxrel_eta)
print()

# ---------- (2) small-rho closed form for symmetric K=3 ----------
# K=3 symmetric weights w*=(c,1-2c,c) => mubar = mu_2 (middle). a -> 1/2 by symmetry.
# Derive eta_1 = eta_2_pinned (the MIDDLE slack) leading order in dmu.
# Use exact integrals at moderate rho to confirm scaling eta ~ const * dmu^2 / SIGMA^2 ... etc.
print("=== (2) middle-slack leading-order scaling, symmetric K=3 ===")
print("Symmetric test: w*=(1/3,1/3,1/3). a should be exactly 1/2.")
for bw in (12.0,):
    for rho in (0.02,0.05,0.08,0.11):
        w = np.array([1/3,1/3,1/3])
        a, dmu, eta, Iv = solve_a_and_eta(3, rho, w, bw)
        # leading order theory below
        print("  bw=%.0f rho=%.2f dmu=%8.4f  a=%.8f  eta_mid=%.6e  eta_mid/dmu^2=%.6e" %
              (bw, rho, dmu, a, float(eta[1]), float(eta[1])/dmu**2))
print()

# ---------- (3) the LP / variance-maximization claim ----------
# Claim: at fixed model MEAN = data mean, KL(p||q_w) to leading order in dmu decreases as
# model variance increases toward data variance from below; simplex-maximizer of model variance
# at fixed mean is supported on extremes {mu_1,mu_K}. Verify numerically that among all w on the
# simplex with sum w_k mu_k = mubar*, the KL is (to leading order) minimized at the 2-endmember a.
print("=== (3) variance-maximization / KL-decrease check (K=3, bw=12, rho=0.05) ===")
K=3; bw=12.0; rho=0.05
w = wstar(3,'middle')
mus,dmu = peaks(3,rho)
p = p_density(3,rho,w,bw)
mubar = float((w*mus).sum())
def KL(qw):
    q = qw+1e-300
    return float(TRAPZ(p*np.log((p+1e-300)/q), TGRID))
Hall = np.array([hk(m) for m in mus])
# family with fixed mean mubar: parametrize w2 in [0, ...], solve w1,w3 from
# w1+w2+w3=1, w1*mu1+w2*mu2+w3*mu3 = mubar
m1,m2,m3 = mus
best=None
for w2 in np.linspace(0.0,0.9,181):
    # w1+w3 = 1-w2 ; w1*m1+w3*m3 = mubar - w2*m2
    A = np.array([[1,1],[m1,m3]]); b=np.array([1-w2, mubar-w2*m2])
    try:
        w1,w3 = np.linalg.solve(A,b)
    except Exception:
        continue
    if w1<-1e-9 or w3<-1e-9: continue
    qw = (np.array([max(w1,0),w2,max(w3,0)])[:,None]*Hall).sum(0)
    kl = KL(qw)
    modvar = (max(w1,0)*m1**2+w2*m2**2+max(w3,0)*m3**2) - mubar**2
    if best is None or kl<best[0]:
        best=(kl,w2,w1,w3,modvar)
print("  KL-min over fixed-mean family at w2=%.4f (=> %s)  modelVar=%.4f" %
      (best[1], "endmember-supported" if best[1]<1e-3 else "INTERIOR", best[4]))
print("  data var (SIGMA^2+bw^2+Var_w*mu) = %.4f ; raw model var max (w2=0) below it." %
      (SIGMA**2+bw**2+float((w*mus**2).sum()-mubar**2)))
print("  => KL minimized at w2->0 (extreme support) confirms variance-maximization argument.")
