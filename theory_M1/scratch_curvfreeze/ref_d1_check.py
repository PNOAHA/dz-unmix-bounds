"""
Referee independent re-derivation / re-check of D1-KKT.
ASCII-only. Creates a NEW file only. Does not modify any existing repo file.

We check, INDEPENDENTLY of the author's scripts:
  (A) The symmetric-K3 expansion q0 = N0*(1 + (d^2/2)(t^2/s^4 - 1/s^2) + O(d^4)).
  (B) The leading constant in eta_mid = d^2 * deficit / (2 s^4) + ...  (author claims 1/2).
      We extract the constant by Richardson extrapolation in d -> 0 AND we compute the
      O(d^2)-relative-correction sign, to test whether "1/2" is the true leading constant
      or whether the author's "0.499818" is actually approaching 1/2 from a biased grid.
  (C) The mean-matching value a = (muK - mubar)/(muK - mu1) from I_1 = I_K at leading order,
      for the ASYMMETRIC K=3 'middle' case, and compare to n2 abar / abar_pred.
  (D) eta_j vs n2 eta for several pinned cells (independent quadrature, high-accuracy
      Gauss-Hermite instead of trapezoid, to make sure the agreement is not grid-coincidence).
  (E) The deficit / moment criterion in physical units, and where it predicts release.
"""
import numpy as np, math, json, os
from scipy.special import erfinv as _erfinv
from numpy.polynomial.hermite_e import hermegauss  # probabilists' Hermite for Gaussian integrals

SIGMA = 25.0
START = 200.0
SQRT2 = math.sqrt(2.0)
HERE = os.path.dirname(os.path.abspath(__file__))

def dmu_of_rho(rho):
    return SIGMA*2.0*SQRT2*float(_erfinv(rho))

# ---- High-accuracy Gaussian integral helper via Gauss-Hermite (probabilists') ----
# E_{X~N(mu,s^2)}[f(X)] = sum_i wq_i f(mu + s*xq_i), with hermegauss nodes for weight exp(-x^2/2)/sqrt(2pi)
NQ = 200
_xq, _wq = hermegauss(NQ)
_wq = _wq/ math.sqrt(2*math.pi)  # normalize so sum wq = 1

def gauss_pdf(t, mu, s):
    return np.exp(-0.5*((t-mu)/s)**2)/(s*math.sqrt(2*math.pi))

# (A)+(B): symmetric K=3, exact via Gauss-Hermite over the data measure p.
# p = sum_k w*_k N(mu_k, seff). q0 = a h1 + (1-a) h3 (raw sigma). a=1/2 by symmetry.
# I2 = E_p[ h2(t)/q0(t) ].  Compute as sum over the 3 data components, each integrated by GH.
def I2_symmetric(c, d, bw):
    seff = math.sqrt(SIGMA**2 + bw**2)
    mus = np.array([START-d, START, START+d])
    w = np.array([c, 1-2*c, c])
    def h(mu, t): return gauss_pdf(t, mu, SIGMA)
    a = 0.5
    # integrate component by component: E_p[g] = sum_k w_k E_{N(mu_k,seff)}[g]
    total = 0.0
    for k in range(3):
        nodes = mus[k] + seff*_xq
        q0 = a*h(mus[0],nodes) + (1-a)*h(mus[2],nodes)
        g = h(mus[1],nodes)/q0
        total += w[k]*np.sum(_wq*g)
    return total

def eta_mid_sym(c, rho, bw):
    d = dmu_of_rho(rho)
    I2 = I2_symmetric(c, d, bw)
    eta = 1.0 - I2
    Varw = 2*c*d**2
    deficit = bw**2 + Varw - d**2   # = bw^2 - (1-2c) d^2
    return eta, d, deficit

print("=== (B) symmetric K=3: leading constant of eta_mid / (d^2 deficit / s^4) ===")
print("author claims this -> 1/2 = 0.5, c-independent. We Richardson-extrapolate in d->0.")
print("%-6s %-10s %-14s %-14s %-14s"%("c","rho","d","eta_mid","ratio_to_half"))
for c in (1/3, 0.25, 0.40):
    rhos = [0.02, 0.01, 0.005, 0.0025]
    ratios=[]
    for rho in rhos:
        eta,d,deff = eta_mid_sym(c, rho, 12.0)
        ratio = eta/(d**2*deff/SIGMA**4)
        ratios.append(ratio)
        print("%-6.3f %-10.4f %-14.6f %-14.6e %-14.8f"%(c,rho,d,eta,ratio))
    # Richardson: ratio(d) = R0 + A d^2 + ...  use two smallest d
    # d roughly halves each step; fit R0 from last two
    r1,r2 = ratios[-2], ratios[-1]
    # d ratio:
    d1 = dmu_of_rho(0.005); d2 = dmu_of_rho(0.0025)
    # ratio = R0 + A d^2 ; solve
    A = (r1-r2)/(d1**2-d2**2)
    R0 = r2 - A*d2**2
    print("   -> Richardson R0 (c=%.3f) = %.8f"%(c,R0))
    print()

# (C): asymmetric K=3 'middle' mean matching.
def wstar(K, kind):
    w = np.linspace(1.0, 0.6, K)
    if kind == 'middle' and K >= 3:
        w = np.roll(w, K//2)
    return w/w.sum()

print("=== (C) asymmetric K=3 'middle' mean-match a_pred vs index-mean formula ===")
w = wstar(3,'middle')
print("w*(middle) =", w)
# index means 0,1,2 ; mubar index = sum w*_k k
mbar_idx = float((w*np.array([0,1,2])).sum())
a_pred = (2 - mbar_idx)/2.0
print("index mubar =", mbar_idx, " a_pred=(2-mbar)/2 =", a_pred)
print("(n2 abar_pred reported 0.45833333) -> match:", abs(a_pred-0.45833333)<1e-6)

# (D): independent eta_j cross-check for a few pinned cells using GH quadrature.
# Solve a from I_1 = I_K (asymmetric), then eta_mid = 1 - I_2.
def I_vec_asym(a, mus, w, bw):
    seff = math.sqrt(SIGMA**2+bw**2)
    def h(mu,t): return gauss_pdf(t,mu,SIGMA)
    h1 = lambda t: h(mus[0],t); hK = lambda t: h(mus[-1],t)
    K = len(mus)
    out = np.zeros(K)
    for k in range(K):
        out[k]=0.0
    # E_p[h_j/q0] = sum_m w_m E_{N(mu_m,seff)}[h_j/q0]
    for m in range(K):
        nodes = mus[m]+seff*_xq
        q0 = a*h1(nodes)+(1-a)*hK(nodes)
        for j in range(K):
            out[j]+= w[m]*np.sum(_wq*(h(mus[j],nodes)/q0))
    return out

def solve_a(mus, w, bw):
    lo,hi=1e-6,1-1e-6
    def f(a):
        Iv=I_vec_asym(a,mus,w,bw); return Iv[0]-Iv[-1]
    flo=f(lo)
    for _ in range(100):
        mid=0.5*(lo+hi); fm=f(mid)
        if (flo>0)==(fm>0): lo,flo=mid,fm
        else: hi=mid
    a=0.5*(lo+hi)
    return a, I_vec_asym(a,mus,w,bw)

with open(os.path.join(HERE,'n2_population_kkt.json')) as fh:
    n2=json.load(fh)

print("\n=== (D) independent GH-quadrature eta_mid vs n2 (K=3 middle pinned cells) ===")
print("%-5s %-6s | %-14s %-14s %-10s | %-14s %-14s %-10s"%
      ("bw","rho","a_n2","a_GH","da/a","eta_n2","eta_GH","deta/eta"))
maxda=0.0; maxdeta=0.0
for key,recs in n2['grid'].items():
    if recs[0]['K']!=3 or recs[0]['kind']!='middle': continue
    bw=recs[0]['bw']
    wv=wstar(3,'middle')
    for rec in recs:
        if rec.get('pinned_set')!=[1]: continue
        rho=rec['rho']; d=dmu_of_rho(rho)
        mus=np.array([START, START+d, START+2*d])
        a,Iv=solve_a(mus,wv,bw)
        eta_mid=1.0-Iv[1]
        a_n2=rec['abar']; eta_n2=rec['eta'][1]
        da=abs(a-a_n2)/a_n2; deta=abs(eta_mid-eta_n2)/max(abs(eta_n2),1e-15)
        maxda=max(maxda,da); maxdeta=max(maxdeta,deta)
        if rho in (0.05,0.11,0.20):
            print("%-5.0f %-6.2f | %-14.8f %-14.8f %-10.2e | %-14.6e %-14.6e %-10.2e"%
                  (bw,rho,a_n2,a,da,eta_n2,eta_mid,deta))
print("max rel-err a (GH vs n2):   %.3e"%maxda)
print("max rel-err eta (GH vs n2): %.3e"%maxdeta)

# (E) sign of the O(d^2) relative correction to eta_mid (is 'hump' explained, and is 1/2 exact?)
print("\n=== (E) exact eta_mid/(d^2 deficit/s^4) at very small d, single c, high-order probe ===")
for c in (1/3,):
    for rho in (0.02,0.01,0.005,0.0025,0.00125):
        eta,d,deff=eta_mid_sym(c,rho,12.0)
        print("  rho=%.5f d=%.5f ratio=%.10f  (ratio-0.5)=%+.3e  /d^2=%+.4e"%
              (rho,d,eta/(d**2*deff/SIGMA**4), eta/(d**2*deff/SIGMA**4)-0.5,
               (eta/(d**2*deff/SIGMA**4)-0.5)/d**2))
