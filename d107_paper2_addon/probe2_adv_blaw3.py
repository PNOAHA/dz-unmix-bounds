# -*- coding: utf-8 -*-
"""
probe2_adv_blaw3.py -- ADVERSARIAL b(rho), final LOO predictiveness verdict.

Strict leave-one-out CV on the law's domain rho in [0.32,0.47] (where the parent
'a+b ln(K-1)' form has R2>0.84). For each held-out rho, fit each model's single
coefficient on the OTHER rho only, predict b at the held-out rho. Report LOO RMSE%.

Models:
  M1  b = c * p
  M1b b = c * p^2                 (null: replace <exc> by p, since <exc>~a+bp)
  M2  b = k * (p/2) * <exc>       (THE LAW)
All use the in-sample MEASURED p and <exc> at the held-out rho (these are
deterministic curve summaries, available at any rho without fitting the law).
The test isolates: does the law's COEFFICIENT transfer across rho better than
the alternatives? A predictive law must have a stable, transferable coefficient.

Also: the (B)-style hardest test from round 1 but EXCLUDING the flat-window rho=0.55,
i.e. predict b purely from p(rho) [analytic] + <exc>(p) [linear in p], anchored on
{0.40,0.47}, evaluated at {0.32,0.35,0.45}.

Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_adv_blaw3.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None]
def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()
_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f


def full(K, rho):
    h = endmembers(K, rho); w = endpoint_w(K)
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    P = np.eye(K) - np.ones((K, K)) / K
    Gt = P @ G @ P
    lam, V = np.linalg.eigh(Gt)
    ones = np.ones(K) / math.sqrt(K)
    tang = [k for k in np.argsort(lam) if abs(V[:, k] @ ones) < 0.5]
    v = V[:, tang[0]]; lam_min = lam[tang[0]]
    phi = np.stack([smooth(h[i] / q) for i in range(K)])
    Gs = np.array([[TRAPZ(q * phi[i] * phi[j], TGRID) for j in range(K)] for i in range(K)])
    bvec = np.array([TRAPZ(q * phi[i], TGRID) for i in range(K)])
    S = math.sqrt(max(v @ Gs @ v - (v @ bvec) ** 2, 0) / lam_min)
    return lam_min, S


KS = np.array(list(range(2, 8)), float)
LNK = np.log(KS - 1.0)
def summary(rho):
    lm = np.array([full(int(K), rho)[0] for K in KS])
    Sv = np.array([full(int(K), rho)[1] for K in KS])
    exc = np.sqrt(lm[0] / lm) * Sv
    A = np.vstack([np.ones(len(KS)), LNK]).T
    c, *_ = np.linalg.lstsq(A, exc, rcond=None)
    p_meas = -np.polyfit(LNK, np.log(lm / lm[0]), 1)[0]
    return float(c[1]), float(p_meas), float(exc.mean())


def main():
    DOMAIN = [0.32, 0.35, 0.40, 0.45, 0.47]
    rec = {r: summary(r) for r in DOMAIN}
    b = np.array([rec[r][0] for r in DOMAIN])
    p = np.array([rec[r][1] for r in DOMAIN])
    e = np.array([rec[r][2] for r in DOMAIN])

    print("=" * 86)
    print("ROUND 3: leave-one-out CV on domain rho in [0.32,0.47]")
    print("=" * 86)

    models = {
        "M1  b=c*p":          lambda pp, ee: pp,
        "M1b b=c*p^2":        lambda pp, ee: pp ** 2,
        "M2  b=k*(p/2)*<exc>": lambda pp, ee: (pp / 2) * ee,
    }
    print("%-22s | %s" % ("model", " ".join("res%%@rho=%.2f" % r for r in DOMAIN)))
    loo = {}
    for name, xf in models.items():
        errs = []
        for i in range(len(DOMAIN)):
            tr = [j for j in range(len(DOMAIN)) if j != i]
            xtr = xf(p[tr], e[tr]); btr = b[tr]
            k = (btr * xtr).sum() / (xtr * xtr).sum()        # 1-param fit on train
            xte = xf(p[i:i+1], e[i:i+1])[0]
            bpred = k * xte
            errs.append((bpred - b[i]) / b[i] * 100)
        errs = np.array(errs)
        loo[name] = math.sqrt((errs ** 2).mean())
        print("%-22s | %s" % (name, " ".join("%+9.1f" % x for x in errs)))
    print("-" * 86)
    for name in models:
        print("LOO RMSE%%  %-22s = %6.1f%%" % (name, loo[name]))
    best = min(loo, key=loo.get)
    print("BEST LOO model: %s" % best)

    # hardest p-only test excluding flat rho=0.55
    print("\n" + "=" * 86)
    print("HARDEST: predict b from p(rho)[analytic] + <exc>(p)[linear in p], train {0.40,0.47}")
    print("=" * 86)
    TR = [0.40, 0.47]; HE = [0.32, 0.35, 0.45]
    xtr = np.array([1.0 / erfinv(r) ** 2 for r in TR]); ptr = np.array([rec[r][1] for r in TR])
    c1 = (ptr[1] - ptr[0]) / (xtr[1] - xtr[0]); c0 = ptr[0] - c1 * xtr[0]
    pa = lambda r: c1 / erfinv(r) ** 2 + c0
    etr = np.array([rec[r][2] for r in TR]); patr = np.array([pa(r) for r in TR])
    beta = (etr[1] - etr[0]) / (patr[1] - patr[0]); alpha = etr[0] - beta * patr[0]
    ktr = np.mean([rec[r][0] / ((pa(r) / 2) * rec[r][2]) for r in TR])
    print("  p(rho)=%.4f/erfinv^2%+.4f ; <exc>(p)=%.4f%+.4f*p ; k=%.4f" % (c1, c0, alpha, beta, ktr))
    print("  %5s | %8s %10s %8s" % ("rho", "b_fit", "b_pred(p-only)", "res%"))
    errs = []
    for r in HE:
        pap = pa(r); excp = alpha + beta * pap; bp = ktr * (pap / 2) * excp
        bt = rec[r][0]; rr = (bp - bt) / bt * 100; errs.append(rr)
        print("  %5.2f | %8.4f %10.4f %+8.1f%%" % (r, bt, bp, rr))
    print("  p-only held-out RMSE%% (excl flat rho=0.55) = %.1f%%" % math.sqrt(np.mean(np.array(errs) ** 2)))


if __name__ == "__main__":
    main()
