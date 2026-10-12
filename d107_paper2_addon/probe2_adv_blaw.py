# -*- coding: utf-8 -*-
"""
probe2_adv_blaw.py -- ADVERSARIAL LENS on b(rho)=0.679*(p/2)*<excess>.

Question: is b(rho)=0.679*(p/2)*<excess> a real predictive law, or an overfit /
near-tautology? It contains <excess> which is itself a summary statistic of the
SAME CR*S curve whose log-slope is b. So "b = k*(p/2)*<excess>" is structurally
"slope(curve) ~ k*(p/2)*mean(curve)" -- a relation between two summaries of ONE
curve. We stress it three ways:

  (A) TAUTOLOGY DIAGNOSTIC. For a smooth monotone curve on a fixed window K=2..7,
      how tightly is the ln(K-1)-slope b forced to track (p/2)*<excess> just by the
      shape, irrespective of rho? Compute the "geometric" coefficient
      k_geo(rho) = b_fit / [(p_meas/2)*<exc_meas>] per rho and see if it is a near
      CONSTANT (=> the 0.679 is a window/shape artifact, not a rho-law) or varies.

  (B) HELD-OUT PREDICTION from p(rho) ALONE. Anchor the two analytic ingredients on
      the TWO training rho only (0.40, 0.47, the prompt anchors):
        - p(rho) = c1/erfinv(rho)^2 + c0   refit on {0.40,0.47} K=2..7 windows
        - <excess>(rho): independent estimate. Two variants:
            (b1) interpolate <exc> linearly in p between the two anchors
                 <exc> ~ alpha + beta*p   (so the predictor uses ONLY p, no curve)
            (b2) "honest tautology": use the in-sample measured <exc> (leaks curve)
        Predict b at HELD-OUT rho in {0.32,0.35,0.45,0.55} and compare to the
        directly-fitted b. Report residuals for (b1) vs (b2). If (b1) (p-only) is
        far worse than (b2), the <excess> term is doing the work => circular.

  (C) IS b ~ c*p (NO <excess>) DISTINGUISHABLE? Fit b=c*p and b=k*(p/2)*<exc> on
      ALL rho; compare R2, AIC, and held-out RMSE. If the 2-ingredient law does not
      beat the 1-ingredient law out of sample, <excess> is not earning its place.

Engine: deterministic CR*S sandwich via full(K,rho) (same as probe_reconcile3.py,
verified <0.03% vs prompt targets). No Monte Carlo, no real-core dependence -- the
b(rho) law is a statement about the deterministic CR*S curve, so we test it there.

English prints. Standalone; does not overwrite existing files.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe2_adv_blaw.py
"""
import math
import numpy as np
from scipy.special import erfinv

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 0.5
TGRID = np.arange(50.0, 3000.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def dgauss(t, mu, s): return -(t - mu) / s ** 2 * gauss(t, mu, s)


def endmembers(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus])
    return h / TRAPZ(h, TGRID, axis=1)[:, None], mus


def endpoint_w(K):
    w = np.linspace(1.0, 0.6, K); return w / w.sum()


_KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
def smooth(f): return _KS @ f


def full(K, rho):
    """Return (lam_min, S) of the worst tangent G-mode -- identical to probe_reconcile3.full."""
    h, mus = endmembers(K, rho); w = endpoint_w(K)
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


KS = np.array(list(range(2, 8)), float)   # K=2..7 window (the fit window)
LNK = np.log(KS - 1.0)


def curve_summary(rho):
    """Return per-rho summary on K=2..7: b_fit (slope of a+b ln(K-1) on CR*S),
    p_meas (in-sample log-slope of lam_min/lam_min[2]), <exc> (window mean of CR*S)."""
    lm = np.array([full(int(K), rho)[0] for K in KS])
    S = np.array([full(int(K), rho)[1] for K in KS])
    CR = np.sqrt(lm[0] / lm)
    exc = CR * S
    A = np.vstack([np.ones(len(KS)), LNK]).T
    c, *_ = np.linalg.lstsq(A, exc, rcond=None)
    b_fit = c[1]
    p_meas = -np.polyfit(LNK, np.log(lm / lm[0]), 1)[0]
    exc_mean = exc.mean()
    return dict(b_fit=float(b_fit), p_meas=float(p_meas), exc_mean=float(exc_mean),
                exc=exc.tolist())


def p_analytic(rho, c1, c0):
    return c1 / erfinv(rho) ** 2 + c0


TRAIN = [0.40, 0.47]
HELD = [0.32, 0.35, 0.45, 0.55]
ALL = sorted(set(TRAIN + HELD + [0.30, 0.55]))


def main():
    print("=" * 96)
    print("ADVERSARIAL b(rho): is b=0.679*(p/2)*<excess> predictive or circular?")
    print("engine: deterministic CR*S sandwich, K=2..7 window; train rho=%s, held-out=%s"
          % (TRAIN, HELD))
    print("=" * 96)

    # ---- gather summaries for all rho we need
    rhos = sorted(set(TRAIN + HELD))
    summ = {rho: curve_summary(rho) for rho in rhos}

    # =========================================================================
    # (A) TAUTOLOGY DIAGNOSTIC: k_geo(rho) = b_fit / [(p_meas/2)*<exc_meas>]
    # =========================================================================
    print("\n" + "-" * 96)
    print("(A) GEOMETRIC/TAUTOLOGY coefficient  k_geo = b_fit / [(p_meas/2)*<exc>]  (uses in-sample p,<exc>)")
    print("    If k_geo is ~constant across rho, the '0.679' is a fixed window-SHAPE ratio, not a rho-law.")
    print("-" * 96)
    print("%5s | %8s %8s %8s | %8s" % ("rho", "b_fit", "p_meas", "<exc>", "k_geo"))
    kgeo = []
    for rho in rhos:
        s = summ[rho]
        kg = s["b_fit"] / ((s["p_meas"] / 2) * s["exc_mean"])
        kgeo.append(kg)
        print("%5.2f | %8.4f %8.4f %8.4f | %8.4f" % (rho, s["b_fit"], s["p_meas"], s["exc_mean"], kg))
    kgeo = np.array(kgeo)
    print("  k_geo: mean=%.4f  std=%.4f  CV=%.1f%%  range=[%.4f,%.4f]"
          % (kgeo.mean(), kgeo.std(), 100 * kgeo.std() / kgeo.mean(), kgeo.min(), kgeo.max()))
    print("  VERDICT(A): k_geo CV<~8%% => '0.679' is essentially a CONSTANT shape ratio")
    print("              (b is forced to ~track (p/2)*<exc> by curve geometry; rho enters via p,<exc> not k).")

    # =========================================================================
    # (B) HELD-OUT PREDICTION from p(rho) trained on 2 anchors only
    # =========================================================================
    print("\n" + "-" * 96)
    print("(B) HELD-OUT: anchor p(rho)=c1/erfinv^2+c0 on TRAIN {0.40,0.47} ONLY; predict b at held-out rho.")
    print("-" * 96)
    # fit p(rho) on the two training rho (p_meas there)
    x_tr = np.array([1.0 / erfinv(r) ** 2 for r in TRAIN])
    p_tr = np.array([summ[r]["p_meas"] for r in TRAIN])
    # 2 points, 2 params -> exact line through them
    c1 = (p_tr[1] - p_tr[0]) / (x_tr[1] - x_tr[0])
    c0 = p_tr[0] - c1 * x_tr[0]
    print("  p(rho) = %.4f/erfinv^2 + %.4f  (exact through train anchors; prompt window-fit ~0.241/-0.194)"
          % (c1, c0))
    # k coefficient fit on TRAIN only (b = k*(p/2)*<exc>); use analytic p, and <exc> at train
    k_tr = np.mean([summ[r]["b_fit"] / ((p_analytic(r, c1, c0) / 2) * summ[r]["exc_mean"]) for r in TRAIN])
    print("  k (deflation) fit on TRAIN only = %.4f" % k_tr)

    # <excess>(rho) INDEPENDENT estimate variants:
    # (b1) <exc> ~ alpha + beta*p  (linear in analytic p, anchored on TRAIN) -> predictor uses ONLY p
    e_tr = np.array([summ[r]["exc_mean"] for r in TRAIN])
    pa_tr = np.array([p_analytic(r, c1, c0) for r in TRAIN])
    beta = (e_tr[1] - e_tr[0]) / (pa_tr[1] - pa_tr[0])
    alpha = e_tr[0] - beta * pa_tr[0]
    print("  <exc>(p) = %.4f + %.4f*p   (anchored on TRAIN; makes predictor p-ONLY)" % (alpha, beta))

    print("\n  %5s | %8s | %10s %10s | %10s %10s"
          % ("rho", "b_fit", "b_pred_b1", "res%_b1", "b_pred_b2", "res%_b2"))
    res_b1, res_b2 = [], []
    for rho in HELD:
        s = summ[rho]
        b_true = s["b_fit"]
        pa = p_analytic(rho, c1, c0)
        # (b1) p-only: <exc> from the p-line
        exc_b1 = alpha + beta * pa
        b_b1 = k_tr * (pa / 2) * exc_b1
        # (b2) honest-tautology: use in-sample measured <exc> (leaks the curve)
        b_b2 = k_tr * (pa / 2) * s["exc_mean"]
        r1 = (b_b1 - b_true) / b_true * 100
        r2 = (b_b2 - b_true) / b_true * 100
        res_b1.append(r1); res_b2.append(r2)
        print("  %5.2f | %8.4f | %10.4f %9.1f%% | %10.4f %9.1f%%"
              % (rho, b_true, b_b1, r1, b_b2, r2))
    res_b1 = np.array(res_b1); res_b2 = np.array(res_b2)
    print("  RMSE%%(p-only b1)=%.1f%%   RMSE%%(leak-<exc> b2)=%.1f%%"
          % (np.sqrt((res_b1 ** 2).mean()), np.sqrt((res_b2 ** 2).mean())))
    print("  INTERPRETATION: if b1 >> b2, the held-out skill comes from <excess> (in-sample leak),")
    print("                  i.e. the 'law' needs the curve it claims to predict => CIRCULAR for prediction.")

    # =========================================================================
    # (C) IS b~c*p (NO <excess>) DISTINGUISHABLE FROM b=k*(p/2)*<exc> ?
    # =========================================================================
    print("\n" + "-" * 96)
    print("(C) MODEL COMPARISON on ALL rho: b=c*p  vs  b=k*(p/2)*<exc>  (in-sample fit quality + held-out)")
    print("-" * 96)
    allr = sorted(set(TRAIN + HELD))
    bb = np.array([summ[r]["b_fit"] for r in allr])
    pp = np.array([summ[r]["p_meas"] for r in allr])
    ee = np.array([summ[r]["exc_mean"] for r in allr])

    # Model 1: b = c*p (1 param)
    c_p = (bb * pp).sum() / (pp * pp).sum()
    pred1 = c_p * pp
    r2_1 = 1 - ((bb - pred1) ** 2).sum() / ((bb - bb.mean()) ** 2).sum()
    rss1 = ((bb - pred1) ** 2).sum()

    # Model 2: b = k*(p/2)*<exc> (1 param)
    x2 = (pp / 2) * ee
    k2 = (bb * x2).sum() / (x2 * x2).sum()
    pred2 = k2 * x2
    r2_2 = 1 - ((bb - pred2) ** 2).sum() / ((bb - bb.mean()) ** 2).sum()
    rss2 = ((bb - pred2) ** 2).sum()

    n = len(allr)
    def aic(rss, kpar): return n * math.log(rss / n) + 2 * kpar
    print("  Model 1  b=c*p          c=%.4f      R2=%.5f  RSS=%.4f  AIC=%.2f" % (c_p, r2_1, rss1, aic(rss1, 1)))
    print("  Model 2  b=k*(p/2)*<exc> k=%.4f     R2=%.5f  RSS=%.4f  AIC=%.2f" % (k2, r2_2, rss2, aic(rss2, 1)))
    print("  per-rho: %5s | %8s | %8s %8s | %8s %8s" % ("rho", "b_fit", "M1", "res%", "M2", "res%"))
    for i, r in enumerate(allr):
        print("           %5.2f | %8.4f | %8.4f %+7.1f%% | %8.4f %+7.1f%%"
              % (r, bb[i], pred1[i], (pred1[i] - bb[i]) / bb[i] * 100,
                 pred2[i], (pred2[i] - bb[i]) / bb[i] * 100))

    # Honest held-out for both models: train each on TRAIN, predict HELD
    print("\n  HELD-OUT (train on {0.40,0.47}, predict held-out):")
    bt = np.array([summ[r]["b_fit"] for r in TRAIN])
    pt = np.array([summ[r]["p_meas"] for r in TRAIN])
    et = np.array([summ[r]["exc_mean"] for r in TRAIN])
    c_p_tr = (bt * pt).sum() / (pt * pt).sum()
    x2t = (pt / 2) * et
    k2_tr = (bt * x2t).sum() / (x2t * x2t).sum()
    print("  %5s | %8s | %8s %8s | %8s %8s" % ("rho", "b_fit", "M1pred", "res%", "M2pred", "res%"))
    e1, e2 = [], []
    for r in HELD:
        s = summ[r]
        m1 = c_p_tr * s["p_meas"]
        m2 = k2_tr * (s["p_meas"] / 2) * s["exc_mean"]
        rr1 = (m1 - s["b_fit"]) / s["b_fit"] * 100
        rr2 = (m2 - s["b_fit"]) / s["b_fit"] * 100
        e1.append(rr1); e2.append(rr2)
        print("  %5.2f | %8.4f | %8.4f %+7.1f%% | %8.4f %+7.1f%%" % (r, s["b_fit"], m1, rr1, m2, rr2))
    e1 = np.array(e1); e2 = np.array(e2)
    print("  HELD-OUT RMSE%%:  M1 (c*p)=%.1f%%   M2 (k*(p/2)*<exc>)=%.1f%%"
          % (np.sqrt((e1 ** 2).mean()), np.sqrt((e2 ** 2).mean())))
    print("  Note: M1 and M2 both use the IN-SAMPLE measured p,<exc> at the held-out rho;")
    print("        this isolates whether <excess> adds info GIVEN p. (B) is the harder p-only test.")


if __name__ == "__main__":
    main()
