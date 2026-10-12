# -*- coding: utf-8 -*-
"""
probe_excess_growth.py -- LENS: excess_growth_law (skeptic).

Characterize excess(K) = sigma_err(K) / [C/(sqrt(N)*rho)] quantitatively in the
UNPINNED regime, using the REAL dz-unmix-cg core via 'import sigma_err_scan as S'.

- Grid: K = 2..7, at rho in {0.47, 0.40}, n=350, w=endpoint (matches diag_kge4_*).
- Seeds: 3 per cell (independent reps each R=600), report mean +/- spread.
- Floor C: K=2 measured pairwise floor C=0.50 (consistent with diag scripts).
- Fits: excess(K) vs candidate forms
    linear : a + b*K
    sqrt   : a + b*sqrt(K-1)
    log    : a + b*ln(K-1)
  Report R^2 each, pick best, and judge saturate-vs-grow via the K=6->7 increment.
- Also report rho-dependence: excess(rho=0.40) vs excess(rho=0.47) per K.

English prints. Does NOT modify any diag/scan file or out_phase. Standalone driver.
Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_excess_growth.py
"""
import math, time, json
import numpy as np
import sigma_err_scan as S

R = 600
SIGMA, BW = S.SIGMA_DEFAULT, S.BW_DEFAULT
C_K2 = 0.500
KS = [2, 3, 4, 5, 6, 7]
RHOS = [0.47, 0.40]
N = 350
SEEDS = [101, 202, 303]


def sigma_err_cell(K, rho, N, seed, n_iter=2000):
    """sigma_err from the real core, endpoint true w (matches diag_kge4_*)."""
    s = S.measure_cell("numpy", K, rho, N, R, S.fixed_interior_w(K, "endpoint"),
                       sigma=SIGMA, bw=BW, noise=0.0, seed=seed, n_iter=n_iter)
    return s["sigma_err"]


def excess_cell(K, rho, N, seed, n_iter=2000):
    floor = C_K2 / (math.sqrt(N) * rho)
    return sigma_err_cell(K, rho, N, seed, n_iter) / floor


def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_forms(Ks, ex):
    """Fit linear a+bK, sqrt a+b*sqrt(K-1), log a+b*ln(K-1). Return params, R2, RMSE."""
    Ks = np.asarray(Ks, float); ex = np.asarray(ex, float)
    out = {}
    # linear
    b, a = np.polyfit(Ks, ex, 1)
    yh = a + b * Ks
    out["linear"] = dict(a=float(a), b=float(b), R2=r2(ex, yh),
                         rmse=float(np.sqrt(np.mean((ex - yh) ** 2))),
                         form="a + b*K")
    # sqrt
    x = np.sqrt(Ks - 1.0)
    b, a = np.polyfit(x, ex, 1)
    yh = a + b * x
    out["sqrt"] = dict(a=float(a), b=float(b), R2=r2(ex, yh),
                       rmse=float(np.sqrt(np.mean((ex - yh) ** 2))),
                       form="a + b*sqrt(K-1)")
    # log
    x = np.log(Ks - 1.0)
    b, a = np.polyfit(x, ex, 1)
    yh = a + b * x
    out["log"] = dict(a=float(a), b=float(b), R2=r2(ex, yh),
                      rmse=float(np.sqrt(np.mean((ex - yh) ** 2))),
                      form="a + b*ln(K-1)")
    return out


def main():
    results = {}
    print("=" * 92)
    print("excess(K) = sigma_err / [%.3f/(sqrt(N)*rho)]   N=%d, w=endpoint, R=%d, seeds=%s"
          % (C_K2, N, R, SEEDS))
    print("=" * 92)
    for rho in RHOS:
        print("\n--- rho = %.2f ---" % rho)
        print("%2s | %s | %8s %8s %8s | %8s"
              % ("K", "  excess per seed (101/202/303) ", "mean", "std", "rng", "sig_err_mean"))
        Ks_ok, ex_mean = [], []
        per_cell = {}
        for K in KS:
            t0 = time.time()
            exs, sigs = [], []
            for sd in SEEDS:
                seed = 20260615 + K * 1000 + sd + int(rho * 1e4)
                sig = sigma_err_cell(K, rho, N, seed)
                floor = C_K2 / (math.sqrt(N) * rho)
                exs.append(sig / floor); sigs.append(sig)
            exs = np.array(exs)
            print("%2d | %6.3f %6.3f %6.3f               | %8.3f %8.4f %8.4f | %8.5f  (%.0fs)"
                  % (K, exs[0], exs[1], exs[2], exs.mean(), exs.std(ddof=0),
                     exs.max() - exs.min(), np.mean(sigs), time.time() - t0))
            Ks_ok.append(K); ex_mean.append(float(exs.mean()))
            per_cell[K] = dict(excess_seeds=[float(e) for e in exs],
                               excess_mean=float(exs.mean()),
                               excess_std=float(exs.std(ddof=0)),
                               sigma_err_mean=float(np.mean(sigs)))
        # monotonicity
        diffs = np.diff(ex_mean)
        mono = bool(np.all(diffs > 0))
        # fits
        fits = fit_forms(Ks_ok, ex_mean)
        best = max(fits, key=lambda k: fits[k]["R2"])
        print("  monotone increasing in K (mean): %s   increments dK: %s"
              % (mono, " ".join("%+.3f" % d for d in diffs)))
        for name in ["linear", "sqrt", "log"]:
            f = fits[name]
            print("  fit %-7s %-16s a=%+.4f b=%+.4f  R2=%.5f  rmse=%.4f"
                  % (name, f["form"], f["a"], f["b"], f["R2"], f["rmse"]))
        print("  BEST FIT: %s (%s)  R2=%.5f" % (best, fits[best]["form"], fits[best]["R2"]))
        # saturation diagnostic: K=6->7 increment vs K=2->3
        d23 = ex_mean[1] - ex_mean[0]
        d67 = ex_mean[-1] - ex_mean[-2]
        print("  saturation check: d(2->3)=%+.3f  d(6->7)=%+.3f  ratio=%.3f -> %s"
              % (d23, d67, (d67 / d23 if d23 else float("nan")),
                 "SATURATING (decreasing increments)" if d67 < d23 else "still growing ~linearly"))
        results["rho=%.2f" % rho] = dict(Ks=Ks_ok, excess_mean=ex_mean, monotone=mono,
                                         increments=[float(d) for d in diffs],
                                         fits=fits, best=best, per_cell=per_cell)

    # rho dependence
    print("\n--- rho dependence: excess(0.40) - excess(0.47) per K ---")
    r47 = results["rho=0.47"]; r40 = results["rho=0.40"]
    print("%2s | %8s %8s %8s" % ("K", "ex@0.47", "ex@0.40", "diff(40-47)"))
    for i, K in enumerate(KS):
        e47, e40 = r47["excess_mean"][i], r40["excess_mean"][i]
        print("%2d | %8.3f %8.3f %+8.3f" % (K, e47, e40, e40 - e47))
    mean_abs_diff = float(np.mean(np.abs(np.array(r40["excess_mean"]) - np.array(r47["excess_mean"]))))
    print("  mean |excess(0.40)-excess(0.47)| over K = %.3f" % mean_abs_diff)

    with open("probe_excess_growth_out.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nwrote probe_excess_growth_out.json")


if __name__ == "__main__":
    main()
