# -*- coding: utf-8 -*-
"""
probe_judge_synth.py -- INDEPENDENT synthesis-judge spot-check of claims (A)/(B)/(C).
Does NOT overwrite any diag_/scan file or out_phase. Real core via `import sigma_err_scan as S`.

Three lenses, all measured (not reasoned):
  (A) worst-mode degree: top tangent eigenvector overlap with orthonormal contrast deg 1..K-1.
      Confirm dom_deg == K-1 for K=2..5 at rho=0.47,n=350 (one fresh seed).
  (B) excess(K) monotonicity + K=7 turnover. excess = sig_err/(0.50/(sqrt(N)*rho)).
      K=2..7 at rho=0.47 AND rho=0.40, n=350, fresh seeds, with n_iter convergence guard at K=7.
  (C) bw sign. Sweep bw in {6,12,18,24,30} at K=4,5 rho=0.47 n=350. Claim says larger bw -> larger
      excess; skeptic says OPPOSITE. Build H bw-free (from sigma) and assert H invariant across bw.
"""
import math
import numpy as np
import sigma_err_scan as S

R, SIGMA, C_K2 = 600, S.SIGMA_DEFAULT, 0.500


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean()
    B = []
    for deg in range(1, K):
        v = x ** deg; v = v - v.mean()
        for b in B:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            B.append(v / nv)
    return B


def cell(K, rho, N, seed, bw, n_iter=2000, return_H=False):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)   # bw-free: built from sigma only
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=bw)
    what = core.supervised_unmix(X, H, n_iter=n_iter, tol=1e-13)
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)
    top = V[:, -1]
    sig = math.sqrt(max(lam[-1], 0.0))
    ov = [abs(float(top @ b)) for b in contrast_polys(K)]
    dom = int(np.argmax(ov)) + 1
    out = dict(sig=sig, excess=sig / (C_K2 / (math.sqrt(N) * rho)), dom=dom, ov=ov,
               l2l1=(max(lam[-2], 0.0) / max(lam[-1], 1e-300)))
    if return_H:
        out["H"] = H
    return out


print("=" * 70)
print("(A) worst-mode degree  rho=0.47 n=350 bw=12 (fresh seed 777)")
print("%2s | %7s %6s | %7s %5s | overlaps deg=1..K-1" % ("K", "sig", "exc", "dom", "=K-1?"))
for K in [2, 3, 4, 5]:
    r = cell(K, 0.47, 350, 777 + K * 31, 12)
    print("%2d | %7.4f %6.2f | deg=%d %5s | %s"
          % (K, r["sig"], r["excess"], r["dom"], "YES" if r["dom"] == K - 1 else "no",
             " ".join(f"{o:.2f}" for o in r["ov"])))

print("\n" + "=" * 70)
print("(B) excess(K) K=2..7, two rho, two seeds. Watch for K=7 turnover.")
for rho in [0.47, 0.40]:
    print(f"--- rho={rho} n=350 bw=12 ---")
    print("%2s | %s" % ("K", "excess per seed [s=11,s=22]  mean   dom_deg"))
    means = {}
    for K in [2, 3, 4, 5, 6, 7]:
        excs, doms = [], []
        for seed in [11, 22]:
            r = cell(K, rho, 350, 1000 * K + seed, 12)
            excs.append(r["excess"]); doms.append(r["dom"])
        m = float(np.mean(excs)); means[K] = m
        print("%2d | [%.3f, %.3f]  %.3f   deg=%d(K-1=%d)"
              % (K, excs[0], excs[1], m, doms[0], K - 1))
    incs = [means[k + 1] - means[k] for k in range(2, 7)]
    print("   increments K2->K7: " + " ".join(f"{d:+.3f}" for d in incs))
    print("   monotone up through K=7? " + ("YES" if all(d > 0 for d in incs) else "NO -> turnover"))

print("\n" + "=" * 70)
print("(B-guard) K=7 convergence: n_iter 2000 vs 6000 vs 12000 (turnover not under-conv?)")
for rho in [0.47, 0.40]:
    for nit in [2000, 6000, 12000]:
        r6 = cell(6, rho, 350, 6011, 12, n_iter=nit)
        r7 = cell(7, rho, 350, 7011, 12, n_iter=nit)
        print("  rho=%.2f nit=%5d | K6 exc=%.3f  K7 exc=%.3f  (K7<K6? %s)"
              % (rho, nit, r6["excess"], r7["excess"], "YES" if r7["excess"] < r6["excess"] else "no"))

print("\n" + "=" * 70)
print("(C) bw sign. Claim: larger bw -> LARGER excess. Skeptic: OPPOSITE.")
# H invariance check first
H6 = cell(4, 0.47, 350, 4099, 6, return_H=True)["H"]
H30 = cell(4, 0.47, 350, 4099, 30, return_H=True)["H"]
print("  H bw-invariant? max|H(bw6)-H(bw30)| = %.3e  (should be ~0)"
      % float(np.max(np.abs(H6 - H30))))
print("%2s | bw=6   bw=12  bw=18  bw=24  bw=30  | trend  dom@bw6 dom@bw30" % "K")
for K in [4, 5]:
    row, doms = [], {}
    for bw in [6, 12, 18, 24, 30]:
        r = cell(K, 0.47, 350, 5000 + K * 17, bw)
        row.append(r["excess"]); doms[bw] = r["dom"]
    trend = "DOWN" if all(row[i + 1] < row[i] for i in range(len(row) - 1)) else \
            ("UP" if all(row[i + 1] > row[i] for i in range(len(row) - 1)) else "mixed")
    print("%2d | %s | %s  deg=%d   deg=%d"
          % (K, "  ".join(f"{x:.3f}" for x in row), trend, doms[6], doms[30]))
print("\nDONE")
