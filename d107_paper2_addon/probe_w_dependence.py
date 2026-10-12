# -*- coding: utf-8 -*-
"""
probe_w_dependence.py -- SKEPTIC lens "w_dependence".

Tests whether claim (A) worst-mode = degree-(K-1) contrast AND claim (B) excess grows with K
SURVIVE across DIFFERENT TRUE WEIGHT VECTORS w, for K=4 and K=5 at rho=0.47, n=350.

Four true-w families per K:
  endpoint   : S.fixed_interior_w(K,'endpoint')  (linspace 1.0->0.6, max at index 0)
  middle     : S.fixed_interior_w(K,'middle')    (same, rolled so max is interior)
  uniform    : [1/K, ..., 1/K]
  asymmetric : linspace 1.0->0.2 then np.roll by 1 (asymmetric, max NOT at an endpoint or center)

For each (K, w) we measure:
  - sig_err = sqrt(top tangent-cov eigenvalue) over R reps  (same estimator as _summarize)
  - excess  = sig_err / (C_K2/(sqrt(N)*rho)),  C_K2=0.50  (the K=2 pairwise floor)
  - dominant contrast degree of the top tangent eigenvector (Gram-Schmidt contrast basis 1..K-1)
  - full overlap vector per degree

Verdict logic:
  (A) holds for a w if dom_deg == K-1.
  (B) holds if, for EACH w-family, excess(K=4) > 1 and excess(K=5) > 1 AND excess(K=5) > excess(K=4).

Reuses sigma_err_scan real dz-unmix-cg core (read-only). Does NOT overwrite any diag/scan file.
Run:  & "C:/Users/Administrator/miniconda3/python.exe" probe_w_dependence.py
"""
import math
import numpy as np
import sigma_err_scan as S

R, SIGMA, BW, C_K2 = 600, S.SIGMA_DEFAULT, S.BW_DEFAULT, 0.500
RHO, N = 0.47, 350


def contrast_polys(K):
    """orthonormal sum-zero contrast basis, degree 1..K-1 (Gram-Schmidt of monomials)."""
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


def w_families(K):
    fams = {}
    fams["endpoint"] = S.fixed_interior_w(K, "endpoint")
    fams["middle"] = S.fixed_interior_w(K, "middle")
    fams["uniform"] = np.full(K, 1.0 / K)
    asym = np.linspace(1.0, 0.2, K)        # steeper, more asymmetric
    asym = np.roll(asym, 1)                 # roll so max is not at an endpoint
    fams["asym_rolled"] = asym / asym.sum()
    return fams


def measure(K, rho, N, w, seed, n_iter=2000):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    what = core.supervised_unmix(X, H, n_iter=n_iter, tol=1e-13)
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)
    top = V[:, -1]
    Bpoly = contrast_polys(K)
    ov = [abs(float(top @ b)) for b in Bpoly]
    sig = math.sqrt(max(lam[-1], 0.0))
    dom_deg = int(np.argmax(ov)) + 1
    excess = sig / (C_K2 / (math.sqrt(N) * rho))
    l2l1 = (max(lam[-2], 0.0) / max(lam[-1], 1e-300)) if lam[-1] > 0 else float("nan")
    return dict(sig=sig, excess=excess, dom_deg=dom_deg, ov=ov, l2l1=l2l1)


print(f"w_dependence probe  rho={RHO} n={N} R={R} sigma={SIGMA} bw={BW} C_K2={C_K2}")
print("claim (A): dom_deg should == K-1 for all non-pathological w")
print("claim (B): excess>1 and K-increasing for each w-family")
print()
print("%2s %-12s %5s %5s | %8s %6s | %7s %5s | %5s | overlaps deg=1..K-1"
      % ("K", "w_family", "rho", "seed", "sig_err", "excess", "dom_deg", "=K-1?", "l2/l1"))

results = {}   # (K, fam) -> excess, dom_deg
for K in [4, 5]:
    fams = w_families(K)
    for fam, w in fams.items():
        for seed in [101, 202]:
            r = measure(K, RHO, N, w, 20260615 + K * 1000 + seed)
            ok = "YES" if r["dom_deg"] == K - 1 else "NO"
            print("%2d %-12s %5.2f %5d | %8.4f %6.2f | deg=%d %5s | %5.2f | %s"
                  % (K, fam, RHO, seed, r["sig"], r["excess"], r["dom_deg"], ok, r["l2l1"],
                     " ".join(f"{o:.2f}" for o in r["ov"])))
            results.setdefault((K, fam), []).append((r["excess"], r["dom_deg"]))
    print()

# ----- aggregate verdict -----
print("=" * 70)
print("AGGREGATE (seed-averaged excess, dom_deg per seed):")
agg = {}
for (K, fam), rows in results.items():
    exc = np.mean([x[0] for x in rows])
    degs = [x[1] for x in rows]
    agg[(K, fam)] = exc
    deg_ok = all(d == K - 1 for d in degs)
    print(f"  K={K} {fam:12s} excess_mean={exc:5.2f}  dom_degs={degs}  all=K-1? {deg_ok}")

print()
print("Claim (B) per-family K-monotonicity (excess(K5) > excess(K4) and both > 1):")
for fam in ["endpoint", "middle", "uniform", "asym_rolled"]:
    e4 = agg[(4, fam)]; e5 = agg[(5, fam)]
    mono = e5 > e4
    gt1 = e4 > 1.0 and e5 > 1.0
    print(f"  {fam:12s}: excess K4={e4:5.2f} K5={e5:5.2f}  | K5>K4? {mono}  both>1? {gt1}")

print()
print("Claim (A) degree summary (dom_deg == K-1 for all w, both seeds):")
all_A = all(all(d == K - 1 for d in [x[1] for x in results[(K, fam)]])
            for K in [4, 5] for fam in ["endpoint", "middle", "uniform", "asym_rolled"])
print(f"  ALL w-families dom_deg == K-1 (A holds universally)? {all_A}")
