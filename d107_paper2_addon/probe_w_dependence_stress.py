# -*- coding: utf-8 -*-
"""
probe_w_dependence_stress.py -- SKEPTIC follow-up: try to BREAK (A)/(B) with stress w-vectors.

Beyond the 4 standard families, test edge-of-pathological true-w for K=4,5 at rho=0.47,n=350:
  near_drop_one : one component tiny (0.02) -> effectively K-1 active endmembers
  strong_skew   : heavily front-loaded geometric weights (ratio ~10x across the simplex)
  two_block     : two clusters of weight (could excite a lower-degree contrast)
  random_dirichlet x3 : random simplex points (Dirichlet alpha=1), to sample generic w

For each: dominant contrast degree of top tangent eigenvector + excess.
If dom_deg DROPS below K-1 for near_drop_one (pathological) but holds for the rest,
that DELIMITS the qualification, not refutes the claim.
Reuses real core. Run: & "C:/Users/Administrator/miniconda3/python.exe" probe_w_dependence_stress.py
"""
import math
import numpy as np
import sigma_err_scan as S

R, SIGMA, BW, C_K2 = 600, S.SIGMA_DEFAULT, S.BW_DEFAULT, 0.500
RHO, N = 0.47, 350


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


def stress_w(K, rng):
    fams = {}
    # one component tiny -> effectively K-1 endmembers active
    w = np.full(K, (1.0 - 0.02) / (K - 1)); w[K // 2] = 0.02
    fams["near_drop_one"] = w / w.sum()
    # strong geometric skew
    g = np.array([0.55 ** i for i in range(K)], dtype=float)
    fams["strong_skew"] = g / g.sum()
    # two-block
    tb = np.ones(K); tb[: K // 2] = 3.0
    fams["two_block"] = tb / tb.sum()
    # random Dirichlet points
    for j in range(3):
        d = rng.dirichlet(np.ones(K))
        fams[f"dirichlet{j}"] = d
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
    return sig, excess, dom_deg, ov


print(f"STRESS w_dependence  rho={RHO} n={N} R={R}")
print("%2s %-14s | %8s %6s | %7s %5s | overlaps deg=1..K-1   (w rounded)"
      % ("K", "w_family", "sig_err", "excess", "dom_deg", "=K-1?"))
brk = []
for K in [4, 5]:
    rng = np.random.default_rng(7000 + K)
    fams = stress_w(K, rng)
    for fam, w in fams.items():
        sig, exc, dom, ov = measure(K, RHO, N, w, 20260615 + K * 1000 + 303)
        ok = "YES" if dom == K - 1 else "NO"
        if dom != K - 1:
            brk.append((K, fam, dom))
        print("%2d %-14s | %8.4f %6.2f | deg=%d %5s | %s   w=%s"
              % (K, fam, sig, exc, dom, ok, " ".join(f"{o:.2f}" for o in ov),
                 np.array2string(w, precision=2, suppress_small=True)))
    print()

print("=" * 70)
if brk:
    print("dom_deg != K-1 in:", brk)
else:
    print("dom_deg == K-1 held even under all stress/pathological w (A robust).")
