# -*- coding: utf-8 -*-
"""
diag_seed_stability.py -- SKEPTIC seed_stability lens.

Re-runs diag_k3_mode.py logic for the LARGE-rho cells (rho=0.47 n=350; rho=0.35 n=350)
under 3 DIFFERENT random seeds x R in {300, 1200}, to test whether the
"structural curvature-dominated" claim is robust or a sampling artifact.

For each (rho, N, seed, R) it forms Cov(what) over R reps and reports:
  - sqrt(top eigenvalue) = sigma_err (worst tangent direction)
  - |top.u1| (pairwise overlap), |top.u2| (curvature overlap)
  - C_u1, C_u2  = sqrt(var along u_i)*sqrt(N)*rho  (structural prefactor along each mode)
  - C_top = sigma_err*sqrt(N)*rho                  (prefactor of the worst mode)

Claim under test (large-rho regime):
  |top.u2| ~ 1.0, |top.u1| ~ 0.05, C_curv ~ 0.70 +/- 0.1, stable across seeds/R.
"""
import math
import numpy as np
import sigma_err_scan as S

K, SIGMA, BW = 3, S.SIGMA_DEFAULT, S.BW_DEFAULT
CELLS = [(0.47, 350), (0.35, 350)]
SEEDS = [11111, 22222, 33333]   # three DIFFERENT seeds (unrelated to diag_k3_mode's derived seed)
RS = [300, 1200]

u1 = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)   # extreme pairwise
u2 = np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0)   # curvature


def what_for(rho, N, seed, R):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, rho, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    return core.supervised_unmix(X, H, n_iter=2000, tol=1e-13), w


print("seed_stability sweep  (K=3, w=endpoint, u1=pairwise, u2=curvature)")
print("%5s %4s %6s %5s | %8s | %5s %5s | %6s %6s | %6s"
      % ("rho", "n", "seed", "R", "sig_err", "|.u1|", "|.u2|", "C_u1", "C_u2", "C_top"))

agg = {}  # (rho,N) -> lists
for rho, N in CELLS:
    for R in RS:
        for seed in SEEDS:
            what, w = what_for(rho, N, seed, R)
            dev = what - what.mean(0, keepdims=True)
            cov = dev.T @ dev / what.shape[0]
            lam, V = np.linalg.eigh(cov)
            top = V[:, -1]
            o1, o2 = abs(top @ u1), abs(top @ u2)
            sig_err = math.sqrt(max(lam[-1], 0.0))
            var_u1 = float(u1 @ cov @ u1); var_u2 = float(u2 @ cov @ u2)
            C_u1 = math.sqrt(max(var_u1, 0)) * math.sqrt(N) * rho
            C_u2 = math.sqrt(max(var_u2, 0)) * math.sqrt(N) * rho
            C_top = sig_err * math.sqrt(N) * rho
            print("%5.2f %4d %6d %5d | %8.5f | %5.2f %5.2f | %6.3f %6.3f | %6.3f"
                  % (rho, N, seed, R, sig_err, o1, o2, C_u1, C_u2, C_top))
            agg.setdefault((rho, N), {"o2": [], "C_u2": [], "C_top": [], "o1": []})
            agg[(rho, N)]["o2"].append(o2)
            agg[(rho, N)]["o1"].append(o1)
            agg[(rho, N)]["C_u2"].append(C_u2)
            agg[(rho, N)]["C_top"].append(C_top)

print("\n--- aggregate across 6 runs (3 seeds x 2 R) per cell ---")
for (rho, N), d in agg.items():
    o2 = np.array(d["o2"]); o1 = np.array(d["o1"])
    cu2 = np.array(d["C_u2"]); ctop = np.array(d["C_top"])
    print(f"rho={rho:.2f} n={N}:  |top.u2| mean={o2.mean():.3f} sd={o2.std():.3f} min={o2.min():.3f} max={o2.max():.3f}")
    print(f"             |top.u1| mean={o1.mean():.3f} sd={o1.std():.3f} max={o1.max():.3f}")
    print(f"             C_u2     mean={cu2.mean():.3f} sd={cu2.std():.3f} range=[{cu2.min():.3f},{cu2.max():.3f}]")
    print(f"             C_top    mean={ctop.mean():.3f} sd={ctop.std():.3f} range=[{ctop.min():.3f},{ctop.max():.3f}]")
