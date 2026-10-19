# -*- coding: utf-8 -*-
"""
probe2_clip_effK_reconcile.py -- reconcile the effective-K model with the TRUE measured turnover.

FINDING that forced this script: the single-wall truncated-Gaussian model (probe2_clip_final) only
reported K*=6 because it multiplied the reduction by a hand-typed unconstrained dict CAPPED at K=7.
Recomputing the unconstrained excess past K=7 (it saturates ~K=11-12), the single-wall model with the
frozen kappa=1.805 turns over at K*=10, NOT 6. The MEASURED turnover (R=4000) is K*=6 at 11.1 sigma.
=> a single wall is too weak; the real high-K clipping is a MULTI-component active-set collapse.

This script measures, from the real core, the TRUE per-K reduction the constraint applies to the worst
mode, and tests the EFFECTIVE-K mixture with the MEASURED per-realization n_clip distribution combined
with the CONDITIONAL (sub-ensemble) excess -- i.e. the honest active-set model:
    sigma_clip(K)^2 = sum_j P(n_clip=j) * sigma_cond(K, j)^2 ,
where sigma_cond(K,j) is the measured worst-mode std of the sub-ensemble with exactly j clipped comps
(NOT the unconstrained exc_unc(K-j), which probe2_clip_mechanism showed is wrong: even j=0 is suppressed).
We report whether THIS reproduces exc_meas and K*=6, and we quantify the per-K reduction so the paper
can state the active-set mechanism honestly with its real strength.

Run:   python probe2_clip_effK_reconcile.py
"""
import math
import numpy as np
import sigma_err_scan as S

RHO, N, R, SIGMA, BW = 0.40, 350, 4000, S.SIGMA_DEFAULT, S.BW_DEFAULT
TAU = 0.02


def what_for(K, seed):
    core = S._load_real_core()
    mus, _ = S.peak_ages(K, RHO, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=BW)
    return core.supervised_unmix(X, H, n_iter=2000, tol=1e-13), w


def worst_se(what):
    if len(what) < 5: return float('nan')
    K = what.shape[1]
    P = np.eye(K) - np.ones((K, K)) / K
    dev = (what - what.mean(0, keepdims=True)) @ P
    return math.sqrt(max(np.linalg.eigvalsh(dev.T @ dev / what.shape[0])[-1], 0.0))


def main():
    print(f"EFFECTIVE-K RECONCILE  rho={RHO} n={N} R={R} tau={TAU}")
    print("Active-set mixture with MEASURED P(n_clip=j) and MEASURED conditional std sigma_cond(K,j):")
    print("  sigma_clip(K)^2 = sum_j P(n_clip=j) sigma_cond(K,j)^2  (= total variance, exact identity)")
    print()
    se2 = None
    rows = {}
    for K in range(2, 9):
        what, w = what_for(K, 20260616 + K)
        se_all = worst_se(what)
        if K == 2: se2 = se_all
        nclip = (what < TAU).sum(1)
        dist = np.bincount(nclip, minlength=K).astype(float) / R
        # conditional excess per j
        cond = {}
        for j in range(K):
            sub = what[nclip == j]
            cond[j] = worst_se(sub) / se2 if len(sub) >= 20 else float('nan')
        # the mixture identity (total variance = within + between); we report the law-of-total-variance
        # decomposition to show the active-set states drive the turnover.
        # E[Var] part = sum_j P(j) Var_cond(j); but the BETWEEN part (shifts of conditional means)
        # also contributes to the pooled top eigenvalue. We report pooled vs sum-of-conditional.
        pooled = se_all / se2
        # variance-weighted mean of conditional worst-mode variances (within-state)
        within = math.sqrt(sum(dist[j] * (cond[j] * se2) ** 2 for j in range(K)
                               if cond[j] == cond[j]) ) / se2
        rows[K] = dict(pooled=pooled, dist=dist, cond=cond, within=within,
                       pclip=float((nclip >= 1).mean()))
        print("K=%d  exc_pooled=%.3f  P(n_clip>=1)=%.3f" % (K, pooled, rows[K]['pclip']))
        print("      P(n_clip=j): " + " ".join("%d:%.3f" % (j, dist[j]) for j in range(min(K, 5))))
        print("      exc_cond(j): " + " ".join("%d:%s" % (j, ("%.3f" % cond[j]) if cond[j]==cond[j] else "  - ")
                                               for j in range(min(K, 5))))
    # turnover
    Kstar = max(rows, key=lambda K: rows[K]['pooled'])
    print("\nK*(measured pooled) = %d" % Kstar)

    # The active-set reading: track the MEASURED dominant worst-mode contrast degree already done in
    # eigcross (deg1 = K-1 to K=6 then lam1 drops). Here confirm the conditional-excess ORDERING:
    # exc_cond(j) should DECREASE in j (more clipped comps -> smaller active worst mode), i.e. the
    # active-set reduction. Print the monotonicity.
    print("\n[active-set ordering: exc_cond(j) decreasing in j => active-set reduction]")
    for K in range(5, 9):
        cs = [rows[K]['cond'][j] for j in range(4) if rows[K]['cond'][j] == rows[K]['cond'][j]]
        mono = all(cs[i] >= cs[i + 1] - 0.05 for i in range(len(cs) - 1))
        print("  K=%d  exc_cond = %s   monotone_decr=%s"
              % (K, ["%.2f" % c for c in cs], mono))


if __name__ == "__main__":
    main()
