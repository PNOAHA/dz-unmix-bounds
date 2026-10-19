# -*- coding: utf-8 -*-
"""
probe2_clip_mechanism.py -- measure the ACTUAL clipping mechanism from the real core, to fix the
effective-K model. The naive "drop 1 endmember a fraction frac<.01 of the time" (probe2_clip_effectiveK)
under-predicts the K=7 reduction (R_pred 0.99 vs R_meas 0.69). Two suspected reasons:
  (1) frac<.01 is a PER-ENTRY fraction (over R*K w_hat entries); the per-REALIZATION boundary-hitting
      probability p_clip (>=1 entry clipped in that realization) is larger.
  (2) when the worst mode (degree-(K-1) contrast) is clipped, the active set can drop by MORE than one,
      and the variance projects onto a LOWER worst mode -> effective active K can be K-2, K-3...

This script measures, per realization (rho=0.40, n=350, endpoint w), via the real core:
  - n_clip = # of w_hat entries < tau   (tau in {.01,.05}) -> effective active K = K - n_clip
  - p_clip = P(>=1 entry clipped)        (per-realization, the right mixture weight)
  - the worst-mode sigma_err RESTRICTED to interior (no clip) vs clipped subensembles
  - excess of the K-n_clip active set, to test excess_clip^2 = sum_j P(n_clip=j)*exc_unc(K-j)^2

Run:   python probe2_clip_mechanism.py
"""
import math
import numpy as np
import sigma_err_scan as S

RHO, N, R, SIGMA, BW = 0.40, 350, 1500, S.SIGMA_DEFAULT, S.BW_DEFAULT


def what_for(K, rho, N, seed):
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


def worst_se(what):
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    return math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0))


def main():
    tau = 0.02
    print(f"CLIP MECHANISM  rho={RHO} n={N} R={R} tau={tau} (endpoint w)")
    print("%2s | %8s %8s %9s %9s | %9s %9s %9s | %9s"
          % ("K", "exc_all", "p_clip", "n_clip_mu", "frac_ent",
             "exc_int", "exc_clip", "exc_int/all", "Pn1"))
    se2 = None
    rows = {}
    for K in range(2, 9):
        what, w = what_for(K, RHO, N, 20260616 + K)
        se_all = worst_se(what)
        if K == 2: se2 = se_all
        exc_all = se_all / se2
        nclip = (what < tau).sum(1)                       # per-realization #clipped entries
        p_clip = float((nclip >= 1).mean())               # per-realization boundary prob
        frac_ent = float((what < tau).mean())             # per-ENTRY frac (= probe_turnover frac)
        nclip_mu = float(nclip.mean())
        # split sub-ensembles
        interior = what[nclip == 0]
        clipped = what[nclip >= 1]
        exc_int = (worst_se(interior) / se2) if len(interior) > 5 else float('nan')
        exc_clip = (worst_se(clipped) / se2) if len(clipped) > 5 else float('nan')
        Pn1 = float((nclip == 1).mean())
        rows[K] = dict(exc_all=exc_all, p_clip=p_clip, nclip_mu=nclip_mu, frac_ent=frac_ent,
                       exc_int=exc_int, exc_clip=exc_clip,
                       dist=np.bincount(nclip, minlength=K) / len(nclip))
        print("%2d | %8.3f %8.3f %9.3f %9.4f | %9.3f %9.3f %9.3f | %9.3f"
              % (K, exc_all, p_clip, nclip_mu, frac_ent, exc_int, exc_clip,
                 (exc_int / exc_all if exc_all else float('nan')), Pn1))

    # the interior (no-clip) subensemble excess SHOULD track the unconstrained sandwich.
    print("\n[interior subensemble = unconstrained?]  exc_int vs sandwich 1/1.44/1.91/2.37/2.77/3.09")
    sand = {2: 1.00, 3: 1.44, 4: 1.91, 5: 2.37, 6: 2.77, 7: 3.09}
    for K in range(2, 8):
        ei = rows[K]["exc_int"]
        print("  K=%d  exc_int=%.3f  sand=%.2f  ratio=%.3f" % (K, ei, sand[K], ei / sand[K]))

    # n_clip distribution (how many drop at once)
    print("\n[n_clip distribution P(n_clip=j)]")
    for K in range(2, 9):
        d = rows[K]["dist"]
        print("  K=%d: " % K + " ".join("%d:%.3f" % (j, d[j]) for j in range(min(len(d), 5))))


if __name__ == "__main__":
    main()
