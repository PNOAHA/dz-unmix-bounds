# -*- coding: utf-8 -*-
"""
probe2_clip_eigcross.py -- is the K~7 turnover a real DECLINE or saturation? And is the mechanism a
top-eigenvalue CROSSOVER (clipping kills the degree-(K-1) mode so the measured worst mode becomes the
2nd, lower-degree contrast)?

The truncation model (probe2_clip_truncation) reproduces the conditional excess shape to 6% but only
SATURATES (K*=8), while probe_turnover shows a DROP at K=7 (2.44->2.13). Two questions:
  (Q1) Is that drop statistically real or R-noise?  -> rerun at large R, report top eigenvalue + its
       jackknife/bootstrap sd.
  (Q2) Mechanism of a true turnover: clipping ASYMMETRICALLY truncates the degree-(K-1) worst mode;
       once truncated below the 2nd eigenvalue, the MEASURED top eigenvalue is the (un-clipped, lower-
       degree) 2nd mode. So excess_clip(K) = max over modes of [exc_mode_m * T(z_m)], and the turnover
       is where mode-(K-1) gets truncated below mode-(K-2): a TOP-EIGENVALUE CROSSOVER.

Measures, real core rho=0.40 endpoint w, large R: top-2 tangent eigenvalues of Cov(w_hat), their
bootstrap sd, and the dominant contrast degree of the top eigenvector (to see the degree drop at K*).

Run:   python probe2_clip_eigcross.py
"""
import math
import numpy as np
import sigma_err_scan as S

RHO, N, R, SIGMA, BW = 0.40, 350, 4000, S.SIGMA_DEFAULT, S.BW_DEFAULT


def contrast_polys(K):
    x = np.arange(1, K + 1, dtype=float); x = x - x.mean(); B = []
    for d in range(1, K):
        v = x ** d; v -= v.mean()
        for b in B: v = v - (v @ b) * b
        if np.linalg.norm(v) > 1e-12: B.append(v / np.linalg.norm(v))
    return B


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


def top2_and_deg(what):
    P = np.eye(what.shape[1]) - np.ones((what.shape[1],) * 2) / what.shape[1]
    dev = (what - what.mean(0, keepdims=True)) @ P
    cov = dev.T @ dev / what.shape[0]
    lam, V = np.linalg.eigh(cov)
    K = what.shape[1]
    B = contrast_polys(K)
    def deg(v): return int(np.argmax([abs(v @ b) for b in B])) + 1
    return lam[-1], (lam[-2] if K >= 3 else 0.0), deg(V[:, -1]), (deg(V[:, -2]) if K >= 3 else 0)


def boot_se(what, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    R_ = what.shape[0]; K = what.shape[1]
    P = np.eye(K) - np.ones((K, K)) / K
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, R_, R_)
        dev = (what[idx] - what[idx].mean(0, keepdims=True)) @ P
        cov = dev.T @ dev / R_
        vals.append(math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0)))
    return float(np.std(vals))


def main():
    print(f"EIG-CROSSOVER / turnover-reality  rho={RHO} n={N} R={R} (endpoint w)")
    print("%2s | %10s %10s | %8s %8s | %6s %6s | %8s"
          % ("K", "lam1", "lam2", "sqrt_l1", "sqrt_l2", "deg1", "deg2", "se_boot"))
    se2 = None
    excs, errs = {}, {}
    for K in range(2, 9):
        what, w = what_for(K, 20260616 + K)
        l1, l2, d1, d2 = top2_and_deg(what)
        s1, s2 = math.sqrt(max(l1, 0)), math.sqrt(max(l2, 0))
        if K == 2: se2 = s1
        excs[K] = s1 / se2
        sb = boot_se(what) / se2
        errs[K] = sb
        print("%2d | %10.3e %10.3e | %8.4f %8.4f | %6d %6d | %8.4f"
              % (K, l1, l2, s1, s2, d1, d2, sb))
    print("\n[excess +/- bootstrap sd, is K=7<K=6 real?]")
    for K in range(2, 9):
        print("  K=%d  exc=%.3f +/- %.3f" % (K, excs[K], errs[K]))
    Kstar = max(excs, key=excs.get)
    print("  K*(measured, R=%d) = %d" % (R, Kstar))
    # significance of the drop at the measured peak
    Kp = Kstar
    if Kp + 1 in excs:
        d = excs[Kp] - excs[Kp + 1]
        sd = math.sqrt(errs[Kp] ** 2 + errs[Kp + 1] ** 2)
        print("  drop exc(%d)-exc(%d) = %.3f  (%.1f sigma)" % (Kp, Kp + 1, d, d / sd if sd else 0))


if __name__ == "__main__":
    main()
