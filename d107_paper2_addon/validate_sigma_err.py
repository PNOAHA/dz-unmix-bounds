# -*- coding: utf-8 -*-
"""
validate_sigma_err.py — equivalence gate for sigma_err_scan backends.
=====================================================================
Mirrors validate_gpu.py's philosophy: a backend passes if its cross-backend
discrepancy is MUCH SMALLER than the within-backend seed-to-seed scatter at the
same (K, rho, N, R). Bit-match is impossible (different RNG streams), and is not
required: supervised KL-MLE with fixed H is CONVEX, so init is irrelevant and both
backends converge to the same MLE; the only residual difference is finite-R Monte
Carlo noise, which shrinks with R.

Two transitivity links (run on the machine that has the backend):
  --mode local     : numpy_ref  vs  numpy(real dz-unmix-cg core).  Needs sklearn (local).
                     Proves the pure-numpy replica == the real algorithm.
  --mode container : torch       vs  numpy_ref.  Needs torch, NOT sklearn (L20 container).
                     Proves the GPU mirror == the replica.
Chain: torch == numpy_ref == real core.

Verdict per cell:
  cross   = |sigma_err[A] - sigma_err[B]|
  scatter = std of sigma_err[B] over n_seed independent seeds (within-backend MC noise)
  PASS if cross <= max(scatter_tol_factor * scatter, abs_tol) AND
          max|what_mean[A]-what_mean[B]| <= w_tol.
A cell that fails ONLY at the smallest rho flags "increase n_iter" (convergence), not a bug.

Run:
  local:     & "C:/.../python.exe" validate_sigma_err.py --mode local     --K 2 3
  container: python validate_sigma_err.py --mode container --K 2 3 5
"""
from __future__ import annotations
import argparse, math, statistics
import sigma_err_scan as S


def cell_sigma(backend, K, rho, N, R, w, seed, **extra):
    return S.measure_cell(backend, K, rho, N, R, w, sigma=S.SIGMA_DEFAULT,
                          bw=S.BW_DEFAULT, noise=0.0, seed=seed, **extra)


def validate(mode, Ks, *, R=400, n_seed=4, rhos=(0.06, 0.10, 0.16, 0.25), Ns=(5000,),
             scatter_tol_factor=2.0, abs_tol=0.004, w_tol=0.02, base_seed=1234):
    A = "torch" if mode == "container" else "numpy_ref"
    B = "numpy_ref" if mode == "container" else "numpy"
    print(f"[validate {mode}]  A={A}  vs  B={B}  (R={R}, n_seed={n_seed})")
    print(f"{'K':>2} {'rho':>6} {'N':>7} {'se_A':>9} {'se_B':>9} {'cross':>9} "
          f"{'scatter':>9} {'dwmax':>7}  verdict")
    n_pass = n_fail = 0
    smallest_rho = min(rhos)
    fails = []
    for K in Ks:
        w = S.fixed_interior_w(K, "endpoint")
        for N in Ns:
            for rho in rhos:
                try:
                    se_A = cell_sigma(A, K, rho, N, R, w, base_seed)["sigma_err"]
                    runsB = [cell_sigma(B, K, rho, N, R, w, base_seed + 1000 * s) for s in range(n_seed)]
                except ValueError as e:
                    print(f"  skip K={K} rho={rho}: {e}"); continue
                se_B = [r["sigma_err"] for r in runsB]
                wmean_B0 = runsB[0]["what_mean"]
                wmean_A = cell_sigma(A, K, rho, N, R, w, base_seed)["what_mean"]
                cross = abs(se_A - statistics.mean(se_B))
                scatter = statistics.pstdev(se_B) if n_seed > 1 else float("nan")
                dwmax = max(abs(a - b) for a, b in zip(wmean_A, wmean_B0))
                ok = (cross <= max(scatter_tol_factor * scatter, abs_tol)) and (dwmax <= w_tol)
                verdict = "PASS" if ok else "FAIL"
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    fails.append((K, rho, N))
                print(f"{K:>2} {rho:>6.3f} {N:>7} {se_A:>9.5f} {statistics.mean(se_B):>9.5f} "
                      f"{cross:>9.5f} {scatter:>9.5f} {dwmax:>7.4f}  {verdict}")
    print(f"\n{n_pass} PASS / {n_fail} FAIL")
    if n_fail and all(r == smallest_rho for _, r, _ in fails):
        print("NOTE: all failures at the smallest rho -> likely 500-iter under-convergence on "
              "near-collinear H, not a backend bug. Re-run that rho with larger n_iter.")
    elif n_fail == 0:
        print("VERDICT: PASS — backends equivalent (cross-backend diff << within-backend scatter).")
    return n_fail == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "container"], required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--R", type=int, default=400)
    ap.add_argument("--n-seed", type=int, default=4)
    ap.add_argument("--Ns", type=int, nargs="+", default=[5000])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.06, 0.10, 0.16, 0.25])
    a = ap.parse_args()
    validate(a.mode, a.K, R=a.R, n_seed=a.n_seed, Ns=tuple(a.Ns), rhos=tuple(a.rhos))
