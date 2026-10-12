# -*- coding: utf-8 -*-
"""
probe_bw_mechanism_guard.py -- robustness guards for the bw-sweep verdict:
  (1) convergence: n_iter 2000 vs 6000 at the extreme bw=6 and bw=24 (rule out under-convergence
      flipping the sign).
  (2) extra/finer bw grid incl. bw=3 and bw=30 to see the full trend and whether excess keeps
      shrinking below 1 as bw grows.
  (3) absolute sigma_err (not just excess) vs bw -- excess divides by a bw-independent floor, so
      sign of d(excess)/d(bw) == sign of d(sigma_err)/d(bw); print both to be explicit.

Reuses probe_bw_mechanism.measure (real core). Run:
  & "C:/Users/Administrator/miniconda3/python.exe" probe_bw_mechanism_guard.py
"""
import math
import numpy as np
import probe_bw_mechanism as P

RHO, N = 0.47, 350
print("(1) convergence guard at extreme bw (sign must not flip with n_iter)")
print("%2s %5s %5s | %8s %7s | dom_deg" % ("K", "bw", "nit", "sig_err", "excess"))
for K in [4, 5]:
    for bw in [6.0, 24.0]:
        for nit in [2000, 6000]:
            sig, exc, dom, ov = P.measure(K, RHO, N, bw, 20260615 + K * 1000 + 101, n_iter=nit)
            print("%2d %5.1f %5d | %8.4f %7.2f | deg=%d" % (K, bw, nit, sig, exc, dom))
    print()

print("(2) finer/wider bw grid (seed 101): does excess keep falling as bw rises?")
print("%2s %5s | %8s %7s | dom_deg =K-1?" % ("K", "bw", "sig_err", "excess"))
for K in [4, 5]:
    prev = None
    for bw in [3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0]:
        sig, exc, dom, ov = P.measure(K, RHO, N, bw, 20260615 + K * 1000 + 101)
        arrow = "" if prev is None else ("UP" if exc > prev else "DOWN")
        print("%2d %5.1f | %8.4f %7.2f | deg=%d %5s  %s"
              % (K, bw, sig, exc, dom, "YES" if dom == K - 1 else "no", arrow))
        prev = exc
    print()
