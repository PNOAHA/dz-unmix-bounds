"""Solver-convergence diagnosis for the supervised KL estimator (compstat-fix, 2026-10-20).

For each (K, rho, N) cell, R KDE rows are generated once (noise 0, sigma 25, bw 12, endpoint
weights, as in sigma_err_scan.py) and solved repeatedly on these identical inputs with:
  ref      : certified projected-Newton solver (kl_certified.py), KKT residual <= 1e-11
  ref_s1/2 : certified solver from two random warm starts (uniqueness check)
  mu500    : dz_unmixing_experiment.supervised_unmix defaults (500 it, tol 1e-7)  [scans, atlas]
  mu500_b  : same, rows solved in 5 batches instead of one batch (stopping rule is batch-global)
  mu2000   : n_iter 2000, tol 1e-13                                               [calibration]
  mu2000_sK: n_iter 2000, tol 1e-13, init seeds 1..4 (multi-start)
  mu20000, mu40000 : tol 1e-13
Outputs per setting: RMS and max solver error ||w - w_ref||, sigma_err (sqrt lambda_max of the
covariance with divisor R), the ratio RMS error / sigma_err(ref), and top-1 agreement.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
from _common import setup, core
from kl_certified import kl_unmix_certified, mu_iterations


def mu(X, H, n_iter, tol, seed=0):
    n, k = X.shape[0], H.shape[0]
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 1.0, size=(n, k)); eps = 1e-10
    Hsum = H.sum(axis=1, keepdims=True).T; prev = np.inf; used = n_iter
    for it in range(n_iter):
        WH = W @ H + eps
        W = W * ((X / WH) @ H.T) / (Hsum + eps)
        if it % 25 == 0:
            WH = W @ H + eps
            loss = np.sum(X * np.log((X + eps) / WH) - X + WH)
            if abs(prev - loss) < tol * max(1.0, abs(prev)):
                used = it + 1; break
            prev = loss
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1
    return W / rs, used


def sig(W):
    C = np.cov(W.T, bias=True)
    return float(np.sqrt(max(np.linalg.eigvalsh(np.atleast_2d(C)).max(), 0)))


def cell(args):
    K, rho, N, R, seed = args
    t0 = time.time()
    X, H, w = setup(K, rho, N, R, seed=seed)
    out = {}
    Wref, info = kl_unmix_certified(X, H, return_info=True)
    s = X.sum(1, keepdims=True) / H.sum(1).mean()
    sols = {'ref': (Wref, 0)}
    for j in (1, 2):
        rng = np.random.default_rng(100 + j)
        W0 = rng.uniform(0.1, 1, (R, K)); W0 = W0 / W0.sum(1, keepdims=True) * s
        # random warm start then the same certified Newton polish
        W1 = mu_iterations(W0, X, H, 50)
        from kl_certified import kl_unmix_certified as kc
        Wr, inf2 = _polish_from(W1, X, H)
        sols[f'ref_s{j}'] = (Wr, 0)
    sols['mu500'] = mu(X, H, 500, 1e-7)
    parts = np.array_split(np.arange(R), 5)
    Wb = np.zeros_like(Wref); ub = []
    for p in parts:
        Wb[p], u = mu(X[p], H, 500, 1e-7); ub.append(u)
    sols['mu500_b'] = (Wb, int(np.mean(ub)))
    sols['mu2000'] = mu(X, H, 2000, 1e-13)
    for sd in (1, 2, 3, 4):
        sols[f'mu2000_s{sd}'] = mu(X, H, 2000, 1e-13, seed=sd)
    sols['mu20000'] = mu(X, H, 20000, 1e-13)
    sols['mu40000'] = mu(X, H, 40000, 1e-13)
    sref = sig(Wref)
    k_star = int(np.argmax(w))
    for name, (W, used) in sols.items():
        e = np.linalg.norm(W - Wref, axis=1)
        out[name] = dict(iters=int(used), rms_err=float(np.sqrt(np.mean(e ** 2))), max_err=float(e.max()),
                         sigma_err=sig(W), err_over_sigma=float(np.sqrt(np.mean(e ** 2)) / sref),
                         top1=float(np.mean(np.argmax(W, 1) == k_star)),
                         top1_agree=float(np.mean(np.argmax(W, 1) == np.argmax(Wref, 1))))
    # multi-start spread among the 5 mu2000 starts
    Ws = np.stack([sols['mu2000'][0]] + [sols[f'mu2000_s{sd}'][0] for sd in (1, 2, 3, 4)])
    spread = np.linalg.norm(Ws - Ws.mean(0), axis=2).max(0)
    out['mu2000_multistart_maxspread'] = dict(rms=float(np.sqrt(np.mean(spread ** 2))), max=float(spread.max()))
    out['ref_kkt_max'] = float(info['kkt'].max()); out['ref_certified_frac'] = float(info['certified'].mean())
    out['ref_newton_max'] = int(info['newton_iters'].max())
    return dict(K=K, rho=rho, N=N, R=R, seed=seed, sigma_ref=sref, sec=time.time() - t0, res=out)


def _polish_from(W1, X, H):
    import kl_certified as kc
    # reuse the Newton loop by temporarily replacing the deterministic warm start
    orig = kc.mu_iterations
    try:
        kc.mu_iterations = lambda W, X_, H_, n: W1
        return kc.kl_unmix_certified(X, H, return_info=True)
    finally:
        kc.mu_iterations = orig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', nargs='+', type=int, default=[2, 3, 4, 5, 6, 7])
    ap.add_argument('--rhos', nargs='+', type=float, default=[0.05, 0.10, 0.19, 0.35, 0.47])
    ap.add_argument('--Ns', nargs='+', type=int, default=[20, 160, 350])
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--out', default='solver_convergence.json')
    a = ap.parse_args()
    jobs = [(K, r, N, a.R, a.seed + K * 1000 + int(r * 1e4) + N) for K in a.K for r in a.rhos for N in a.Ns]
    jobs.sort(key=lambda j: -j[0])
    res = []
    with Pool(a.procs) as p:
        for i, c in enumerate(p.imap_unordered(cell, jobs)):
            res.append(c); print(f"{i+1}/{len(jobs)} K={c['K']} rho={c['rho']} N={c['N']} {c['sec']:.0f}s "
                                 f"mu500 e/s={c['res']['mu500']['err_over_sigma']:.3f} mu2000 e/s={c['res']['mu2000']['err_over_sigma']:.3f} "
                                 f"mu20000 e/s={c['res']['mu20000']['err_over_sigma']:.4f} refs={c['res']['ref_s1']['max_err']:.1e}", flush=True)
            json.dump(dict(args=vars(a), cells=res), open(a.out, 'w'), indent=1)
