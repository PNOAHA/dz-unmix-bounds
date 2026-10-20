"""Certified supervised KL unmixing (compstat-fix branch, 2026-10-20).

Solves  min_{w >= 0}  f(w) = sum_t [ (wH)_t - x_t log (wH)_t ]   for each row x of X,
then normalizes w to the simplex (identical target to dz_unmixing_experiment.supervised_unmix,
whose output is the row-normalized multiplicative-update iterate).  Because every row of H
has the same grid sum, the normalized minimizer equals the simplex-constrained KL minimizer.

Algorithm: deterministic warm start (uniform weights, row-independent), a short block of
KL multiplicative updates, then a batched projected (active-set) Newton method with
ratio test and Armijo backtracking.  Each row is certified by the KKT residual
    r = max( max_{k free} |g_k| , max_{k at 0} max(-g_k, 0) ) / sum_t H_kt ,
with g the gradient of f.  Rows not certified within the iteration budget are reported.
"""
from __future__ import annotations
import numpy as np

FLOOR = 1e-300


def _f(W, X, H):
    M = np.maximum(W @ H, FLOOR)
    return (M - X * np.log(M)).sum(-1)


def _grad_hess(W, X, H, hess=True):
    M = np.maximum(W @ H, FLOOR)
    R = X / M
    G = H.sum(1)[None, :] - R @ H.T
    if not hess:
        return G, None
    # x / m^2 for the Newton direction only; guarded so that x > 0 at a numerically empty grid
    # point cannot give inf * 0 = nan (the objective, gradient and KKT certificate are unchanged)
    A = np.where(X > 0, X / np.maximum(M, 1e-150) ** 2, 0.0)
    Hs = np.einsum('kt,nt,jt->nkj', H, A, H, optimize=True)
    return G, Hs


def kkt_residual(W, X, H, zero_tol=0.0):
    G, _ = _grad_hess(W, X, H, hess=False)
    scale = H.sum(1).mean()
    free = W > zero_tol
    r_free = np.where(free, np.abs(G), 0.0).max(1)
    r_act = np.where(~free, np.maximum(-G, 0.0), 0.0).max(1)
    return np.maximum(r_free, r_act) / scale


def mu_iterations(W, X, H, n_iter):
    Hsum = H.sum(1)[None, :]
    for _ in range(n_iter):
        M = np.maximum(W @ H, FLOOR)
        W = W * ((X / M) @ H.T) / Hsum
    return W


def kl_unmix_certified(X, H, mu_warm=200, max_newton=200, tol=1e-11, return_info=False):
    X = np.asarray(X, float); H = np.asarray(H, float)
    n, K = X.shape[0], H.shape[0]
    scale = X.sum(1, keepdims=True) / H.sum(1).mean()
    W = np.full((n, K), 1.0 / K) * scale          # row-independent deterministic start
    W = mu_iterations(W, X, H, mu_warm)
    hs = H.sum(1).mean()
    alphas = 0.5 ** np.arange(0, 40)
    done = np.zeros(n, bool)
    it_used = np.zeros(n, int)
    for it in range(max_newton):
        idx = np.where(~done)[0]
        if idx.size == 0:
            break
        Wi, Xi = W[idx], X[idx]
        G, Hs = _grad_hess(Wi, Xi, H)
        pos = Wi > 0
        res = np.maximum(np.where(pos, np.abs(G), 0).max(1),
                         np.where(~pos, np.maximum(-G, 0), 0).max(1)) / hs
        conv = res <= tol
        done[idx[conv]] = True
        it_used[idx] = it
        keep = ~conv
        if not keep.any():
            break
        idx, Wi, Xi, G, Hs = idx[keep], Wi[keep], Xi[keep], G[keep], Hs[keep]
        # Bertsekas (1982) projected Newton: epsilon-active set
        # infinity norm of the projected-gradient step (the Euclidean norm underflows to 0 when the
        # only violating coordinate is a denormal-scale weight such as 5e-241, which then never
        # enters the epsilon-active set; fixed 2026-10-20 after the scan rerun flagged 0.05% to 0.8% of configs)
        epsk = np.minimum(1e-3 * Wi.sum(1), np.abs(Wi - np.maximum(Wi - G, 0)).max(1))[:, None]
        act = (Wi <= epsk) & (G > 0)
        free = ~act
        Fm = free[:, :, None] & free[:, None, :]
        diagH = np.einsum('nkk->nk', Hs)
        Msys = np.where(Fm, Hs, 0.0) + np.where(act, diagH, 0.0)[:, :, None] * np.eye(K)[None]
        D = -np.linalg.solve(Msys, G[..., None])[..., 0]
        f0 = _f(Wi, Xi, H)
        best = Wi.copy()
        found = np.zeros(len(idx), bool)
        for a in alphas:
            Wn = np.maximum(Wi + a * D, 0.0)
            fn = _f(Wn, Xi, H)
            dec = (G * (Wn - Wi)).sum(1)
            ok = (~found) & ((fn <= f0 + 1e-4 * dec) | ((a == 1.0) & (fn - f0 <= 1e-15 * np.abs(f0))))
            best[ok] = Wn[ok]; found |= ok
            if found.all():
                break
        stuck = ~found
        W[idx] = best
        if stuck.any():
            done[idx[stuck]] = True
    res = kkt_residual(W, X, H)
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1.0
    Wn = W / rs
    if return_info:
        return Wn, dict(kkt=res, newton_iters=it_used, certified=res <= tol)
    return Wn
