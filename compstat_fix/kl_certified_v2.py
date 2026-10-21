"""Algorithm 2 + exact-certificate repair stage ("Algorithm 2r", 2026-10-21; atlas_rerun bundle).

Root cause of the uncertified rows in job 269468: Algorithm 2 evaluates m = wH with a floor
m >= 1e-300.  With high multiplicative noise the KDE x has mass at grid points far from every
source peak, where the Gaussian endmembers are ~1e-300 or denormal (down to 5e-324) or exactly 0.
There the floored surrogate is (i) non-smooth (kink at m = 1e-300) and (ii) its gradient
x*H/1e-300 is not the gradient of the objective actually being minimised, so the projected-Newton
iteration stalls (200-step budget) or, when a weight is exactly 0 and the remaining mixture is
below the floor, produces |g| up to 1e16 and an Armijo failure (row abandoned).

Repair: an exactly equivalent, floor-free formulation.  With S = {t : x_t > 0, max_j H_jt > 0}
and column scaling Hhat_kt = H_kt / max_j H_jt (so max_k Hhat_kt = 1 on S),
    f(w) = sum_k w_k Hsum_k - sum_{t in S} x_t log( sum_k w_k Hhat_kt ) + const,
which is the KL objective sum_t [m_t - x_t log m_t] up to an additive constant (grid points with
x_t > 0 but H_.t = 0 for all k contribute a w-independent constant; points with x_t = 0 contribute
only through sum_t m_t = w.Hsum).  Gradient and Hessian are exact; no floor is needed because
m_hat_t >= max_k w_k Hhat_kt and the iterates are kept strictly feasible for m_hat > 0 on S.
Certificate (unchanged criterion, exact gradient):
    r = max( max_{w_k>0} |g_k| , max_{w_k=0} max(-g_k,0) ) / mean_k Hsum_k  <= 1e-11.
"""
import numpy as np
try:
    from kl_certified import kl_unmix_certified, kkt_residual as kkt_floored
except ImportError:  # package import when living under compstat_fix/
    from .kl_certified import kl_unmix_certified, kkt_residual as kkt_floored

TOL = 1e-11


def _prep(x, H):
    Hmax = H.max(0)
    S = (x > 0) & (Hmax > 0)
    Hh = H[:, S] / Hmax[S]
    return x[S], Hh, H.sum(1)


def grad_exact(w, xs, Hh, Hsum):
    m = w @ Hh
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        r = xs / m
    return Hsum - Hh @ r, m


def kkt_exact_row(w, x, H):
    xs, Hh, Hsum = _prep(x, H)
    g, m = grad_exact(w, xs, Hh, Hsum)
    if np.any(m <= 0):
        return np.inf
    pos = w > 0
    r = max(np.abs(g[pos]).max(initial=0.0), np.maximum(-g[~pos], 0).max(initial=0.0))
    return r / Hsum.mean()


def kkt_exact(W, X, H, scale_rows=True):
    """exact certificate for each row; W may be normalised (rescaled to the row's optimal mass first:
    for H with equal row sums the minimiser has sum_k w_k Hsum_k = sum_{t in S} x_t)."""
    out = np.empty(len(X))
    for i in range(len(X)):
        w = W[i].copy()
        if scale_rows:
            xs, Hh, Hsum = _prep(X[i], H)
            s = (w * Hsum).sum()
            w = w * (xs.sum() / s) if s > 0 else w
        out[i] = kkt_exact_row(w, X[i], H)
    return out


def _f_delta(wn, w, xs, Hh, Hsum, m):
    mn = wn @ Hh
    if np.any(mn <= 0):
        return np.inf, mn
    return float((wn - w) @ Hsum - xs @ np.log1p((mn - m) / m)), mn


def repair_row(x, H, w0, max_iter=500, mu_iter=2000, tol=TOL):
    """Projected Newton (Bertsekas eps-active set) on the exact column-scaled objective.
    Returns unnormalised w, exact kkt, info."""
    xs, Hh, Hsum = _prep(x, H)
    K = H.shape[0]
    hs = Hsum.mean()
    # optimal mass: sum_k w_k Hsum_k = sum xs (stationarity summed over the support)
    w = np.asarray(w0, float).copy()
    s = (w * Hsum).sum()
    w = w * (xs.sum() / s) if s > 0 else np.full(K, xs.sum() / Hsum.sum())
    m = w @ Hh
    if np.any(m <= 0) or not np.all(np.isfinite(m)):
        # restore strict feasibility: mix with uniform (all m_hat > 0 since max_k Hhat = 1)
        w = 0.999 * w + 0.001 * xs.sum() / Hsum.sum()
        m = w @ Hh
    stage = "newton"
    nfail = 0
    hist = []
    for it in range(max_iter):
        g, m = grad_exact(w, xs, Hh, Hsum)
        pos = w > 0
        res = max(np.abs(g[pos]).max(initial=0), np.maximum(-g[~pos], 0).max(initial=0)) / hs
        if res <= tol:
            return w, res, dict(it=it, stage=stage, nfail=nfail)
        A = xs / m ** 2
        Hs = (Hh * A) @ Hh.T
        eps = min(1e-3 * w.sum(), np.abs(w - np.maximum(w - g, 0)).max())
        act = (w <= eps) & (g > 0)
        free = ~act
        Msys = np.where(free[:, None] & free[None, :], Hs, 0.0)
        Msys[act, act] = np.maximum(np.diag(Hs)[act], 1e-300)
        try:
            D = -np.linalg.solve(Msys, g)
        except np.linalg.LinAlgError:
            D = -np.linalg.lstsq(Msys, g, rcond=None)[0]
        ok = False
        if res < 1e-6:
            # local regime: f-differences are below floating-point resolution (|df| ~ res^2 << eps|f|),
            # so Armijo is decided by rounding noise.  Accept the (projected) Newton step if it
            # decreases the exact KKT residual (Newton on the optimality system), else backtrack on it.
            a = 1.0
            for _ in range(30):
                wn = np.maximum(w + a * D, 0.0)
                gn, mn = grad_exact(wn, xs, Hh, Hsum)
                if np.all(mn > 0):
                    pn = wn > 0
                    rn = max(np.abs(gn[pn]).max(initial=0), np.maximum(-gn[~pn], 0).max(initial=0)) / hs
                    if rn < res:
                        ok = True
                        break
                a *= 0.5
        if not ok:
            a = 1.0
            for _ in range(60):
                wn = np.maximum(w + a * D, 0.0)
                df, mn = _f_delta(wn, w, xs, Hh, Hsum, m)
                dec = g @ (wn - w)
                if np.isfinite(df) and df <= 1e-4 * dec and df < 0:
                    ok = True
                    break
                a *= 0.5
        if not ok:
            # Newton step not acceptable in floating point: take exact-MU (EM) steps, which are
            # monotone and keep w > 0, then resume Newton
            nfail += 1
            stage = "newton+mu"
            for _ in range(50):
                w = w * (Hh @ (xs / (w @ Hh))) / Hsum
            continue
        w = wn
    g, m = grad_exact(w, xs, Hh, Hsum)
    pos = w > 0
    res = max(np.abs(g[pos]).max(initial=0), np.maximum(-g[~pos], 0).max(initial=0)) / hs
    return w, res, dict(it=max_iter, stage=stage, nfail=nfail)


def kl_unmix_certified_v2(X, H, return_info=False, max_iter=500):
    """Algorithm 2 unchanged; then every row whose EXACT certificate exceeds 1e-11 is repaired.
    Rows that Algorithm 2 certified and that pass the exact certificate are returned bit-identical."""
    X = np.asarray(X, float); H = np.asarray(H, float)
    W, info = kl_unmix_certified(X, H, return_info=True)
    W_alg2 = W.copy()
    k_floor = info["kkt"]
    k_ex = kkt_exact(W, X, H)
    need = np.where(~(k_ex <= TOL) | ~(k_floor <= TOL))[0]
    k_final = k_ex.copy()
    rep = []
    for i in need:
        cands = []
        # (a) continue from Algorithm 2's point; (b) from a fresh long-EM start (uniform 1/K)
        for start in ("alg2", "em"):
            if start == "alg2":
                w0 = W[i]
            else:
                xs, Hh, Hsum = _prep(X[i], H)
                w0 = np.full(H.shape[0], xs.sum() / Hsum.sum())
                for _ in range(2000):
                    w0 = w0 * (Hh @ (xs / (w0 @ Hh))) / Hsum
            w, r, inf = repair_row(X[i], H, w0, max_iter=max_iter)
            xs, Hh, Hsum = _prep(X[i], H)
            fval = float(w @ Hsum - xs @ np.log(w @ Hh))
            cands.append((r <= TOL, -fval if False else fval, r, w, start, inf))
            if r <= TOL:
                break
        # prefer certified, then lower objective
        cands.sort(key=lambda c: (not c[0], c[1]))
        ok, fval, r, w, start, inf = cands[0]
        wn = w / w.sum()
        rep.append(dict(row=int(i), kkt_floor_alg2=float(k_floor[i]), kkt_exact_alg2=float(k_ex[i]),
                        kkt_exact_final=float(r), start=start, f=fval, dw_inf=float(np.abs(wn - W[i]).max()),
                        argmax_changed=bool(np.argmax(wn) != np.argmax(W[i])), **inf))
        W[i] = wn
        k_final[i] = r
    if return_info:
        return W, dict(W_alg2=W_alg2, kkt=k_final, kkt_floor_alg2=k_floor, kkt_exact_alg2=k_ex, repaired=rep)
    return W
