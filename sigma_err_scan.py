# -*- coding: utf-8 -*-
"""
sigma_err_scan.py — M1 main-theorem numerical validation (dz-unmix-bounds, paper 2 §3).
=========================================================================================
Measures the constrained-MLE worst-tangent-direction estimation error sigma_err of the
SUPERVISED KL-NMF unmixer (fixed endmembers H), as a function of endmember KS distance
rho and sample size N, to test the main theorem

    sigma_err(K, rho, N)  ~  c_K / ( sqrt(N) * rho^(K-1) )           (LAN region only)

valid only in the LAN window  N >> rho^{-2(K-1)}.  Three rho-regimes coexist at fixed N
(established by the 2026-06-11 crossover probe):
    (1) ill-conditioned  N << rho^{-2(K-1)} : vertex collapse, slope shallower than -(K-1)
    (2) LAN              N >> rho^{-2(K-1)} : sigma_err ~ rho^{-(K-1)}  <-- theory window
    (3) saturated        rho large          : sigma_err -> rho-independent 1/sqrt(N) floor
The -(K-1) exponent is fit ONLY over rho cells classified LAN-and-not-saturated.

Closes main_theorem_draft.md §6 D107 items:
    #1 exponent K-1 controlled measurement   (this script, --task slope)
    #2 c_K calibration                        (intercept of the LAN-window fit)
    #3 vertex-jump confirmation               (--task vertexjump; w_hat[k*]->0 in failures)
    #4 endpoint vs middle k* prefactor        (--w endpoint|middle)

Two backends, identical math, picked with --backend:
    numpy : imports the REAL dz_unmixing_experiment (generate_sample / kde_density /
            supervised_unmix). Maximal fidelity. Runs anywhere sklearn+numpy exist.
    torch : self-contained GPU mirror (validated bit-for-bit by validate_sigma_err.py).
            For the publication-scale grid (large N, K=5, fine rho) on the L20 container.

The supervised problem is CONVEX in W for fixed H (KL of a linear map), so both backends
converge to the same MLE regardless of init -> the backends are equivalent up to float.
Does NOT import or modify dz-unmix-cg's source; numpy backend imports it read-only, torch
backend MIRRORS its DEFAULT geometry. (Red line: dz-unmix-cg untouched.)

Run (local, numpy):
    & "C:/Users/Administrator/miniconda3/python.exe" sigma_err_scan.py --backend numpy \
        --task slope --K 2 3 --out-dir out_sigma_err --smoke
Run (L20 container, torch, full grid):
    python sigma_err_scan.py --backend torch --task slope --K 2 3 5 --out-dir /private/sigma_err/out
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

try:
    from scipy.special import erfinv as _erfinv
    def erfinv(x): return float(_erfinv(x))
except Exception:                                   # no scipy -> rational approx (Winitzki)
    def erfinv(x):
        a = 0.147
        ln = math.log(1 - x * x)
        t = 2 / (math.pi * a) + ln / 2
        return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)

# ---- geometry shared by both backends (MIRRORS dz-unmix-cg conventions) --------------
SIGMA_DEFAULT = 25.0          # equal common sigma (Ma); matches applicability_scan BASE_SIGMA
START_DEFAULT = 200.0         # start peak so all K peaks stay inside the grid
BW_DEFAULT = 12.0             # KDE bandwidth; matches dz-unmix-cg kde_density default
TGRID = np.arange(50.0, 2500.0 + 1e-9, 5.0)         # MIRRORS dz-unmix-cg T_GRID
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz->trapezoid; container may be 1.x


def standardized_spacing(rho):
    """dmu/sigma giving adjacent-endmember KS distance = rho (lemma1 (i))."""
    return 2.0 * math.sqrt(2.0) * erfinv(rho)


def peak_ages(K, rho, sigma=SIGMA_DEFAULT, start=START_DEFAULT):
    dmu = sigma * standardized_spacing(rho)
    mus = np.array([start + i * dmu for i in range(K)], dtype=np.float64)
    if mus[-1] >= TGRID[-1] - 3 * sigma:
        raise ValueError(f"K={K} rho={rho}: top peak {mus[-1]:.0f} too close to grid top {TGRID[-1]}")
    return mus, dmu


def fixed_interior_w(K, kind="middle"):
    """Deterministic interior true weight (NOT Dirichlet; per main_theorem §6 'fix Dw').
    middle:   argmax k* is an interior endmember (worst-mode overlap ~ aligned).
    endpoint: argmax k* is an end endmember (worst-mode overlap ~ orthogonal). [item #4]"""
    if kind == "middle":
        w = np.linspace(1.0, 0.6, K)
        if K >= 3:                                  # push the max to an interior index
            w = np.roll(w, K // 2)
    elif kind == "endpoint":
        w = np.linspace(1.0, 0.6, K)                # max at index 0 (an endpoint)
    else:
        raise ValueError(kind)
    return w / w.sum()


# ======================================================================================
# numpy backend — uses the REAL dz-unmix-cg core (read-only import)
# ======================================================================================
def _load_real_core():
    cand = os.environ.get("DZ_CORE_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dz-unmix-cg"))
    if cand not in sys.path:
        sys.path.insert(0, cand)
    import dz_unmixing_experiment as core             # noqa
    return core


SOLVER = "mu2000"   # compstat-fix: "mu2000" (archived v1.2.2 behaviour) or "certified"


def measure_cell_numpy(K, rho, N, R, w, *, sigma, bw, noise, seed, n_iter=2000):
    core = _load_real_core()
    mus, _ = peak_ages(K, rho, sigma)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(sigma), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    rng = np.random.default_rng(seed)
    X = np.empty((R, len(core.T_GRID)))
    for r in range(R):
        ages = core.generate_sample(w, n_grains_range=(N, N), noise_pct=noise, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=bw)
    if SOLVER == "certified":
        # compstat-fix: KKT-certified minimiser (see compstat_fix/kl_certified.py); the archived
        # 2000-iteration MU result on the identical rows is also kept for comparison.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compstat_fix"))
        from kl_certified import kl_unmix_certified
        what, info = kl_unmix_certified(X, H, return_info=True)
        out = _summarize(what, w)
        what_old = core.supervised_unmix(X, H, n_iter=n_iter, tol=1e-13)
        old = _summarize(what_old, w)
        out.update(solver="certified", kkt_max=float(info["kkt"].max()),
                   certified_frac=float(info["certified"].mean()),
                   mu2000_sigma_err=old["sigma_err"], mu2000_top1=old["top1"], mu2000_what_mean=old["what_mean"],
                   solver_rms_diff=float(np.sqrt(np.mean(np.sum((what - what_old) ** 2, 1)))))
        return out
    what = core.supervised_unmix(X, H, n_iter=n_iter, tol=1e-13)  # (R,K) 2000-iteration MU (archived v1.2.2)
    return _summarize(what, w)


# ======================================================================================
# numpy_ref backend — self-contained pure-numpy replica (NO sklearn, NO dz-unmix-cg import)
#   Mirrors the pure-numpy parts of dz-unmix-cg: build_source_endmembers / generate_sample /
#   kde_density / supervised_unmix. Lets validate_sigma_err.py check torch == numpy_ref on the
#   air-gapped L20 container (which has no sklearn). Proven == real core locally (sklearn there).
# ======================================================================================
def _H_ref(K, rho, sigma):
    mus, _ = peak_ages(K, rho, sigma)
    h = np.exp(-0.5 * ((TGRID[None, :] - mus[:, None]) / sigma) ** 2)
    return h / _TRAPZ(h, TGRID, axis=1)[:, None]                      # (K,T) area-normalized


def _gen_sample_ref(w, N, sigma, mus, noise, rng):
    counts = rng.multinomial(N, w)
    ages = np.concatenate([np.clip(rng.normal(mus[i], sigma, c), TGRID[0], TGRID[-1])
                           for i, c in enumerate(counts)])
    if noise > 0:
        ages = ages * rng.normal(1.0, noise, ages.shape)
    return np.clip(ages, TGRID[0], TGRID[-1])


def _kde_ref(ages, bw):
    u = (TGRID[:, None] - ages[None, :]) / bw
    d = np.maximum((np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)).sum(1) / (len(ages) * bw), 0.0)
    area = _TRAPZ(d, TGRID)
    return d / area if area > 0 else d


def _supervised_unmix_ref(X, H, n_iter=500, tol=1e-7):
    n, k = X.shape[0], H.shape[0]
    rng = np.random.default_rng(0)                                    # mirror real init (seed 0)
    W = rng.uniform(0.1, 1.0, size=(n, k)); eps = 1e-10
    Hsum = H.sum(1, keepdims=True).T; prev = np.inf
    for it in range(n_iter):
        WH = W @ H + eps
        W = W * ((X / WH) @ H.T) / (Hsum + eps)
        if it % 25 == 0:
            WH = W @ H + eps
            loss = np.sum(X * np.log((X + eps) / WH) - X + WH)
            if abs(prev - loss) < tol * max(1.0, abs(prev)):
                break
            prev = loss
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1.0
    return W / rs


def measure_cell_numpyref(K, rho, N, R, w, *, sigma, bw, noise, seed, n_iter=2000):
    mus, _ = peak_ages(K, rho, sigma)
    H = _H_ref(K, rho, sigma)
    rng = np.random.default_rng(seed)
    X = np.stack([_kde_ref(_gen_sample_ref(w, N, sigma, mus, noise, rng), bw) for _ in range(R)])
    return _summarize(_supervised_unmix_ref(X, H, n_iter=n_iter, tol=1e-13), w)


# ======================================================================================
# torch backend — self-contained GPU mirror (validated by validate_sigma_err.py)
# ======================================================================================
def measure_cell_torch(K, rho, N, R, w, *, sigma, bw, noise, seed, device="cuda",
                       dtype=None, n_iter=2000, chunk=4000, mem_budget=1.5e8):
    import torch
    dtype = dtype or torch.float64
    # adaptive chunk: the (B,T,N) KDE kernel is the memory bottleneck. Keep B*T*N
    # under mem_budget elements so large-N (1e5-1e6) fits the L20's 48 GB.
    chunk = max(1, min(chunk, int(mem_budget // (len(TGRID) * max(N, 1)))))
    g = torch.Generator(device=device).manual_seed(seed)
    tg = torch.as_tensor(TGRID, device=device, dtype=dtype)            # (T,)
    mus, _ = peak_ages(K, rho, sigma)
    mus_t = torch.as_tensor(mus, device=device, dtype=dtype)           # (K,)
    sig_t = torch.full((K,), float(sigma), device=device, dtype=dtype)
    w_t = torch.as_tensor(w, device=device, dtype=dtype)               # (K,)

    # endmember matrix H (K,T), area-normalized (mirror build_source_endmembers)
    H = torch.exp(-0.5 * ((tg[None, :] - mus_t[:, None]) / sig_t[:, None]) ** 2)   # (K,T)
    H = H / torch.trapz(H, tg, dim=1, ).clamp_min(1e-300)[:, None]
    Hsum = H.sum(1)                                                    # (K,)

    what = torch.empty(R, K, device=device, dtype=dtype)
    for c0 in range(0, R, chunk):
        B = min(chunk, R - c0)
        # --- generate B samples of N grains from the mixture (mirror generate_sample) ---
        labels = torch.multinomial(w_t, B * N, replacement=True, generator=g).reshape(B, N)
        ages = torch.normal(mus_t[labels], sig_t[labels], generator=g)             # (B,N)
        if noise > 0:
            ages = ages * (1.0 + noise * torch.randn(B, N, device=device, dtype=dtype, generator=g))
        ages = ages.clamp(float(TGRID[0]), float(TGRID[-1]))
        # --- KDE density on TGRID (mirror kde_density), area-normalized ---
        u = (tg[None, :, None] - ages[:, None, :]) / bw                            # (B,T,N)
        Xc = torch.exp(-0.5 * u * u).sum(-1) / (N * bw * math.sqrt(2 * math.pi))    # (B,T)
        Xc = Xc / torch.trapz(Xc, tg, dim=1).clamp_min(1e-300)[:, None]
        # --- supervised KL multiplicative update with fixed H (mirror supervised_unmix) ---
        W = torch.full((B, K), 1.0 / K, device=device, dtype=dtype)                # convex -> init-free
        eps = 1e-10
        for _ in range(n_iter):
            WH = W @ H + eps                                                       # (B,T)
            W = W * ((Xc / WH) @ H.T) / (Hsum[None, :] + eps)
        W = W / W.sum(1, keepdim=True).clamp_min(1e-300)
        what[c0:c0 + B] = W
    return _summarize(what.detach().cpu().numpy(), w)


# ======================================================================================
# shared summary: sigma_err + top1 + vertex-jump
# ======================================================================================
def _summarize(what, w):
    K = what.shape[1]
    kstar = int(np.argmax(w))
    dev = what - what.mean(0, keepdims=True)
    cov = dev.T @ dev / what.shape[0]
    lam = np.linalg.eigvalsh(cov)
    sigma_err = math.sqrt(max(float(lam[-1]), 0.0))                    # worst tangent direction
    pred_star = what.argmax(1)
    top1 = float((pred_star == kstar).mean())
    fail = pred_star != kstar
    n_fail = int(fail.sum())
    what_star = what[:, kstar]
    return dict(
        sigma_err=sigma_err,
        sigma_err_tangent_eigs=[float(x) for x in lam[::-1]],
        top1=top1,
        n_fail=n_fail,
        wstar_true=float(w[kstar]),
        wstar_mean_in_fail=(float(what_star[fail].mean()) if n_fail else float("nan")),
        wstar_collapsed_frac_in_fail=(float((what_star[fail] < 0.10).mean()) if n_fail else float("nan")),
        what_mean=[float(x) for x in what.mean(0)],
    )


def measure_cell(backend, *a, **k):
    return {"numpy": measure_cell_numpy, "numpy_ref": measure_cell_numpyref,
            "torch": measure_cell_torch}[backend](*a, **k)


# ======================================================================================
# regime classification + LAN-window slope fit
# ======================================================================================
def lan_threshold(K, rho):
    return rho ** (-2 * (K - 1))


def classify(K, rho, N, *, lan_margin=3.0):
    """ill if N < margin*rho^{-2(K-1)}; else LAN. (saturation flagged later from the curve.)"""
    return "LAN" if N >= lan_margin * lan_threshold(K, rho) else "ill"


def fit_lan_slope(rhos, sigma_errs, regimes, *, drop_saturated=True):
    """Fit log sigma_err vs log rho over LAN cells, dropping the saturated tail
    (where local slope flattens toward 0). Returns dict with slope, intercept (=> c_K), window."""
    idx = [i for i, r in enumerate(regimes) if r == "LAN"]
    if len(idx) < 2:
        return dict(slope=float("nan"), n_points=len(idx), window=[], note="too few LAN points")
    lr = np.log(np.array([rhos[i] for i in idx]))
    ls = np.log(np.array([sigma_errs[i] for i in idx]))
    order = np.argsort(lr); lr, ls = lr[order], ls[order]
    keep = np.ones(len(lr), dtype=bool)
    if drop_saturated and len(lr) >= 3:
        # drop high-rho points whose local (forward) slope is shallower than -0.5 (saturated floor)
        for j in range(len(lr) - 1):
            local = (ls[j + 1] - ls[j]) / (lr[j + 1] - lr[j])
            if local > -0.5:                       # flattening -> saturated
                keep[j + 1:] = False
                break
    lr2, ls2 = lr[keep], ls[keep]
    if len(lr2) < 2:
        lr2, ls2 = lr, ls
    slope, intercept = np.polyfit(lr2, ls2, 1)
    return dict(slope=float(slope), intercept=float(intercept), n_points=int(len(lr2)),
                window=[float(math.exp(lr2.min())), float(math.exp(lr2.max()))])


# ======================================================================================
# scan driver
# ======================================================================================
def run_slope_task(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rhos = args.rhos
    Ns = args.Ns
    results = {"meta": dict(task="slope", backend=args.backend, sigma=args.sigma, bw=args.bw,
                            noise=args.noise, R=args.R, w_kind=args.w, rhos=list(rhos), Ns=list(Ns)),
               "cells": [], "fits": []}
    for K in args.K:
        w = fixed_interior_w(K, args.w)
        for N in Ns:
            row_se, row_reg = [], []
            t0 = time.time()
            for rho in rhos:
                try:
                    s = measure_cell(args.backend, K, rho, N, args.R, w,
                                     sigma=args.sigma, bw=args.bw, noise=args.noise,
                                     seed=args.seed + K * 1000 + int(rho * 1e4))
                except ValueError as e:
                    print(f"  skip K={K} rho={rho} N={N}: {e}", flush=True); continue
                reg = classify(K, rho, N)
                row_se.append(s["sigma_err"]); row_reg.append(reg)
                s.update(K=K, rho=float(rho), N=int(N), regime=reg, lan_threshold=lan_threshold(K, rho))
                results["cells"].append(s)
                print(f"  K={K} N={N:>8} rho={rho:.3f} [{reg}]  sigma_err={s['sigma_err']:.5f}  "
                      f"top1={s['top1']:.3f}  wstar_fail={s['wstar_mean_in_fail']:.3f}", flush=True)
            valid = [(r, se, rg) for r, se, rg in zip(rhos, row_se, row_reg)]
            if valid:
                fr = fit_lan_slope([v[0] for v in valid], [v[1] for v in valid],
                                   [v[2] for v in valid])
                fr.update(K=K, N=int(N), theory_slope=-(K - 1), c_K_proxy=math.exp(fr["intercept"]) *
                          math.sqrt(N) if not math.isnan(fr["slope"]) else float("nan"))
                results["fits"].append(fr)
                print(f"  => K={K} N={N}: LAN-window slope={fr['slope']:+.3f} "
                      f"(theory {-(K-1)})  over rho in {fr.get('window')}  "
                      f"[{fr['n_points']} pts]  ({time.time()-t0:.1f}s)", flush=True)
    results["meta"]["solver"] = SOLVER
    sfx = "_certified" if SOLVER == "certified" else ""
    out = os.path.join(args.out_dir, f"sigma_err_slope_{args.backend}{sfx}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}  ({len(results['cells'])} cells, {len(results['fits'])} fits)", flush=True)
    return results


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="M1 main-theorem sigma_err numerical validation")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--task", choices=["slope"], default="slope")
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--Ns", type=int, nargs="+", default=None)
    ap.add_argument("--rhos", type=float, nargs="+", default=None)
    ap.add_argument("--R", type=int, default=600, help="reps per cell (covariance sample size)")
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    ap.add_argument("--bw", type=float, default=BW_DEFAULT)
    ap.add_argument("--noise", type=float, default=0.0, help="measurement noise_pct (0 = clean exponent test)")
    ap.add_argument("--w", choices=["middle", "endpoint"], default="middle")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--out-dir", default="out_sigma_err")
    ap.add_argument("--smoke", action="store_true", help="tiny grid sanity")
    ap.add_argument("--solver", choices=["mu2000", "certified"], default="mu2000",
                    help="compstat-fix: supervised solver for the numpy backend")
    args = ap.parse_args(argv)
    if args.smoke:
        args.Ns = args.Ns or [2000, 20000]
        args.rhos = args.rhos or [0.06, 0.09, 0.13, 0.18, 0.25]
        args.R = min(args.R, 300)
    else:
        args.Ns = args.Ns or [2000, 20000, 200000]
        args.rhos = args.rhos or [0.05, 0.07, 0.09, 0.12, 0.16, 0.21, 0.28]
    return args


if __name__ == "__main__":
    a = parse_args()
    SOLVER = a.solver
    print(f"[sigma_err_scan] backend={a.backend} K={a.K} Ns={a.Ns} rhos={a.rhos} R={a.R} "
          f"w={a.w} noise={a.noise}", flush=True)
    run_slope_task(a)
