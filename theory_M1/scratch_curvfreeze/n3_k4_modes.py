import numpy as np, math, json, time, os

try:
    from scipy.special import erfinv as _erfinv
    def erfinv(x): return float(_erfinv(x))
except Exception:
    def erfinv(x):
        a = 0.147; ln = math.log(1 - x*x); t = 2/(math.pi*a) + ln/2
        return math.copysign(math.sqrt(math.sqrt(t*t - ln/a) - t), x)

SIGMA = 25.0; START = 200.0; BW = 12.0
TGRID = np.arange(50.0, 2500.0 + 1e-9, 5.0)
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz

def peaks(K, rho):
    dmu = SIGMA * 2.0 * math.sqrt(2.0) * erfinv(rho)
    return np.array([START + i*dmu for i in range(K)]), dmu

def H_of(K, rho):
    mus, _ = peaks(K, rho)
    h = np.exp(-0.5*((TGRID[None,:] - mus[:,None])/SIGMA)**2)
    return h / TRAPZ(h, TGRID, axis=1)[:,None]

def gen(w, N, mus, rng):
    counts = rng.multinomial(N, w)
    return np.concatenate([np.clip(rng.normal(mus[i], SIGMA, c), TGRID[0], TGRID[-1])
                           for i, c in enumerate(counts)])

def kde(ages, bw=BW):
    d = np.zeros_like(TGRID)
    for s in range(0, len(ages), 20000):
        u = (TGRID[:,None] - ages[None, s:s+20000]) / bw
        d += np.exp(-0.5*u*u).sum(1)
    d /= (len(ages) * bw * math.sqrt(2*math.pi))
    return d / TRAPZ(d, TGRID)

def unmix(X, H, n_iter):
    R, K = X.shape[0], H.shape[0]
    rng = np.random.default_rng(0)
    W = rng.uniform(0.1, 1.0, size=(R, K)); eps = 1e-10
    Hsum = H.sum(1)[None, :]
    for _ in range(n_iter):
        WH = W @ H + eps
        W = W * ((X / WH) @ H.T) / (Hsum + eps)
    return W / W.sum(1, keepdims=True)

# ---- experiment N3: K=4 mode structure fingerprint ----
K = 4
wstar = np.roll(np.linspace(1.0, 0.6, 4), 2); wstar = wstar / wstar.sum()
N = 20000; R = 300; n_iter = 40000
rhos = [0.05, 0.10]

# modes (normalized)
s2 = math.sqrt(2.0); s6 = math.sqrt(6.0)
modes = {
    't':  np.array([1.0, 0.0, 0.0, -1.0]) / s2,
    'c1': np.array([1.0, -2.0, 1.0, 0.0]) / s6,
    'c2': np.array([0.0, 1.0, -2.0, 1.0]) / s6,
    'm':  np.array([0.0, 1.0, -1.0, 0.0]) / s2,
    'p1': np.array([1.0, -1.0, 0.0, 0.0]) / s2,
    'p3': np.array([0.0, 0.0, 1.0, -1.0]) / s2,
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n3_k4_modes.json")

results = {
    "setup": {
        "K": K, "wstar": wstar.tolist(), "N": N, "R": R, "n_iter": n_iter,
        "rhos": rhos, "SIGMA": SIGMA, "BW": BW, "START": START,
    },
    "per_rho": {}
}

print("wstar =", wstar.tolist())
mubar = float(np.dot(wstar, peaks(K, 0.10)[0]))  # mubar depends on rho; recompute per rho below

for rho in rhos:
    t0 = time.time()
    seed = 5678 + int(rho*1000)
    rng = np.random.default_rng(seed)
    mus, dmu = peaks(K, rho)
    H = H_of(K, rho)

    # build R observed KDE replicates
    X = np.empty((R, len(TGRID)))
    for r in range(R):
        ages = gen(wstar, N, mus, rng)
        X[r] = kde(ages)

    What = unmix(X, H, n_iter)  # (R,K), simplex rows

    # projections onto modes
    proj = {name: What @ v for name, v in modes.items()}
    stds = {name: float(np.std(proj[name])) for name in modes}

    # interior weights
    w1 = What[:, 1]; w2 = What[:, 2]
    interior = {
        "w1_max": float(w1.max()), "w1_mean": float(w1.mean()),
        "w2_max": float(w2.max()), "w2_mean": float(w2.mean()),
        "w1_min": float(w1.min()), "w2_min": float(w2.min()),
    }

    # tangent covariance (4x4 centered cov of What), eigenvalues
    Wc = What - What.mean(0, keepdims=True)
    cov = (Wc.T @ Wc) / (R - 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(evals)[::-1]
    evals_sorted = evals[order]
    evecs_sorted = evecs[:, order]
    top_vec = evecs_sorted[:, 0]
    # sign-normalize top vec
    if top_vec[0] < 0: top_vec = -top_vec

    # the simplex tangent space is 3-dim (sum=0). report top 3 eigenvalues of cov
    top3 = evals_sorted[:3].tolist()

    # std ratios relative to t
    st = stds['t']
    ratios = {name: (float(stds[name]/st) if st > 0 else float('inf'))
              for name in modes}

    mubar_rho = float(np.dot(wstar, mus))
    varmu_rho = float(np.dot(wstar, (mus - mubar_rho)**2))
    pin_lhs = float((mus[-1] - mubar_rho) * (mubar_rho - mus[0]))
    pin_rhs = float(BW**2 + varmu_rho)

    rho_res = {
        "seed": seed, "dmu": float(dmu), "mus": mus.tolist(),
        "mean_what": What.mean(0).tolist(),
        "stds": stds,
        "ratios_to_t": ratios,
        "interior": interior,
        "cov_eigs_top3_desc": top3,
        "cov_eigs_all_desc": evals_sorted.tolist(),
        "top_eigvec": top_vec.tolist(),
        "lambda2_over_lambda1": float(evals_sorted[1]/evals_sorted[0]) if evals_sorted[0] > 0 else float('inf'),
        "pinning_lhs": pin_lhs, "pinning_rhs": pin_rhs,
        "pinning_predicts_pinned": bool(pin_lhs < pin_rhs),
        "elapsed_s": time.time() - t0,
    }
    results["per_rho"][f"{rho:.2f}"] = rho_res

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== rho={rho} (seed={seed}, dmu={dmu:.4f}) elapsed={rho_res['elapsed_s']:.1f}s ===")
    print("mean w_hat =", [f"{x:.6e}" for x in What.mean(0)])
    print("interior w1: max=%.6e mean=%.6e ; w2: max=%.6e mean=%.6e"
          % (interior['w1_max'], interior['w1_mean'], interior['w2_max'], interior['w2_mean']))
    print("stds:", {k: f"{v:.6e}" for k, v in stds.items()})
    print("ratios to t:", {k: f"{v:.6f}" for k, v in ratios.items()})
    print("cov eigs (desc):", [f"{x:.6e}" for x in evals_sorted])
    print("lambda2/lambda1 =", f"{rho_res['lambda2_over_lambda1']:.6e}")
    print("top eigvec =", [f"{x:.6f}" for x in top_vec])
    print("pinning: lhs=%.4f rhs=%.4f -> pinned=%s" % (pin_lhs, pin_rhs, rho_res['pinning_predicts_pinned']))

print("\nSaved ->", OUT)
