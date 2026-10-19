# -*- coding: utf-8 -*-
# N4: active-set stability vs N + pinned-coordinate decay rate
# Self-contained, mirrors sigma_err_scan.py numpy_ref backend EXACTLY
# except: NO early stopping, fixed n_iter. ASCII-only, float64.
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n4_stability_decay.json')

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

# ---- Part 1 ----------------------------------------------------------------
def part1():
    K = 3
    w = np.array([0.6, 1.0, 0.8]) / 2.4
    rho = 0.09
    n_iter = 40000
    mus, _ = peaks(K, rho)
    H = H_of(K, rho)
    grid = [(500, 200), (2000, 200), (8000, 200), (32000, 200), (128000, 100)]
    rows = []
    for N, R in grid:
        t0 = time.time()
        rng = np.random.default_rng(9999 + N)
        X = np.stack([kde(gen(w, N, mus, rng)) for _ in range(R)])
        what = unmix(X, H, n_iter)        # (R,K), rows sum to 1
        w1 = what[:, 1]                    # interior (pinned) coordinate
        curv = what[:, 0] - 2*what[:, 1] + what[:, 2]   # (1,-2,1)
        pair = what[:, 0] - what[:, 2]                  # (1,0,-1)
        row = dict(
            N=int(N), R=int(R),
            P_w1_gt_1e4=float(np.mean(w1 > 1e-4)),
            P_w1_gt_1e2=float(np.mean(w1 > 1e-2)),
            curv_std=float(np.std(curv)),
            pair_std=float(np.std(pair)),
            pair_std_sqrtN=float(np.std(pair) * math.sqrt(N)),
            w1_mean=float(np.mean(w1)),
            w1_max=float(np.max(w1)),
            what_mean=[float(x) for x in what.mean(0)],
            secs=round(time.time()-t0, 1),
        )
        rows.append(row)
        print("P1 N=%7d R=%3d  P(w1>1e-4)=%.4f P(w1>1e-2)=%.4f  curv_std=%.3e pair_std=%.3e  pw*sqrtN=%.4e  (%.1fs)"
              % (N, R, row['P_w1_gt_1e4'], row['P_w1_gt_1e2'], row['curv_std'], row['pair_std'],
                 row['pair_std_sqrtN'], row['secs']), flush=True)
        _save({"part1_partial": rows})
    # S3: fit pair_std ~ N^alpha
    Ns = np.array([r['N'] for r in rows], dtype=float)
    ps = np.array([r['pair_std'] for r in rows], dtype=float)
    alpha, logc = np.polyfit(np.log(Ns), np.log(ps), 1)
    return rows, dict(pair_std_exponent_alpha=float(alpha),
                      pair_std_intercept_logc=float(logc),
                      note="pair_std ~ exp(logc) * N^alpha; expect alpha ~ -0.5")

# ---- Part 2 ----------------------------------------------------------------
def gbar(w, sigma_eff):
    # population KDE-smoothed mixture: sum_k w_k N(mu_k, sigma_eff), area-normalized on TGRID
    mus, _ = peaks(3, _P2_RHO)
    g = np.exp(-0.5*((TGRID[None,:] - mus[:,None])/sigma_eff)**2)
    g = g / TRAPZ(g, TGRID, axis=1)[:,None]
    d = (w[:,None] * g).sum(0)
    return d / TRAPZ(d, TGRID)

_P2_RHO = None
def unmix_record(X, H, n_iter, every=100):
    # R=1 unmix recording w_2 (interior, index 1) every `every` iters
    R, K = X.shape[0], H.shape[0]
    rng = np.random.default_rng(0)
    W = rng.uniform(0.1, 1.0, size=(R, K)); eps = 1e-10
    Hsum = H.sum(1)[None, :]
    iters = []; w2 = []
    for it in range(n_iter):
        WH = W @ H + eps
        W = W * ((X / WH) @ H.T) / (Hsum + eps)
        if (it+1) % every == 0:
            Wn = W / W.sum(1, keepdims=True)
            iters.append(it+1); w2.append(float(Wn[0, 1]))
    # KKT ratio for the interior coord at converged W (multiplicative update factor)
    WH = W @ H + eps
    ratio = ((X / WH) @ H.T) / (Hsum + eps)   # (1,K)
    ratio_1 = float(ratio[0, 1])
    Wn = W / W.sum(1, keepdims=True)
    return np.array(iters), np.array(w2), ratio_1, float(Wn[0,1])

def part2():
    global _P2_RHO
    K = 3
    w = np.array([0.6, 1.0, 0.8]) / 2.4
    sigma_eff = math.sqrt(SIGMA**2 + BW**2)
    n_iter = 60000; every = 100
    out = []
    for rho in (0.05, 0.09, 0.13):
        _P2_RHO = rho
        t0 = time.time()
        H = H_of(K, rho)
        X = gbar(w, sigma_eff)[None, :]    # (1,T)
        iters, w2, ratio_1, w2_final = unmix_record(X, H, n_iter, every)
        # geometric decay in the tail: use last segment where w2 still > 0 (avoid underflow zeros)
        valid = w2 > 0
        iv = iters[valid]; wv = w2[valid]
        # tail: last ~30% of valid samples but ensure >=4 points
        n = len(wv)
        if n >= 6:
            t_start = max(1, int(n*0.6))
        else:
            t_start = 1
        # fit r = w2(t+every)/w2(t) as geometric mean ratio over the tail
        seg = wv[t_start:]
        if len(seg) >= 2:
            r = float((seg[-1]/seg[0]) ** (1.0/(len(seg)-1)))   # ratio per `every` steps
        else:
            r = float(wv[-1]/wv[-2]) if n >= 2 else float('nan')
        eta_decay = 1.0 - r ** (1.0/every)
        eta_kkt = 1.0 - ratio_1
        out.append(dict(
            rho=float(rho),
            sigma_eff=float(sigma_eff),
            w2_first=float(w2[0]), w2_final=float(w2_final),
            n_valid_pos=int(n), tail_start_idx=int(t_start),
            r_per_block=float(r),
            eta_decay=float(eta_decay),
            eta_kkt=float(eta_kkt),
            ratio_1=float(ratio_1),
            abs_diff=float(abs(eta_decay - eta_kkt)),
            eta_4000=float(4000*eta_kkt),
            secs=round(time.time()-t0,1),
        ))
        print("P2 rho=%.2f  w2_first=%.4e w2_final=%.4e  r=%.6f  eta_decay=%.6e eta_kkt=%.6e  |diff|=%.3e  4000*eta=%.2f  (%.1fs)"
              % (rho, w2[0], w2_final, r, eta_decay, eta_kkt, abs(eta_decay-eta_kkt), 4000*eta_kkt, out[-1]['secs']), flush=True)
        _save({"part2_partial": out})
    return out

def _save(extra):
    data = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(extra)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT)

if __name__ == "__main__":
    T0 = time.time()
    print("== Part 1 ==", flush=True)
    p1_rows, p1_fit = part1()
    print("== Part 2 ==", flush=True)
    p2 = part2()
    final = {
        "meta": dict(SIGMA=SIGMA, START=START, BW=BW, T=len(TGRID),
                     w_star=[0.6/2.4, 1.0/2.4, 0.8/2.4],
                     part1_rho=0.09, part1_n_iter=40000,
                     part2_n_iter=60000, part2_sigma_eff=math.sqrt(SIGMA**2+BW**2)),
        "part1_rows": p1_rows,
        "part1_S3_fit": p1_fit,
        "part2_rows": p2,
        "total_secs": round(time.time()-T0, 1),
    }
    _save(final)
    print("\nDONE total=%.1fs  wrote %s" % (time.time()-T0, OUT), flush=True)
