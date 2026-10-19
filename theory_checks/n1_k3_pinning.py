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

# ---------------- experiment N1 ----------------
K = 3
W_STAR = np.array([0.6, 1.0, 0.8]) / 2.4   # [0.25, 0.416667, 0.333333]
N = 20000
R = 200
N_ITER = 40000
RHOS = [0.05, 0.09, 0.13, 0.18, 0.25, 0.30, 0.35]
SQRT6 = math.sqrt(6.0); SQRT2 = math.sqrt(2.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n1_k3_pinning.json')

def build_X(rho, n_samples, seed_int):
    mus, dmu = peaks(K, rho)
    rng = np.random.default_rng(1234 + seed_int)
    X = np.empty((n_samples, TGRID.shape[0]))
    for r in range(n_samples):
        ages = gen(W_STAR, N, mus, rng)
        X[r] = kde(ages)
    return X, mus, dmu

def analyze(Wh):
    w0, w1, w2 = Wh[:,0], Wh[:,1], Wh[:,2]
    c = (w0 - 2*w1 + w2) / SQRT6
    c_alt = (1.0 - 3.0*w1) / SQRT6
    p = (w0 - w2) / SQRT2
    cov = np.cov(np.vstack([c, p]))          # 2x2, ddof=1
    evals = np.linalg.eigvalsh(cov)          # ascending
    lam2, lam1 = float(evals[0]), float(evals[1])
    return {
        "w2_max": float(w1.max()), "w2_mean": float(w1.mean()), "w2_min": float(w1.min()),
        "curv_mean": float(c.mean()), "curv_std": float(c.std(ddof=0)),
        "curv_identity_maxabsdiff": float(np.max(np.abs(c - c_alt))),
        "pair_std": float(p.std(ddof=0)),
        "lambda1": lam1, "lambda2": lam2,
        "lambda2_over_lambda1": float(lam2/lam1) if lam1 != 0 else float('nan'),
        "mean_what": [float(w0.mean()), float(w1.mean()), float(w2.mean())],
    }

results = {"meta": {"K": K, "w_star": W_STAR.tolist(), "N": N, "R": R, "n_iter": N_ITER,
                    "rhos": RHOS, "SIGMA": SIGMA, "BW": BW, "T": int(TGRID.shape[0])},
           "cells": {}, "rho005_niter20000": None}

t_all = time.time()
for rho in RHOS:
    t0 = time.time()
    seed_int = int(rho*1000)
    X, mus, dmu = build_X(rho, R, seed_int)
    t_kde = time.time() - t0
    t1 = time.time()
    Wh = unmix(X, H_of(K, rho), N_ITER)
    t_unmix = time.time() - t1
    stats = analyze(Wh)
    stats["rho"] = rho; stats["dmu"] = float(dmu); stats["seed_int"] = seed_int
    stats["t_kde_s"] = round(t_kde,1); stats["t_unmix_s"] = round(t_unmix,1)
    results["cells"]["%.2f" % rho] = stats
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("rho=%.2f dmu=%.3f seed=%d | w2[max=%.3e mean=%.3e min=%.3e] "
          "curv[mean=%.10f std=%.3e id=%.2e] pair_std=%.4e lam2/lam1=%.3e | kde=%.1fs unmix=%.1fs"
          % (rho, dmu, 1234+seed_int, stats["w2_max"], stats["w2_mean"], stats["w2_min"],
             stats["curv_mean"], stats["curv_std"], stats["curv_identity_maxabsdiff"],
             stats["pair_std"], stats["lambda2_over_lambda1"], t_kde, t_unmix), flush=True)

# rho=0.05 convergence-stability check at n_iter=20000 (same seeds/data)
rho = 0.05; seed_int = int(rho*1000)
X, mus, dmu = build_X(rho, R, seed_int)
Wh2 = unmix(X, H_of(K, rho), 20000)
s2 = analyze(Wh2)
s2["rho"] = rho; s2["n_iter"] = 20000
results["rho005_niter20000"] = s2
print("rho=0.05 n_iter=20000 | w2_max=%.3e curv_std=%.3e curv_mean=%.10f pair_std=%.4e"
      % (s2["w2_max"], s2["curv_std"], s2["curv_mean"], s2["pair_std"]), flush=True)

# -------- verdicts --------
def is_pinned(c): return c["w2_max"] < 1e-6
pinned = {r:c for r,c in results["cells"].items() if is_pinned(c)}
interior = {r:c for r,c in results["cells"].items() if not is_pinned(c)}

# P3 fit: log(pair_std) vs log(rho) over pinned cells
if len(pinned) >= 2:
    rs = np.array([c["rho"] for c in pinned.values()])
    ps = np.array([c["pair_std"] for c in pinned.values()])
    order = np.argsort(rs); rs, ps = rs[order], ps[order]
    slope, intercept = np.polyfit(np.log(rs), np.log(ps), 1)
    p3_slope = float(slope)
else:
    p3_slope = None

verd = {}
# P1: rho<=0.25 -> max w2 < 1e-6
p1_cells = {r:c for r,c in results["cells"].items() if c["rho"] <= 0.25}
verd["P1_max_w2_rho_le_0.25"] = {r: c["w2_max"] for r,c in sorted(p1_cells.items())}
verd["P1_pass"] = all(c["w2_max"] < 1e-6 for c in p1_cells.values())
# P2: pinned cells curv_mean == 1/sqrt(6) to >=6 dp AND curv_std<1e-8
TARGET = 1.0/SQRT6
verd["P2_target_curv_mean"] = TARGET
verd["P2_pinned_curv_mean_absdiff"] = {r: abs(c["curv_mean"]-TARGET) for r,c in sorted(pinned.items())}
verd["P2_pinned_curv_std"] = {r: c["curv_std"] for r,c in sorted(pinned.items())}
verd["P2_pass"] = all(abs(c["curv_mean"]-TARGET) < 5e-7 and c["curv_std"] < 1e-8 for c in pinned.values())
# P3
verd["P3_pair_std_pinned"] = {r: c["pair_std"] for r,c in sorted(pinned.items())}
verd["P3_fit_slope"] = p3_slope
verd["P3_pass"] = (p3_slope is not None and -1.4 < p3_slope < -0.6
                   and all(1e-4 < c["pair_std"] < 5e-2 for c in pinned.values()))
# P4 transition
verd["P4_w2max_all"] = {r: c["w2_max"] for r,c in sorted(results["cells"].items())}
verd["P4_curvstd_all"] = {r: c["curv_std"] for r,c in sorted(results["cells"].items())}
verd["P4_rho030_interior"] = (results["cells"]["0.30"]["w2_max"] > 1e-3 and results["cells"]["0.30"]["curv_std"] > 1e-8)
verd["P4_rho035_interior"] = (results["cells"]["0.35"]["w2_max"] > 1e-3 and results["cells"]["0.35"]["curv_std"] > 1e-8)
verd["P4_rho025_pinned"] = is_pinned(results["cells"]["0.25"])
# transition boundary = last pinned rho -> first interior rho
sorted_cells = sorted(results["cells"].values(), key=lambda c: c["rho"])
last_pinned = max([c["rho"] for c in sorted_cells if is_pinned(c)], default=None)
first_interior = min([c["rho"] for c in sorted_cells if not is_pinned(c)], default=None)
verd["P4_last_pinned_rho"] = last_pinned
verd["P4_first_interior_rho"] = first_interior
verd["P4_pass"] = (verd["P4_rho025_pinned"] and verd["P4_rho030_interior"] and verd["P4_rho035_interior"])
# P5: pinned tangent cov rank-1
verd["P5_lambda2_over_lambda1_pinned"] = {r: c["lambda2_over_lambda1"] for r,c in sorted(pinned.items())}
verd["P5_pass"] = all(c["lambda2_over_lambda1"] < 1e-10 for c in pinned.values())

results["verdicts"] = verd
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print("\n==== VERDICTS ====")
for k in ["P1_pass","P2_pass","P3_pass","P4_pass","P5_pass"]:
    print(k, verd[k])
print("P3_fit_slope", p3_slope)
print("last_pinned_rho", last_pinned, "first_interior_rho", first_interior)
print("pinned cells:", sorted(pinned.keys()), "interior cells:", sorted(interior.keys()))
print("total time %.1fs" % (time.time()-t_all))
print("SAVED", OUT)
