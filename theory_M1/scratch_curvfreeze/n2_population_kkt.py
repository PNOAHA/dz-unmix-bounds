import numpy as np, math, json, time, os
from scipy.special import erfinv as _erfinv, erf as _erf
def erfinv(x): return float(_erfinv(x))
def erf(x): return float(_erf(x))

SIGMA = 25.0; START = 200.0
TGRID = np.arange(50.0, 2500.0 + 1e-9, 5.0)
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz
NITER = 400000
EPS = 1e-10
PIN_THR = 1e-9                 # wbar below this => pinned (threshold based)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n2_population_kkt.json')

RHO_GRID = [0.02,0.05,0.08,0.11,0.14,0.17,0.20,0.24,0.28,0.32,0.36,0.40,0.45]
SQRT2 = math.sqrt(2.0)

def peaks(K, rho):
    dmu = SIGMA * 2.0 * SQRT2 * erfinv(rho)
    return np.array([START + i*dmu for i in range(K)]), dmu

def H_of(K, rho):
    mus, _ = peaks(K, rho)
    h = np.exp(-0.5*((TGRID[None,:]-mus[:,None])/SIGMA)**2)
    return h / TRAPZ(h, TGRID, axis=1)[:,None]

def wstar(K, kind):
    w = np.linspace(1.0, 0.6, K)
    if kind == 'middle' and K >= 3:
        w = np.roll(w, K//2)
    return w / w.sum()

def gbar_row(K, rho, w, bw):
    mus, _ = peaks(K, rho)
    seff = math.sqrt(SIGMA**2 + bw**2)
    comp = np.exp(-0.5*((TGRID[None,:]-mus[:,None])/seff)**2)
    comp = comp / TRAPZ(comp, TGRID, axis=1)[:,None]
    g = (w[:,None]*comp).sum(0)
    return g / TRAPZ(g, TGRID)

def unmix_cert(X, H, n_iter, track_every=1000):
    """X:(R,T), H:(K,T). KL mult. updates, NO early stop. Returns:
       Wn (R,K normalized), ratio (R,K at converged raw W), trace (S,R,K normalized), its (S,)"""
    R, K = X.shape[0], H.shape[0]
    rng = np.random.default_rng(0)
    W = rng.uniform(0.1, 1.0, size=(R, K))
    Hsum = H.sum(1)[None, :]; HT = H.T
    trace = []; its = []
    for it in range(n_iter):
        WH = W @ H + EPS
        ratio = ((X / WH) @ HT) / (Hsum + EPS)
        W = W * ratio
        if (it % track_every) == 0:
            s = W.sum(1, keepdims=True)
            trace.append((W/s).copy()); its.append(it)
    WH = W @ H + EPS
    ratio = ((X / WH) @ HT) / (Hsum + EPS)     # certificate ratio at converged W
    Wn = W / W.sum(1, keepdims=True)
    return Wn, ratio, np.array(trace), np.array(its)

def tail_rate(its, vals):
    """asymptotic geometric decay rate of one coordinate trace. returns (rate, pinned_bool, slope)."""
    v = np.asarray(vals, float); n = len(v)
    lo = n // 2
    si = its[lo:].astype(float); sv = v[lo:]
    m = np.isfinite(sv) & (sv > 1e-280)
    if m.sum() < 3:
        return 0.0, True, -np.inf            # underflowed -> decayed to 0
    xi = si[m]; yv = np.log(sv[m])
    slope = float(np.polyfit(xi, yv, 1)[0])
    rate = math.exp(slope) if slope > -700 else 0.0
    return rate, (slope < -1e-7), slope

def moment_pred(K, kind, bw):
    """continuous moment-matching prediction. returns dict incl predicted rho*."""
    w = wstar(K, kind)
    k = np.arange(K, dtype=float)
    mbar = float((w*k).sum())
    Vk = float((w*k*k).sum() - mbar*mbar)
    C = (K-1-mbar)*mbar - Vk          # coefficient of dmu^2 in (LHS - Var)
    if C > 0:
        dmu_thr = bw / math.sqrt(C)
        rho_pred = erf(dmu_thr / (2.0*SQRT2*SIGMA))
    else:
        rho_pred = float('nan')       # interior never pinned even at rho->0
    return dict(mbar_k=mbar, Vk=Vk, C=C, rho_pred=rho_pred)

def cell_record(K, rho, bw, kind, Wn_row, ratio_row, trace_rc, its):
    """assemble per-cell record. Wn_row,(K,) ratio_row,(K,) trace_rc:(S,K)"""
    mus, dmu = peaks(K, rho)
    w = wstar(K, kind)
    mubar = float((w*mus).sum())
    eta = (1.0 - ratio_row)
    wbar = Wn_row.copy()
    # rate based classification per coord
    rates = []; pinned_rate = []; slopes = []
    for j in range(K):
        r, pj, sl = tail_rate(its, trace_rc[:, j])
        rates.append(r); pinned_rate.append(bool(pj)); slopes.append(sl)
    pinned_thr = [bool(wbar[j] < PIN_THR) for j in range(K)]
    survivors = [j for j in range(K) if not pinned_rate[j]]
    pinned_set = [j for j in range(K) if pinned_rate[j]]
    eta_pinned = [float(eta[j]) for j in pinned_set]
    rec = dict(K=K, bw=bw, kind=kind, rho=rho, dmu=float(dmu),
               mus=[float(x) for x in mus], mubar=mubar,
               wbar=[float(x) for x in wbar],
               ratio=[float(x) for x in ratio_row],
               eta=[float(x) for x in eta],
               tail_rate=[float(x) for x in rates],
               eta_fit=[float(1.0-x) for x in rates],
               pinned_thr=pinned_thr, pinned_rate=pinned_rate,
               survivors=survivors, pinned_set=pinned_set,
               min_eta_pinned=(float(min(eta_pinned)) if eta_pinned else None))
    # endmember-only survivor prediction
    if survivors == [0, K-1]:
        abar = float(wbar[0])
        abar_pred = float((mus[K-1]-mubar)/(mus[K-1]-mus[0]))
        rec['abar'] = abar; rec['abar_pred'] = abar_pred
        rec['abar_relerr'] = float(abs(abar-abar_pred)/abar_pred)
    return rec

# ---- configs ----
CONFIGS = []
for K in (3,4,5):
    for bw in (6.0,12.0,24.0):
        CONFIGS.append(dict(K=K, bw=bw, kind='middle'))
CONFIGS.append(dict(K=3, bw=12.0, kind='endpoint'))

def cfg_key(c): return (c['K'], c['bw'], c['kind'])

results = dict(meta=dict(SIGMA=SIGMA, START=START, BW_grid=[6.0,12.0,24.0],
                         n_iter=NITER, rho_grid=RHO_GRID, pin_thr=PIN_THR,
                         tgrid_n=len(TGRID)),
               grid={}, threshold={}, predictions={})

def save():
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=1, allow_nan=True)
    os.replace(tmp, OUT)

# ---- GRID (batched by K, rho) ----
t_start = time.time()
for K in (3,4,5):
    cfgsK = [c for c in CONFIGS if c['K'] == K]
    for rho in RHO_GRID:
        H = H_of(K, rho)
        X = np.array([gbar_row(K, rho, wstar(K, c['kind']), c['bw']) for c in cfgsK])
        Wn, ratio, trace, its = unmix_cert(X, H, NITER)
        for r, c in enumerate(cfgsK):
            rec = cell_record(K, rho, c['bw'], c['kind'], Wn[r], ratio[r], trace[:, r, :], its)
            results['grid'].setdefault(str(cfg_key(c)), []).append(rec)
        save()
        print('grid K=%d rho=%.2f done  t=%.1fs' % (K, rho, time.time()-t_start), flush=True)

# ---- THRESHOLD bisection per config ----
def run_single(K, rho, bw, kind):
    H = H_of(K, rho)
    X = np.array([gbar_row(K, rho, wstar(K, kind), bw)])
    Wn, ratio, trace, its = unmix_cert(X, H, NITER)
    rec = cell_record(K, rho, bw, kind, Wn[0], ratio[0], trace[:, 0, :], its)
    return rec

def all_interior_pinned(rec):
    return rec['survivors'] == [0, rec['K']-1]

for c in CONFIGS:
    K, bw, kind = c['K'], c['bw'], c['kind']
    key = str(cfg_key(c))
    recs = results['grid'][key]
    flags = [(r['rho'], all_interior_pinned(r)) for r in recs]
    # find bracket: last pinned rho (True) and first released rho (False) after it
    lo = hi = None
    for i in range(len(flags)-1):
        if flags[i][1] and not flags[i+1][1]:
            lo = flags[i][0]; hi = flags[i+1][0]; break
    pred = moment_pred(K, kind, bw)
    if lo is None:
        # either all pinned or all released
        if all(f[1] for f in flags):
            obs = dict(status='all_pinned_in_grid', rho_star=('>%.3f' % RHO_GRID[-1]))
        else:
            obs = dict(status='none_pinned_in_grid', rho_star=('<%.3f' % RHO_GRID[0]))
        results['threshold'][key] = dict(K=K, bw=bw, kind=kind,
                                         rho_pred=pred['rho_pred'], **obs)
        results['predictions'][key] = pred
        save()
        print('thr %s %s' % (key, obs['status']), flush=True)
        continue
    a, b = lo, hi
    steps = 0
    while (b - a) > 0.005 and steps < 9:
        mid = 0.5*(a+b)
        rec = run_single(K, mid, bw, kind)
        if all_interior_pinned(rec):
            a = mid
        else:
            b = mid
        steps += 1
        print('  bisect %s [%.4f,%.4f] mid=%.4f pinned=%s' %
              (key, a, b, mid, all_interior_pinned(rec)), flush=True)
    rho_star = 0.5*(a+b)
    # moment LHS/RHS at observed
    _, dmu = peaks(K, rho_star)
    w = wstar(K, kind); k = np.arange(K, dtype=float)
    mbar = float((w*k).sum()); Vk = float((w*k*k).sum()-mbar*mbar)
    LHS = dmu*dmu*(K-1-mbar)*mbar
    RHS = bw*bw + dmu*dmu*Vk
    results['threshold'][key] = dict(K=K, bw=bw, kind=kind, status='bracketed',
        rho_star_obs=rho_star, bracket=[a, b], rho_pred=pred['rho_pred'],
        rel_err=(float(abs(rho_star-pred['rho_pred'])/pred['rho_pred'])
                 if pred['rho_pred']==pred['rho_pred'] else None),
        moment_LHS=LHS, moment_RHS=RHS)
    results['predictions'][key] = pred
    save()
    print('thr %s rho*_obs=%.4f rho*_pred=%.4f t=%.1fs' %
          (key, rho_star, pred['rho_pred'], time.time()-t_start), flush=True)

results['meta']['total_seconds'] = time.time()-t_start
save()
print('ALL DONE  total=%.1fs  -> %s' % (time.time()-t_start, OUT), flush=True)
