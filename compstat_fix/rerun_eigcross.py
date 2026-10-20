"""Re-run theory_checks/probe2_clip_eigcross.py (rho=.40, N=350, R=4000, K=2..8, original seeds)
with the certified solver; the original 2000-iteration MU fit on identical rows is kept."""
import os, sys, json, math
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'theory_checks')); sys.path.insert(0, HERE)
os.environ['DZ_CORE_DIR'] = ROOT
import numpy as np
import sigma_err_scan as S
import probe2_clip_eigcross as E
from kl_certified import kl_unmix_certified
core = S._load_real_core()
res = {}
for tag in ('old', 'new'):
    res[tag] = {}
cache = {}
for K in range(2, 9):
    mus, _ = S.peak_ages(K, E.RHO, E.SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(E.SIGMA), f"syn-{i+1}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = S.fixed_interior_w(K, "endpoint")
    rng = np.random.default_rng(20260616 + K)
    X = np.empty((E.R, len(core.T_GRID)))
    for r in range(E.R):
        ages = core.generate_sample(w, n_grains_range=(E.N, E.N), noise_pct=0.0, sources=src, rng=rng)
        X[r] = core.kde_density(ages, t_grid=core.T_GRID, bandwidth=E.BW)
    W_old = core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)
    W_new, info = kl_unmix_certified(X, H, return_info=True)
    for tag, W in (('old', W_old), ('new', W_new)):
        l1, l2, d1, d2 = E.top2_and_deg(W)
        res[tag][K] = dict(lam1=float(l1), lam2=float(l2), deg1=d1, deg2=d2, s1=math.sqrt(max(l1, 0)),
                           boot_sd=E.boot_se(W), frac_below_001=float((W < 0.01).mean()))
    res['new'][K]['kkt_max'] = float(info['kkt'].max())
    print(K, {t: (round(res[t][K]['lam1'], 5), round(res[t][K]['lam2'], 5)) for t in ('old', 'new')}, flush=True)
for tag in ('old', 'new'):
    s2 = res[tag][2]['s1']
    for K in res[tag]:
        res[tag][K]['excess'] = res[tag][K]['s1'] / s2; res[tag][K]['excess_sd'] = res[tag][K]['boot_sd'] / s2
    ex = {K: res[tag][K]['excess'] for K in res[tag]}
    Ks = max(ex, key=ex.get); res[tag]['Kstar'] = Ks
    if Ks + 1 in ex:
        d = ex[Ks] - ex[Ks + 1]; sd = math.hypot(res[tag][Ks]['excess_sd'], res[tag][Ks + 1]['excess_sd'])
        res[tag]['drop'] = d; res[tag]['drop_z'] = d / sd
    print(tag, 'K*', Ks, 'drop', res[tag].get('drop'), 'z', res[tag].get('drop_z'), {K: round(ex[K], 3) for K in ex})
json.dump(res, open(os.path.join(HERE, 'eigcross_certified.json'), 'w'), indent=1)
