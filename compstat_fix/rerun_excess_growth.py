"""Re-run theory_checks/probe_excess_growth.py grid (K=2..7, rho=.47/.40, N=350, R=600, seeds as in
the original) with the certified solver; the 2000-iteration MU value on identical rows is kept."""
import os, sys, json, math, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'theory_checks')); os.environ['DZ_CORE_DIR'] = ROOT
import numpy as np
import sigma_err_scan as S
import probe_excess_growth as P
S.SOLVER = 'certified'
out = {}
for rho in P.RHOS:
    for K in P.KS:
        new, old, kk = [], [], []
        for sd in P.SEEDS:
            seed = 20260615 + K * 1000 + sd + int(rho * 1e4)
            s = S.measure_cell('numpy', K, rho, P.N, P.R, S.fixed_interior_w(K, 'endpoint'), sigma=P.SIGMA, bw=P.BW,
                               noise=0.0, seed=seed, n_iter=2000)
            fl = P.C_K2 / (math.sqrt(P.N) * rho)
            new.append(s['sigma_err'] / fl); old.append(s['mu2000_sigma_err'] / fl); kk.append(s['kkt_max'])
        out[f'{rho:.2f}_{K}'] = dict(rho=rho, K=K, excess_new=new, excess_old=old, new_mean=float(np.mean(new)),
                                     old_mean=float(np.mean(old)), kkt_max=float(max(kk)))
        print(rho, K, 'old %.3f new %.3f' % (np.mean(old), np.mean(new)), flush=True)
json.dump(out, open(os.path.join(HERE, 'excess_growth_certified.json'), 'w'), indent=1)
