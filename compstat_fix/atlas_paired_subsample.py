"""Paired solver comparison on a subsample of the joint atlas (compstat-fix; box only).

The full atlas (1,541,100 configurations) cannot be rerun on the box.  For every populated atlas cell
at eta in {0.06, 0.20, 0.40} (the three Fig. 3 slices) the configurations with seeds 1 and 2 are
regenerated exactly as in scan/phase3d_scan.py, and the supervised fold fits are computed on identical
inputs with (a) the archived solver (500 it, tol 1e-7) and (b) the certified solver.  The unsupervised
half is not affected and is not recomputed.  Wall time of the certified supervised half is recorded for
the cluster estimate.
"""
import os, sys, csv, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scan')); sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np
import applicability_scan as A
import phase3d_scan as P
from kl_certified import kl_unmix_certified

def run(job):
    K, sep, eta, seed = job
    cfg = P.build_phase3d_configs([K], [sep], [eta], [seed])[0]
    out = dict(K=K, sep=sep, eta=eta, seed=seed)
    orig = A.supervised_unmix
    for tag, fn in (('old', orig), ('new', lambda X, H: kl_unmix_certified(X, H))):
        A.supervised_unmix = fn
        t0 = time.time()
        try:
            r = P.run_phase3d_config(cfg, methods=('supervised',))
        finally:
            A.supervised_unmix = orig
        out[tag] = dict(top1_dom=r['sup_top1_dom_mean'], top1=r['sup_top1_mean'], mae=r['sup_mae_mean'], sec=time.time() - t0)
    return out

if __name__ == '__main__':
    cells = list(csv.DictReader(open(os.path.join(ROOT, 'scan', 'phase3d_cellsummary.csv'))))
    jobs = []
    for c in cells:
        eta = float(c['noise_pct'])
        if any(abs(eta - e) < 1e-9 for e in (0.06, 0.20, 0.40)):
            for s in (1, 2):
                jobs.append((int(c['n_sources_request']), float(c['sep_index']), eta, s))
    jobs.sort(key=lambda j: -j[0])
    print('jobs', len(jobs), flush=True)
    res = []; t0 = time.time()
    with Pool(int(os.environ.get('PROCS', '8'))) as p:
        for i, r in enumerate(p.imap_unordered(run, jobs)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(i + 1, len(jobs), f'{time.time()-t0:.0f}s', flush=True)
                json.dump(res, open(os.path.join(HERE, 'atlas_paired_subsample.json'), 'w'))
    json.dump(res, open(os.path.join(HERE, 'atlas_paired_subsample.json'), 'w'))
    print('done', time.time() - t0)
