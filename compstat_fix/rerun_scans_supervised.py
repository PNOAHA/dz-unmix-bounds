"""Re-run the supervised half of the single-axis scans with the certified solver (compstat-fix).

For every row of results/applicability_{sep,noise,srcK}.csv the dataset is regenerated with
the archived seed and grid value (scan/applicability_scan.run_one_ext, identical data path and
KFold partition).  Each test fold is solved twice on identical inputs:
  (a) dz_unmixing_experiment.supervised_unmix defaults (500 it, tol 1e-7) -> must reproduce the
      archived sup_* columns (reproducibility check);
  (b) kl_certified.kl_unmix_certified (KKT residual <= 1e-11).
The unsupervised columns are copied unchanged from the archive (they do not use this solver).
"""
import os, sys, csv, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scan')); sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np
import applicability_scan as A
from kl_certified import kl_unmix_certified

SUPK = ['sup_top1_mean', 'sup_top1_std', 'sup_top1_dom_mean', 'sup_top1_dom_std', 'sup_mae_mean',
        'sup_mae_std', 'sup_r2_mean', 'sup_r2_std']


def cfg_from_row(r):
    axis = r['axis']; seed = int(r['seed'])
    if axis == 'sep':
        cfgs = A.build_axis_configs('sep', [seed], sep_grid=[float(r['axis_value'])])
    elif axis == 'noise':
        cfgs = A.build_axis_configs('noise', [seed], noise_grid=[float(r['axis_value'])])
    else:
        k = int(float(r['axis_value'])); cfgs = A.build_axis_configs('srcK', [seed], k_values=[k])
    return cfgs[0]


def run(row):
    t0 = time.time(); cfg = cfg_from_row(row)
    out = {}
    kk = []
    def cert(X, H):
        W, info = kl_unmix_certified(X, H, return_info=True); kk.append(info['kkt'].max()); return W
    for tag, fn in (('old', A.supervised_unmix.__wrapped__ if hasattr(A.supervised_unmix, '__wrapped__') else A.supervised_unmix), ('new', cert)):
        orig = A.supervised_unmix
        A.supervised_unmix = fn
        try:
            res = A.run_one_ext(sources=cfg['sources'], dirichlet_alpha=cfg['dirichlet_alpha'],
                                noise_pct=cfg['noise_pct'], bandwidth=cfg['bandwidth'], seed=cfg['seed'],
                                methods=('supervised',))
        finally:
            A.supervised_unmix = orig
        out[tag] = {k: res[k] for k in SUPK}
    dev = max(abs(out['old'][k] - float(row[k])) for k in SUPK)
    return dict(config_idx=row['config_idx'], axis=row['axis'], old_reprod_maxdev=dev, kkt_max=float(max(kk)),
                new=out['new'], sec=time.time() - t0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--axes', nargs='+', default=['sep', 'noise', 'srcK'])
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results_certified'))
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    for axis in a.axes:
        src = os.path.join(ROOT, 'results', f'applicability_{axis}.csv')
        rows = list(csv.DictReader(open(src, newline='')))
        if a.limit: rows = rows[:a.limit]
        t0 = time.time(); res = {}
        with Pool(a.procs) as p:
            for i, r in enumerate(p.imap_unordered(run, rows, chunksize=2)):
                res[r['config_idx']] = r
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print(f'{axis} {i+1}/{len(rows)} {time.time()-t0:.0f}s maxdev={max(v["old_reprod_maxdev"] for v in res.values()):.2e}', flush=True)
        fields = list(rows[0].keys()) + ['sup_solver_kkt_max', 'archived_sup_reprod_maxdev']
        with open(os.path.join(a.out_dir, f'applicability_{axis}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in rows:
                q = dict(r); n = res[r['config_idx']]
                for k in SUPK: q[k] = repr(float(n['new'][k]))
                q['threshold_met'] = str(bool(n['new']['sup_top1_dom_mean'] >= 0.85))
                q['sup_solver_kkt_max'] = repr(n['kkt_max']); q['archived_sup_reprod_maxdev'] = repr(n['old_reprod_maxdev'])
                w.writerow(q)
        print(axis, 'done', time.time() - t0, flush=True)
