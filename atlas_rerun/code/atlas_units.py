# -*- coding: utf-8 -*-
"""Atlas rerun with the KKT-certified supervised solver (compstat-fix, Algorithm 2).

Grid, seeds and config ordering are IDENTICAL to the archived atlas
(reference/phase3d_scan_hpc.py defaults): 11 K x 25 sep x 12 noise x 1000 seeds = 3,300,000
configs in that loop order (K, sep, noise, seed), originally split into 1000 tasks of 3300 configs.
Only the 467 original tasks that completed on D107 (reference/target_tasks.json) form the archived
atlas (1,541,100 configs, 1618 observed cells). These target configs are split into
N_UNITS = 1000 work units (about 1542 configs each, about 0.85 core-h).

Each unit writes results/units/unit_XXXX.json atomically (tmp file + rename); an existing file means
the unit is done and will be skipped.  Only the supervised method is computed (the unsupervised
baseline does not use this solver; its archived cell means are kept).
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DZ_CORE_DIR", HERE)

import numpy as np
if not hasattr(np, "trapezoid"):  # numpy<2 compat
    np.trapezoid = np.trapz
import applicability_scan as A           # imports dz_unmixing_experiment from HERE
from phase3d_scan import build_phase3d_configs, run_phase3d_config
from kl_certified import kl_unmix_certified
from kl_certified_v2 import kl_unmix_certified_v2
# Solver: "v2" (default since 2026-10-21) = Algorithm 2 unchanged + exact (floor-free) KKT certificate and
# repair of any row failing it (see kl_certified_v2.py); "alg2" = Algorithm 2 alone (job 269468 behaviour).
SOLVER = os.environ.get("DZ_SOLVER", "v2")

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
SEP_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0,
              2.2, 2.5, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0]
NOISE_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
N_SEEDS = 1000
ORIG_TASK_SIZE = 3300           # ceil(3,300,000 / 1000) as in phase3d_scan_hpc.slice_for_task
N_UNITS = 1000
KKT_TOL = 1e-11
N_CELLS = len(K_VALUES) * len(SEP_VALUES) * len(NOISE_VALUES)    # 3300
N_CONFIGS = N_CELLS * N_SEEDS                                       # 3,300,000


def cell_of_index(i):
    c, s = divmod(i, N_SEEDS)
    k, rem = divmod(c, len(SEP_VALUES) * len(NOISE_VALUES))
    j, e = divmod(rem, len(NOISE_VALUES))
    return K_VALUES[k], SEP_VALUES[j], NOISE_VALUES[e], s + 1


def target_indices():
    tasks = json.load(open(os.path.join(ROOT, "reference", "target_tasks.json")))["completed_original_tasks"]
    idx = []
    for t in sorted(tasks):
        idx.extend(range(t * ORIG_TASK_SIZE, min((t + 1) * ORIG_TASK_SIZE, N_CONFIGS)))
    return idx


def unit_slice(u, n_target):
    per = (n_target + N_UNITS - 1) // N_UNITS
    return u * per, min((u + 1) * per, n_target)


def run_config_certified(i):
    K, sep, eta, seed = cell_of_index(i)
    cfg = build_phase3d_configs([K], [sep], [eta], [seed])[0]
    kk, kf, ke, rep = [], [], [], []

    def cert(X, H):
        if SOLVER == "alg2":
            W, info = kl_unmix_certified(X, H, mu_warm=200, max_newton=200, tol=KKT_TOL, return_info=True)
            kk.append(info["kkt"])
        else:
            W, info = kl_unmix_certified_v2(X, H, return_info=True)
            kk.append(info["kkt"]); kf.append(info["kkt_floor_alg2"]); ke.append(info["kkt_exact_alg2"])
            rep.extend(info["repaired"])
        return W

    orig = A.supervised_unmix
    A.supervised_unmix = cert
    try:
        t0 = time.time()
        r = run_phase3d_config(cfg, methods=("supervised",))
        dt = time.time() - t0
    finally:
        A.supervised_unmix = orig
    k_all = np.concatenate(kk) if kk else np.array([np.nan])
    out = dict(config_idx=i, K=K, sep=sep, noise=eta, seed=seed,
               sup_top1_dom_mean=r["sup_top1_dom_mean"], sup_top1_mean=r["sup_top1_mean"],
               sup_mae_mean=r["sup_mae_mean"], sup_r2_mean=r["sup_r2_mean"],
               kkt_max=float(np.nanmax(k_all)), n_rows=int(k_all.size),
               n_uncertified=int(np.sum(~(k_all <= KKT_TOL))), sec=dt, solver=SOLVER)
    if SOLVER != "alg2":
        kf = np.concatenate(kf); ke = np.concatenate(ke)
        out.update(kkt_definition="exact", alg2_floor_kkt_max=float(np.max(kf)),
                   alg2_floor_n_unc=int(np.sum(~(kf <= KKT_TOL))), alg2_exact_kkt_max=float(np.max(ke)),
                   alg2_exact_n_unc=int(np.sum(~(ke <= KKT_TOL))), n_repaired=len(rep),
                   n_argmax_changed=int(sum(x["argmax_changed"] for x in rep)),
                   repair_dw_inf_max=float(max([x["dw_inf"] for x in rep], default=0.0)))
    return out


def run_unit(u, outdir, max_configs=0):
    """Run work unit u; returns the output path (skips if it already exists)."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"unit_{u:04d}.json")
    if os.path.exists(path):
        return path, "skipped"
    cfg_file = os.environ.get("DZ_CONFIG_LIST")       # subset rerun: unit u = lines [u*per, (u+1)*per)
    if cfg_file:
        idx = [int(l) for l in open(cfg_file) if l.strip()]
        per = int(os.environ.get("DZ_CONFIGS_PER_UNIT", "50"))
        a, b = u * per, min((u + 1) * per, len(idx))
    else:
        idx = target_indices()
        a, b = unit_slice(u, len(idx))
    sel = idx[a:b]
    if max_configs:
        sel = sel[:max_configs]
    t0 = time.time()
    rows = [run_config_certified(i) for i in sel]
    out = dict(unit=u, n_units=N_UNITS, target_slice=[a, b], max_configs=max_configs,
               n_configs=len(rows), wall_sec=time.time() - t0, kkt_tol=KKT_TOL,
               host=os.uname().nodename, solver=SOLVER, config_list=os.environ.get("DZ_CONFIG_LIST", ""), rows=rows)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    return path, f"done {len(rows)} configs in {out['wall_sec']:.0f}s"


if __name__ == "__main__":
    u = int(sys.argv[1]); outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "units")
    print(run_unit(u, outdir))
