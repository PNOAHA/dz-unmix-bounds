# -*- coding: utf-8 -*-
"""One Slurm array element = one whole node: runs its share of the 1000 work units in a process pool.

Usage: python code/run_node.py ELEMENT N_ELEMENTS [--procs 32] [--outdir results/units]
                               [--units 0,1] [--max-configs N]
Element e owns units u with u % N_ELEMENTS == e. Finished units (existing JSON) are skipped, so
resubmitting the same array resumes. Logs one line per unit.
"""
import os, sys, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from multiprocessing import Pool
import atlas_units as AU


def _work(args):
    u, outdir, maxc = args
    t0 = time.time()
    try:
        p, msg = AU.run_unit(u, outdir, maxc)
        return u, msg, time.time() - t0
    except Exception as e:                       # report, keep the pool alive
        return u, f"FAILED {type(e).__name__}: {e}", time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("element", type=int); ap.add_argument("n_elements", type=int)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "units"))
    ap.add_argument("--units", default="", help="explicit comma list (overrides element ownership)")
    ap.add_argument("--max-configs", type=int, default=0, help="smoke test: configs per unit")
    a = ap.parse_args()
    n_units = AU.N_UNITS
    if os.environ.get("DZ_CONFIG_LIST"):                 # subset rerun (see README section 9)
        n_cfg = sum(1 for l in open(os.environ["DZ_CONFIG_LIST"]) if l.strip())
        per = int(os.environ.get("DZ_CONFIGS_PER_UNIT", "50"))
        n_units = (n_cfg + per - 1) // per
    units = [int(x) for x in a.units.split(",")] if a.units else \
        [u for u in range(n_units) if u % a.n_elements == a.element]
    todo = [u for u in units if not os.path.exists(os.path.join(a.outdir, f"unit_{u:04d}.json"))]
    print(f"element {a.element}/{a.n_elements}: solver={AU.SOLVER} list={os.environ.get('DZ_CONFIG_LIST', '') or 'full target'} {len(units)} units, {len(todo)} to run, procs={a.procs}", flush=True)
    t0 = time.time(); nfail = 0
    with Pool(min(a.procs, max(1, len(todo)))) as p:
        for u, msg, dt in p.imap_unordered(_work, [(u, a.outdir, a.max_configs) for u in todo]):
            nfail += msg.startswith("FAILED")
            print(f"unit {u:04d}: {msg} ({dt:.0f}s)", flush=True)
    print(f"element {a.element} finished in {time.time()-t0:.0f}s, failures={nfail}", flush=True)
    sys.exit(1 if nfail else 0)
