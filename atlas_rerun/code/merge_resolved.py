# -*- coding: utf-8 -*-
"""Merge re-solved configurations into a COPY of the unit files (originals are never modified).

  python code/merge_resolved.py --units-dir results/units --out results_fixed/units RESOLVED_DIR [RESOLVED_DIR ...]

RESOLVED_DIR holds JSON files with a "rows" list (unit_*.json from run_node.py with DZ_CONFIG_LIST, or
chunk_*.json from the box re-solve); each row has config_idx and the v2 fields. For every re-solved
config the row in the copy is replaced by the re-solved row (statistics + exact KKT fields); the
Algorithm 2 (job 269468) values are kept in the row as orig_* for traceability. Later directories win.
"""
import os, sys, json, glob, argparse

KEEP = ("sup_top1_dom_mean", "sup_top1_mean", "sup_mae_mean", "sup_r2_mean", "kkt_max", "n_rows", "n_uncertified", "sec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("resolved", nargs="+")
    a = ap.parse_args()
    if os.path.abspath(a.out) == os.path.abspath(a.units_dir):
        sys.exit("refusing to overwrite the original unit files")
    new = {}
    for d in a.resolved:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            for r in json.load(open(f))["rows"]:
                new[r["config_idx"]] = r
    os.makedirs(a.out, exist_ok=True)
    n_rep = 0
    for f in sorted(glob.glob(os.path.join(a.units_dir, "unit_*.json"))):
        u = json.load(open(f))
        for j, r in enumerate(u["rows"]):
            q = new.get(r["config_idx"])
            if q is None:
                r.setdefault("kkt_definition", "floored")
                continue
            assert (q["K"], q["seed"]) == (r["K"], r["seed"]) and abs(q["sep"] - r["sep"]) < 1e-9 and abs(q["noise"] - r["noise"]) < 1e-9
            m = {k: r[k] for k in ("config_idx", "K", "sep", "noise", "seed")}
            for k in KEEP:
                m[k] = q[k]
                m["orig_" + k] = r[k]
            for k, v in q.items():
                if k not in m and k not in ("unresolved",):
                    m[k] = v
            m["kkt_definition"] = "exact"
            u["rows"][j] = m; n_rep += 1
        u["merged_from"] = [os.path.abspath(d) for d in a.resolved]
        tmp = os.path.join(a.out, os.path.basename(f) + ".tmp")
        json.dump(u, open(tmp, "w")); os.replace(tmp, os.path.join(a.out, os.path.basename(f)))
    print(f"re-solved configs available: {len(new)}; replaced in copy: {n_rep}")
    if n_rep != len(new):
        sys.exit("some re-solved configs were not found in the unit files")


if __name__ == "__main__":
    main()
