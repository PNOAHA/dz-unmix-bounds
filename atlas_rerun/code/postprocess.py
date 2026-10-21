# -*- coding: utf-8 -*-
"""Post-process the certified atlas rerun.

  python code/postprocess.py [--units-dir results/units] [--out results] [--smoke]

1. Completeness: all 1000 unit files present; 1,541,100 configs; config_idx set == target set;
   per-cell config counts == archived n_seeds for all 1618 observed cells (1464 x 1000 + 154 partial).
2. KKT pass check: every supervised row must have KKT residual <= 1e-11; uncertified rows/configs
   are counted and listed (results/kkt_uncertified.csv). Uncertified configs are kept but flagged.
3. Rebuilds results/phase3d_cellsummary_certified.csv (same columns as the archived file;
   sup_top1_dom_mean = nanmean over seeds with the certified solver; uns_top1_dom_mean copied from
   the archived summary because the unsupervised baseline was not rerun), and
   results/phase3d_cell_compare.csv (archived vs certified per cell, threshold 0.85 flips).
4. Fig. 3: results/figures/Fig3_atlas_certified.{pdf,png,svg} (and the archived summary rendered by the
   same script, Fig3_atlas_archived_samescript.*, for a like-for-like visual comparison).
5. results/postprocess_report.json + console summary. Exit code 0 iff complete (always 0 in --smoke mode).
   Certification is reported separately (ALL_ROWS_CERTIFIED); uncertified rows are listed, never dropped.
"""
import os, sys, json, glob, csv, argparse, subprocess, shutil, math
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np
import atlas_units as AU

THRESH = 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", default=os.path.join(ROOT, "results", "units"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.units_dir, "unit_*.json")))
    rows, units, wall = [], [], 0.0
    for f in files:
        d = json.load(open(f)); units.append(d["unit"]); rows.extend(d["rows"]); wall += sum(r["sec"] for r in d["rows"])
    rep = dict(units_present=len(units), units_expected=AU.N_UNITS, configs=len(rows))
    target = AU.target_indices()
    got = [r["config_idx"] for r in rows]
    rep["duplicate_configs"] = len(got) - len(set(got))
    rep["configs_outside_target"] = len(set(got) - set(target))
    rep["configs_missing"] = len(set(target) - set(got))
    missing_units = sorted(set(range(AU.N_UNITS)) - set(units))
    rep["missing_units"] = missing_units[:50] + (["..."] if len(missing_units) > 50 else [])
    rep["n_missing_units"] = len(missing_units)
    # archived summary
    arch = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "reference", "phase3d_cellsummary_archived.csv"))):
        key = (int(r["n_sources_request"]), round(float(r["sep_index"]), 4), round(float(r["noise_pct"]), 4))
        arch[key] = r
    # per-cell aggregation
    cell = defaultdict(list)
    for r in rows:
        cell[(r["K"], round(r["sep"], 4), round(r["noise"], 4))].append(r)
    cnt_mismatch = [(k, len(v), int(arch[k]["n_seeds"]) if k in arch else 0) for k, v in cell.items()
                    if k not in arch or len(v) != int(arch[k]["n_seeds"])]
    rep["cells_with_results"] = len(cell)
    rep["cells_expected"] = len(arch)
    rep["cells_full_1000"] = sum(1 for v in cell.values() if len(v) == 1000)
    rep["cell_count_mismatches"] = len(cnt_mismatch)
    # KKT
    kk = np.array([r["kkt_max"] for r in rows]) if rows else np.array([])
    unc = [r for r in rows if r["n_uncertified"] > 0]
    rep["rows_total"] = int(sum(r["n_rows"] for r in rows))
    rep["rows_uncertified"] = int(sum(r["n_uncertified"] for r in rows))
    rep["configs_with_uncertified_row"] = len(unc)
    rep["kkt_max"] = float(kk.max()) if kk.size else None
    # certificate type per config: "exact" = floor-free KKT (solver v2, kl_certified_v2.py);
    # "floored" = Algorithm 2's certificate computed with m >= 1e-300 (job 269468 rows)
    kd = [r.get("kkt_definition", "exact" if r.get("solver") == "v2" else "floored") for r in rows]
    rep["configs_kkt_exact"] = sum(1 for x in kd if x == "exact")
    rep["configs_kkt_floored"] = sum(1 for x in kd if x == "floored")
    rep["rows_uncertified_exact"] = int(sum(r["n_uncertified"] for r, x in zip(rows, kd) if x == "exact"))
    rep["rows_uncertified_floored"] = int(sum(r["n_uncertified"] for r, x in zip(rows, kd) if x == "floored"))
    if any("alg2_exact_n_unc" in r for r in rows):
        rep["resolved_configs_alg2_rows_failing_exact_kkt"] = int(sum(r.get("alg2_exact_n_unc", 0) for r in rows))
        rep["resolved_configs_rows_repaired"] = int(sum(r.get("n_repaired", 0) for r in rows))
        rep["resolved_configs_rows_argmax_changed"] = int(sum(r.get("n_argmax_changed", 0) for r in rows))
    with open(os.path.join(a.out, "kkt_uncertified.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config_idx", "K", "sep", "noise", "seed", "kkt_max", "n_uncertified", "n_rows"])
        for r in unc:
            w.writerow([r["config_idx"], r["K"], r["sep"], r["noise"], r["seed"], r["kkt_max"], r["n_uncertified"], r["n_rows"]])
    rep["core_hours_measured"] = wall / 3600.0
    # cell summary (certified) + comparison
    summ = os.path.join(a.out, "phase3d_cellsummary_certified.csv")
    comp = os.path.join(a.out, "phase3d_cell_compare.csv")
    flips, diffs = 0, []
    with open(summ, "w", newline="") as f1, open(comp, "w", newline="") as f2:
        w1 = csv.writer(f1); w2 = csv.writer(f2)
        w1.writerow(["n_sources_request", "sep_index", "noise_pct", "sup_top1_dom_mean", "uns_top1_dom_mean", "n_seeds"])
        w2.writerow(["n_sources_request", "sep_index", "noise_pct", "n_seeds_archived", "n_seeds_certified",
                     "sup_dom_archived", "sup_dom_certified", "diff", "pass_archived", "pass_certified", "flip",
                     "configs_with_uncertified_row"])
        for key in sorted(arch):
            v = cell.get(key, [])
            vals = np.array([r["sup_top1_dom_mean"] for r in v], float)
            vals = vals[np.isfinite(vals)] if vals.size else vals
            new = float(vals.mean()) if vals.size else float("nan")
            old = float(arch[key]["sup_top1_dom_mean"])
            if v:
                w1.writerow([key[0], key[1], key[2], repr(new), arch[key]["uns_top1_dom_mean"], len(v)])
            if v and math.isfinite(new) and math.isfinite(old):
                fl = (old >= THRESH) != (new >= THRESH); flips += fl; diffs.append(new - old)
            else:
                fl = ""
            w2.writerow([key[0], key[1], key[2], arch[key]["n_seeds"], len(v), old, new,
                         (new - old) if v else "", old >= THRESH, (new >= THRESH) if v else "", fl,
                         sum(1 for r in v if r["n_uncertified"] > 0)])
    d = np.array(diffs)
    rep["cells_compared"] = int(d.size)
    rep["threshold_flips_0.85"] = int(flips)
    if d.size:
        rep["cell_diff_mean"] = float(d.mean()); rep["cell_diff_mean_abs"] = float(np.abs(d).mean())
        rep["cell_diff_max_abs"] = float(np.abs(d).max())
    complete = (rep["n_missing_units"] == 0 and rep["configs_missing"] == 0 and rep["duplicate_configs"] == 0
                and rep["configs_outside_target"] == 0 and rep["cell_count_mismatches"] == 0
                and rep["cells_with_results"] == 1618 and rep["cells_full_1000"] == 1464 and rep["configs"] == 1541100)
    rep["COMPLETE"] = bool(complete)
    rep["ALL_ROWS_CERTIFIED"] = rep["rows_uncertified"] == 0
    rep["smoke"] = a.smoke
    # figures
    if not a.no_figure and rows:
        figdir = os.path.join(a.out, "figures"); os.makedirs(figdir, exist_ok=True)
        for tag, csvp in (("certified", summ), ("archived_samescript", os.path.join(ROOT, "reference", "phase3d_cellsummary_archived.csv"))):
            env = dict(os.environ, PHASE3D_CSV=csvp, DZ_FIGOUT=figdir, MPLBACKEND="Agg")
            r = subprocess.run([sys.executable, os.path.join(HERE, "make_fig6_phase_atlas.py")], env=env,
                               capture_output=True, text=True)
            rep[f"figure_{tag}_exit"] = r.returncode
            if r.returncode == 0:
                for ext in ("pdf", "png", "svg"):
                    src = os.path.join(figdir, f"Fig6_phase_atlas.{ext}")
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(figdir, f"Fig3_atlas_{tag}.{ext}"))
            else:
                rep[f"figure_{tag}_stderr"] = r.stderr[-2000:]
    json.dump(rep, open(os.path.join(a.out, "postprocess_report.json"), "w"), indent=1)
    for k, v in rep.items():
        if not k.endswith("stderr"):
            print(f"{k}: {v}")
    ok = complete or a.smoke
    print("RESULT:", ("PASS (complete)" if complete else "SMOKE (completeness not required)") if ok else "INCOMPLETE (see report)")
    print("KKT certificate type:", f"{rep['configs_kkt_exact']} configs exact (floor-free), {rep['configs_kkt_floored']} configs floored (Algorithm 2, m>=1e-300)")
    print("KKT:", "all rows certified" if rep["ALL_ROWS_CERTIFIED"] else
          f"WARNING {rep['rows_uncertified']} uncertified rows in {rep['configs_with_uncertified_row']} configs (kkt_max {rep['kkt_max']:.2e}); see kkt_uncertified.csv")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
