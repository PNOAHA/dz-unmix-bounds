# -*- coding: utf-8 -*-
"""
合并 phase3d_scan_hpc.py 的 part_*.json
=========================================
输出：
  - phase3d.csv               逐 config 平铺
  - phase3d.json              结构化 + 按 (K, sep, noise) tuple 聚合 + 95% CI
  - 控制台按 noise 切片打印 phase 表

跑法：
  DZ_OUTDIR=results_phase3d /public/software/Anaconda3/bin/python merge_phase3d.py
"""

import os
import json
import glob
import csv
import math
import statistics
from collections import defaultdict

OUTDIR = os.environ.get("DZ_OUTDIR", "results_phase3d")


def _safe(xs):
    return [x for x in xs if x is not None and x == x]


def _mean(xs):
    xs = _safe(xs)
    return statistics.mean(xs) if xs else float("nan")


def _ci95(xs):
    """95% CI half-width using normal approx (xs 通常 N≥30 即可)."""
    xs = _safe(xs)
    if len(xs) < 2:
        return float("nan")
    sd = statistics.stdev(xs)
    return 1.96 * sd / math.sqrt(len(xs))


def _to_tuple(v):
    return tuple(v) if isinstance(v, (list, tuple)) else (v,)


def main():
    parts = sorted(glob.glob(os.path.join(OUTDIR, "part_*.json")))
    if not parts:
        print(f"[ERROR] {OUTDIR}/ 下没有 part_*.json")
        return 1

    rows = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as f:
            rows.extend(json.load(f))
    rows.sort(key=lambda r: r.get("config_idx", 0))
    print(f"合并 {len(parts)} parts -> {len(rows)} configs (axis=phase3d)")

    # ---- 逐 config CSV ----
    keys = list(rows[0].keys())
    csv_rows = []
    for r in rows:
        rc = dict(r)
        if isinstance(rc.get("axis_value"), (list, tuple)):
            rc["axis_value"] = json.dumps(list(rc["axis_value"]))
        csv_rows.append(rc)
    csv_path = "phase3d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rc in csv_rows:
            w.writerow(rc)
    print(f"写 {csv_path}")

    # ---- (K, sep, noise) 聚合 ----
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[_to_tuple(r["axis_value"])].append(r)

    summary = []
    for cell in sorted(by_cell):
        grp = by_cell[cell]
        k, sep, noise = cell
        sup_dom = [g.get("sup_top1_dom_mean") for g in grp]
        uns_dom = [g.get("uns_top1_dom_mean") for g in grp]
        summary.append({
            "K": int(k),
            "sep_index": float(sep),
            "noise_pct": float(noise),
            "n_seeds": len(grp),
            "sup_top1_dom_mean": _mean(sup_dom),
            "sup_top1_dom_ci95": _ci95(sup_dom),
            "uns_top1_dom_mean": _mean(uns_dom),
            "uns_top1_dom_ci95": _ci95(uns_dom),
            "sup_mae_mean":      _mean([g.get("sup_mae_mean") for g in grp]),
            "uns_mae_mean":      _mean([g.get("uns_mae_mean") for g in grp]),
            "pass_rate": sum(1 for g in grp if g.get("threshold_met")) / max(len(grp), 1),
        })

    json_path = "phase3d.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Paper2-Phase3D-KxSepxNoise",
            "axis": "phase3d",
            "n_config": len(rows),
            "n_parts": len(parts),
            "summary_by_cell": summary,
            "results": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"写 {json_path}")

    # ---- 按 noise 切片打印 phase 表 ----
    ks = sorted({s["K"] for s in summary})
    seps = sorted({s["sep_index"] for s in summary})
    noises = sorted({s["noise_pct"] for s in summary})
    cell_lookup = {(s["K"], s["sep_index"], s["noise_pct"]): s for s in summary}

    for noise in noises:
        print(f"\n=== sup_top1_dom @ noise={noise:.2f}  (K × sep) ===")
        header = "  K \\ sep " + "".join(f"{s:>7.2g}" for s in seps)
        print(header)
        for k in ks:
            row = f"  K={k:<4d}   "
            for sep in seps:
                cell = cell_lookup.get((k, sep, noise))
                v = cell.get("sup_top1_dom_mean", float("nan")) if cell else float("nan")
                row += f"{v:>7.3f}"
            print(row)

    # ---- Failure region ----
    failing = [s for s in summary if not (s["sup_top1_dom_mean"] >= 0.85)]
    if failing:
        print(f"\n[!] Failure cells (sup dom < 0.85): {len(failing)}/{len(summary)}")
        for s in failing[:15]:
            print(f"    K={s['K']:d}  sep={s['sep_index']:.2f}  noise={s['noise_pct']:.2f}  "
                  f"sup_dom={s['sup_top1_dom_mean']:.3f}±{s['sup_top1_dom_ci95']:.3f}")
        if len(failing) > 15:
            print(f"    ... and {len(failing) - 15} more")
    else:
        print("\n全 3D 网格 sup dom ≥ 0.85（grid 可能不够 stress）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
