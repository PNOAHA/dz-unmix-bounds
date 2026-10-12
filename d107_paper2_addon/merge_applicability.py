# -*- coding: utf-8 -*-
"""
合并 applicability_scan_hpc.py 各 array task 的 part_*.json
============================================================
-> applicability_<axis>.csv    (机器可读, 逐 config)
-> applicability_<axis>.json   (结构化 + 按 axis_value 聚合的摘要)
+ 控制台打印：按 axis_value 跨 seed 聚合的 sup/uns 准确率 + 阈值通过率 + 失效边界定位

跑法（每轴一次，指定该轴的 OUTDIR）:
  DZ_OUTDIR=results_appl_sep python merge_applicability.py
"""

import os
import json
import glob
import csv
import statistics
from collections import defaultdict

OUTDIR = os.environ.get("DZ_OUTDIR", "results_appl")


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and x == x]  # 去 None / nan
    return statistics.mean(xs) if xs else float("nan")


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

    axis = rows[0].get("axis", "unknown")
    print(f"合并 {len(parts)} 个 part -> {len(rows)} 个 config (axis={axis})")

    # ---- 逐 config CSV ----
    keys = list(rows[0].keys())
    csv_path = f"applicability_{axis}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"写  {csv_path}")

    # ---- 按 axis_value 跨 seed 聚合 ----
    by_val = defaultdict(list)
    for r in rows:
        by_val[r["axis_value"]].append(r)

    summary = []
    for val in sorted(by_val):
        grp = by_val[val]
        s = {
            "axis_value": val,
            "n_seeds": len(grp),
            "n_sources": grp[0].get("n_sources"),
            "chance_top1": grp[0].get("chance_top1"),
            "sup_top1_mean":      _safe_mean([g.get("sup_top1_mean") for g in grp]),
            "sup_top1_dom_mean":  _safe_mean([g.get("sup_top1_dom_mean") for g in grp]),
            "sup_mae_mean":       _safe_mean([g.get("sup_mae_mean") for g in grp]),
            "uns_top1_mean":      _safe_mean([g.get("uns_top1_mean") for g in grp]),
            "uns_top1_dom_mean":  _safe_mean([g.get("uns_top1_dom_mean") for g in grp]),
            "pass_rate": sum(1 for g in grp if g.get("threshold_met")) / max(len(grp), 1),
        }
        summary.append(s)

    # ---- JSON（含聚合摘要）----
    json_path = f"applicability_{axis}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Paper2-Applicability",
            "axis": axis,
            "n_config": len(rows),
            "n_parts": len(parts),
            "summary_by_axis_value": summary,
            "results": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"写  {json_path}")

    # ---- 控制台摘要 ----
    print(f"\n按 {axis} 聚合（跨 {summary[0]['n_seeds']} seed 均值）:")
    print(f"  {axis:>10} | sup_top1 | sup_dom | uns_top1 | pass% | chance")
    for s in summary:
        print(f"  {s['axis_value']!s:>10} | {s['sup_top1_mean']:.4f}   | "
              f"{s['sup_top1_dom_mean']:.4f}  | {s['uns_top1_mean']:.4f}   | "
              f"{100*s['pass_rate']:5.1f} | {s['chance_top1']:.3f}")

    # ---- 失效边界定位（supervised dominant 首次/末次跌破 0.85）----
    failing = [s for s in summary if not (s["sup_top1_dom_mean"] >= 0.85)]
    if failing:
        vals = [s["axis_value"] for s in failing]
        print(f"\n[!] 失效区（sup dominant < 0.85）: {axis} in {vals}")
    else:
        print(f"\n全 {axis} 网格 sup dominant 均 ≥ 0.85（无失效区）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
