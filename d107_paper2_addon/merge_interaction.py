# -*- coding: utf-8 -*-
"""
合并 interaction_scan_hpc.py 的 part_*.json
============================================
输出：
  - interaction.csv          逐 config 平铺
  - interaction.json         结构化 + 按 (K, sep) tuple 聚合
  - 控制台打印 phase 表 (K × sep) of sup_top1_dom

跑法：
  DZ_OUTDIR=results_interaction python merge_interaction.py

paper2 merge_applicability 的姐妹脚本——区别只在 axis_value 是 [K, sep] tuple
而不是 scalar，所以按 tuple 聚合。
"""

import os
import json
import glob
import csv
import statistics
from collections import defaultdict

OUTDIR = os.environ.get("DZ_OUTDIR", "results_interaction")


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.mean(xs) if xs else float("nan")


def _to_tuple(v):
    """axis_value 是 JSON list [K, sep] —— 转 tuple 作 dict key。"""
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

    print(f"合并 {len(parts)} parts -> {len(rows)} configs (axis=interaction)")

    # ---- 逐 config CSV ----
    keys = list(rows[0].keys())
    # tuple axis_value 需 stringify 给 CSV
    csv_rows = []
    for r in rows:
        rc = dict(r)
        if isinstance(rc.get("axis_value"), (list, tuple)):
            rc["axis_value"] = json.dumps(list(rc["axis_value"]))
        csv_rows.append(rc)
    csv_path = "interaction.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rc in csv_rows:
            w.writerow(rc)
    print(f"写 {csv_path}")

    # ---- 按 (K, sep) tuple 聚合 ----
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[_to_tuple(r["axis_value"])].append(r)

    summary = []
    for cell in sorted(by_cell):
        grp = by_cell[cell]
        k, sep = cell
        summary.append({
            "K": int(k),
            "sep_index": float(sep),
            "n_seeds": len(grp),
            "chance_top1": grp[0].get("chance_top1"),
            "sup_top1_mean":      _safe_mean([g.get("sup_top1_mean") for g in grp]),
            "sup_top1_dom_mean":  _safe_mean([g.get("sup_top1_dom_mean") for g in grp]),
            "sup_mae_mean":       _safe_mean([g.get("sup_mae_mean") for g in grp]),
            "uns_top1_mean":      _safe_mean([g.get("uns_top1_mean") for g in grp]),
            "uns_top1_dom_mean":  _safe_mean([g.get("uns_top1_dom_mean") for g in grp]),
            "pass_rate": sum(1 for g in grp if g.get("threshold_met")) / max(len(grp), 1),
        })

    json_path = "interaction.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Paper2-Interaction-KxSep",
            "axis": "interaction",
            "n_config": len(rows),
            "n_parts": len(parts),
            "summary_by_cell": summary,
            "results": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"写 {json_path}")

    # ---- Phase tables (K × sep) ----
    ks = sorted({s["K"] for s in summary})
    seps = sorted({s["sep_index"] for s in summary})
    cell_lookup = {(s["K"], s["sep_index"]): s for s in summary}

    def _print_phase(metric_key, label):
        print(f"\n=== {label} by (K, sep) [supervised] ===")
        header = "  K \\ sep " + "".join(f"{s:>9.3g}" for s in seps)
        print(header)
        for k in ks:
            row = f"  K={k:<4d}   "
            for sep in seps:
                cell = cell_lookup.get((k, sep))
                v = cell.get(metric_key, float("nan")) if cell else float("nan")
                row += f"{v:>9.3f}"
            print(row)

    _print_phase("sup_top1_dom_mean", "sup_top1_dom (primary metric)")
    _print_phase("uns_top1_dom_mean", "uns_top1_dom (unsupervised baseline)")

    # ---- Failure region ----
    failing = [s for s in summary if not (s["sup_top1_dom_mean"] >= 0.85)]
    if failing:
        print(f"\n[!] Failure cells (sup dominant < 0.85): {len(failing)}/{len(summary)}")
        for s in failing[:10]:
            print(f"    K={s['K']:d}  sep={s['sep_index']:.3f}  "
                  f"sup_dom={s['sup_top1_dom_mean']:.3f}  "
                  f"uns_dom={s['uns_top1_dom_mean']:.3f}  "
                  f"chance={s['chance_top1']:.3f}")
        if len(failing) > 10:
            print(f"    ... and {len(failing) - 10} more")
    else:
        print("\n全 (K, sep) 网格 sup dominant 均 ≥ 0.85（无失效 cell — grid 可能不够 stress）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
