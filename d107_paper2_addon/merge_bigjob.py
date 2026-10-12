# -*- coding: utf-8 -*-
"""
合并 bigjob_scan_hpc.py 的 part_*.json
=======================================
- bigjob.csv         逐 config 平铺
- bigjob.json        per-sample × LOO summary
- 控制台打印 each sample × LOO variant 的 weights mean/std + 端元 bias

跑法：
  DZ_OUTDIR=results_bigjob python merge_bigjob.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict

OUTDIR = os.environ.get("DZ_OUTDIR", "results_bigjob")


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.mean(xs) if xs else float("nan")


def _safe_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def _flatten_list_field(d, key):
    v = d.get(key)
    if isinstance(v, list):
        for i, val in enumerate(v):
            d[f"{key}_{i}"] = val
        del d[key]


def main():
    parts = sorted(glob.glob(os.path.join(OUTDIR, "part_*.json")))
    if not parts:
        print(f"[ERROR] {OUTDIR}/ 下没 part_*.json")
        return 1

    rows = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as f:
            rows.extend(json.load(f))
    rows.sort(key=lambda r: r.get("config_idx", 0))
    print(f"合并 {len(parts)} parts -> {len(rows)} configs")

    # Flatten list fields for CSV
    list_fields = ["bayes_weights_mean", "bayes_weights_std",
                   "bayes_weights_q05", "bayes_weights_q95",
                   "bayes_mus_mean", "bayes_mus_std",
                   "bayes_sigmas_mean", "bayes_sigmas_std"]
    flat_rows = []
    for r in rows:
        rc = dict(r)
        for key in list_fields:
            _flatten_list_field(rc, key)
        flat_rows.append(rc)
    keys = sorted({k for r in flat_rows for k in r.keys()})

    csv_path = "bigjob.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for fr in flat_rows:
            w.writerow(fr)
    print(f"写 {csv_path}")

    # Aggregate by (sample, loo_idx)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["sample_name"], r["loo_idx"])].append(r)

    summary = []
    for cell, grp in sorted(by_cell.items()):
        sname, loo_idx = cell
        K_used = grp[0]["K_used"]
        # bayes_weights_mean is list of length K_used
        w_per_k = list(zip(*[g["bayes_weights_mean"] for g in grp]))
        s = {
            "sample": sname,
            "loo_idx": loo_idx,
            "K_used": K_used,
            "n_configs": len(grp),
            "avg_acceptance": _safe_mean([g["bayes_acceptance_fraction"] for g in grp]),
            "avg_log_evidence": _safe_mean([g["bayes_log_evidence_proxy"] for g in grp]),
        }
        for k in range(K_used):
            s[f"w_{k}_mean"] = _safe_mean(w_per_k[k])
            s[f"w_{k}_std"] = _safe_std(w_per_k[k])
        summary.append(s)

    json_path = "bigjob.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Paper2-Bigjob-LOO",
            "n_configs": len(rows),
            "n_parts": len(parts),
            "per_sample_loo_summary": summary,
        }, f, ensure_ascii=False, indent=2)
    print(f"写 {json_path}")

    # 控制台 summary
    print(f"\n=== Per-sample × LOO posterior weights ===")
    print(f"{'sample':25s}  loo  K  acpt   w_0          w_1          w_2")
    last_sample = None
    for s in summary:
        if s["sample"] != last_sample:
            print("  " + "-" * 80)
            last_sample = s["sample"]
        loo_str = "FULL" if s["loo_idx"] < 0 else f"d{s['loo_idx']}"
        K = s["K_used"]
        cols = []
        for k in range(K):
            cols.append(f"{s[f'w_{k}_mean']:.3f}±{s[f'w_{k}_std']:.3f}")
        # Pad to 3 cols for alignment
        while len(cols) < 3:
            cols.append("       -      ")
        print(f"  {s['sample']:23s}  {loo_str:4s}  {K}  {s['avg_acceptance']:.3f}  "
              f"{cols[0]}  {cols[1]}  {cols[2]}")

    # LOO bias 解读：full K=3 vs LOO 比较 (assuming weights re-index in LOO)
    print(f"\n=== LOO bias 解读 ===")
    print("注：LOO drop endmember i 后，posterior 是 K=2 的 (w_0', w_1')，对应剩下 endmember 的权重。")
    print("    LOO 缺失维度的权重隐性分到其他 endmember 上——通过对照 FULL K=3 可量化 bias。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
