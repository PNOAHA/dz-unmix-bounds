# -*- coding: utf-8 -*-
"""
合并 realdata_scan_hpc.py 的 part_*.json
=========================================
- realdata.csv      逐 config 平铺（含 bayesian weights mean/std/CI + 端元 μ/σ posterior）
- realdata.json     结构化 + per-sample summary
- 控制台打印 phase 表 (mean_shift × sigma_shift) by sample

跑法：
  DZ_OUTDIR=results_realdata python merge_realdata.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict

OUTDIR = os.environ.get("DZ_OUTDIR", "results_realdata")


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.mean(xs) if xs else float("nan")


def _flatten_list_field(d, key, k_dim=3):
    """List 字段 (e.g. bayes_weights_mean = [0.32, 0.44, 0.24]) → flat columns w0, w1, w2"""
    v = d.get(key)
    if isinstance(v, list):
        for i, val in enumerate(v):
            d[f"{key}_{i}"] = val
        del d[key]


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
    print(f"合并 {len(parts)} parts -> {len(rows)} configs")

    # Flatten list fields for CSV
    list_fields = [
        "bayes_weights_mean", "bayes_weights_std",
        "bayes_weights_q05", "bayes_weights_q95",
        "bayes_mus_mean", "bayes_mus_std",
        "bayes_sigmas_mean", "bayes_sigmas_std",
    ]
    flat_rows = []
    for r in rows:
        rc = dict(r)
        for key in list_fields:
            _flatten_list_field(rc, key)
        flat_rows.append(rc)

    keys = sorted({k for r in flat_rows for k in r.keys()})
    csv_path = "realdata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for fr in flat_rows:
            w.writerow(fr)
    print(f"写 {csv_path}")

    # ----- Per-sample summary -----
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample_name"]].append(r)

    summary = []
    for sname, grp in sorted(by_sample.items()):
        # 跨 bootstrap × perturbation 聚合（每 sample 的"鲁棒"权重估计）
        w0_all = [g["bayes_weights_mean"][0] for g in grp]
        w1_all = [g["bayes_weights_mean"][1] for g in grp]
        w2_all = [g["bayes_weights_mean"][2] for g in grp]
        summary.append({
            "sample": sname,
            "n_configs": len(grp),
            "w0_mean": _safe_mean(w0_all),
            "w0_std": statistics.stdev(w0_all) if len(w0_all) > 1 else float("nan"),
            "w1_mean": _safe_mean(w1_all),
            "w1_std": statistics.stdev(w1_all) if len(w1_all) > 1 else float("nan"),
            "w2_mean": _safe_mean(w2_all),
            "w2_std": statistics.stdev(w2_all) if len(w2_all) > 1 else float("nan"),
            "avg_acceptance": _safe_mean([g["bayes_acceptance_fraction"] for g in grp]),
            "avg_log_evidence": _safe_mean([g["bayes_log_evidence_proxy"] for g in grp]),
        })

    json_path = "realdata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Paper2-Realdata-Bayesian",
            "n_configs": len(rows),
            "n_parts": len(parts),
            "per_sample_summary": summary,
            # 不写 raw rows 到 JSON（容易上 GB）；CSV 已平铺
        }, f, ensure_ascii=False, indent=2)
    print(f"写 {json_path}")

    # ----- 控制台 summary -----
    print(f"\n=== Per-sample posterior weights (跨 {grp and len(grp) or 0} perturbation×bootstrap 平均) ===")
    print(f"  {'sample':30s}  {'w_0':>14s}  {'w_1':>14s}  {'w_2':>14s}  accept_frac")
    for s in summary:
        print(f"  {s['sample']:30s}  "
              f"{s['w0_mean']:.3f}±{s['w0_std']:.3f}  "
              f"{s['w1_mean']:.3f}±{s['w1_std']:.3f}  "
              f"{s['w2_mean']:.3f}±{s['w2_std']:.3f}  "
              f"{s['avg_acceptance']:.3f}")

    # ----- Perturbation phase 表 (per sample, 选 1 sample) -----
    if summary:
        first_sample = summary[0]["sample"]
        cells = defaultdict(list)
        for r in rows:
            if r["sample_name"] != first_sample:
                continue
            cells[(r["mean_shift"], r["sigma_shift"])].append(r["bayes_weights_mean"][0])

        if cells:
            ms_vals = sorted({k[0] for k in cells})
            ss_vals = sorted({k[1] for k in cells})
            print(f"\n=== w_0 (S1=130Ma) by (mean_shift × sigma_shift) for {first_sample} ===")
            print("  ms↓ \\ ss→  " + "  ".join(f"{s:+.2f}" for s in ss_vals))
            for ms in ms_vals:
                row = f"  ms={ms:+.2f}    "
                for ss in ss_vals:
                    vals = cells.get((ms, ss), [])
                    if vals:
                        row += f"  {_safe_mean(vals):.3f}"
                    else:
                        row += "    -  "
                print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
