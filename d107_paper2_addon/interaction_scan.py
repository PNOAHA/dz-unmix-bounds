# -*- coding: utf-8 -*-
"""
第二篇方法论文 · K × sep 二维 interaction phase scan
=====================================================
对 paper2/applicability_scan.py 的 *扩展模块* —— 不修改 applicability_scan.py
任何字，只 import 它的 base case + 物理模型 (make_equispaced_sources, run_one_ext)。

为什么需要：applicability_scan 是一次扫一个轴（其它固定 base case），看每个
单轴的失效边界。但 supervised NMF 的真正失效模式是 *K × sep interaction*——
单维度 stress 不致失败，两维联合才会触发（local dz-unmix-bounds v1.1 pilot 实证）。

输出： run 在 axis="interaction" namespace，axis_value 是 tuple (K, sep)。
merge_interaction.py 按这个 tuple 聚合，得到 §4 phase diagram 的 (K, sep) → top1 表。

跟 paper2 sep 轴的 cross-validation：(K=3, sep=3.0) 这一格点应该与
applicability_sep.csv 中 K=3 (固定) / sep=3.0 (base case) 的数据点一致——
是 phase diagram 与 axis sweep 的 sanity bridge。

本地冒烟（小网格 6 configs）：
  & "C:/Users/Administrator/miniconda3/python.exe" interaction_scan.py --smoke

作者: 潘路加 — 2026-05-26
"""

import argparse

from applicability_scan import (   # paper2 single source of truth
    BASE_ALPHA, BASE_NOISE, BASE_BW, BASE_SIGMA,
    make_equispaced_sources, run_one_ext,
)


def build_interaction_configs(k_values, sep_values, seeds, *,
                              sigma=BASE_SIGMA, dirichlet_alpha=BASE_ALPHA,
                              noise_pct=BASE_NOISE, bandwidth=BASE_BW):
    """K × sep 笛卡尔积 × seeds → config list。

    其它轴固定在 paper2 base case 以便与 applicability_*  cross-compare。
    axis_value 设为 tuple (K, sep) — 2D 网格上的位置。
    """
    cfgs = []
    for k in k_values:
        for sep in sep_values:
            k_int, sep_f = int(k), float(sep)
            for s in seeds:
                cfgs.append(dict(
                    axis="interaction",
                    axis_value=[k_int, round(sep_f, 4)],   # tuple → list for JSON-safe
                    n_sources_request=k_int,
                    sep_index=round(sep_f, 4),
                    sources=make_equispaced_sources(k_int, sep_f, sigma),
                    dirichlet_alpha=dirichlet_alpha, noise_pct=noise_pct,
                    bandwidth=bandwidth, seed=int(s),
                ))
    return cfgs


def run_interaction_config(cfg, methods=("supervised", "unsupervised")):
    """跑一个 K×sep config。返回带 axis 标注的扁平 dict（与 paper2 run_config 同结构）。"""
    res = run_one_ext(
        sources=cfg["sources"], dirichlet_alpha=cfg["dirichlet_alpha"],
        noise_pct=cfg["noise_pct"], bandwidth=cfg["bandwidth"], seed=cfg["seed"],
        methods=methods,
    )
    res["axis"] = cfg["axis"]
    res["axis_value"] = cfg["axis_value"]          # [K, sep] list
    res["n_sources_request"] = cfg["n_sources_request"]
    res["sep_index"] = cfg["sep_index"]
    return res


def _smoke():
    """3 K × 2 sep × 1 seed = 6 configs ~ 半分钟"""
    import time
    seeds = [1]
    k_values = [2, 3, 5]
    sep_values = [0.5, 3.0]
    cfgs = build_interaction_configs(k_values, sep_values, seeds)
    print(f"=== interaction smoke: {len(cfgs)} configs "
          f"(K={list(k_values)} × sep={sep_values} × {len(seeds)} seed) ===")
    t0 = time.time()
    for c in cfgs:
        r = run_interaction_config(c, methods=("supervised", "unsupervised"))
        kv, sv = r["axis_value"]
        print(f"  (K={kv}, sep={sv})  seed={r['seed']}  "
              f"sup_top1={r.get('sup_top1_mean', float('nan')):.4f}  "
              f"sup_dom={r.get('sup_top1_dom_mean', float('nan')):.4f}  "
              f"uns_top1={r.get('uns_top1_mean', float('nan')):.4f}  "
              f"thr={r['threshold_met']}  chance={r['chance_top1']:.3f}")
    print(f"({time.time()-t0:.1f}s)")
    print("[smoke] OK — pipeline 通畅")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K × sep 二维 interaction phase scan")
    ap.add_argument("--smoke", action="store_true", help="跑 6 config 冒烟测试")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        print(__doc__)
        print("--smoke 跑冒烟；大规模扫描用 interaction_scan_hpc.py")
