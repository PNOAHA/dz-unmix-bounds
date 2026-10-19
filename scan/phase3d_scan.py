# -*- coding: utf-8 -*-
"""
第二篇方法论文 · K × sep × noise 三维 phase scan
=================================================
interaction_scan.py 的三维扩展——再加 noise 轴。

为什么需要：
  - paper2 §4 现有 5×7 K×sep 表（jobid 21017）固定 noise=BASE_NOISE=0.05。
    但实验室真实 DZ 数据噪声水平差异大（2%–30% 都有），
    "在干净数据上的 phase 边界"对应用者不够。
  - 三维相图把 (K, sep, noise) 全空间映射出来，
    论文 §4 从单一 heatmap 升级为"按 noise 切片的 phase atlas"。

axis_value = [K, sep, noise]   (3-tuple)
"""

import argparse

from applicability_scan import (
    BASE_ALPHA, BASE_BW, BASE_SIGMA,
    make_equispaced_sources, run_one_ext,
)


def build_phase3d_configs(k_values, sep_values, noise_values, seeds, *,
                          sigma=BASE_SIGMA, dirichlet_alpha=BASE_ALPHA,
                          bandwidth=BASE_BW):
    """K × sep × noise 笛卡尔积 × seeds → config list。

    其它轴（alpha, bw, sigma）固定 base case 与 paper2 cross-compare。
    axis_value 设为 tuple (K, sep, noise) — 3D 网格位置。
    """
    cfgs = []
    for k in k_values:
        for sep in sep_values:
            for noise in noise_values:
                k_int = int(k)
                sep_f = float(sep)
                noise_f = float(noise)
                sources = make_equispaced_sources(k_int, sep_f, sigma)
                for s in seeds:
                    cfgs.append(dict(
                        axis="phase3d",
                        axis_value=[k_int, round(sep_f, 4), round(noise_f, 4)],
                        n_sources_request=k_int,
                        sep_index=round(sep_f, 4),
                        noise_pct=noise_f,
                        sources=sources,
                        dirichlet_alpha=dirichlet_alpha,
                        bandwidth=bandwidth,
                        seed=int(s),
                    ))
    return cfgs


def run_phase3d_config(cfg, methods=("supervised", "unsupervised")):
    res = run_one_ext(
        sources=cfg["sources"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        noise_pct=cfg["noise_pct"],
        bandwidth=cfg["bandwidth"],
        seed=cfg["seed"],
        methods=methods,
    )
    res["axis"] = cfg["axis"]
    res["axis_value"] = cfg["axis_value"]
    res["n_sources_request"] = cfg["n_sources_request"]
    res["sep_index"] = cfg["sep_index"]
    res["noise_pct"] = cfg["noise_pct"]
    return res


def _smoke():
    """2 K × 2 sep × 2 noise × 1 seed = 8 configs ~ 1 分钟"""
    import time
    seeds = [1]
    cfgs = build_phase3d_configs(
        k_values=[2, 5],
        sep_values=[0.5, 3.0],
        noise_values=[0.05, 0.20],
        seeds=seeds,
    )
    print(f"=== phase3d smoke: {len(cfgs)} configs ===")
    t0 = time.time()
    for c in cfgs:
        r = run_phase3d_config(c)
        kv, sv, nv = r["axis_value"]
        print(f"  (K={kv}, sep={sv}, noise={nv})  "
              f"sup_dom={r.get('sup_top1_dom_mean', float('nan')):.4f}  "
              f"uns_dom={r.get('uns_top1_dom_mean', float('nan')):.4f}")
    print(f"({time.time()-t0:.1f}s) — pipeline OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K × sep × noise 3D phase scan")
    ap.add_argument("--smoke", action="store_true", help="8-config 冒烟")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        print(__doc__)
        print("--smoke 跑冒烟；大规模扫描用 phase3d_scan_hpc.py")
