import json
with open("results_interaction/part_00000.json") as f:
    d = json.load(f)
print(f"n={len(d)}  keys={list(d[0].keys())[:12]}...")
print()
for r in d[:5]:
    k, sep = r["axis_value"]
    seed = r["seed"]
    sup_top1 = r.get("sup_top1_mean", 0)
    sup_dom = r.get("sup_top1_dom_mean", 0)
    uns_top1 = r.get("uns_top1_mean", 0)
    print(f"  (K={k}, sep={sep:.2f})  seed={seed}  sup_top1={sup_top1:.3f}  sup_dom={sup_dom:.3f}  uns_top1={uns_top1:.3f}")
print()
print("--- cells in part_00000 ---")
cells = sorted(set(tuple(r["axis_value"]) for r in d))
for c in cells:
    print(f"  K={c[0]}, sep={c[1]:.2f}")
