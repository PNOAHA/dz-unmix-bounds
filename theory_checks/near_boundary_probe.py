"""Reconstructed B.6 Dirichlet check; the original ad-hoc script/seed was not found.
This new, explicitly seeded check is not the historical experiment.
"""
import json
import numpy as np
rng=np.random.default_rng(20261019)
rows=[]
for K in range(3,7):
    w=rng.dirichlet(np.full(K,0.3),size=1_000_000)
    rows.append(dict(K=K,alpha=0.3,draws=len(w),near_boundary_threshold=1e-3,
                     fraction=float((w.min(1)<=1e-3).mean()),median_min_weight=float(np.median(w.min(1)))))
print(json.dumps(dict(status='reconstructed check; not historical seed',seed=20261019,rows=rows),indent=2))
