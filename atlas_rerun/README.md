# Certified joint atlas (CompStat Fig. 3)

Recomputes the supervised half of the archived 1,541,100-configuration joint atlas
with the KKT-certified solver (Algorithm 2 + floor-free column-rescaled certificate,
`code/kl_certified_v2.py`). Prepared for dz-unmix-bounds **v1.2.3**.

## What is included

```
code/
  kl_certified.py           Algorithm 2 (floored surrogate; warm-start / Newton)
  kl_certified_v2.py        exact floor-free certificate + repair stage (solver v2)
  atlas_units.py            grid / target set / work-unit split / single-config run
  run_node.py               node process-pool runner
  postprocess.py            completeness + KKT + cell summary + Fig. 3
  merge_resolved.py         merge repaired configs into a unit set
  dz_unmixing_experiment.py, applicability_scan.py, phase3d_scan.py
  make_fig6_phase_atlas.py  Fig. 3 plotting (DZ_FIGOUT)
atlas_rerun.slurm           portable Slurm array wrapper
smoke_test.sh               tiny local smoke (8 configs × 2 units)
reference/
  phase3d_cellsummary_archived.csv
  target_tasks.json         the 467 completed original task indices
results_v2/                 full v2 postprocess outputs (no per-unit JSONs)
  phase3d_cellsummary_certified.csv
  phase3d_cell_compare.csv
  kkt_uncertified.csv
  postprocess_report.json
  figures/Fig3_atlas_certified.*
  figures/Fig3_atlas_archived_samescript.*
```

## What is omitted (size)

Per-unit result JSONs (`results_v2/units/unit_XXXX.json`, ~816 MB for 1000 units;
comparable earlier unit dumps ~450–500 MB) are **not** archived here, for the same
reason as the archived per-seed `phase3d.csv` (560 MB): they exceed a practical
code-repository limit. They may be requested from the author. Paper Fig. 3
(CompStat numbering) redraws from `results_v2/phase3d_cellsummary_certified.csv`.

## Headline numbers (from `results_v2/postprocess_report.json`)

- COMPLETE: true; 1000/1000 units; 1,541,100 configs; 1618 cells (1464 with 1000 seeds)
- ALL_ROWS_CERTIFIED: false — 1 uncertified row in 1 config (see `kkt_uncertified.csv`)
- Measured core-hours ≈ 924.3 (D107 job 269547)
- vs archived Algorithm-1 atlas: 34 cells flip the 0.85 label; mean |Δ| ≈ 0.008

## Smoke

```
# from repository root, with Python deps from requirements.txt
bash atlas_rerun/smoke_test.sh
# or, if cluster python path differs:
PYTHON=python3 bash atlas_rerun/smoke_test.sh
```

Outputs go to `atlas_rerun/results_smoke/` (gitignored).

## Full regeneration

See comments in `atlas_rerun.slurm` and `code/run_node.py`. Submit from this
directory on an HPC site after adapting partition/account. Post-process with:

```
python code/postprocess.py --units-dir results_v2/units --out results_v2
```
