# Applicability boundaries of NMF detrital-zircon unmixing — reproducibility package

Code and data accompanying:

> Pan, L. *Where does detrital-zircon source unmixing break? An analytic
> identifiability criterion and its large-scale Monte-Carlo verification for NMF
> provenance unmixing.* (submitted to *Minerals*).

This package reproduces every figure in the paper. The study (i) derives an
analytic identifiability criterion linking the Kolmogorov–Smirnov separability of
two source signatures to the conditioning of the unmixing problem, and a two-mode
weight-error law; and (ii) maps supervised and unsupervised NMF unmixing
performance across three data-difficulty axes — **number of sources $K$**,
**measurement noise**, and **source separability $\Delta/\sigma$** — on a
tunable-difficulty synthetic detrital-zircon framework at HPC scale (≈4250
single-axis cross-validated configurations, 50 seeds each), plus a joint 3D phase
atlas over $K\times\Delta/\sigma\times$ noise (~1.5M configurations). The
theory-verification figure (Fig. 7) is the directly measured weight-error
$\sigma_{\mathrm{err}}$ on the $(\rho,n)$ plane that calibrates the two-mode law.
Fig. 8 demonstrates the criterion as an a-priori diagnostic on fifteen real Lower
Cretaceous samples (Songliao Basin and northern North China Craton margin,
compiled from published open datasets; per-sample source DOIs in the CSV
headers), included here under `data/realdata_k1_samples/`.

## Contents

```
applicability_scan.py        core: data generator (sources/noise tunable),
                             equispaced-source factory, 5-fold CV, sup/unsup NMF
applicability_scan_hpc.py    Slurm job-array sharding of one axis
applicability.slurm          submit script (--nice low priority)
merge_applicability.py       merge per-task JSON -> CSV/JSON + per-axis summary
make_paper2_figures.py       Figs 2-5 from results/*.csv
make_fig6_phase_atlas.py     Fig 6 (3D phase atlas) from scan/phase3d.csv
make_fig1_framework.py       Fig 1 (conceptual framework schematic)
sigma_err_scan.py            theory verification: direct sigma_err measurement
                             (supervised KL-NMF) on the (rho, n) plane
fig_sigma_err_phase.py       Fig 7 (two-mode law / pairwise-degeneracy stitch)
make_fig8_real_diagnostic.py Fig 8 (a-priori diagnostic on 15 real Lower
                             Cretaceous samples; emits fig8_diagnostic_numbers.json)
data/realdata_k1_samples/    the 15 real per-grain age samples (CSV; provenance
                             and licences in the folder README)
dz_unmixing_experiment.py    core unmixing implementation (KDE, supervised/
                             unsupervised NMF, metrics); shared with the
                             companion methods paper, included for completeness
scan/phase3d_cellsummary.csv 3D atlas per-cell aggregate (1618 cells, K x d/s x
                             noise) underlying Fig 6 -- this is the figure-ready
                             summary; the full per-seed raw (1.54M rows, 559 MB;
                             187 MB gzipped) is archived as a separate data file
                             (see "Large data" below), not in the code repository
results/                     per-axis results (CSV + JSON) underlying Figs 2-5
out_phase/                   sigma_err_slope_numpy.json underlying Fig 7
figures/                     Fig1-Fig8 (pdf/png)
references.bib               bibliography
LICENSE                      MIT
```

## Large data (separate archive)

The full per-seed 3D-atlas results (`phase3d.csv`, 1,541,100 rows, 559 MB; 187 MB
gzipped) exceed a code-repository's practical limits and are archived as a separate
Zenodo data file. Figure 6 is fully reproducible from the per-cell aggregate
`scan/phase3d_cellsummary.csv` included here; the raw is provided only for
provenance / re-aggregation.

## Requirements

Python >= 3.9 with `numpy`, `scikit-learn`, `matplotlib`. No GPU needed.

```
pip install numpy scikit-learn matplotlib
```

## Reproduce

**1. Smoke test (local, ~1 min):**

```
python applicability_scan.py --smoke
```

**2. Full single-axis sweep.** Each axis is an embarrassingly parallel set of
single-core tasks. On a Slurm cluster (place `dz_unmixing_experiment.py`
alongside, or set `DZ_CORE_DIR` to its directory):

```
mkdir -p logs
DZ_AXIS=sep   sbatch --array=0-499 applicability.slurm   # 40 d/s   x 50 seeds
DZ_AXIS=noise sbatch --array=0-499 applicability.slurm   # 40 noise x 50 seeds
DZ_AXIS=srcK  sbatch --array=0-249 applicability.slurm   #  5 K     x 50 seeds
```

Merge each axis into `results/`:

```
DZ_OUTDIR=results_appl_sep   python merge_applicability.py
DZ_OUTDIR=results_appl_noise python merge_applicability.py
DZ_OUTDIR=results_appl_srcK  python merge_applicability.py
```

(Without a cluster, the same grids run serially via
`DZ_AXIS=sep python applicability_scan_hpc.py <task_id>` over all task ids.)

**3. Figures** — from the merged `results/`, `scan/phase3d.csv`, and
`out_phase/sigma_err_slope_numpy.json`:

```
python make_paper2_figures.py     # Fig 2 (K), Fig 3 (noise), Fig 4 (d/s cliff), Fig 5 (advantage)
python make_fig6_phase_atlas.py   # Fig 6 (3D phase atlas; reads scan/phase3d_cellsummary.csv)
python make_fig1_framework.py     # Fig 1 (framework schematic)
python fig_sigma_err_phase.py --json out_phase/sigma_err_slope_numpy.json --out figures/Fig7_sigma_err_phase
python make_fig8_real_diagnostic.py   # Fig 8 (real-sample diagnostic; reads data/realdata_k1_samples/)
```

The theory-verification scan that produces `out_phase/sigma_err_slope_numpy.json`
is regenerated by `sigma_err_scan.py` (local, numpy backend, minutes).

## Key parameters (env vars, see `applicability_scan_hpc.py`)

`DZ_AXIS` (srcK|noise|sep) · `DZ_N_SEEDS` (50) · `DZ_METHODS`
(supervised,unsupervised) · base case `DZ_ALPHA` (1.0), `DZ_BW` (12.0),
`DZ_SIGMA` (25.0).

## Headline results

1. An analytic identifiability criterion: σ_err = max(σ_pair, σ_curv), with the
   error exponent robust to source number (no high-K catastrophe) and the binding
   limit set by adjacent pairwise degeneracy; the pinned-regime law
   σ_err ≈ C/(√N·ρ₁ₖ) is directly calibrated to C = 0.500 ± 0.026.
2. Supervised dominant-source accuracy is robust to ≤20% measurement noise and to
   2–6 sources, but collapses below a **critical separability Δ/σ ≈ 1.10**
   (source age modes closer than ~1 peak width cannot be reliably unmixed).

## Licence

MIT (see `LICENSE`).
