# Applicability boundaries of NMF detrital-zircon unmixing — reproducibility package

Code and data accompanying:

> Pan, L. *Where does detrital-zircon source unmixing break? An analytic
> identifiability criterion and its large-scale Monte-Carlo verification for NMF
> provenance unmixing.* Submitted to *Sedimentary Geology*.

Archive: Zenodo concept DOI [10.5281/zenodo.20768131](https://doi.org/10.5281/zenodo.20768131)
(always resolves to the latest version). This is release **v1.2.0**; see `CHANGELOG.md`.

This package reproduces every figure in the paper. The study (i) derives an
identifiability criterion linking the Kolmogorov–Smirnov separability of source
signatures to the conditioning of the supervised unmixing problem, and characterises
the sampling spread of the weight estimates (σ_err, bias excluded) together with the
separately measured bias; and (ii) maps supervised and unsupervised NMF unmixing
performance across three data-difficulty axes — **number of sources K**,
**measurement noise**, and **source separability Δ/σ** — on a tunable-difficulty
synthetic detrital-zircon framework at HPC scale (≈4250 single-axis cross-validated
configurations, 50 seeds each), plus a joint 3D phase atlas over K × Δ/σ × noise
(~1.5M configurations). Paper Fig. 3 is the directly measured σ_err on the (ρ, n)
plane; `analysis/` adds its bias / RMSE decomposition. Paper Fig. 8 demonstrates the
criterion as an a-priori screen on fifteen real Mesozoic samples — eleven from
Cretaceous strata of the Songliao Basin and four from Jurassic–Lower Cretaceous
strata of the northern North China Craton margin — compiled from published open
datasets (per-sample source DOIs, formations and ages in
`data/realdata_k1_samples/README.md`).

## Contents

```
scan/applicability_scan.py      core: data generator (sources/noise tunable),
                                equispaced-source factory, 5-fold CV, sup/unsup NMF
scan/applicability_scan_hpc.py  Slurm job-array sharding of one axis
scan/applicability.slurm        submit script (--nice low priority)
scan/merge_applicability.py     merge per-task JSON -> CSV/JSON + per-axis summary
scan/phase3d_cellsummary.csv    3D atlas per-cell aggregate (1618 cells, K x d/s x
                                noise) underlying paper Fig. 2 (figure-ready summary;
                                full per-seed raw archived separately, see below)
make_paper2_figures.py          paper Figs. 4-7 from results/*.csv
make_fig6_phase_atlas.py        paper Fig. 2 (3D phase atlas; pcolormesh on true
                                cell edges)
make_fig1_framework.py          paper Fig. 1 (framework schematic)
sigma_err_scan.py               theory verification: direct sigma_err measurement
                                (supervised KL-NMF) on the (rho, n) plane
fig_sigma_err_phase.py          paper Fig. 3 (two-mode law / pairwise-degeneracy stitch)
make_fig8_real_diagnostic.py    paper Fig. 8 (a-priori screen on 15 real samples;
                                emits fig8_diagnostic_numbers.json)
make_graphical_abstract.py      graphical abstract (sampling SD vs RMSE; cliff)
analysis/bias_rmse.py           bias / RMSE decomposition of the Fig. 3 harness
                                -> analysis/bias_rmse_allcells.csv
analysis/pop_kl_example.py      population-level KL-NMF bias example
data/realdata_k1_samples/       the 15 real per-grain age samples (CSV; provenance,
                                ages and licences in the folder README)
dz_unmixing_experiment.py       core unmixing implementation (KDE, supervised/
                                unsupervised NMF, metrics); shared with the
                                companion methods paper, included for completeness
results/                        per-axis results (CSV + JSON) underlying Figs. 4-7
out_phase/                      sigma_err_slope_numpy.json underlying Fig. 3 and
                                analysis/bias_rmse.py
figures/                        all figures (pdf/png/svg) + GraphicalAbstract
references.bib                  bibliography
requirements.txt                dependencies (tested versions noted)
CHANGELOG.md                    release history
LICENSE                         MIT (code and synthetic results; see Licence)
```

## Figure numbering (paper vs repository files)

The manuscript numbers figures by order of first citation; repository file and
script names keep their original (development) numbering. Mapping:

| Paper figure | Repository file | Script |
|---|---|---|
| Figure 1 | figures/Fig1_framework.* | make_fig1_framework.py |
| Figure 2 | figures/Fig6_phase_atlas.* | make_fig6_phase_atlas.py |
| Figure 3 | figures/Fig7_sigma_err_phase.* | fig_sigma_err_phase.py |
| Figure 4 | figures/Fig2_source_number_K.* | make_paper2_figures.py |
| Figure 5 | figures/Fig3_noise_robustness.* | make_paper2_figures.py |
| Figure 6 | figures/Fig4_separability_cliff.* | make_paper2_figures.py |
| Figure 7 | figures/Fig5_supervised_advantage.* | make_paper2_figures.py |
| Figure 8 | figures/Fig8_real_diagnostic.* | make_fig8_real_diagnostic.py |
| Graphical abstract | figures/GraphicalAbstract.* | make_graphical_abstract.py |

## Large data (separate archive)

The full per-seed 3D-atlas results (`phase3d.csv`, 1,541,100 rows, 559 MB; 187 MB
gzipped) exceed a code-repository's practical limits and are archived as a separate
Zenodo data file. Paper Fig. 2 is fully reproducible from the per-cell aggregate
`scan/phase3d_cellsummary.csv` included here; the raw is provided only for
provenance / re-aggregation.

## Requirements

Python >= 3.10 with NumPy, SciPy, scikit-learn and Matplotlib. No GPU needed.

```
pip install -r requirements.txt
```

NumPy note: the core uses `numpy.trapezoid` (NumPy >= 2.0, where `numpy.trapz` was
renamed). On NumPy 1.x, use a NumPy >= 2.0 environment or replace `np.trapezoid` with
`np.trapz` in `dz_unmixing_experiment.py` and `analysis/pop_kl_example.py`
(`sigma_err_scan.py` already falls back automatically). Tested on 2026-10-19 with
Python 3.12.13, NumPy 2.4.3, SciPy 1.17.1, Matplotlib 3.10.9, scikit-learn 1.8.0.

## Reproduce

**1. Smoke test (local, ~1 min):**

```
DZ_CORE_DIR=. python scan/applicability_scan.py --smoke
```

**2. Full single-axis sweep** (run from `scan/`). Each axis is an embarrassingly parallel set of
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

**3. Figures and analysis** (paper numbering):

```
python make_paper2_figures.py                                   # Figs. 4-7
PHASE3D_CSV=scan/phase3d_cellsummary.csv python make_fig6_phase_atlas.py   # Fig. 2
python make_fig1_framework.py                                   # Fig. 1
python fig_sigma_err_phase.py --json out_phase/sigma_err_slope_numpy.json --out figures/Fig7_sigma_err_phase   # Fig. 3
python make_fig8_real_diagnostic.py                             # Fig. 8
python make_graphical_abstract.py                               # graphical abstract
python analysis/bias_rmse.py                                    # bias / RMSE table
python analysis/pop_kl_example.py                               # population KL example
```

The theory-verification scan that produces `out_phase/sigma_err_slope_numpy.json`
is regenerated by `sigma_err_scan.py` (local, numpy backend, minutes).

## Key parameters (env vars, see `scan/applicability_scan_hpc.py`)

`DZ_AXIS` (srcK|noise|sep) · `DZ_N_SEEDS` (50) · `DZ_METHODS`
(supervised,unsupervised) · base case `DZ_ALPHA` (1.0), `DZ_BW` (12.0),
`DZ_SIGMA` (25.0). Measurement noise η is relative Gaussian 1σ noise.

## Headline results

1. In the pinned (overlapping-signature) regime the sampling spread of the
   supervised weight estimates follows σ_err ≈ C/(√N·ρ₁ₖ), calibrated to
   C = 0.500 ± 0.026 (K = 2); σ_err excludes bias, and for K = 3 overlap-induced
   pinning leaves an interior-weight bias of ≈ 0.4 that does not shrink with N
   (`analysis/bias_rmse_allcells.csv`).
2. Supervised dominant-source accuracy is robust to ≤ 20% (1σ) measurement noise and
   to 2–6 sources, but collapses below a **critical separability Δ/σ ≈ 1.10** in the
   tested design (source age modes closer than ~1 peak width cannot be reliably
   unmixed).

## Licence

The code and the synthetic results in this repository are released under the MIT
licence (see `LICENSE`). The fifteen real per-grain age samples in
`data/realdata_k1_samples/` are **not** re-licensed: they are redistributed from
published open datasets under their original terms (CC-BY or equivalent), and reuse
requires attribution to the original studies listed per sample in that folder's
README and in each CSV header (`source_doi`).
