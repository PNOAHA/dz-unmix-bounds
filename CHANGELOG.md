# Changelog

All notable changes to this reproducibility package. Version DOIs are minted by
Zenodo from the GitHub releases (concept DOI: https://doi.org/10.5281/zenodo.20768131,
which always resolves to the latest version).

## [1.2.1] - 2026-10-19 (prepared, not released)

- Prepared on local branch `release-v1.2.1` from `release-v1.2.0`; no push, tag,
  GitHub release or Zenodo action. Published v1.2.0 remains the citation target
  until Louise authorizes a release.
- Added selected SI B.6/C.6-C.7/D.1-D.6/E.1-E.9 probes, eight small historical
  summaries, source hashes, exact per-script commands and isolated execution checks.
  The B.6 Dirichlet probe is a labelled reconstruction, not a recovered original.
- Added atlas generator, array dispatcher, merge script and portable Slurm wrapper;
  recorded the distinction between the planned full grid and partial saved atlas.
- Documented all Fig.3 CLI flags and seed; verified four R=600 cells against the
  saved JSON to floating-point precision. Existing scan defaults are unchanged.
- Withdrew the unverified claim that per-seed atlas raw has a separate public
  archive; raw is omitted and may be requested from the author. No DOI invented.
- Fixed Fig.1 companion label to `Pan, submitted-a` and re-rendered PDF/PNG/SVG.
  Added Barthelme & Usevich (2021), using the paper's existing reference data.
- Real-sample record corrections (data/realdata_k1_samples): 05FW007 restored to 59
  analyses (two analyses lost by a cross-sample de-duplication error); 100821-1 / -3
  rebuilt within the original supplementary-table boundaries (81 / 110). Fig. 8 and
  fig8_diagnostic_numbers.json re-rendered: total 1091 analyses, 31-110 per sample,
  predicted spread 0.0667-0.1527, rho_crit 0.1287-0.2425; library KS distances unchanged.
- Updated version metadata; removed `date-released` from this unreleased candidate.
- Recorded numerical matches, mismatches and reduced-test limits; this preparation
  does not certify that every quoted supplementary number has been reproduced.

## [1.2.0] - 2026-10-19

Accompanies the manuscript as submitted to *Sedimentary Geology*.

### Added
- `analysis/bias_rmse.py` and its output `analysis/bias_rmse_allcells.csv`: bias / RMSE
  decomposition of the Fig. 3 sigma_err harness (`out_phase/sigma_err_slope_numpy.json`).
  Per cell: sampling spread (sqrt of the covariance trace and the leading-eigenvalue
  sigma_err), L2 bias of the Monte-Carlo mean against the design weights, its Monte-Carlo
  standard error, and RMSE. Reproduces the K = 2 calibration C = 0.500 +/- 0.026 and the
  persistent K = 3 pinning bias (about 0.4) reported in the paper.
- `analysis/pop_kl_example.py`: population-level (infinite-grain) KL-NMF example showing
  estimator bias for a raw-sigma dictionary versus a broadened (sigma^2 + bw^2) dictionary.
- `make_graphical_abstract.py` and `figures/GraphicalAbstract.{pdf,png}`: graphical abstract
  rebuilt from measured data (sampling SD and RMSE vs rho; separability cliff).
- SVG versions of the figures; `figures/Fig7_sigma_err_phase_report.txt`.
- `requirements.txt` with tested versions; this `CHANGELOG.md`.

### Changed
- `make_fig6_phase_atlas.py` (paper Fig. 2): cells drawn with `pcolormesh` on true cell
  edges (midpoints between grid values), so cells are no longer shifted by half a cell;
  nanmean over seeds (untested / timed-out cells stay grey); reads
  `scan/phase3d_cellsummary.csv` via `PHASE3D_CSV`.
- `make_paper2_figures.py` (paper Figs. 4-7): noise axis of paper Fig. 5 relabelled from
  "multiplicative 2-sigma" to relative Gaussian 1-sigma noise eta (the generator's
  definition); in-figure titles moved to captions; 400 dpi PNG; cosmetic polish.
- `make_fig8_real_diagnostic.py` (paper Fig. 8): panel annotation "weights recoverable"
  replaced by "far from overlap regime" (the diagnostic is a variance / separability screen,
  not a total-error certificate).
- `fig_sigma_err_phase.py` (paper Fig. 3), `make_fig1_framework.py` (paper Fig. 1): label
  wording ("pairwise-law regime", "vertex-collapse", lower-case panel letters) and styling.
- All figures re-rendered from the archived inputs (numbers unchanged;
  `fig8_diagnostic_numbers.json` identical to v1.1).
- README / CITATION.cff / .zenodo.json: target journal now *Sedimentary Geology*
  (submitted); "Minerals" removed; paper-vs-repository figure numbering map updated;
  dependencies (scikit-learn, SciPy, NumPy >= 2.0 for `numpy.trapezoid`) documented;
  data-licence note (MIT code vs. original licences of the real samples).
- `data/realdata_k1_samples/README.md`: stratigraphic age wording revised (see
  "Sample ages" in that file); the blanket "Lower Cretaceous" label is no longer used.

## [1.1] - 2026-08-01
- Real-sample a-priori diagnostic (Fig. 8) and the 15 real per-grain age samples.

## [1.0] - 2026-06-20
- Initial reproducibility package.
