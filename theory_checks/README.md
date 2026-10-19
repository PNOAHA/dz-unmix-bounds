# Supplementary numerical checks (v1.2.1)

Added in release v1.2.1 (2026-10-19).

Only selected verification code and small result snapshots were extracted. No theory manuscript, working notes, other-paper drafts or full development directory is included. `SOURCES.json` records source-relative paths, SHA-256 hashes and portability fixes.

Historical source comments preserve exploratory hypotheses and superseded proof language. Current scope: finite-grid numerical checks; same-face rank-one algebra is conditional; at-model sandwich and one-direction clipping/projection are approximations. A successful run is not a proof or a validation outside its tested window.

## Commands and SI mapping

Run from the repository root in the environment from `requirements.txt`. Each command below makes a temporary copy, sets `DZ_CORE_DIR` and `PYTHONPATH` to that copy, runs the named original script with its full original constants, and writes the specified validation JSON. It never overwrites the bundled historical snapshots. The default timeout is 540 seconds per script: use a suitably larger `--timeout` for a full large Monte-Carlo job. For an execution-only reduced run, add `--smoke`; every changed AST literal is recorded in the result.

| Script | Reproduces / scope | Exact full command |
|---|---|---|
| `analytic_worst_mode.py` | E.4: finite-rho Fisher/contrast alignment | `python theory_checks/run_checks.py --only analytic_worst_mode.py --output analytic_worst_mode.validation.json` |
| `c1_prefactor.py` | D.1-D.3: K=3-6 constants and N slopes | `python theory_checks/run_checks.py --only c1_prefactor.py --output c1_prefactor.validation.json` |
| `c3_top1_split.py` | D.6: endpoint/interior top-1 mechanism | `python theory_checks/run_checks.py --only c3_top1_split.py --output c3_top1_split.validation.json` |
| `c_calib_analytic.py` | D.4-D.5: reduced Fisher and calibrated KDE correction | `python theory_checks/run_checks.py --only c_calib_analytic.py --output c_calib_analytic.validation.json` |
| `compute_factors.py` | E.2: CR and smoothing decomposition | `python theory_checks/run_checks.py --only compute_factors.py --output compute_factors.validation.json` |
| `compute_sandwich.py` | E.1: at-model sandwich approximation | `python theory_checks/run_checks.py --only compute_sandwich.py --output compute_sandwich.validation.json` |
| `d2_moment_envelope.py` | C.6: variance envelope and release proxy | `python theory_checks/run_checks.py --only d2_moment_envelope.py --output d2_moment_envelope.validation.json` |
| `d3_mode_algebra_check.py` | C.6: endpoint-face mode algebra | `python theory_checks/run_checks.py --only d3_mode_algebra_check.py --output d3_mode_algebra_check.validation.json` |
| `final_analysis.py` | D.1-D.3: original N-scan normalization of reported constants | `python theory_checks/run_checks.py --only final_analysis.py --output final_analysis.validation.json` |
| `n1_k3_pinning.py` | C.6-C.7: K=3 pinning | `python theory_checks/run_checks.py --only n1_k3_pinning.py --output n1_k3_pinning.validation.json` |
| `n2_population_kkt.py` | C.6-C.7: population KKT grid and 82 selected face cells | `python theory_checks/run_checks.py --only n2_population_kkt.py --output n2_population_kkt.validation.json` |
| `n3_k4_modes.py` | C.6-C.7: K=4 mode fingerprint | `python theory_checks/run_checks.py --only n3_k4_modes.py --output n3_k4_modes.validation.json` |
| `n4_stability_decay.py` | C.6: finite-N stability and iterative decay | `python theory_checks/run_checks.py --only n4_stability_decay.py --output n4_stability_decay.validation.json` |
| `probe2_clip_effK_reconcile.py` | E.9: effective-K versus clipping comparison | `python theory_checks/run_checks.py --only probe2_clip_effK_reconcile.py --output probe2_clip_effK_reconcile.validation.json` |
| `probe2_clip_effectiveK.py` | E.9: alternative effective-K mixture | `python theory_checks/run_checks.py --only probe2_clip_effectiveK.py --output probe2_clip_effectiveK.validation.json` |
| `probe2_clip_eigcross.py` | E.9: high-R turnover bootstrap and eigenvalue crossover | `python theory_checks/run_checks.py --only probe2_clip_eigcross.py --output probe2_clip_eigcross.validation.json` |
| `probe2_clip_final.py` | E.9: kappa=1.805 calibrated clipping curve | `python theory_checks/run_checks.py --only probe2_clip_final.py --output probe2_clip_final.validation.json` |
| `probe2_clip_mechanism.py` | E.9: boundary-hit and coordinate-spread checks | `python theory_checks/run_checks.py --only probe2_clip_mechanism.py --output probe2_clip_mechanism.validation.json` |
| `probe2_clip_model.py` | E.9: one-wall model and calibration helper | `python theory_checks/run_checks.py --only probe2_clip_model.py --output probe2_clip_model.validation.json` |
| `probe2_clip_truncgauss.py` | E.9: unconstrained covariance and clipping helper | `python theory_checks/run_checks.py --only probe2_clip_truncgauss.py --output probe2_clip_truncgauss.validation.json` |
| `probe2_deflation3.py` | E.8: exact source for reported empirical prefactors | `python theory_checks/run_checks.py --only probe2_deflation3.py --output probe2_deflation3.validation.json` |
| `probe2_lncoincidence_a.py` | E.7: log-fit window drift | `python theory_checks/run_checks.py --only probe2_lncoincidence_a.py --output probe2_lncoincidence_a.validation.json` |
| `probe2_lncoincidence_b.py` | E.8: empirical prefactors | `python theory_checks/run_checks.py --only probe2_lncoincidence_b.py --output probe2_lncoincidence_b.validation.json` |
| `probe2_lncoincidence_b2.py` | E.8: leverage coefficient and rho fits | `python theory_checks/run_checks.py --only probe2_lncoincidence_b2.py --output probe2_lncoincidence_b2.validation.json` |
| `probe2_omega_edge.py` | E.8: exact source for reported empirical prefactors | `python theory_checks/run_checks.py --only probe2_omega_edge.py --output probe2_omega_edge.validation.json` |
| `probe2_omega_mobius.py` | E.6: Mobius frequency fit | `python theory_checks/run_checks.py --only probe2_omega_mobius.py --output probe2_omega_mobius.validation.json` |
| `probe2_pR_cap_final.py` | E.8: p_R cap and drift | `python theory_checks/run_checks.py --only probe2_pR_cap_final.py --output probe2_pR_cap_final.validation.json` |
| `probe2_synth.py` | E.3/E.8: independent structural spot checks | `python theory_checks/run_checks.py --only probe2_synth.py --output probe2_synth.validation.json` |
| `probe3_metric_projection.py` | E.9: Euclidean/G/widened-Hessian projection comparison | `python theory_checks/run_checks.py --only probe3_metric_projection.py --output probe3_metric_projection.validation.json` |
| `probe_S_exact.py` | E.6: directional quadratic-form smoothing identity | `python theory_checks/run_checks.py --only probe_S_exact.py --output probe_S_exact.validation.json` |
| `probe_S_final.py` | E.6: frequency and smoothing fits | `python theory_checks/run_checks.py --only probe_S_final.py --output probe_S_final.validation.json` |
| `probe_final_p.py` | E.5: finite-window Fisher-decay fit | `python theory_checks/run_checks.py --only probe_final_p.py --output probe_final_p.validation.json` |
| `probe_reconcile3.py` | E.8: exact source for reported empirical prefactors | `python theory_checks/run_checks.py --only probe_reconcile3.py --output probe_reconcile3.validation.json` |
| `probe_toeplitz.py` | E.3/E.5: Gaussian Toeplitz overlap and symbol | `python theory_checks/run_checks.py --only probe_toeplitz.py --output probe_toeplitz.validation.json` |
| `probe_turnover.py` | E.9: measured clipping/turnover | `python theory_checks/run_checks.py --only probe_turnover.py --output probe_turnover.validation.json` |
| `ref_d1_check.py` | C.6: symmetric small-spacing slack coefficient | `python theory_checks/run_checks.py --only ref_d1_check.py --output ref_d1_check.validation.json` |
| `top1_confirm.py` | D.6: 36-cell endpoint flip grid | `python theory_checks/run_checks.py --only top1_confirm.py --output top1_confirm.validation.json` |
| `diag_k3_mode.py` | C.6/E.1/E.4: measured K=3 pair/curvature modes | `python theory_checks/run_checks.py --only diag_k3_mode.py --output diag_k3_mode.validation.json` |
| `diag_kge4_mode.py` | E.4/E.9: measured K=2-5 unpinned covariance modes | `python theory_checks/run_checks.py --only diag_kge4_mode.py --output diag_kge4_mode.validation.json` |
| `diag_kge4_poly.py` | E.4: measured K=3-7 contrast degrees and convergence | `python theory_checks/run_checks.py --only diag_kge4_poly.py --output diag_kge4_poly.validation.json` |
| `probe_excess_growth.py` | E.9: K=2-7, rho=.40/.47, three-seed excess and turnover | `python theory_checks/run_checks.py --only probe_excess_growth.py --output probe_excess_growth.validation.json` |
| `near_boundary_probe.py` | B.6: explicitly reconstructed Dirichlet alpha=0.3 check, seed 20261019; original ad-hoc script/seed not found | `python theory_checks/run_checks.py --only near_boundary_probe.py --output near_boundary.validation.json` |
| `verification_summary.py` | Saved-output audit for C.6/D.1-D.6; distinguishes snapshots from fresh MC | `python theory_checks/run_checks.py --only verification_summary.py --output saved_summary.validation.json` |

```bash
python theory_checks/run_checks.py --smoke --output all_smoke.validation.json
python theory_checks/verification_summary.py
```

`run_checks.py` is new orchestration code. `near_boundary_probe.py` and `verification_summary.py` are new checks, not recovered historical scripts. The former makes 1,000,000 draws for each K=3..6 and records its seed. Reduced MC caps N at 2000, R at 24, iterations at 4000 (population KKT at 12000), projection draws at 40, and process count at 2; selected N grids/range-based calibration replicates are also reduced. Details are emitted per run. Imported dependencies retain their original constants unless they are the selected script. Deterministic probes are run in full.

## Small historical snapshots and numeric verification

The eight JSON files in `reference_results/` are original saved outputs. They are input to postprocessors and remain immutable during isolated runs. They support arithmetic rechecks, not independent regeneration of the original Monte-Carlo populations.

- D.1/D.2 use the N-scan normalization in **`final_analysis.py`**, not the free-slope rho-fit intercept. Recomputed C=0.466034/0.483113/0.479577/0.496889 for K=3/4/5/6 and N-slopes -0.520122/-0.515819/-0.506983/-0.498229 match the SI at its precision. Adjacent-normalized constants span 0.467354-0.508166.
- C.6 K=4 saved mode ratios match 1/sqrt(12)=0.288675 and 0.5; the interior first-difference ratio is below 2.4e-49. `d3_mode_algebra_check.py` independently checks the algebra against the snapshot.
- **C.6 KKT selection recovered; endpoint-only wording mismatches**: the saved n2 JSON has 130 grid cells. Exactly 82 have a nonempty original `pinned_set`, with minimum inactive slack 1.113618575e-6, reproducing the reported count and minimum. Only 65 have `survivors == [0,K-1]` (minimum slack 7.071200839e-5); the other 17 retain at least one interior survivor. Example: K=5, bw=6, rho=.14 has zero-based survivors [0,2,4]. Thus the original 82-cell selection is traceable, but describing all 82 as endpoint-only is not supported by this snapshot. These are the original tail-rate classifications, not an independently proven exact active set.
- D.6 has all 36 saved flip cells. The stated outlier is recovered (K=5, rho=.12, N=300: observed .8475, predicted .694412837). Using a strict absolute tolerance .02, 32/36 agree; .03 gives 34/36. Thus "35/36 to about 1-2%" is not a verified strict tolerance statement. Do not silently change manuscript numbers on this packaging branch.
- D.4/D.5 deterministic calibration reads a rounded measured table embedded in the historical source. Its kappa is a comparison with measured MC, not a wholly deterministic first-principles estimator covariance derivation.
- E.9 full high-R bootstrap at rho=.40, N=350, R=4000 reproduces the drop 0.297 / 11.1 bootstrap standard errors, with leading covariance eigenvalues .02663 (K=6) and .02032 (K=7). The original three-seed R=600 growth script reproduces a K=6 mean peak at both rho=.40 and .47. The calibrated one-wall model reproduces the rho=.40 curve but its rho=.47 predicted peak is K=7; it is not an exact universal peak rule.
- E.9 full NSAMP=1500 metric projection reproduces G-metric .737 (+7%) and Euclidean .852 (+23%) at K=7. The widened-Hessian branch prints .829 but also emits an invalid-divide RuntimeWarning in `A_widened`; this comparison is not certified as a valid finite Hessian/projection result. The historical algorithm is retained and the warning is preserved in the log.
- E.1 `compute_sandwich.py` runs in full, but its uncorrected rho=.40 K=7 excess is 3.09 versus embedded measured 2.19 (+41%). The approximate 10% wording does not cover that clipped cell; the claimed global R2=.994 is not independently reconstructed by this table. E.7 window drift (b:34/45/92%, a up to1328%) and E.8 leverage coefficient .6789 (R2=.9987) do reproduce.
- B.6 reconstructed seed-20261019 check gives near-boundary fractions .280668/.412638/.528689 for K=3/4/5, median minimum weight .008620 for K=3. K=6 gives .629741; the original probe's complete K grid/seed is unavailable. Agreement in scale does not recover the original run.
- E.1-E.9 deterministic versus reduced/full Monte-Carlo run status and console output are in `validation_20261019.json`. Original high-R jobs are not replaced by smoke results as scientific evidence.

## Atlas and Figure 3

See the main README for the complete Fig.3 command (including endpoint weights, R=600, eight N values, eight rho values, sigma=25, bw=12, noise=0 and seed=20260611) and the four-cell seed verification. No defaults were changed.

Atlas entry points are `scan/phase3d_scan.py --smoke`, `scan/phase3d_scan_hpc.py`, `scan/merge_phase3d.py` and `scan/phase3d.slurm`; exact commands and environment flags are in the main README. The planned grid is not the saved partial atlas. Per-seed raw is omitted, no separate raw archive DOI was found, and redrawing Fig.2 from the cell summary does not re-run or re-aggregate that raw.
