# Which mixing weights can you trust? GPU-accelerated Bayesian uncertainty quantification for detrital-zircon provenance unmixing of Lower Cretaceous samples from NE Asia

**Lujia Pan**¹
¹ School of Mathematical Sciences, Hebei Normal University, No. 20 East 2nd Ring South Road, Shijiazhuang 050024, Hebei Province, China.
Corresponding author. E-mail: peter205834@hebtu.edu.cn

---

## Highlights

- A batched GPU sampler completes 181,500 Bayesian unmixing fits in 74 min, not days.
- GPU posteriors match emcee fit-by-fit; the difference is below emcee's own noise.
- Weight credibility is graded per weight, per sample — not assumed globally.
- The soft arc:basement split equals the multinomial floor, scaling as n^(−1/2).
- Priors contribute <3% of the between-fit variance in 13 of 15 samples.

## Graphical abstract

A two-panel figure built from Fig. 2: *left*, the ternary diagram of the fifteen
K1 samples with 2-σ posterior covariance ellipses — Songliao samples elongated
along the arc–basement (S1–S2) edge, the craton (S3) axis tight; *right*, the
per-sample dot-and-whisker weights showing the hard craton weight against the
soft, anti-correlated arc/basement pair, annotated with the two governing facts:
"softness = counting-noise floor (∝ n^(−1/2), reducible by dating)" and "prior
contributes <3% of between-fit variance (13/15 samples)".

---

## Abstract

Detrital-zircon provenance unmixing is routinely reported as a single vector of
point-estimate mixing weights, leaving unquantified how trustworthy each weight
actually is. We cast three-end-member unmixing of fifteen Lower Cretaceous (K1)
samples from NE Asia — four from the North China Craton (NCC) north margin,
eleven from the Songliao Basin — as a Bayesian inference problem and propagate
the full posterior uncertainty through a systematic scan: 121 perturbed prior
configurations × 100 bootstrap resamples of the grains per sample, 181,500 MCMC
fits in total. To make the scan tractable we implement the Goodman–Weare
ensemble sampler as a batched GPU kernel, completing in 74 min what would take the reference CPU implementation days,
and validate it fit-by-fit
against emcee (maximum posterior-mean difference 0.020, posterior widths
matching to the same tolerance — an order of magnitude
below emcee's own re-seeding scatter of ~0.19). The three weights are not
equally identifiable — credibility must be graded
per weight: the cratonic weight w_S3 is hard (scan std 0.000–0.085), whereas the
arc:basement partition is soft, anti-correlated, and confined to a
one-dimensional ridge. Subsampling demonstrates that this softness is a
finite-grain counting-noise floor — it scales as n^(−1/2) (median exponent
−0.507) and matches the multinomial floor to a median factor of 1.02 — rather
than any intrinsic overlap of the well-separated 130 and 280 Ma end-members, and
is therefore reducible by further dating. The reported uncertainty is
prior-insensitive (prior share <3% of the between-fit variance in 13 of 15
samples), and a single fit's
credible interval already carries about half of it. Geologically, the hard
craton axis largely separates the craton-fed NCC north-margin samples from the
arc + basement-fed Songliao fill. The framework turns "how confident are we?"
into "what would it take to be more confident?" — and answers it per sample.

**Keywords:** detrital zircon; provenance unmixing; Bayesian inference;
uncertainty quantification; GPU computing; Songliao Basin.

---

## 1. Introduction

Detrital-zircon U–Pb geochronology has become a primary tool for sediment
provenance analysis, and a natural way to interpret a measured age spectrum is
to decompose it into contributions from a small number of geologically defined
source domains — a mixing, or "unmixing", problem (Sharman & Johnstone, 2017;
Sundell & Saylor, 2017; Pan, submitted-a). When the candidate sources are known
a priori, the quantity of interest is the vector of mixing weights: the
fractional contribution of each source to the sample. Recovering these weights,
however, is only half of the task. For the result to be geologically usable one
must also know how trustworthy each weight is, and prevailing practice —
reporting a single point estimate per sample — leaves this unquantified. Where uncertainty is reported, it is usually a bootstrap confidence interval on the weights (Sundell & Saylor, 2017); we extend this in three ways: the full Bayesian posterior is propagated jointly with a systematic prior-perturbation scan, identifiability is graded per weight rather than per sample, and the irreducible finite-grain counting-noise floor is separated from the prior- and shape-driven contributions — a decomposition a single bootstrap interval does not provide. Two
difficulties are especially acute. First, when two source domains have
overlapping age modes their relative contributions can be only weakly
identifiable even when their sum is well constrained, and a point estimate
conceals this. Second, the answer may depend on analyst choices (the assumed
source spectra) or on the finite number of grains dated, and a single fit cannot
separate a robust conclusion from a fragile one.

We address this by casting three-end-member detrital-zircon unmixing as a
Bayesian inference problem and propagating the full posterior uncertainty.
Rather than fitting each sample once, we combine a systematic perturbation of
the source priors (121 configurations) with bootstrap resampling of the grains
(100 resamples), yielding a per-sample posterior for the mixing weights that
jointly reflects prior sensitivity and finite-sample noise. To make this
tractable — the reference ensemble sampler would require several days for the
full 181,500-fit scan — we implement the sampler as a batched GPU kernel,
validated fit-by-fit against the CPU gold standard (Section 3). Applying it to
fifteen Lower Cretaceous (K1) samples from NE Asia, we show that the three
source weights are not equally identifiable, that the identifiability structure
can be diagnosed per sample and per weight rather than assumed, and that the
dominant source of the reported uncertainty is the finite grain count rather
than the prior. The outcome is a provenance interpretation where each
conclusion carries an explicit, source-specific confidence, and where the
route to a more confident answer — more grains, or tighter source constraints —
is made quantitative.

## 2. Geological setting and samples

The fifteen samples fall into two groups spanning the eastern Central Asian
Orogenic Belt (CAOB) and the northern North China Craton (NCC) (Liu et al.,
2017; Zhao et al., 2012): a North-China-Craton north-margin group and a Songliao
Basin group (Fig. 1; coordinates and grain counts in Table 1). Their Lower
Cretaceous clastic fill draws potentially on three compositionally and
geochronologically distinct source domains, adopted here as the mixing
end-members (Section 3):

- **S1 — Mesozoic arc (reference peak ≈ 130 Ma).** Early Cretaceous arc
  magmatism — driven predominantly by rollback of the Palaeo-Pacific (Izanagi)
  slab along the eastern continental margin, with a subordinate Mongol–Okhotsk
  contribution in the northwest during the earliest Cretaceous (Huang et al.,
  2024; Li et al., 2021; Wu et al., 2022) — is widespread across NE Asia
  (arc-magmatic peak ≈ 124–133 Ma; Li et al., 2021) and contributes a young,
  narrow age mode.
- **S2 — Palaeozoic CAOB basement (≈ 280 Ma).** The accretionary and arc
  terranes of the CAOB — the world's largest accretionary orogen, amalgamated
  through the Palaeozoic with final closure of the Palaeo-Asian Ocean along the
  Solonker Suture near the northern NCC margin — provide a broad late-Palaeozoic
  source (Liu et al., 2017; Xiao & Santosh, 2014).
- **S3 — Archaean–Proterozoic craton (≈ 1900 Ma).** The crystalline basement of
  the North China Craton carries the characteristic ~1.85 Ga signature of its
  final Palaeoproterozoic amalgamation along the Trans-North China Orogen (Zhao
  et al., 2012; Zhao, 2001), a source separated from the two younger domains by
  a wide age margin.

These three domains are geologically expected to contribute in different
proportions to the two groups — the NCC north-margin samples being proximal to
cratonic basement, the Songliao Basin being dominated by its encircling arc and
CAOB terranes — furnishing an independent, qualitative check on the
quantitative unmixing (Section 5.5). Since the arc (≈ 130 Ma) and basement
(≈ 280 Ma) modes are the most closely spaced of the three, their relative
partition is the component most exposed to the identifiability limits examined
here, whereas the cratonic mode, isolated near 1900 Ma, is expected to be the
most robustly resolved — an expectation borne out by the posterior analysis
(Section 4).

## 3. Methods

### 3.1 Three-end-member Gaussian mixture model

We model each detrital-zircon U–Pb age spectrum as a mixture of three
geologically defined source end-members: S1, Mesozoic arc magmatism (reference
peak μ ≈ 130 Ma, σ ≈ 15 Myr); S2, Palaeozoic basement of the Central Asian
Orogenic Belt (CAOB; μ ≈ 280 Ma, σ ≈ 30 Myr); and S3, Archaean–Proterozoic
cratonic crust of the North China Craton north margin (μ ≈ 1900 Ma, σ ≈ 100
Myr). For a sample of grain ages a = {a_z}, mixing weights **w** = (w₁, w₂, w₃)
live on the 2-simplex (Σ w_k = 1). Rather than assigning grains to sources, we
marginalise over the latent source label of each zircon, giving the per-grain
mixture likelihood

  log p(a | **w**, **μ**, **σ**) = Σ_z log [ Σ_k w_k · N(a_z | μ_k, σ_k) ],

where N(· | μ_k, σ_k) is the Gaussian density of end-member k. The posterior is
8-dimensional: two free weights (K − 1 = 2) plus the three centres μ_k and three
widths σ_k, co-inferred rather than fixed.

### 3.2 Priors

End-member centres are taken from the reference end-members reported in the
companion supervised-unmixing study (Pan, submitted-a). We place independent
Gaussian priors on each parameter, with relative standard deviations of ±10% on
the μ_k and ±20% on the σ_k, allowing the data to refine the end-member shapes
while anchoring them to geologically meaningful values. A symmetric Dirichlet(1)
prior on **w** imposes a uniform distribution over the simplex, expressing no
prior preference among sources.

### 3.3 Prior-perturbation × bootstrap scan

To separate robust conclusions from those that are sensitive to analyst
choices, we do not fit each sample once. Instead, we systematically perturb the
prior centres over a grid of mean_shift ∈ [−10%, +10%] (11 levels) × sigma_shift
∈ [−20%, +20%] (11 levels) = 121 prior configurations, and at each configuration
draw 100 with-replacement bootstrap resamples of the grains. This yields 12,100
independent MCMC fits per sample, and 15 × 121 × 100 = 181,500 fits in total.
The spread of the posterior weight means across this 121 × 100 grid (reported as
the standard deviation) directly quantifies how much each conclusion depends
jointly on prior specification and on sampling of the finite grain population,
rather than relying on a single posterior credible interval.

### 3.4 MCMC and GPU implementation

Posteriors were sampled with the Goodman–Weare affine-invariant ensemble
sampler (Goodman & Weare, 2010; emcee, Foreman-Mackey et al., 2013) using 64
walkers × 5000 steps with a 1000-step burn-in. The reference emcee
implementation requires ~21 s per fit on CPU, making the full scan infeasible
(~44 CPU-core-days). We therefore re-implemented the sampler as a batched GPU kernel
(NVIDIA L20), completing all 181,500 fits in 4475 s (~74 min). We validated the
GPU sampler fit-by-fit against the emcee gold standard: the maximum absolute
difference in posterior weight means was 0.020, and the posterior standard
deviations agreed to within the same 0.02 tolerance — so the credible-interval
widths underpinning the uncertainty analysis, not merely the means, are
reproduced. Re-seeding emcee itself
produced inter-seed scatter of ~0.19 along the degenerate (adjacent S1/S2)
direction. The GPU–emcee discrepancy is thus an order of magnitude smaller than
emcee's own Monte Carlo noise, establishing equivalence. Bootstrap convergence
was checked at the central prior: the ratio of weight std from b < 30 subsets to
that from the full 100 had a median of ≈ 1.0, confirming that ~30 resamples
already converge and 100 provides a robustness margin (quantified per b in
Table 2). The ensemble sampler itself remained healthy throughout the scan: the
acceptance fraction lay within the recommended 0.2–0.5 range for 99.3% of all
181,500 fits (median 0.42), the only depressed value being on the most
degenerate sample, R31370-S12652 (Fig. 6).

**Table 2. Bootstrap convergence at the central prior configuration.**

| b (resamples) | median (all 45) | IQR (all 45) | median (soft 35) | IQR (soft 35) |
|---|---|---|---|---|
| 10 | 0.90 | 0.82–1.10 | 0.90 | 0.83–1.11 |
| 20 | 0.99 | 0.91–1.12 | 1.01 | 0.93–1.12 |
| 30 | 0.99 | 0.92–1.10 | 0.99 | 0.92–1.09 |
| 50 | 1.02 | 0.97–1.06 | 1.03 | 0.99–1.06 |
| 75 | 1.02 | 0.97–1.05 | 1.02 | 0.98–1.04 |

*Notes:* Entries are the ratio σ_b / σ_100 of the bootstrap standard deviation
of the posterior mean weight from the first *b* resamples to that from all 100,
at the central prior configuration (mean_shift = sigma_shift = 0), aggregated
over the 15 samples × 3 end-member weights (45 pairs) and over the 35 "soft"
pairs with σ_100 > 0.02 — the latter excludes the hard weights whose σ_100 ≈
0.001–0.003 make the ratio a division-noise artefact. The median ratio is
already ≈ 1.0 by *b* ≈ 20 and the interquartile range tightens steadily, i.e.
the bootstrap standard deviation has essentially converged well before 100
resamples; 100 is therefore a comfortable robustness margin, not excess.

## 4. Results

### 4.1 Posterior mixing weights and their uncertainty

Per-sample three-end-member posterior mean weights and their associated
uncertainties (the standard deviation of the posterior mean across the 121 ×
100 prior-perturbation × bootstrap grid) are reported in Table 1 and summarised
in Fig. 2. Across the 15 samples all posteriors resolve geologically
interpretable mixtures of the Mesozoic-arc (S1, 130 Ma), Palaeozoic-basement/
CAOB (S2, 280 Ma) and Archaean–Proterozoic cratonic (S3, 1900 Ma) end-members.
The central message of our uncertainty analysis, however, is that the three
weights are *not* equally well constrained, and that the appropriate unit of
"robustness" is the individual weight rather than the sample as a whole.

The cratonic weight w_S3 is the hard, universally well-resolved component. Its
scan std is small for every sample (0.000–0.085) and effectively
zero for several Songliao samples (≤ 0.002; e.g. R31354-S12519, R31370-S12650,
R31370-S12651). The presence or absence — and, where present, the magnitude —
of an Archaean–Proterozoic cratonic contribution is therefore recovered as a
conclusion insensitive to both prior perturbation and bootstrap resampling.

By contrast, the partition of the remaining budget between the arc (S1) and
basement (S2) end-members is the soft component. The two standard deviations
std(w_S1) and std(w_S2) are large (0.05–0.22), anti-correlated, and nearly
equal within every sample (e.g. 05FW007 0.118/0.121; SL0930 0.055/0.061;
R31354-S12518 0.069/0.069): the data pin the combined arc + basement
contribution w_S1 + w_S2 while leaving its internal split loosely determined,
so the posterior cloud collapses to a one-dimensional ridge along the S1–S2
join (Fig. 2a). A per-sample covariance eigen-analysis makes this explicit —
the leading eigenvector of the weight covariance lies within ~2° of the S1–S2
direction for 11 of the 15 samples, and for the most arc-poor Songliao samples
the anisotropy λ₁/λ₂ reaches 10³–10⁴ with corr(w_S1, w_S2) = −1.000, i.e. an
essentially pure one-dimensional degeneracy once the craton weight is fixed
near zero.

### 4.2 Anatomy of the arc–basement softness

Three properties of this soft direction, established by resampling and
eigen-decomposition of the scan, bear directly on how the weights should be
read.

*The softness is a finite-grain counting-noise floor, not an intrinsic overlap
of the end-member age modes.* Within each sample, subsampling the dated grains
to size m and refitting under fixed end-member shapes gives a soft-direction
spread that scales as σ_soft ∝ m^(−1/2) (median exponent −0.507, interquartile
range [−0.511, −0.490]; prediction −0.500), and whose full-n magnitude matches
the multinomial counting-noise floor of a perfectly separable mixture to a
median factor of 1.02 (14 of 15 samples in 0.97–1.12). The 130 Ma arc and 280
Ma basement age modes are well separated at the data level — their
Kolmogorov–Smirnov resolvability is ≈ 0.999 and the cosine-normalised
Gaussian-overlap Gram matrix of the reference end-members is essentially the
identity. The arc–basement split is thus limited simply by how many grains were dated
(n = 31–111) rather than by any inability of the method to distinguish the two
sources; it is a *reducible* uncertainty that additional dating would tighten
predictably.

*A single Bayesian fit already reports most of this uncertainty.* Decomposing
the pooled posterior variance into the mean within-fit credible variance and
the between-fit scatter of the means (a law-of-total-variance split) shows the
two terms to be comparable: the between-fit scatter accounts for a median of
46% of the total, so a single fit's own credible interval understates the full
scan uncertainty by only a median factor of ≈ 1.4 — not by an order of
magnitude. The 121 × 100 scan therefore refines, rather than overturns, the
single-fit credible intervals.

*The reported uncertainty is a data-resampling quantity, essentially
independent of the prior.* Splitting the between-fit scatter into its bootstrap
and prior-perturbation components, the ±10% (μ) / ±20% (σ) prior shifts across
the 121 configurations contribute a median of under 3% of the between-fit
variance; ≥ 97% is bootstrap resampling of the grains. This holds for 13 of the
15 samples and constitutes a direct robustness check — the posterior weight
means are stable against realistic perturbations of the reference end-member
centres and widths. The two exceptions coincide with the two least-constrained
samples (Section 4.4).

Consistent with a data-driven rather than a model-imposed degeneracy, the soft
direction tracks whichever end-member pair is exchangeable: for the two NCC
north-margin samples that carry almost no arc component (100821-1 and 100821-3,
w_S1 ≈ 0.01) the leading eigenvector rotates off the S1–S2 join and onto the
S2–S3 join (to within 0.0°), the covariance following the degeneracy as it
migrates to the next adjacent pair.

### 4.3 Regional mixing patterns

The two groups show distinct mixing patterns. Most Songliao samples are
dominated by arc plus basement with a negligible cratonic component; the Puetz
R-series in particular yields w_S3 = 0.012–0.016 (R31354-S12518/S12519,
R31370-S12650/S12651/S12654), indicating sediment sourced from Mesozoic arc and
Palaeozoic orogenic terranes with essentially no Archaean–Proterozoic cratonic
input. Three of the four NCC north-margin samples instead carry a resolvable cratonic
signature (w_S3 = 0.443 for 100821-1, 0.687 for 100821-3, 0.431 for 05FW007),
consistent with their position on the northern margin of the North China
Craton; the fourth, 05FW008, is arc-dominated (w_S3 = 0.026). One Songliao sample, SL0932, is exceptionally craton-dominated (w_S3 =
0.872): its arc and basement weights are pinned near zero as the hard pair, so
the craton dominance is robust (its small residual std of 0.034 merely
propagates those two). We interpret this exception in Section 5.5.

### 4.4 The two least-constrained samples

Two samples carry the largest weight uncertainties: 05FW007 (w =
0.133/0.436/0.431, std 0.118/0.121/0.070) and R31370-S12652 (w =
0.247/0.659/0.093, std 0.219/0.177/0.063), in both cases driven by the soft
S1↔S2 partition rather than by w_S3. These are also the only two samples
where prior perturbation contributes materially to the spread (bootstrap
fraction ≈ 0.5–0.6, versus ≈ 0.97–1.0 elsewhere) and where co-inferring the
end-member shapes inflates the soft-direction width beyond the counting-noise
floor (by a factor of ≈ 8.8 for R31370-S12652). For these two samples the
arc–basement split is limited by end-member-shape flexibility as well as by
grain count and — unlike the reducible, counting-noise-limited majority —
additional grains alone would not fully resolve it. This distinction, between
samples whose softness is a reducible consequence of finite grain count and the
minority whose softness reflects genuine end-member-shape ambiguity, is the
practical pay-off of the per-weight uncertainty analysis: it tells the analyst
which sub-populations more dating would help and which it would not.

## 5. Discussion

### 5.1 Not every mixing weight deserves equal interpretive weight

The recurring temptation in detrital-zircon unmixing is to read a single point
estimate of the source proportions as if all of its components were equally
well determined. Our posterior analysis shows that they are not, and — more
usefully — that *which* components are resolved varies from sample to sample in
a way that can be diagnosed rather than assumed. Across the fifteen K1 samples
the cratonic weight w_S3 is recovered as a hard quantity (scan std 0.000–0.085,
effectively zero in several Songliao samples), whereas the arc-versus-basement
partition w_S1↔w_S2 is soft (std 0.05–0.22), anti-correlated, and concentrated
on a one-dimensional ridge in the simplex. Since the covariance is estimated
per sample, the diagnosis adapts automatically: where the arc component is
essentially absent the soft direction rotates onto the S2–S3 join (100821-1,
100821-3), and where the craton dominates the arc/basement pair becomes the
hard one instead (SL0932). We therefore report resolvability per weight (Table
1) rather than as a single per-sample grade, and we suggest that downstream
interpretation follow the same rule: the presence or absence of a craton source
and the combined arc + basement contribution are safe to read, whereas the
internal arc:basement ratio of an individual soft sample is not, unless it is
backed by additional dating.

### 5.2 The softness is a finite-sample counting-noise floor, not a failure of the method

A central result is that the arc–basement softness, at the grain counts
available here (n = 31–111), is set by multinomial counting statistics rather
than by any intrinsic inability to separate the 130 and 280 Ma sources. Three
independent lines of evidence converge. First, subsampling the grains within
each sample and refitting yields a soft-direction spread that scales as
n^(−1/2) (median exponent −0.507), the signature of a counting-noise floor.
Second, that spread equals the exact multinomial floor of a perfectly separable
mixture to within a median of 2%. Third, the reference end-members are
themselves well separated at the data level — their Kolmogorov–Smirnov
resolvability is ≈ 0.999 and the normalised Gaussian-overlap matrix is
essentially the identity. The corollary is optimistic: for the great majority
of samples the arc:basement uncertainty is *reducible*, and a target grain
count to reach any desired precision can be read directly off the n^(−1/2) law.
The exceptions are the two samples (05FW007, R31370-S12652) where
co-inferring the end-member shapes inflates the soft direction beyond the
counting-noise floor (by ≈ 8.8× for R31370-S12652); for these, additional grains
alone would not suffice, and either tighter shape priors or independent
constraints on the end-member spectra would be needed.

### 5.3 The reported uncertainty is honest and prior-independent

Two further properties make the reported uncertainties trustworthy in their own
right. First, a law-of-total-variance decomposition of the pooled posterior
shows that a single fit's own credible interval already accounts for about half
of the total (median within-fit variance share 54%); the 121 × 100 perturbation
scan inflates the standard deviation by only ≈ 1.4× (Fig. 3), so the scan
refines rather than overturns the single-fit posterior — a single Bayesian fit
is not badly overconfident. Second, of the uncertainty that the scan does add,
essentially all is data-resampling: the ±10% μ / ±20% σ prior perturbation
contributes under 3% of the between-fit variance in thirteen of the fifteen
samples (Fig. 3b). The literature-anchored end-member centres and widths are
therefore not load-bearing, and a standard objection to any prior-based
unmixing — that the answer is a restatement of the prior — is here answered
quantitatively. Tellingly, the only two samples where the prior does
contribute materially are exactly the two flagged as least constrained on
independent grounds (Section 5.2). Consistently, the end-member shape
parameters themselves are visibly updated by the data — the posterior on each
peak age μ narrows to 0.3–0.5× the prior width, and the basement and craton
peak widths shift to +1.4 and +1.6 prior-σ (Fig. 5) — so the reference priors
act as constraints that the data refines, not as assumptions that propagate
unchanged into the weights.

### 5.4 Relation to the companion papers

To position the Bayesian treatment relative to the supervised point-estimate
approach of the companion methods study (Pan, submitted-a) — the source of the
reference end-members — we applied that method (identical
end-members, KL multiplicative updates) to the same four real
North-China-Craton north-margin grain-age spectra and compared its point
estimates with our Bayesian posterior means (Fig. 4). The two methods agree
closely: the mean absolute weight difference is 0.03, tightest on the hard
craton weight (w_S3, MAE 0.01) and larger on the soft arc/basement split (w_S1,
w_S2, MAE 0.04), where the point estimate falls within the Bayesian scan
uncertainty. The Bayesian framework therefore leaves the central
estimate unmoved and calibrates it: the soft arc:basement direction — where the two
methods differ most, e.g. sample 05FW007 — is precisely the direction where
any single point estimate is least reproducible, so attaching an
uncertainty to it is what the Bayesian treatment contributes. The comparison is
limited to these four samples: the eleven Puetz/OneDZ Songliao samples were
never run through that pipeline, and the companion study's own case study used
four literature-grounded *synthetic* basin proxies rather than these real
samples, so the point estimates here are obtained by applying its method to the
real data, not read from it.

The weak identifiability documented here is also the real-data counterpart of
the identifiability limit derived analytically in the companion methods paper
on the applicability bounds of fixed-dictionary detrital-zircon unmixing (Pan,
submitted-b; reproducibility code archived at Zenodo,
doi:10.5281/zenodo.20768132). Its identifiability theorem bounds the
weight-estimation error by the end-member Gram condition number through an
explicit Kolmogorov–Smirnov-distance → Gaussian-overlap bridge (with
λ_min(HHᵀ) ∝ ρ^{2(K−1)} and, in the ideal density-Fisher limit, σ_err ≍
1/(√N ρ^{K−1})), and identifies adjacent-end-member pairwise degeneracy as the
controlling bottleneck — the same arc–basement (S1↔S2) pair resolved here. The
connection is specific and complementary. The analytic Gram / condition-number
analysis predicts the *direction* of the ill-conditioning exactly — the weakest
eigenvector of the normalised end-member overlap matrix is (1, −1, 0)/√2,
coinciding with the empirically observed soft axis to within a fraction of a
degree — whereas it does not predict its *magnitude*: the bare density-overlap
condition number (κ ≈ 1 + 4.5 × 10⁻⁵) drastically understates the softness
actually observed, since at finite n the softness is governed by the sampling
of the likelihood, not by the geometric overlap of the end-member densities.
The division of labour is therefore clean: the methods paper supplies the
analytic direction and the sufficient conditions, the present study supplies
the finite-sample magnitude, and the two can be cross-referenced without
re-deriving either. (This also explains why re-seeding the emcee sampler
scatters the posterior mean by ~0.19 along this same axis — the sampler is
tracing a genuinely flat likelihood ridge, not failing to converge.)

### 5.5 Provenance of the NE-Asia Lower Cretaceous basins

Geologically, the hard cratonic weight largely separates the two sample groups, with one exception on each side (below).
The Songliao samples are dominated by arc plus basement with a negligible
cratonic component — the Puetz R-series in particular yields w_S3 =
0.012–0.016 — indicating K1 sediment derived from Mesozoic arc rocks and
Palaeozoic orogenic (CAOB) terranes, with essentially no first-cycle
Archaean–Proterozoic cratonic input, consistent with the Songliao Basin drawing
on its encircling arc and accretionary terranes rather than the distant craton.
Three of the four NCC north-margin samples instead carry a resolvable cratonic signature
(w_S3 = 0.43–0.69), as expected from their position on the northern edge of the
North China Craton; the fourth, 05FW008, is arc-dominated (w_S3 = 0.026), the
NCC-side counterpart to the craton-rich Songliao exception. That exception, SL0932 (w_S3 = 0.87), is a robustly
craton-dominated source — a hard result, not a soft-direction artefact
(Section 4). It is also the only sample of the SL09xx cluster from the Quantou
Formation (the others Denglouku); we interpret its dominant ~1900 Ma component
as proximal sourcing from a local basement high of exposed North China cratonic
basement. Since the craton weight is the hard component throughout, all of
these group-level provenance statements rest on the well-determined part of the
posterior; what should not be over-read is the precise arc:basement split of
any individual soft sample (Section 5.1).

### 5.6 Limitations and sampling implications

The analysis rests on three assumptions, stated plainly. (i) The mixture
is three-end-member with geologically fixed reference spectra; co-inferring the
end-member shapes relaxes this only mildly (±10% μ / ±20% σ) and, as shown,
matters for only two samples. (ii) The symmetric Dirichlet(1) prior on the
weights is deliberately uninformative; a sparsity-promoting Dirichlet(α < 1) is
a natural extension, albeit not required by the present data. (iii) Grain counts
are modest (n = 31–111). The most actionable consequence follows from (iii):
since the arc:basement uncertainty obeys an n^(−1/2) counting-noise law for
thirteen of fifteen samples, future sampling can be designed to a target
precision, whereas the two shape-inflated samples call instead for better
independent constraints on the end-member age spectra. In this sense the
uncertainty analysis is not merely descriptive; it is prescriptive — it turns "how
confident are we?" into "what would it take to be more confident?", and answers
that question differently for reducible and irreducible samples.

## 6. Conclusions

1. Casting three-end-member detrital-zircon unmixing as a Bayesian inference
   problem, with a 121-prior × 100-bootstrap uncertainty scan (181,500 MCMC
   fits over fifteen NE-Asia K1 samples), yields mixing weights whose
   credibility is graded per weight and per sample rather than assumed.
2. A batched GPU implementation of the Goodman–Weare ensemble sampler makes the
   scan practical — 74 min instead of days — and is validated fit-by-fit
   against emcee: the maximum posterior-mean difference (0.020) is an order of
   magnitude below emcee's own re-seeding scatter (~0.19).
3. The cratonic weight w_S3 is hard (scan std 0.000–0.085); the arc:basement
   partition is soft, anti-correlated, and confined to a one-dimensional ridge
   whose orientation adapts to the sample (rotating onto S2–S3 where the arc is
   absent, and becoming hard where the craton dominates).
4. The softness is a finite-grain counting-noise floor — scaling as n^(−1/2)
   (median exponent −0.507) and matching the multinomial floor to a median
   factor of 1.02 — not an intrinsic overlap of the well-separated 130 and 280
   Ma end-members; it is therefore reducible by further dating, except in the
   two shape-inflated samples (05FW007, R31370-S12652) where tighter
   end-member constraints are needed instead.
5. The reported uncertainty is prior-insensitive (prior share < 3% of
   between-fit variance in 13 of 15 samples) and a single fit's credible
   interval already carries about half of it, so the scan refines rather than
   overturns single-fit inference.
6. Geologically, the hard craton axis largely separates the craton-fed NCC north-margin
   samples (w_S3 = 0.43–0.69 in three of the four; 05FW008 is arc-dominated) from the arc + basement-fed Songliao fill (w_S3 ≤
   0.10 for most samples), with the craton-dominated Quantou-Formation sample
   SL0932 (w_S3 = 0.87) interpreted as proximal sourcing from a local basement
   high.

## CRediT authorship contribution statement

**Lujia Pan:** Conceptualization, Methodology, Software, Validation, Formal
analysis, Investigation, Data curation, Writing – original draft, Writing –
review & editing, Visualization.

## Declaration of competing interest

The author declares that he has no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

## Data and code availability

**Name of the code/library:** dz-bayes-uq
**Developer and contact:** Lujia Pan (peter205834@hebtu.edu.cn)
**Year first available:** 2026
**Program language:** Python 3
**Software required:** Python ≥3.10 with numpy, pandas, pyarrow, scipy, emcee, scikit-learn, and matplotlib; the batched sampler additionally requires a CUDA-capable GPU.
**Program size:** ~40 MB (release bundle, dominated by the scan output)
**Availability:** MIT licence; source at GitHub (https://github.com/PNOAHA/dz-bayes-uq), archived at Zenodo (doi:10.5281/zenodo.21242245).

The archive contains the batched GPU
sampler (`bayes_gpu.py`), the scan driver (`run_scan_gpu.py`), the fit-by-fit
GPU-versus-emcee validation (`validate_gpu.py`), the emcee reference
implementation (`bayesian_realdata.py`), all figure-generation scripts, the
fifteen input grain-age samples (CSV, with per-sample source DOIs), and the
full 181,500-fit scan output (`grains_bayes_gpu.parquet`) underlying every
figure and table. Bootstrap seeds are deterministic (SHA-1-derived), so
the resampling is reproducible. The input samples are compiled from published
open datasets: NCC north-margin samples from doi:10.1016/j.epsl.2006.04.029 and
doi:10.1080/00206814.2016.1168323; Songliao samples from the OneDZ compilation
(doi:10.1016/j.gr.2012.04.002) and the Puetz compilation
(doi:10.1016/j.sedgeo.2022.106133; doi:10.1016/j.gsf.2022.101377).

## Acknowledgements

The numerical scan was carried out on an NVIDIA L20 GPU; the exploratory and
validation computations used the D107 cluster of the School of Mathematical
Sciences, Hebei Normal University.

## Funding

This work was supported by the Science and Technology Project of the Hebei
Education Department (grant number QN2025106) and the Doctoral Research Startup
Fund of Hebei Normal University (grant number L2024B05).

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author used Claude (Anthropic) in order to improve the language and readability of the manuscript. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

## References

- Foreman-Mackey, D., Hogg, D.W., Lang, D., Goodman, J., 2013. emcee: The MCMC
  Hammer. Publications of the Astronomical Society of the Pacific 125 (925),
  306–312. doi:10.1086/670067
- Goodman, J., Weare, J., 2010. Ensemble samplers with affine invariance.
  Communications in Applied Mathematics and Computational Science 5 (1), 65–80.
  doi:10.2140/camcos.2010.5.65
- Huang, H., Wang, T., Guo, L., Tong, Y., He, Z.-Y., Yin, J., Wu, H.-H., 2024.
  Crustal modification influenced by multiple convergent systems: insights from
  Mesozoic magmatism in northeastern China. Earth-Science Reviews 252, 104737.
  doi:10.1016/j.earscirev.2024.104737
- Li, H., Han, J., Yin, Z., 2021. Early Cretaceous arc-related volcanic rocks in the
  northern Great Xing'an Range, NE China: records of Paleo-Pacific ocean
  subduction. International Journal of Earth Sciences 110, 1233–1263.
  doi:10.1007/s00531-021-02014-y
- Liu, Y.-J., Li, W., Feng, Z., Wen, Q., Neubauer, F., Liang, C., 2017. A
  review of the Paleozoic tectonics in the eastern part of the Central Asian
  Orogenic Belt. Gondwana Research 43, 123–148. doi:10.1016/j.gr.2016.03.013
- Pan, L., submitted-a. Detrital-zircon provenance unmixing with supervised
  NMF: a forward modelling framework validated on Cretaceous basins of
  Northeast Asia. Submitted to Computers & Geosciences. Preprint:
  https://ssrn.com/abstract=6955308; code:
  https://github.com/PNOAHA/dz-unmix-cg
- Pan, L., submitted-b. Where does detrital-zircon source unmixing break? An
  analytic identifiability criterion and its large-scale Monte-Carlo
  verification for NMF provenance unmixing. Submitted to Computers &
  Geosciences. Code: doi:10.5281/zenodo.20768132;
  https://github.com/PNOAHA/dz-unmix-bounds
- Sharman, G.R., Johnstone, S.A., 2017. Sediment unmixing using detrital
  geochronology. Earth and Planetary Science Letters 477, 183–194.
  doi:10.1016/j.epsl.2017.07.044
- Sundell, K.E., Saylor, J.E., 2017. Unmixing detrital geochronology age
  distributions. Geochemistry, Geophysics, Geosystems 18 (8), 2872–2886.
  doi:10.1002/2016GC006774
- Wu, J.T.-J., Wu, J., Okamoto, K., 2022. Intra-oceanic arc accretion along
  Northeast Asia during Early Cretaceous provides a plate tectonic context for
  North China craton destruction. Earth-Science Reviews 226, 103952.
  doi:10.1016/j.earscirev.2022.103952
- Xiao, W., Santosh, M., 2014. The western Central Asian Orogenic Belt: a
  window to accretionary orogenesis and continental growth. Gondwana Research
  25, 1429–1444. doi:10.1016/j.gr.2014.01.008
- Zhao, G., 2001. Palaeoproterozoic assembly of the North China Craton.
  Geological Magazine 138 (1), 87–91. doi:10.1017/S0016756801005040
- Zhao, G., Cawood, P.A., Li, S., Wilde, S.A., Sun, M., Zhang, J., He, Y., Yin,
  C., 2012. Amalgamation of the North China Craton: key issues and discussion.
  Precambrian Research 222–223, 55–76. doi:10.1016/j.precamres.2012.09.016

---

## Tables

**Table 1.** Posterior mixing weights and per-weight resolvability for the
fifteen K1 samples. (Assembled from the materials pack §1 — full table with
Lat/Lon/n and per-weight mean ± std; notes as drafted, with the internal
cross-reference updated to Fig. 3.)

| Sample | Group | Lat (°N) | Lon (°E) | n | w_S1 (Arc) mean±std | w_S2 (Basement) mean±std | w_S3 (Craton) mean±std | Resolvability (per weight) |
|---|---|---|---|---|---|---|---|---|
| 05FW007 | NCC north margin | 40.833 | 115.833 | 57 | 0.133 ± 0.118 | 0.436 ± 0.121 | 0.431 ± 0.070 | S1,S2 very soft; S3 soft |
| 05FW008 | NCC north margin | 40.833 | 115.833 | 36 | 0.940 ± 0.033 | 0.034 ± 0.033 | 0.026 ± 0.002 | S3 hard; S1,S2 soft |
| 100821-1 | NCC north margin | 40.733 | 113.817 | 82 | 0.012 ± 0.001 | 0.545 ± 0.055 | 0.443 ± 0.055 | S1 hard; S2,S3 soft |
| 100821-3 | NCC north margin | 40.733 | 113.817 | 111 | 0.009 ± 0.001 | 0.304 ± 0.042 | 0.687 ± 0.042 | S1 hard; S2,S3 soft |
| SL0929 | Songliao | 43.090 | 124.488 | 76 | 0.326 ± 0.101 | 0.587 ± 0.105 | 0.087 ± 0.031 | S1,S2 very soft; S3 soft |
| SL0930 | Songliao | 43.100 | 124.692 | 74 | 0.337 ± 0.055 | 0.560 ± 0.061 | 0.103 ± 0.031 | all soft |
| SL0931 | Songliao | 43.100 | 124.696 | 81 | 0.214 ± 0.052 | 0.343 ± 0.057 | 0.443 ± 0.047 | all soft |
| SL0932 | Songliao | 43.100 | 124.701 | 75 | 0.071 ± 0.026 | 0.057 ± 0.023 | 0.872 ± 0.034 | S1,S2 hard; S3 soft |
| R31354-S12518 | Songliao | 43.290 | 122.318 | 79 | 0.606 ± 0.069 | 0.379 ± 0.069 | 0.015 ± 0.007 | S3 hard; S1,S2 soft |
| R31354-S12519 | Songliao | 43.290 | 122.318 | 78 | 0.690 ± 0.060 | 0.298 ± 0.060 | 0.012 ± 0.000 | S3 hard; S1,S2 soft |
| R31370-S12650 | Songliao | 45.873 | 125.532 | 78 | 0.460 ± 0.069 | 0.528 ± 0.069 | 0.012 ± 0.000 | S3 hard; S1,S2 soft |
| R31370-S12651 | Songliao | 45.873 | 125.532 | 64 | 0.567 ± 0.057 | 0.418 ± 0.057 | 0.015 ± 0.000 | S3 hard; S1,S2 soft |
| R31370-S12652 | Songliao | 43.881 | 125.779 | 91 | 0.247 ± 0.219 | 0.659 ± 0.177 | 0.093 ± 0.063 | S1,S2 very soft; S3 soft |
| R31370-S12653 | Songliao | 46.465 | 125.803 | 31 | 0.214 ± 0.071 | 0.195 ± 0.072 | 0.591 ± 0.085 | all soft |
| R31370-S12654 | Songliao | 45.637 | 125.179 | 78 | 0.719 ± 0.050 | 0.264 ± 0.050 | 0.016 ± 0.010 | S3 hard; S1,S2 soft |

*Notes:* Posterior weight means and standard deviations are aggregated over
12,100 independent MCMC fits per sample (121 systematically perturbed prior
configurations — end-member prior centres shifted by mean_shift ∈ [−10%, +10%]
× sigma_shift ∈ [−20%, +20%], 11 × 11 grid — combined with 100 bootstrap
resamples of the detrital-zircon ages); *n* is the number of dated grains,
and Lat/Lon (WGS84) are the sample coordinates compiled from the dz-global-db
compilation (OneDZ / Puetz sources). The reported std captures the combined
prior-sensitivity and data-resampling uncertainty (not the within-chain MCMC
error alone), and for 13 of the 15 samples it is ≥ 97% data-resampling and
< 3% prior-sensitivity; the exceptions are 05FW007 and R31370-S12652, where
prior / end-member-shape flexibility contributes roughly half (see the
decomposition in Fig. 3). End-members: S1 = Mesozoic arc (peak 130 Ma), S2 =
Palaeozoic basement / CAOB (280 Ma), S3 = Archaean–Proterozoic craton (1900
Ma). **Resolvability is graded per weight** by its scan std — *hard* (< 0.03;
insensitive to both prior and bootstrap), *soft* (0.03–0.10), *very soft*
(> 0.10) — rather than by a single sample-level grade, since *which* weights
are resolved differs from sample to sample: the craton weight w_S3 is hard in
most samples, whereas the soft (exchangeable) pair rotates onto w_S2 ↔ w_S3 where
the arc is essentially absent (100821-1, 100821-3, w_S1 ≈ 0.01), and w_S1 ↔
w_S2 become the *hard* pair where the craton dominates (SL0932, w_S3 = 0.87).
The softness of the arc–basement split is set by the finite grain count (n =
31–111; the soft-direction spread scales as n^(−1/2) and equals the multinomial
counting-noise floor to within ~2%), *not* by the proximity of the 130 and 280
Ma end-members, well resolved at the data level; it is
therefore reducible by additional dating — except in 05FW007 and R31370-S12652,
where end-member-shape ambiguity also contributes and more grains alone would
not resolve it.

**Table 2.** (In Section 3.4 above.)

## Figure captions

**Figure 1.** Location of the fifteen Lower Cretaceous (K1) detrital-zircon
samples on a schematic tectonic basemap of NE Asia. The North-China-Craton
(NCC) north-margin group (reddish-purple squares, n = 4) lies on the craton
south of the Solonker Suture; the Songliao Basin group (grey circles, n = 11)
occupies the Songliao wide-rift basin. Regional framework after Pan
(submitted-a): the Siberian and North China cratons, the intervening Central
Asian Orogenic Belt (CAOB), the Mongol–Okhotsk and Solonker sutures, and the
sinistral Tanlu Fault Zone; the blue vector marks NW-directed Palaeo-Pacific
subduction. Near-coincident samples share one symbol (the label gives the
site); individual sample coordinates are listed in Table 1 (compiled from the
dz-global-db compilation, OneDZ / Puetz sources). Line styles follow FGDC TM
11A02; the categorical palette is Okabe-Ito.

**Figure 2.** Detrital-zircon unmixing of fifteen Lower Cretaceous (K1) samples
from NE Asia, resolved as a three-end-member Gaussian mixture under a Bayesian
framework, with full posterior uncertainty propagated through a 121-prior ×
100-bootstrap scan (181,500 GPU-accelerated MCMC fits). (a) Ternary diagram of
posterior mean mixing weights for the arc (S1, ~130 Ma), basement/CAOB (S2,
~280 Ma) and craton (S3, ~1900 Ma) end-members; each sample is a single point
(NCC north-margin group, magenta squares; Songliao group, grey circles)
enclosed by a 2-σ covariance ellipse whose size and orientation encode the
posterior spread, with Songliao samples elongated along the S1–S2 edge. (b)
Dot-and-whisker plot of per-sample end-member weights ± one standard deviation
across the scan grid, coloured by end-member (S1 orange, S2 green, S3 blue);
whisker length is inversely proportional to confidence. Together the panels
show that the craton weight w_S3 is tightly constrained (near-zero spread),
whereas the arc-versus-basement partition w_S1↔w_S2 is soft: the data fix the
arc + basement sum while leaving its internal split limited by the finite number of
dated grains — a counting-noise floor that scales as n^(−1/2) and is reducible
by further dating — rather than by any intrinsic overlap of the well-separated
130 and 280 Ma end-members.

**Figure 3.** Variance-source decomposition of the posterior mixing-weight
uncertainty for the fifteen K1 samples (NCC north-margin group left of the
dashed divider, Songliao right; the two shape-inflated samples 05FW007 and
R31370-S12652 highlighted in orange, with *n* the number of dated grains). For
each sample the pooled posterior weight covariance across the 121-prior ×
100-bootstrap scan has total variance tr(Σ) = Σ_k Var(w_k), partitioned by the
law of total variance into three additive sources: the mean within-fit credible
variance (blue) and the between-fit scatter of the posterior means contributed
by bootstrap resampling of the grains (grey) and by the ±10 % μ / ±20 % σ prior
perturbation (orange). (a) Absolute variance; the value above each bar is the
total posterior spread √tr(Σ). (b) The same, normalised to 100 %. The
within-fit credible width supplies about half of the total variance in every
sample (median 54 %), showing that a single Bayesian fit's own credible
interval already captures most of the reported uncertainty; the
prior-sensitivity share is negligible (median 0.8 %, ≤ 6 % for thirteen of the
fifteen samples) except in 05FW007 and R31370-S12652 (~25 % and ~20 %), where
co-inference of the end-member shapes inflates the arc–basement degeneracy. For
the remaining samples the reported uncertainty is essentially a data-resampling
(finite-grain) quantity, independent of the prior.

**Figure 4.** Agreement between the supervised-NMF point estimate of the
companion methods study (Pan, submitted-a) and this paper's Bayesian posterior
mean, for the four North-China-Craton north-margin samples where both can be
run (05FW007, 05FW008, 100821-1, 100821-3; 12 points = 4 samples × 3
end-members, coloured by end-member). The supervised-NMF weights are computed
by applying that method (identical end-members and KL multiplicative updates)
to the same real grain-age spectra; error bars are the Bayesian scan standard
deviation (this paper). Points fall on the 1:1 line — the mean absolute weight
difference is 0.03 — with the closest agreement on the hard craton weight w_S3
(MAE 0.01) and the largest deviation on the soft arc/basement split (w_S1,
w_S2; MAE 0.04), where the point estimate nonetheless lies within the Bayesian
uncertainty (e.g. sample 05FW007). The comparison is limited to these four
samples as the eleven Puetz/OneDZ Songliao samples were never run through
that pipeline.

**Figure 5.** Prior versus posterior for the end-member shape parameters —
peak age μ (top row) and peak width σ (bottom row) of the arc (S1), basement
(S2) and craton (S3) end-members — testing whether the reference priors
dominate the posterior. In each panel the grey curve is the Gaussian prior
(centred on the reference value, dashed line; relative std ±10 % on μ, ±20 % on
σ) and the coloured curve is a representative single-fit posterior (median over
the 15 samples of the posterior mean and within-fit std, at the central prior
configuration mean_shift = sigma_shift = 0); coloured ticks are the 15
per-sample posterior means. The data appreciably updates the shapes rather than
reproducing the prior: the posterior on μ narrows to 0.31–0.53× the prior width
(tightest for the craton, μ_S3) and shifts by up to 0.7 prior-σ (arc +0.7,
basement −0.5), while σ_S2 and σ_S3 shift to +1.4 and +1.6 prior-σ (the
basement and craton age modes are broader than assumed: σ_S2 ≈ 38 vs 30 Ma,
σ_S3 ≈ 132 vs 100 Ma). That the end-member shapes are data-constrained is the
flip side of the weights' prior-insensitivity (Section 5.3): the data pins the
shapes, so the mixing weights do not depend on the exact prior.

**Figure 6.** MCMC sampling diagnostics for the 181,500-fit GPU scan
(Goodman–Weare affine-invariant ensemble sampler; 64 walkers × 5000 steps,
1000-step burn-in). (a) Distribution of the acceptance fraction over all fits:
99.3 % fall within the emcee-recommended healthy band [0.2, 0.5] (shaded),
median 0.42 (dashed line). (b) Per-sample box plots (NCC north-margin above the
divider, Songliao below); every sample sits in the healthy band. The only
sample with a depressed acceptance is R31370-S12652 (highlighted; median ≈
0.28), the most degenerate / shape-inflated sample (Section 4.4), whose flatter
posterior is intrinsically harder to sample — still healthy. Per-fit integrated
autocorrelation times were not stored; convergence is corroborated by the
burn-in, the acceptance fraction shown here, and the fit-by-fit GPU-versus-emcee
equivalence check (Section 3.4).
