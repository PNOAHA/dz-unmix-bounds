# Highlights

- A batched GPU sampler completes 181,500 Bayesian unmixing fits in 74 min, not days.
- GPU posteriors match emcee fit-by-fit; the difference is below emcee's own noise.
- Weight credibility is graded per weight, per sample — not assumed globally.
- The soft arc:basement split matches the multinomial floor, scaling as n^(−1/2).
- Priors contribute a median of < 3% of the between-fit variance across samples.
