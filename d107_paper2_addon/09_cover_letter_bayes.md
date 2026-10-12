Lujia Pan
School of Mathematical Sciences
Hebei Normal University
No. 20 East 2nd Ring South Road, Shijiazhuang 050024, Hebei, China
peter205834@hebtu.edu.cn

2 September 2026

To the Editors
*Computers & Geosciences*

Dear Editors,

I am pleased to resubmit my manuscript, **"Grading the credibility of
detrital-zircon mixing weights: GPU-accelerated Bayesian uncertainty
quantification of Jurassic–Cretaceous samples from NE Asia"**, for
consideration as a research article in *Computers & Geosciences*. This
resubmission adds the requested continuous line numbering, together with minor
editorial corrections — including re-deriving the bootstrap-convergence ratios (Table 2) and the prior-sensitivity summaries from the archived scan output; the scientific conclusions are unchanged.

Detrital-zircon provenance unmixing is normally reported as a single vector of
point-estimate mixing weights, leaving unquantified how trustworthy each weight
is. This manuscript addresses that gap methodologically and demonstrates the
method on real data. Casting three-end-member unmixing of fifteen Jurassic–Cretaceous samples from NE Asia as a Bayesian inference problem, I propagate the
full posterior uncertainty through a systematic 121-prior × 100-bootstrap scan —
181,500 MCMC fits in total. Two contributions are central to the journal's scope
in computational geoscience:

1. *A batched GPU sampler that makes full-posterior uncertainty quantification
   practical.* A GPU re-implementation of the Goodman–Weare ensemble sampler
   completes the 181,500-fit scan in 74 minutes rather than the several days the
   reference CPU implementation would require, and is validated fit-by-fit
   against emcee (maximum posterior-mean difference 0.020, an order of magnitude
   below emcee's own re-seeding scatter). The complete scan output is released so
   that reviewers and readers can reproduce every figure without re-running the
   sampler.

2. *Per-weight, per-sample credibility grading.* The analysis reveals that the
   three source weights are not equally identifiable and that credibility must be
   graded per weight: the cratonic weight is hard, whereas the arc:basement split
   is soft. Subsampling demonstrates that this softness is a finite-grain
   counting-noise floor (scaling as n^(−1/2)), hence largely reducible by further dating,
   rather than an intrinsic overlap of the end-members; and the reported
   uncertainty is prior-insensitive (median prior share below 3% of the
   between-fit variance).
   The framework turns "how confident are we?" into "what would it take to be
   more confident?", answered per sample.

The work is accompanied by two companion studies: a supervised-NMF
forward-modelling method submitted to *Computers & Geosciences* (from which the
reference end-members here are taken) and an analytic identifiability-bounds
paper in preparation (whose adjacent-pair degeneracy
result this study confirms empirically on real data). The three are complementary and independently readable; the present
manuscript contributes the finite-sample uncertainty magnitude that the analytic
paper's direction-only result does not predict. I note this openly to avoid any
appearance of redundant publication: the numerical uncertainty-quantification
results reported here appear in no other submission.

I confirm that this manuscript is original, has not been published previously,
and is not under consideration for publication elsewhere. I have no competing
interests to declare. All code and the complete posterior scan are openly
archived at Zenodo (DOI: 10.5281/zenodo.21242245) and maintained at
https://github.com/PNOAHA/dz-bayes-uq; the input grain-age samples are compiled
from openly licensed (CC-BY) published databases, each retaining its source DOI.
I will serve as corresponding author and can be reached at
peter205834@hebtu.edu.cn.

Thank you for considering my submission. I look forward to the reviewers'
feedback.

Sincerely,

Lujia Pan
School of Mathematical Sciences, Hebei Normal University
