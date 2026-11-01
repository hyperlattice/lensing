# Statistical prior work and finite-seed limits

Primary sources checked 2026-10-27; no new final outcomes have been viewed.

[Owen (2007), The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111)
develops independent row/column resampling for crossed data. Our seed/parent
resampling is this style of two-factor bootstrap, with all condition views
kept inside a parent. It avoids pretending that every seed-parent-condition
error is independent. Exact general bootstrap validity for crossed random
effects is unavailable. We describe percentile bounds as approximate, retain
variance components and seed effects, and promise no exact ten-seed coverage.

[Owen and Eckles (2012), Bootstrapping data arrays of arbitrary order](https://arxiv.org/abs/1106.2125)
studies independent factor reweighting with product weights, including
conservatism under specified crossed-effects conditions. This supports the
factor structure, not a theorem that our P95/ratio percentile bounds have
nominal finite-sample coverage. Interaction-dominated settings can be
conservative; a seed-only t sensitivity remains conditional on observed
parents and cannot replace the main crossed analysis.

[Knopov and Shvai (October 2026), Equivalence testing and run-count design for
stochastic machine-learning experiments](https://arxiv.org/abs/2610.23700)
uses a seed-by-item variance model for practical-equivalence testing and run
planning. Its abstract describes simulation/case-study checks and warns that
pilot variance point estimates can undersize a later experiment. Their AUROC
equivalence results are not our MAE meaningful-superiority test. We use this
as motivation for a larger pilot and variance-uncertainty diagnostics, without
claiming to reproduce or extend their statistical method.

## Consequences for this study, declared before fits

Use five pilot training seeds instead of the earlier three-seed minimum:
720811–720815, disjoint from confirmation 720821–720840. Methods remain frozen
before pilot target-family predictions. Parent count remains 120. Estimate
paired seed, parent and interaction variance without using the observed
treatment mean to set the assumed effect or change the hypothesis.

Primary size selection remains the declared Gaussian planning simulation
with clipped moment variances and actual crossed-bootstrap rejection, plus
Monte Carlo uncertainty. Report both variance doubling and conservative
Gaussian upper-component diagnostics. For the latter, form Bonferroni 90%
upper bounds using variance across seed means, variance across parent means,
and residual mean square with their balanced-model chi-square degrees of
freedom. The first two contain residual noise and are conservative bounds
for corresponding main-effect variances under that model. These are not
distribution-free bounds or substitutes for bootstrap validation.

Disclose power fragility and the 20-seed/2,000-parent cap. A larger test sample
cannot eliminate training-seed uncertainty. Failure to pass superiority
establishes no equivalence or absence of a useful effect. No variance,
bootstrap or sampling method is claimed as our statistical novelty.
