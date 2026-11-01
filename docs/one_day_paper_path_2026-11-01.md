# One-day paper path (decision memo, 2026-11-01)

Status: proposed rapid study; no final images, labels, predictions, or result
have been opened. The full confirmatory protocol remains an alternative, and
its completed development artifacts are preserved. Do not cite this memo as a
completed experiment.

## Why the full route misses a day

At 2026-11-01 16:50 UTC, 5/240 initial tuning fits had completed. Median fit
time was about 177 seconds, implying about 11.8 hours for 240 fits alone.
The full workflow then requires boundary review, 60 budget-curve fits, a
five-seed pilot, power analysis, 10-20 seed confirmatory fits for ten arms,
two-renderer final data, one-way scoring, figures, and manuscript. It cannot
reasonably be called a one-day result.

## Smallest defensible question and claim

Question: under equal CNN, source-parent schedule, optimizer updates and
source exposures, does source-only failure-weighted sampling R improve
Einstein-radius MAE over uniform augmentation G on new simulated lens systems
with an unseen Moffat PSF rendered by engine B? Report canonical A/B MAE and
tail errors as guardrails. A second, descriptive comparison to a common-benchmark
MMD port may be included only if time permits; it receives unlabeled target
images and therefore has different information access.

The paper can claim a measured advantage over G on **our fixed benchmark** if
the declared paired confidence bound and guardrails pass. It cannot claim a
numerical win over published Hezaveh, Swierc or Agarwal results: their data,
targets, metrics, training budgets and information access differ. A direct
published-paper superiority claim requires a faithfully reproduced method on
the same dataset, target, metric and split, which is not ready in a day.
Engine A/B numerical agreement has passed, but they share the project's
integration, PSF and detector recipe. The unseen Moffat family is the actual
observation shift; do not describe B alone as a wholly independent astrophysical
simulator or make a real-telescope transfer claim.

## Rapid protocol to declare before fresh test generation

1. Use the qualified v4 source and target banks unchanged. Fix the current
   source-only policy weights. Fix one common optimizer at the existing
   reference configuration (880 updates, batch 16, learning rate 0.001,
   weight decay 0.0001, Huber 1.0); do not select it from the partial 240-fit
   search. Mark the policy and optimizer as development-informed.
2. Fit C, G and R for ten new paired seeds, with G and R sharing initialization,
   source-parent sequence, supervised examples and optimizer. Include C as a
   descriptive canonical-training baseline. This is 30 fits, about 1.5 hours
   at the observed median before loading and scoring overhead.
3. Fix 600 fresh eligible parents before rendering; preserve the proposal pool
   and every exclusion. Render both engines and the existing nine conditions
   if the unchanged generator is used. Do not inspect test labels while fitting.
4. Seal all predictions, then score R minus G on the three B Moffat conditions
   using paired seed-and-parent resampling. Keep the existing 0.005-arcsec
   meaningful-improvement threshold, 0.005-arcsec canonical allowances and
   0.010-arcsec P95 allowance. Report the one-sided .025 bound, all seeds,
   raw errors, and failed gates. Fixed 10 seeds/600 parents has no demonstrated
   90% power; a non-significant outcome is inconclusive.
5. Write a short workshop paper: one-paragraph problem, exact simulation and
   training controls, prior-work table, held-out result table with uncertainty,
   one error-by-condition figure, limitations and reproducibility links.

If this fast track is executed, version its protocol, fit outputs, generation
design and scorer separately from the full confirmatory study. Do not recast
earlier source-development scores as held-out results or silently reuse any
final set from another role. A negative or inconclusive result is still a
quantified result, but the paper must be framed as a controlled robustness
study rather than a superiority demonstration.

## Work still missing

- A versioned rapid freeze and runner that enforce the three-arm/ten-seed
  budget, fixed optimizer and paired source schedules.
- Fresh paired test images and isolated truth; sealed predictions.
- Paired crossed uncertainty, canonical/tail checks and final figures.
- A paper draft grounded in those actual results and a comparison table that
  distinguishes our benchmark from published numbers.

Budget estimate: development freeze and runner 1-2 hours; 30 fits 2-3 hours;
paired generation and validation 2-4 hours; prediction/statistics 1-2 hours;
manuscript/figures 3-4 hours. These are estimates, not a guarantee, and an
unresolved numerical failure or weak effect cannot be solved by shortening
the statistical gate after seeing outcomes.
