# Research priorities: controlled training and cross-simulator transfer

Status checked against saved artifacts on 2026-10-27. This is a proposed new
study, not an amendment to the frozen Week 11 result. No new training or CNN
evaluation was performed for this plan.

## Research question

Does targeted observation-condition randomization improve Einstein-radius
regression when training effort is matched, and does that advantage transfer
from Lenstronomy to an independently implemented rendering path without
training on target-simulator images?

The existing 27.4% result establishes an improvement of the saved randomized
ensemble over the saved canonical ensemble on new same-simulator parents. It
does not isolate the effect of conditions from training effort, and it does
not establish cross-simulator transfer.

## 1. Training fairness and reproducibility

### Evidence

Both saved training configurations use batch size 16, a maximum of 20 epochs,
and patience 6. The baseline uses 700 images; the randomized release uses
1,400 views of the same 700 parents. Baseline training has 44 batches per
epoch. The randomized shape buckets contain 247 images at 214x214, 926 at
256x256, and 227 at 320x320: 89 batches per epoch. Consequently, equal epochs
are unequal optimizer-update budgets. Baseline seeds completed 20, 20 and 18
epochs; randomized seeds completed 20 each. These imply 880/880/792 versus
1,780/1,780/1,780 training updates, not counting evaluation passes. Selected
checkpoints occur at different earlier epochs as well.

The saved improvement is real for those two procedures, but its attribution
requires an additional control.

### Proposed comparison

Train two arms using the same 700 training parents:

| Arm | Available training views | Purpose |
|---|---|---|
| Canonical control | Standard-condition images, revisited as needed | Measures what equal additional training can achieve |
| Targeted randomization | Canonical plus varied-condition images | Measures the intervention with matched training effort |

Give both arms exactly K optimizer updates with the same batch size,
architecture, optimizer, learning-rate schedule expressed in updates, loss,
paired initialization seeds, and validation/checkpoint-selection rule.
Choose K and the validation cadence using source development data before
final evaluation. Use fixed-size shape-homogeneous batches, record sampling
weights, and balance parent exposure. Do not silently give partial shape
batches different effective weights. Equal update counts do not equal wall
time because native image sizes differ; report image exposures, pixel counts
and runtime too.

For the main attribution experiment, use a common normalization derived only
from canonical training parents, including the same target transform. A
separate normalization-policy ablation can follow if needed. Use the same
source-validation criterion and checkpoints at the same update intervals for
both arms. Do not select one arm on canonical validation and the other on
stress validation.

An optional third arm uses extra independent noise realizations under
otherwise standard conditions. This asks whether broader observing conditions
add value beyond generic noise augmentation. It is secondary to completing the
two-arm comparison.

### Reproducibility work

The October 13 full-suite record reports two `run_controls changed` failures.
Inspection identifies a schema mismatch: `run_seed_suite` plans two control
fields, while `train_experiment` now records five. Newly written controls
cannot equal that plan. This diagnosis is from source inspection; no repair
or fresh full-suite pass has been established here.

Implement consistent control serialization in a separately versioned runner.
Retain all scientific controls rather than ignoring provenance mismatches.
Record actual updates, examples, seeds, sampler state, checkpoint selection,
normalization, source/config/data hashes and environment. If resume is
supported, preserve optimizer, scheduler and random states and update count;
otherwise fail clearly. Verify seed-suite creation/resume and that changed
controls are rejected. Preserve frozen source paths and all original results
under the repository's CONTRIBUTING.md requirements.

## 2. Cross-simulator evaluation

### Current boundary

PyAutoLens 2026.10.26.1 is installed in `.venv-autolens` with Python 3.12.
All 12 analytic SIE deflection controls pass (maximum error approximately
5.1e-14 arcsec). Four development parents produced 16 observations with no
generation failures, and six focused checks passed with the backend active.
See `results/reports/cross-simulator-development-v1/controls.json`.

This adapter currently uses PyAutoLens mass and light profiles, with project
root solving, integration, convolution and detector code. It is not a fully
independent instrument simulation. Sersic parameters are copied directly and
their amplitude/radius conventions remain unvalidated. The development images
are therefore ineligible for accuracy evaluation.

### Ordered gates

1. **Establish the same physical problem.** Validate Sersic intensity,
   ellipticity, effective radius and integrated flux on noiseless grids;
   verify units, coordinate orientation, field of view and the SIE target
   conversion. Compare point-image positions, multiplicities and fluxes on
   development parents. Set tolerances before applying the controls.
2. **Validate image formation.** Check pixel-area scaling and convergence
   with finer integration, PSF normalization and boundaries, and repeated-noise
   mean/variance. Inspect difference images for equivalent scenes. Numerical
   conventions should agree within declared tolerances; deliberately different
   PSF families need physical checks rather than identical pixels.
3. **Make renderer independence explicit.** Prefer a backend-native PyAutoLens
   simulation path where feasible. If project integration, PSF and noise remain,
   list them as shared/project components and restrict claims accordingly.
   A package name alone does not prove independent image generation.
4. **Freeze the experiment.** Keep the physical prior fixed for the primary
   renderer comparison. Use a source-only train/validation split and a fresh
   final parent set disjoint from every development release. Render each final
   parent with both A (Lenstronomy) and B (PyAutoLens path), under a declared
   condition matrix. Matched A/B observations belong to the same final parent
   group; they must never cross training and evaluation. Final B images,
   labels and metrics must not influence model or hyperparameter selection.
5. **Evaluate both matched-budget models on A and B.** Include canonical,
   familiar pixel-scale/exposure/noise shifts, then a separately reported
   unseen Moffat-PSF shift. The current four-condition engineering sample is
   not the final benchmark. Determine final sample size from development
   precision requirements rather than treating four parents as sufficient.
6. **Report the transfer result.** Primary comparison: randomized-minus-
   canonical MAE on B, averaged within parent across the declared primary
   conditions. Also report each model's B-minus-A degradation, signed bias,
   high-percentile absolute error, per-condition outcomes and per-seed
   stability. Bootstrap independent parents, not their correlated views;
   report seed variation separately. Record failures and exclusions for both
   renderers, preserving a common paired cohort and reporting selection effects.

Do not alter or re-score the original exactly-once holdout. Any changed
physics prior, target-domain adaptation or additional source morphology is
a distinct follow-up experiment, not part of the primary renderer control.

## 3. Contribution and prior work

These are directly relevant primary sources, not an exhaustive novelty review:

| Work | Established result or scope | Implication for this project |
|---|---|---|
| [Hezaveh et al. (2017)](https://www.nature.com/articles/nature23463) | CNN estimation of strong-lens model parameters | A CNN predicting lens parameters is established. |
| [Pearson et al. (2019)](https://doi.org/10.1093/mnras/stz1750) | Effects of lens light, colour, signal-to-noise and mass/light scatter on parameter estimation | Sensitivity to image properties and training diversity are established topics. |
| [Swierc et al. (2023)](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_121.pdf) | Einstein-radius regression with DANN/MMD adaptation between simulated source and DES-like noisy target domains, both built with deeplenstronomy | This is a close task/domain-shift comparator. It uses unlabeled target data during adaptation. |
| [Agarwal et al. (2024)](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_58.pdf) | Mean-variance prediction plus unsupervised domain adaptation for simulated lensing domains | Domain adaptation and uncertainty under noise shifts already have workshop precedents. |
| [Gawade et al. (2025)](https://arxiv.org/abs/2404.18897) | CNN lens-parameter prediction for HSC samples, examining training-simulation realism | Review its training/test domains and real-data comparisons before positioning our realism claims. |

The proposed contribution is a controlled measurement of whether observation
randomization transfers across rendering implementations after matching
training effort, with no target images used for adaptation. This is a
candidate contribution until the experiment and broader literature review
are complete; do not claim to be the first.

Create a comparison matrix for each close paper: target parameters, data
generator and shared components, train/test physics priors, shift types,
target-domain access during training, update budget, independent parent
counts, uncertainty method, and released artifacts. Compare published metrics
qualitatively unless data, targets and evaluation match. Our 27.4% is not an
improvement over any of these papers.

If a domain-adaptation baseline is added, give it a separate B-adaptation
split disjoint from the B final set, state its extra target-data access, and
match budgets where possible. It answers an additional question and is not
required to establish source-only transfer of the simple intervention.

## Four-week sequence

Implementation update (2026-10-27): the versioned control repair and all six
matched-budget runs are complete. Source-validation stress MAE improved by
about 12.0% under the matched budget, with condition-dependent tradeoffs;
this is not a new final holdout result. The corrected v2 simulator passed 50
light/image/PSF/noise development controls plus 12 mass checks and produced
16 observations. See `prior_work_and_novelty_analysis.md` for the expanded
primary-source comparison and the remaining evidence required for novelty.

Implementation update (2026-11-01): the first 150-parent rapid A/B release
completed rendering, target-free prediction, and one-way scoring, but its
renderer identity audit found 1,350/1,350 Lenstronomy/AutoLens image pairs and
40,500/40,500 paired predictions exactly identical at saved precision. Treat
`cross-simulator-rapid-v2` as a matched-renderer equivalence control plus an
unseen Moffat PSF-family test, not as evidence of simulator-induced transfer.
The next cross-simulator release needs a documented backend-native image-forming
difference and a new immutable release identity.

1. Week 1: versioned training-control repair, update-budget audit and matched
   trainer; light-profile/units/geometry validation; prior-work matrix.
2. Week 2: complete rendering gates and source-only development runs; fix the
   condition matrix, sample-size rationale, selection rule and scoring plan.
3. Week 3: freeze inputs, generate new paired final A/B data, produce and hash
   predictions, then score once. Report negative or mixed transfer honestly.
4. Week 4: build the matched-budget and A/B transfer figures, finalize the
   limitations and comparison table, and polish the manuscript.

If the renderer gates remain open, present the completed original study and
describe cross-simulator work as development. Do not substitute unvalidated
images for a completed transfer result.
