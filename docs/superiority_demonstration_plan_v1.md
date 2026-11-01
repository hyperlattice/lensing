# Plan for testing superiority over existing strong-lensing methods

Date: 2026-10-27. Status: proposed study design, not a frozen preregistration.
This document plans new experiments; it does not report that a published method
has been beaten. Final thresholds, sample sizes and method settings must be
locked before final data are scored. Existing frozen experiments remain intact.

Implementation has started: see [the foundations checkpoint](benchmark_foundations_v1.md)
for the strict engineering loader, augmentation allocation helpers, MMD
primitive and executed development copy. The experiment design below remains
proposed; no weights or methods are frozen and no confirmatory fits have run.

## 1. The claim we should try to establish

Primary question: **At a matched training budget, does a failure-guided
observation-randomization policy reduce Einstein-radius prediction error on a
held-out observing family more than strong uniform augmentation?**

Second question: **Does that policy improve MMD-based adaptation, and can a
source-only version compete with adaptation that has access to target images?**

The intended achievement is a measured improvement over specified, competently
tuned implementations on a shared task. It is not universal superiority over
whole papers, different scientific targets, or every telescope population.

Candidate eventual claim, only if supported:

> Failure-guided observation randomization reduced Einstein-radius MAE versus
> uniform augmentation on a blinded, parent-disjoint synthetic benchmark with
> a held-out PSF family, under matched source-training exposure. The same
> intervention also improved a matched MMD baseline.

If only the first comparison succeeds, omit the MMD improvement claim. If only
the second succeeds, present the intervention as complementary to adaptation.
Real-image accuracy, calibrated uncertainty, data efficiency and faster
inference each require additional evidence below.

## 2. What is already available, and what is missing

Completed: the original frozen study; six equal-update development models;
source-validation stress improvement of about 12.0%; corrected Sersic parameter
mapping; 50 light/image/PSF/noise checks; 12 mass checks; four limited integration
convergence cases; and 16 alternate-renderer development observations.

Missing: strong generic/noise-only comparators, adaptation implementations,
an explicitly specified new targeting policy, broader renderer qualification,
a strict benchmark loader, power analysis for the actual method contrasts,
and a new blinded final sample. The existing validation findings do not fill
these gaps. See [current evidence](research_progress_2026-10-27.md).

The current engineering release retains supersampling factor 2. Passing the
factor-4-to-8 development criterion does not certify factor 2 at that tolerance.
Choose the new final integration policy through numerical development controls.

## 3. Select the papers and the comparison each actually permits

| Prior work | Role in our study | Evidence needed to claim improvement |
|---|---|---|
| [Hezaveh et al. 2017](https://arxiv.org/abs/1708.08842) | Foundational CNN inference and established observational augmentation | A strong augmentation comparator; no claim that lens-parameter CNNs or augmentation are new |
| [Pearson et al. 2019](https://doi.org/10.1093/mnras/stz1750) | Sensitivity to light, noise and training diversity | Common-data robustness comparison, not a comparison of unrelated percentage gains |
| [Swierc et al. 2023](https://arxiv.org/abs/2311.17238) | Closest point-regression adaptation reference: MMD and DANN for Einstein radius | Matched MMD implementation; add DANN before claiming superiority over both methods |
| [Agarwal et al. 2024/2025](https://arxiv.org/html/2411.03334v3) | Mean-variance estimation with MMD and uncertainty evaluation | A matched mean-variance track with proper scoring and coverage, not just our point-regression MAE |
| [Gawade et al. 2025](https://arxiv.org/html/2404.18897v3) | Simulation realism and real HSC modeling comparisons | A separate qualified real-image benchmark, or an explicitly limited synthetic extension |
| [Filipp et al. 2025](https://arxiv.org/abs/2411.05905) | Distribution-shift robustness for population-level dark-matter inference | Context only; scalar Einstein-radius MAE cannot beat their population-posterior task |
| [Poh et al. 2025](https://arxiv.org/abs/2501.08524) | NPE/BNN posterior inference | A separate posterior-inference comparison if we make posterior-method claims |
| [Galan et al. 2024](https://arxiv.org/abs/2406.08484) | Existing blind cross-code classical comparison | Use explicit conventions and blind comparison; changing lensing packages is not itself novel |

The complete evidence ledger, additional 2026 reference and code/data links
are in [prior_work_and_novelty_analysis.md](prior_work_and_novelty_analysis.md).
The roles above are our proposed interpretation of those sources.

Important distinctions from the sources: Swierc uses unlabeled target images
for adaptation. Agarwal adds uncertainty prediction and publishes signed mean
residuals alongside calibration results; those residuals are not our MAE.
Gawade has real-image comparison evidence that we currently lack. None of these
differences establishes an advantage for us without the relevant experiment.

## 4. Define an intervention that differs from standard augmentation

Our existing selected-factor policy samples familiar factors uniformly. It
must remain a historical/control arm, not automatically be called a new
learning algorithm. If a strong generic policy is mathematically identical,
there is no targeting effect to claim.

Proposed new candidate R, **source-failure-weighted sampling**, now has an
implemented allocation helper, but no empirical weights or integrated renderer/
training run. Its purpose is to test allocation of a fixed augmentation budget,
not to introduce another CNN architecture.

1. Fix seven transformation cells: pixel scale 0.04, pixel scale 0.06,
   exposure x0.5, read noise x2, Gaussian PSF FWHM x0.75, Gaussian FWHM x1.25,
   and canonical observing parameters with a fresh detector-noise draw.
   Keep physical lens/source parameters unchanged. Pixel shapes preserve the
   declared angular field as closely as the integer grid permits.
2. Use a canonical source-only reference ensemble and source-development
   parents to measure each cell's paired MAE increase over canonical.
   Average the increases over fixed pilot seeds, giving d_j.
3. Set w_j = max(d_j, 0). With J=7 and exploration fraction epsilon=0.25,
   set p_j = epsilon/J + (1-epsilon)*w_j/sum(w). If all w_j are zero,
   use uniform probabilities. Save the measured scores, resulting weights,
   source IDs and hashes before candidate training.
4. Freeze the probabilities. Do not update them from target labels, final
   images, or the candidate's final performance. This first version has no
   online curriculum or hidden additional hyperparameters.
5. Give R and the generic comparator G exactly the same transformation cells,
   bounds, rendering fidelity, independent-noise opportunity and image budget.
   G uses p_j=1/J. The deliberate difference is the allocation probabilities.

This weighting rule is a proposed testable recipe, not an established new
algorithm or proven lensing novelty. Failure-guided augmentation has broader
precedents; extend the literature search before any priority claim. If the
existing policy suffices for a benchmark paper, R can be omitted, but then
the paper must not claim superiority of a new targeting strategy.

## 5. Core comparison matrix

Use the same image-only CNN backbone for the point-regression track.

| Arm | Training intervention | Target B images in optimization | Purpose |
|---|---|---|---|
| C | Canonical views | None | Equal-budget standard control |
| N | Canonical conditions, fresh independent noise draws | None | Separates noise augmentation from other diversity |
| G | Uniform sampling over the seven declared cells | None | Strong source-only augmentation comparator |
| T | Existing selected-factor policy | None | Connects the new study to the current project |
| R | Frozen failure-weighted sampling over G's cells | None | Proposed intervention |
| C+MMD | Canonical source plus MMD alignment | Unlabeled Gaussian-condition B adaptation split | Literature-inspired adaptation reference |
| G+MMD | Uniform augmentation plus MMD | Same unlabeled B split | Strong adaptation comparator |
| R+MMD | Failure-weighted augmentation plus MMD | Same unlabeled B split | Tests whether targeting improves adaptation |

Do not handicap adaptation by comparing only C+MMD against an augmented R.
G+MMD is necessary to distinguish a targeting advantage from the generic
benefit of supplying adaptation with better source training data.

Primary source-only contrast: R versus G. Secondary source-only contrasts:
R versus N, C and T. Adaptation contribution: R+MMD versus G+MMD, with C+MMD
as context. R versus MMD arms is a separately labeled information-access
comparison; target-free training is a deployment property, not proof of lower
MAE. Zero target access is not itself new.

### Required extension for a DANN superiority claim

Add C+DANN, G+DANN and R+DANN, with the same source budgets and target split.
Do not claim to outperform DANN without running a competent DANN baseline.
This extends the eight-arm core to eleven point-regression arms.

### Required extension for the Agarwal uncertainty comparison

Use an identical mean/variance head and likelihood treatment for six arms:
C, G and R, each with and without MMD. Keep the backbone and normalization
matched within this track. Reproduce the paper's beta-NLL/MMD recipe first,
then tune documented method-specific settings on permitted development data.

Evaluate mean prediction error, negative log predictive density, a proper
distribution score such as CRPS, interval width and empirical coverage at
50/68/90/95%. A narrow interval is not better if it undercovers. Include
any post-training calibration as a named step using a separate source
calibration split, applied consistently. No final-label calibration.

Our point-only CNN cannot establish improved uncertainty calibration. With
this extension, comparisons must match heads/objectives; otherwise label them
as different capabilities. The full point+DANN+MVE design has seventeen arms.

## 6. Reproduce faithfully before claiming to improve a paper

Start with [the Agarwal author repository](https://github.com/deepskies/DomainAdaptiveMVEforLensModeling)
and its [public data](https://zenodo.org/records/13647416). The repository
provides code/models and warns about platform and simulator-version details;
the data record lists two 3.8 GB archives. Pin exact commits, artifact hashes,
package versions and licenses in a separate environment. Do not change the
frozen project environment. No exact reproduction has been completed yet.

There are two deliverables, which must retain different labels:

1. **Paper reproduction:** run the authors' task/data/input channels,
   preprocessing, heads and selection rules. Check a documented reference
   metric, prediction shape, units and calibration behavior. Define the
   acceptable reproduction tolerance before running it from reported seed
   variability/rounding where available. If details are missing, record the
   uncertainty instead of inventing a precise tolerance.
2. **Common-benchmark port:** run the method on our shared task. Our existing
   images are single-band native-resolution data, while the published MVE
   benchmark uses multiband 40x40 images. A channel/backbone/input change is a
   port, not an exact reproduction. Publish a precise deviation table.

Reproducing the public reference is a check on implementation competence; it
does not make unrelated datasets comparable. If reproducing or obtaining an
author implementation is infeasible, use the term literature-inspired
baseline and narrow the claim. Do not contact authors automatically.

MMD implementation checks: kernel symmetry/nonnegativity where appropriate,
the chosen biased/unbiased estimator's definition, finite gradients to the
encoder, no target regression labels entering the loss, and sensible behavior
for identical versus shifted feature samples. DANN checks must verify gradient
reversal and prevent domain IDs from becoming regression inputs. Include
simple synthetic regression recovery checks, not just matching a formula.

## 7. Data roles and separation

| Role | Proposed size | Permitted use |
|---|---:|---|
| A source training | Existing 700 parents initially | All arms' source labels and normalization |
| A source development | Existing 150 parents, clearly labeled reused | Augmentation scores and permitted tuning |
| Additional A calibration | 100 new parents if uncertainty track used | Source-only uncertainty calibration; disjoint from training/development/final |
| B unlabeled adaptation | 500 new parents initially | Only MMD/DANN optimization; Gaussian conditions only |
| B numerical development | Existing four plus at least 100 additional control draws | Renderer/solver validation, no final method ranking |
| A/B power pilot | 120 new paired parents initially | Variance/sample-size planning after method freeze |
| A/B final evaluation | Initially plan 600-2,000 fresh paired parents | One confirmatory scoring pass after freeze |

Pilot sizes are planning choices, not established power guarantees. Increase
them before the final freeze if variance estimation or engineering coverage
is inadequate. Count independent parents separately from rendered views.

The same final physical parent is rendered through A and B and under every
declared condition. Pairing is intentional within final evaluation; no parent,
template, background asset or derived view may cross scientific roles. Check
ancestry and seed identities against existing releases using identity ledgers,
without reading old final pixels or labels. The original historical test and
exactly-once final holdout are not reused for tuning or new claims.

B adaptation contains no Moffat images. Neither target labels nor target
validation labels are available to any optimizer or tuning procedure in the
core study. Store adaptation images behind an image-only interface without
target sidecars. Retain labels separately only for future separately scoped
experiments. Do not give source-only methods B images through normalization,
batch statistics, pseudo-labeling or checkpoint selection.

## 8. Renderer qualification before any new accuracy benchmark

1. Define a common latent SIE radius, light radial function, centers,
   orientation, brightness units, angular field and pixel-area integration.
   Explicitly resolve the central radial-floor difference in new versioned
   renderers. Preserve every original frozen source path.
2. Expand profile, mass, root/magnification and image controls across the
   full source prior and difficult cases: aligned/off-center sources,
   centers on pixel/subpixel locations, near-caustic systems and field edges.
   Include actual canonical/finer/coarser native geometries.
3. Repeat integration convergence across those cases and select/freeze a
   final factor using numerical criteria. Validate A/B equivalence at that
   factor. Do not assume factor 2 or 4 is sufficient from four examples.
4. Check PSF support, normalization, boundaries, signal/background units,
   detector moments, and transformation metadata in both implementations.
   For intentionally different PSF families test physical properties rather
   than demanding equal pixel arrays.
5. Build a strict release manifest and loader with stable identities, parent
   grouping, masks/variance/PSF semantics, hashes, failure ledgers and explicit
   target isolation. No development-only row becomes final merely by renaming
   its split. Test role/target isolation and deterministic regeneration.
6. Freeze rejection rules before generating evaluation systems. Report both
   implementations' rejection rates and causes. A common renderable cohort
   supports a conditional claim; it cannot hide a solver's systematic failures
   or be described as the original unfiltered population.

Declare which components are external, independent project implementations,
or shared. Our current adapter is not native SimulatorImaging. Two nearly
identical equivalent scenes are a consistency control; the new observing
family below supplies the substantive held-out condition.

## 9. Final condition matrix and primary endpoint

Keep the underlying physical prior fixed for the primary experiment. Use
these nine observing conditions for every paired A/B final parent:

| Group | Conditions | Role |
|---|---|---|
| Canonical | Gaussian PSF, original pixel scale/exposure/read noise | Utility/noninferiority guardrail |
| Familiar stresses | Pixel scale 0.04; pixel scale 0.06; half exposure; double read noise | Connects to development and existing project |
| Held-out family | Moffat PSF beta 2.5, 3.5 and 4.5 at matched FWHM and declared finite support | Primary unseen-family endpoint on renderer B |
| Held-out composition | Moffat beta 3.5 plus half exposure | Secondary combined shift |

Moffat is held out from training and model/hyperparameter selection in both
source-only and adaptation arms. Numerical development may inspect its
physical rendering. This is a synthetic observing-family challenge, not an
empirically calibrated telescope benchmark. Keep source morphology, mass-law
and prior changes in separately reported exploratory experiments.

Primary score is single-model expected absolute error, averaged over the three
Moffat conditions, then over parents and training seeds:

`L(method) = mean_seed mean_parent mean_Moffat |prediction - true_theta_E|`.

This averages errors, not predictions. It estimates method performance across
initializations. A fixed-size ensemble is a secondary deployment option; use
the same number of members for every method. Existing three-seed ensemble
results are context and are not interchangeable with this primary score.

Primary contrast is `Delta = L(R) - L(G)`; negative values favor R. Report A
and B separately, each method's B-minus-A change, and the interaction between
method and renderer as secondary diagnostics. Do not pool equivalent A/B views
as if they were independent systems.

## 10. Success thresholds and statistical analysis

Proposed engineering margins, to be reviewed and locked before final scoring:

- **Practically meaningful superiority:** the one-sided upper confidence bound
  for Delta is below -0.005 arcsec (5 mas) on the primary B Moffat endpoint.
- **Canonical utility:** the one-sided upper bound for R-minus-G canonical MAE
  is below +0.005 arcsec, assessed separately on A and B.
- **Error-tail guardrail:** predeclare a +0.01 arcsec noninferiority margin for
  P95 absolute error on the primary condition group; compute P95 across
  parents/conditions within each seed, then average across seeds. Use paired
  resampling and lock the quantile convention before evaluation.
- Report signed bias, relative error divided by true radius, radius-bin
  behavior, each condition, solver/training failures and runtime regardless of
  which method wins. A mean gain does not erase tail or calibration damage.

These margins are benchmark design proposals, not validated cosmological
requirements. Do not change them to accommodate final results. If their
scientific relevance cannot be justified, choose better margins before freeze.

Use at least 10 independently initialized, paired seeds for confirmation,
subject to the power check below. Keep the same source-parent exposure schedule
and initialization pairing within comparable heads. Never select a lucky seed.

Use a crossed paired bootstrap: independently resample parent IDs and paired
seed IDs, carrying all conditions/methods with each selected parent and each
seed. Compute the full estimator in each resample. This retains shared-parent
and shared-initialization structure; do not treat N*S*C rows as independent.
Use at least 10,000 fixed-seed resamples, with the inference implementation
checked against synthetic examples and a crossed-effects sensitivity analysis.
With few seeds, report instability and wider intervals rather than relying
only on parent-conditional uncertainty.

The single predeclared primary R-versus-G endpoint uses one-sided alpha 0.05.
Secondary superiority claims form a prespecified family of contrasts; use
Bonferroni-adjusted one-sided bootstrap bounds or another method fixed before
freeze. Lock the exact contrast list and correction in the scoring config.
Canonical/tail noninferiority are additional required guardrails, not
substitutes for a failed primary test. A claim of beating every eligible
baseline requires passing every corresponding declared comparison.

An interval crossing zero does not prove equivalence. A point estimate below
zero is not a superiority result. Relative percentage improvement is secondary
and must state its comparator/denominator and uncertainty.

## 11. Power and final sample size

After the methods and hyperparameters are frozen, run the separate 120-parent
paired pilot using at least three pilot seeds disjoint from confirmatory
seeds. Use it to estimate variance components, not to choose the winning
method, alter the primary observing family, or retune augmentation weights.

Plan for 90% power when the true improvement is 0.010 arcsec against the
0.005 superiority margin. The difference being detected beyond the margin is
therefore 0.005 arcsec. Simulate power for the actual crossed-bootstrap test,
including multiplicity for secondary claims.

A useful planning approximation is
`Var(Delta_hat) = sigma_seed^2/S + sigma_parent^2/N + sigma_interaction^2/(S*N)`.
Estimate it from paired pilot errors and validate through simulation. As a
purely illustrative zero-seed-variance calculation, parent SD 0.04 arcsec
requires roughly 550 parents for this primary target. That is not a power
claim for our as-yet unrun R-versus-G comparison.

Start planning with N=600 and S=10; allow N up to 2,000 and S up to 20 before
the final freeze. Increase seeds if seed variability is the limiting term:
adding more images cannot remove that component. If the budget cannot attain
the target precision, report the study as lower-powered or restrict the claim
to explicitly fixed deployed ensembles. Do not imply an adequately powered
algorithm-level result from three initializations.

No optional stopping or final-sample enlargement after seeing final method
differences. A later independent replication is a separately registered study.

## 12. Training, tuning and compute fairness

- Match source labels, parent exposure schedules, full batch size, backbone,
  training-only normalization and opportunity for independent noise. Image
  IDs, seeds, truth diagnostics and domain labels never enter the regression
  predictor. Domain labels are permitted only in DANN's domain head.
- Use separate random streams for source sampling, view parameters, target
  batches and initialization. Extra target forwards must not accidentally
  change source exposure counts. Record all actual optimizer updates.
- Use final fixed-update checkpoints for the primary endpoint. Validation-
  selected checkpoints are a named secondary analysis with identical selection
  opportunities; their chosen updates are reported.
- On source development only, inspect learning curves at 880, 1,760 and 3,520
  updates before choosing a common final budget. Give all methods enough room
  to converge, record the decision, and show budget curves. A stalled or
  undertrained reference is not a competent comparator.
- Reserve an initial 12 source-development tuning configurations per arm,
  evaluated over two fixed pilot seeds. Include reported author defaults plus
  sensible optimizer and adaptation-weight/kernel variations. Record all
  trials, failures and selection rules. Extend boundary-limited searches
  before freeze with comparable tuning opportunity for competing families.
- Source-only methods select by the same source-development criterion; UDA
  methods may use unlabeled adaptation images but not target labels for
  selection. Use source-validation predictive loss/MAE consistently within
  the point/MVE tracks, and record the exact choice. No target pseudo-label
  selection unless introduced as a separate declared method.
- MMD/DANN receive the same labeled-source update budget plus their declared
  target batches. Record target examples, pixel counts, FLOP estimates and
  wall time. Add a total-compute-matched comparison before claiming efficiency;
  equal source updates alone do not imply equal compute.
- Include the canonical reference fits and source-development scoring used to
  construct R in its end-to-end development cost. Report both per-candidate
  training cost and total method-construction cost; sharing a reference fit
  across experiments does not make that overhead disappear.
- Training instability is an outcome. Predefine numerical checks and a
  deterministic retry policy for infrastructure failures. Do not replace
  unfavorable seeds. Report algorithmic failures explicitly; a conditional
  MAE on successful runs cannot establish unconditional superiority when
  failure rates differ.

## 13. Two freezes and final scoring

**Freeze 1: methods, before the power pilot.** Lock source data/normalization,
transform definitions and targeting weights, architecture/head, optimizer,
tuning records, selection rule, update budget, target-access policy, metrics,
margins, condition matrix, software and candidate comparison list. No method
retuning after target-family pilot outcomes are visible. Lock final rendering
fidelity and rejection rules here as well, so the power pilot represents the
same observation-generating process as the final benchmark.

**Freeze 2: final evaluation, after variance-only planning.** Lock final N/S,
seed lists, ancestry/seed exclusion rules, final rendering fidelity,
rejection rules, checkpoints, ensemble rule if relevant, bootstrap algorithm,
contrast correction and output identities. Hash code, configurations, inputs,
normalization and checkpoints before final generation/inference.
Rendering fidelity and rejection rules are carried forward unchanged from
Freeze 1; a necessary correction invalidates the pilot and requires a
documented new protocol and power pilot before final generation.

Write target-free final predictions first, verify expected identities/counts
and hash them, then join truth and run one scoring pass. Save an evaluation-
started marker and do not silently resume/repeat final inference. Any approved
recovery procedure must be declared before starting. Prefer independent
review of the freeze/scoring boundary when available; the workflow must also
be auditable if one researcher executes it.

## 14. Ablations that explain an apparent win

Required before attributing a gain to targeting:

1. R versus G with identical transformation support and image budget.
2. R versus N to distinguish broader conditions from fresh detector noise.
3. R versus T to show what the new weighting adds beyond our existing recipe.
4. Same-implementation A versus independently rendered B at matched observing
   conditions, separating numerical implementation from the Moffat challenge.
5. Per-condition, per-radius-bin, bias and tail analysis, including failures.

If R+MMD wins, compare it with both G+MMD and C+MMD; the combination alone does
not prove a new adaptation algorithm. Equal-compute and learning-curve
experiments are required only if claiming speed or sample efficiency, but
their basic cost records are mandatory for every run.

Larger source/mass/prior shifts, correlated detector noise and empirical PSFs
are valuable next experiments. They must be distinct predeclared strata,
not post hoc replacements for a failed primary endpoint.

## 15. Real-data and posterior claims are separate tracks

To compare with Gawade on real images, first obtain compatible calibrated HSC
images and reviewed model references for the same objects. Specify radius
conventions, uncertainty/systematic treatment, center/PSF/noise preparation,
selection function and disjoint tuning/evaluation roles. Compare both methods
on identical objects using the same reference definitions, and disclose that
model-based references are not exact truth. Reference uncertainty is shared
between methods and must be carried through the paired comparison.

The current HST pilot has zero accuracy-ready objects. Twenty or fifty
selected lenses can be a feasibility study; a superiority claim needs a
representative sample and its own power/uncertainty analysis. Do not use a
synthetic win to claim better real-data modeling than Gawade, or better
population inference than Filipp/Poh. These are extension gates, not hidden
requirements for the narrower synthetic paper.

## 16. Deliverables and acceptance checklist

| Work package | Concrete deliverables | Completion gate |
|---|---|---|
| Benchmark contract | Latent conventions, role manifests, ancestry rules, renderer qualification and failure audit | Strict loader and numerical/role-isolation checks pass |
| Literature competence | Pinned author sources, reproduction report, deviations and data licenses | Baseline behavior reproduced or comparison labeled as a port |
| Intervention | Versioned scores/weights, G/R transform recipes, paired schedule tests | Same support/budget; no target influence |
| Training | Every seed/trial, controls, histories, predictions, failures and cost ledger | Method freeze and adequate convergence |
| Power | Pilot variance components, simulated power, chosen N/S and budget rationale | Final precision target assessed before final sample |
| Evaluation | Frozen identities, target-free hashes, scored tables, crossed intervals and corrected contrasts | Single confirmatory analysis; no tuned final decisions |
| Publication | Claim-evidence map, method matrix, limitations, code/data manifests and reproducible figures | Each claim has its required passed comparison |

Required figures: the A/B method-by-condition error matrix; primary paired
effects with simultaneous bounds where appropriate; seed and radius-bin
behavior; error tails/failures; source examples of equivalent and intentionally
shifted images. Add calibration/width/NLL plots only for the uncertainty track,
and matched cost/accuracy curves only for an efficiency claim.

## 17. One-month execution schedule and resources

Assuming work begins October 28, target a complete core package by November 26,
2026. This is a planning target, not a verified workshop deadline.

| Dates | Work | Exit condition |
|---|---|---|
| Oct 28-Nov 3 | Broader renderer/central-floor controls, strict loader, pin/reference implementations, implement G/N/R and MMD | Verified data contract and functioning baselines |
| Nov 4-10 | Source-only targeting scores and tuning; convergence/budget curves; test leakage and failure handling | Freeze 1; candidate and all core baselines fixed |
| Nov 11-14 | Variance-only paired pilot; power simulation; finish confirmatory training under fixed settings; finalize N/S | Freeze 2 with actual checkpoints and powered design |
| Nov 15-17 | Generate fresh paired final systems, hash predictions, score once | Final comparison tables and qualified claims |
| Nov 18-26 | Interpretation, figures, robustness audit, related work, reproducibility package and manuscript | Submission-ready artifact package |

Do the eight-arm core first: initially 80 confirmatory fits at 10 seeds,
plus tuning/pilot fits. Optional DANN adds 30 fits; the MVE track adds 60.
Up to 20 seeds doubles confirmatory counts. Our recent six CPU fits took
about 138-184 seconds each after input loading at 880 updates; do not apply
that timing blindly to longer budgets, new heads or adaptation. Measure
cost on the actual candidate implementations and forecast the full run list
before committing to all extensions. GPU availability is not assumed.

At N=600-2,000, nine conditions and two renderers require 10,800-36,000 final
observations, before development/adaptation products. Measure bytes per
observation from a pilot and reserve space for raw/processed/variance/PSF and
provenance products plus temporary arrays; this may require tens to hundreds
of GB. Free capacity could not be reliably determined from the current drive
query, so storage is an explicit preflight item. Use a declared storage
format compatible with the strict loader; do not drop variance/provenance
silently to fit the disk.

If the core gates are late, drop optional claim tracks before final freeze,
not required baselines because their results look strong. If the core cannot
be completed, submit a bounded development/benchmark paper or defer the
superiority claim. A reproducible negative transfer result remains valid.

## 18. Decision rules for writing the paper

| Observed outcome | Defensible statement |
|---|---|
| R beats G by the primary margin and passes utility/tail gates | Failure weighting improves the declared source-only augmentation benchmark |
| R beats C but not G | Better than canonical training; no advantage over generic augmentation |
| R helps familiar shifts but not the Moffat endpoint | Bounded familiar-condition robustness; no primary unseen-family superiority |
| R+MMD beats G+MMD | Failure-guided augmentation adds value to the matched adaptation method |
| R is similar to MMD with fewer data inputs | Report the measured tradeoff; equivalence/noninferiority needs its own passed test |
| Only an ensemble wins | State the ensemble size/cost and use the ensemble-specific claim |
| MAE improves but tails/calibration fail the declared guardrails | Report the tradeoff; do not claim overall superiority |
| None of the primary comparisons succeeds | Publish the controlled failure/transfer boundary; do not switch the endpoint |

The final manuscript must name the compared implementations, benchmark,
information access and uncertainty. It must not say simply that we are better
than existing papers. The plan is designed to make a narrower superiority
claim defensible if the data support it, and to identify exactly where that
claim fails otherwise.
