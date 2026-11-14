# Prior work, candidate novelty, and defensible improvement claims

Research date: 2026-10-27. Primary papers and author-maintained resources were
consulted. This is a focused comparison, not an exhaustive proof of priority.
No result in this document establishes superiority over a published method.
The original 27.4% improvement is relative to our own canonical-trained
comparator, evaluated on the same simulator/prior as development.

## Assessment

A CNN estimating Einstein radius is established. Varying noise, PSF and other
training conditions is established. Studying distribution mismatch in strong
lensing is established. An auditable pipeline, frozen scoring and parent-safe
splits are important research practices; they are not sufficient scientific
novelty on their own.

Our most defensible candidate contribution is a controlled study of how much
observation randomization helps after matching training effort, and whether
that benefit survives independently implemented image formation without
target-domain training images. The experiment must distinguish implementation
agreement from a meaningful domain shift. Two codes producing essentially the
same images establish numerical consistency, not broad astrophysical transfer.

The reviewed close papers motivate this combination. They do not establish
that it is absent everywhere in the literature. Avoid any first-of-its-kind
claim until a broader search and a reviewer-facing comparison are complete.

## Primary-source evidence ledger

Each entry states what the source establishes and our interpretation of its
relationship to this project. Unknown details are not filled by assumption.

### Hezaveh, Perreault Levasseur and Marshall (2017)

[Paper](https://arxiv.org/abs/1708.08842),
[Nature publication](https://doi.org/10.1038/nature23463),
[authors' Ensai repository](https://github.com/yasharhezaveh/Ensai).

Established CNN inference of strong-lens SIE model parameters. The published
examples include varying noise and PSF blurring, masks and cosmic rays; broad
observational augmentation therefore predates our project. The public
repository provides a demonstration and an image-generation script.

Our inference: neither fast CNN inference nor source-only observation
augmentation is a defensible new-method claim. We should cite this work as a
foundation and use an augmentation baseline inspired by this established
practice. A reproduction would need its exact targets, source populations,
preprocessing and artifact model; it is not a direct numerical comparator to
our current one-target benchmark.

### Pearson, Li and Dye (2019)

[Paper and sections 3.3-3.4](https://doi.org/10.1093/mnras/stz1750).

Investigates CNN mass-parameter inference in simulated LSST/Euclid imaging,
including lens light, colour, signal-to-noise and mass/light misalignment.
Moderate scatter in training yields more consistent results over tested
misalignment. It also compares training and testing at different stacked
exposure depths. Training is reported on 20,000 single-band images for 150
epochs. Its percentage gains concern specific changes to inputs/observations
and multiple parameter errors.

Our inference: sensitivity analysis and training diversity are already
established. Our added test should isolate observing-condition variation from
extra optimization, then assess transfer under a separately validated renderer
and an unseen PSF family. Do not compare their multi-parameter percentage
gains with our four-condition scalar MAE percentage. Budget matching in their
specific experiments would require a dedicated reconstruction; do not allege
an uncontrolled budget from an incomplete review.

### Swierc, Zhao, Ciprijanovic and Nord (2023)

[Workshop paper](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_121.pdf),
[arXiv](https://arxiv.org/abs/2311.17238). See sections 2-5 and Table 1.

This is a very close task comparator: Einstein-radius regression with DANN
and MMD domain adaptation. Both domains use deeplenstronomy, built on
Lenstronomy; source images have no pixel noise and target images emulate DES.
Each domain has 50,000 40x40 images and a 70/10/20 split. Target images enter
adaptation without regression labels. Table 1 reports target R2 of 0.50,
0.94 and 0.95 for no adaptation, DANN and MMD, averaged over ten initializations.
The reported baseline/adaptation learning rates differ. The discussion itself
notes that simulating training noise may suffice for a noise-only shift.

Our inference: the useful question is whether a simple source-only intervention
can transfer to a renderer/observation family absent from training, and how
that compares to MMD when target images are available. Their R2 values cannot
be converted into our MAE or ranked against our 27.4%. We have not established
an exact author-code reproduction for this paper.

### Agarwal, Ciprijanovic and Nord (2024; revised 2025)

[Revised paper](https://arxiv.org/html/2411.03334v3),
[workshop version](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_58.pdf).
See sections 2-4 and Table 2.

Combines mean-variance estimation with MMD-based unsupervised adaptation for
Einstein-radius inference. Each domain has 100,000 40x40 multiband objects,
with a 70/10/20 split and nearly noiseless versus DES-like observing settings.
Target images are used during adaptation. Training uses 150 epochs and batch
size 128; five initializations are discussed. Table 2's 0.0818 versus 0.0425
arcsec target values are explicitly mean signed prediction residuals, not MAE.
Its uncertainty-calibration results are a capability our current predictor
does not establish.

Our inference: source-only transfer is a different information-access setting,
not evidence that adaptation is inferior. Compare methods on identical held-out
parents and calculate MAE from their predictions. Do not claim better
uncertainty calibration using ensemble seed spread.

The [author repository](https://github.com/deepskies/DomainAdaptiveMVEforLensModeling)
provides code and model artifacts and states Linux support, a simulator-version
caveat, and an untested Windows environment. The linked
[Zenodo record](https://zenodo.org/records/13647416) lists two 3.8 GB archives.
These are identified resources, not downloaded or reproduced benchmarks.

### Gawade et al. (2025; first preprint 2024)

[Paper](https://arxiv.org/html/2404.18897v3), sections 2-4, Appendix A and
Data Availability. [YattaLens](https://github.com/astrosonnen/YattaLens).

Uses SIMCT simulations with increasing realism, including arcs inserted into
HSC cutouts, and predicts Einstein radius and ellipticity. It models 182 HSC
lenses to construct comparison measurements and discusses ten systems with
prior literature models. The abstract reports approximately 10-20% radius
accuracy, less than 5% bias and an order-10% outlier fraction for the favored
training sample. Its Appendix A discusses parameter conventions. The CNN and
simulated images are available on request; SuGOHI lenses and YattaLens are
public.

Our inference: this work has real-observation evidence beyond ours. We cannot
claim a stronger simulation-to-real result. Its image-realism ladder is useful
for a later hybrid-data experiment. A numerical comparison requires the same
reviewed real systems, image preparation and radius definition, with model
reference uncertainty retained. We have not obtained its requested code/data.

### Filipp, Hezaveh and Perreault-Levasseur (2025)

[Paper, revised November 2025](https://arxiv.org/abs/2411.05905).

Studies neural ratio and sequential posterior estimators for population-level
dark-matter subhalo inference. Shifts in nuisance distributions, including
source morphology, can bias estimates even when in-distribution performance
is good. This is a different target and inference level from scalar
Einstein-radius regression.

Our inference: distribution-shift sensitivity is not itself novel. This work
motivates reporting source/physics changes separately from observational and
renderer changes. Our scalar MAE cannot establish superiority for their
population-posterior task. Cite it as broader robustness context, not as a
benchmark that we have beaten.

### Poh et al. (2025)

[Published-version preprint](https://arxiv.org/abs/2501.08524).

Compares neural posterior estimation and Bayesian neural networks for 1-, 5-
and 12-parameter DES-like simulated strong-lens inference. It evaluates
posterior accuracy, precision and calibration, reporting advantages for NPE
in that setting.

Our inference: a simple one-target CNN is a useful controlled probe, but is
not a methodological advance over posterior inference. Point prediction and
calibrated posterior coverage answer different questions. A future NPE
baseline would require matching targets and data and evaluating coverage;
it is unnecessary for the initial two-arm attribution experiment.

### Galan et al. (2024): cross-code comparison already exists

[Paper](https://arxiv.org/abs/2406.08484),
[author-institution full text](https://orbi.uliege.be/bitstream/2268/326168/1/2024_AA_692_A87_aa51095-24_Galan_et_al.pdf),
[publication](https://doi.org/10.1051/0004-6361/202451095).

Blindly compares six classical modeling methods on independently simulated
HST-like imaging, with parameterized lens mass and different source
reconstructions. It explicitly excludes deep learning from the methods
compared and uses COOLEST to reconcile code-dependent products and conventions.
It finds method-dependent systematic biases; no single method dominates all
properties. It publicly releases simulated data and modeling products.

Our inference: using two lensing codes is not itself new. Adopt its principle
of explicit code-independent parameter definitions and consider
[COOLEST](https://github.com/aymgal/COOLEST) for future interchange. Our narrower
candidate question concerns source-only CNN training interventions and
transfer, not classical method agreement. Their
[public data](https://github.com/aymgal/LensSourceDegeneracy_public) are a
potential external challenge set, but require mass-profile, radius-convention
and source-population review before any scalar CNN comparison. Our current SIE
target cannot silently stand in for an elliptical power-law radius.

### Additional 2026 check

[Ancona-Flores et al., revised April 2026](https://arxiv.org/abs/2603.06339)
examines dropout configurations in CNN inference of SIE parameters from
simulated images. This reinforces that routine architecture/regularization
changes are not enough to support a broad novelty claim. This source was
screened at the abstract level; it is not part of our direct benchmark plan.

## What can be compared fairly?

### Evidence obtained during this task

The three-seed matched-budget training is now complete: both arms used 880
updates and 14,080 exposures per seed, identical parent schedules and initial
weights, and shared canonical-training normalization. On the existing source
validation cohort, ensemble four-stress MAE changed from approximately 0.1411
to 0.1241 arcsec (12.0% relative improvement). The paired change is -0.01697
arcsec, with a descriptive parent-bootstrap interval [-0.02391, -0.01005].
Pixel-scale conditions account for most of the gain; doubled-read-noise MAE
and some error tails worsen. See
[`matched-budget validation report`](../results/reports/matched-budget-validation-v1/report.md).
This is reused development data, not a new final test. It neither replaces
the original 27.4% holdout result nor partitions that old gain into causal
percentages attributable to extra training.

The corrected alternate renderer passed 50 light/image/PSF/noise controls and
12 deflection controls, and generated 16 new development observations. An
additional four-system integration-refinement gate passed its declared 0.5%
tolerance and monotonic-refinement criterion; broader parameter/center coverage
remains necessary. The
worst convolved-image normalized L1 difference on four equivalent scenes was
6.9e-6. This near-agreement reinforces the novelty limitation: switching
software while keeping all physical and observing choices equivalent is
principally a numerical consistency check. The more substantive research
challenge is a declared new observation family (such as Moffat PSFs), with
generic-augmentation and adaptation comparators where feasible. Passing the
engineering controls does not supply that generalization result.
The current engineering images retain supersampling factor 2. The additional
convergence criterion concerns refinement from factor 4 to 8, so it does not
by itself qualify factor-2 images at the same tolerance.

There are three distinct comparisons. Do not merge them into one leaderboard.

| Comparison | What it can establish | What it cannot establish |
|---|---|---|
| Our canonical versus randomized CNN, matched updates and parent exposures | Attribution to changed training views under declared controls | Superiority to published architectures or real-data methods |
| Literature-inspired methods on one common A/B benchmark | Method performance under our exact data/access/budget setting | Exact reproduction of original published numbers |
| Exact author implementation and public dataset | Reproduction or extension of that particular study | Universal superiority on other populations or observing setups |

The essential common-benchmark arms are:

| Arm | Source labels | Target B images during training | Role |
|---|---|---|---|
| Canonical CNN | Yes | None | Matched-budget control, implemented |
| Targeted randomization | Yes | None | Our intervention, implemented |
| Noise-only augmentation | Yes | None | Separates broad condition variation from extra noise realizations; pending |
| Broader generic observation augmentation | Yes | None | Tests whether failure-guided targeting adds value over established augmentation; pending |
| CNN plus MMD | Yes | Unlabeled adaptation split | Closest practical domain-adaptation comparator; pending |

DANN is an optional second adaptation comparator. Start with MMD to keep scope
manageable. A faithful MVE+MMD comparison adds a variance head and coverage
evaluation; a point-regression MMD baseline must be labeled as inspired by the
papers, not an exact reproduction of their uncertainty model.

All source-only arms need the same source parents, initialization seeds,
fixed-size update budget, optimizer, normalization policy and source-only
checkpoint rule. Use separate random streams for sampling and initialization.
For MMD, preserve the same labeled-source update budget and record the extra
target forward passes, pixel counts and runtime. It has extra information and
compute; do not describe it as an identical-information comparison.

For method tuning, use source validation for source-only arms. Give adaptation
arms an independent target adaptation split and a predeclared selection rule
that does not use final B labels. Any supervised target-validation access
belongs to a separately labeled setting. Freeze the final B set and never
use it to choose augmentation strength, MMD weight, training duration or method.

## Candidate improvements and the evidence each needs

| Proposed claim | Minimum evidence | Current status |
|---|---|---|
| Variation helps beyond extra training | Equal-update paired-parent two-arm result on a fresh final set | Training complete and source validation positive; fresh final result pending |
| Benefit survives a new renderer | Passed physical/image controls and lower MAE on blinded B parents | Development light/image/convergence controls pass; broader coverage, strict release and final scoring pending |
| Robustness extends beyond familiar conditions | Positive result on a preregistered unseen PSF/source condition | Moffat engineering condition exists; final scoring pending |
| Targeted variation beats generic augmentation | Same-budget generic/noise-only arms on the same final systems | Pending; cannot claim yet |
| Source-only variation competes with adaptation | Same benchmark with MMD and declared target access, paired intervals | Pending; cannot claim yet |
| Better than a particular published method | Verified common-task comparison or exact reproduction, uncertainty on the difference | No evidence yet |
| Less data or faster at equal accuracy | Matched learning/compute curves on the same task and hardware | No evidence yet; 700 parents alone does not establish efficiency |
| Accurate on real lenses | Reviewed calibrated images, compatible reference definitions and uncertainties | No qualified real scoring sample yet |

For a superiority statement, use the paired difference in absolute error on
the same final parents, aggregate conditions within each parent, and bootstrap
parents. Report seeds separately and show high-percentile errors and bias.
Predeclare a practical effect size before final scoring. A confidence interval
including zero does not establish equivalence; noninferiority requires a
predeclared margin and appropriate analysis. Select one primary endpoint;
label multiple conditions/methods as secondary or account for multiplicity.

## Numerical traps to avoid

- Our 27.4% is (our baseline MAE minus our randomized MAE) divided by our
  baseline MAE on four familiar synthetic stresses. It is not a paper-to-paper
  performance difference.
- MAE, signed mean residual, residual scatter, R2 and posterior coverage are
  different statistics. A small signed residual can conceal large cancelling
  errors. Use the same prediction-level metrics for direct comparisons.
- Fractional errors may divide by truth or by prediction; state the formula.
- Equal target names do not guarantee equal Einstein-radius conventions,
  particularly for elliptical models. Translate and validate definitions.
- Equal epochs do not imply equal optimizer updates. Equal updates do not
  imply equal pixels or runtime, especially across native image resolutions.
- 750 images of 150 parents are not 750 independent systems. Ten training seeds
  do not multiply the number of independent astronomical test systems either.

## Proposed paper positioning

Candidate title: **Does observation randomization transfer across strong-lens
rendering implementations? A matched-budget study of Einstein-radius regression.**

The paper should separate three effects: optimization exposure, numerical
rendering conventions, and observation-family mismatch. Keep the physical
population fixed for the primary renderer comparison. New source morphologies,
mass models or parameter priors are additional experiments with distinct
claims. Record shared and independent components of both rendering paths.

An acceptable outcome is a precise boundary or a failure: if the advantage
disappears after budget matching, revise the attribution of the historical
gain. If it survives within A but fails on B or Moffat images, report the
transfer limit. Do not keep modifying the final benchmark until the desired
improvement appears.

The minimum workshop package is the completed matched-budget comparison,
validated A/B rendering evidence, a blinded paired transfer table, a generic
augmentation control where feasible, and this related-work analysis. Adding
MMD would materially strengthen any claim of improvement over a literature
method. Without that comparison, frame the contribution as new controlled
evidence and a benchmark, not a better algorithm than those papers.

## Saved evidence and next actions

- Protocol: `docs/matched_training_protocol_v1.md`.
- Trainer/config: `src/ml/matched_training.py`, `configs/ml/matched_budget_v1.yaml`.
- Versioned control repair: `src/ml/experiments_v2.py`; frozen v1 unchanged.
- Training output: `results/ml/theta-e-matched-budget-v1/`.
- Simulator development evidence: `results/reports/cross-simulator-development-v1/`.
- Training completed before simulator implementation changes. Source-only
  stress-validation predictions and their hashes are saved separately.
- Light/image development controls passed; see
  `results/reports/cross-simulator-controls-v2/report.md`. Broader renderer gates
  across a wider parameter/center range and a final scoring release remain open.
- Before final scoring, choose the benchmark arms, target-access rules,
  practical margin and primary endpoint; freeze them in a new protocol.
- Paper-code/data resources above remain external links unless explicitly
  recorded as downloaded and verified. No authors were contacted.
