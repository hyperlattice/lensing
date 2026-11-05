# Matched-training, simulator validation, and prior-work progress

## Completed work

The training-control mismatch was corrected in the separately versioned
`src.ml.experiments_v2` runner. The frozen original runner is unchanged;
its recorded legacy failures are not being relabeled as a passing full suite.
New checks cover completed-run resume and rejection of changed control
records. The matched-budget trainer refuses implicit interrupted-run resume.

All six matched-budget models completed: two training arms, three seeds,
880 updates and 14,080 image exposures each. Each paired arm uses identical
realized parent exposures and initial weights, with shared normalization from
canonical training parents. Primary results use final fixed-update checkpoints.
No test or alternate-simulator image was used in their training/evaluation.

On the existing 150-parent source-validation cohort:

| Condition | Standard training MAE | Varied training MAE |
|---|---:|---:|
| Canonical | 0.0931 | 0.0921 |
| Pixel scale 0.04 | 0.1614 | 0.1261 |
| Pixel scale 0.06 | 0.1460 | 0.1133 |
| Half exposure | 0.1322 | 0.1297 |
| Double read noise | 0.1249 | 0.1275 |
| Four-stress average | 0.1411 | 0.1241 |

All errors are arcseconds. The stress-average relative improvement is about
12.0%, and the parent-paired change is -0.01697 arcsec, with descriptive
95% parent-bootstrap interval [-0.02391, -0.01005]. Each individual training
seed also improves its four-stress average: paired changes are -0.01236,
-0.02273 and -0.00892 arcsec. Seed variation is separate from that bootstrap.

The benefit is primarily pixel-scale robustness; read-noise error, some tails
and some signed biases worsen. This is reused development data, not a new
independent final evaluation. It cannot replace the original 27.4% result or
quantify what fraction of that old result was caused by additional updates.

![Matched-budget source-validation results](../results/reports/matched-budget-validation-v1/figures/matched_validation.png)

## Alternate renderer

New versioned adapters correct the Sersic b_n-convention mismatch by converting
both radius and intensity, and explicitly match Gaussian finite support.
Controls passed on 36 light profiles, 4 complete image/point comparisons,
2 PSFs, 8 detector settings, and 12 SIE deflection cases. An additional
integration refinement check (supersampling 2, 4 and 8) passed its predeclared
0.5% plus monotonic-refinement criterion for all four development draws.

The worst converted profile relative error was 3.3e-13; the worst convolved
image normalized L1 discrepancy against the original at matched sampling was
6.9e-6. The worst 4-to-8 clean-image refinement difference was about 0.358%,
and the worst convolved-image refinement difference was about 0.241%.
These small development controls do not establish arbitrary-parameter accuracy.
The engineering release still uses factor 2 to match the original
discretization. Passing the factor-4-to-8 criterion does not certify factor 2
at that tolerance; a final integration policy must be chosen and frozen after
broader development controls.

The corrected `cross-simulator-development-v2` release contains 16 observations
from four parents and four conditions, with no rejected observations. It is
still ineligible for model accuracy evaluation. The central radial floor
differs from the original; broader center/parameter/condition coverage and a
strict final manifest/loader remain necessary. Rendering uses project-owned
integration, root solving, PSF and noise code with external mass/light profiles;
it is not the native PyAutoLens SimulatorImaging pipeline or a calibrated
telescope model.

## What this means for novelty

The detailed primary-source review is
[`prior_work_and_novelty_analysis.md`](prior_work_and_novelty_analysis.md).
It covers foundational CNN inference, existing augmentation and sensitivity
studies, DANN/MMD adaptation, uncertainty estimation, HSC realism studies,
distribution-shift robustness, classical cross-code comparison, and a 2026
regularization study. It links available code/data and distinguishes methods
we can port from exact paper reproductions.

We have not demonstrated superiority to a published method. Neither a CNN
predicting Einstein radius, observational augmentation, nor comparing two
lensing codes is new by itself. The near-agreement between physically
equivalent renderers also means a package switch alone is a weak domain shift.

The candidate contribution is a controlled robustness study separating
training exposure, implementation conventions and new observing families.
Before claiming that targeted randomization improves on prior methods, the
key comparisons are a strong generic/noise-only augmentation control and an
MMD-inspired adaptation baseline on the same fresh A/B benchmark. Adaptation
must receive a disjoint unlabeled target split, with its additional data and
compute clearly recorded. No superiority claim follows from comparing our
MAE with another paper's R2, mean signed residual or posterior coverage.

## Evidence and verification

- [Training protocol](matched_training_protocol_v1.md).
- [Matched-budget source-validation report](../results/reports/matched-budget-validation-v1/report.md).
- [Canonical training summary](../results/reports/matched-budget-v1/report.md).
- [Light/image/PSF/noise controls](../results/reports/cross-simulator-controls-v2/report.md).
- [Integration convergence](../results/reports/cross-simulator-convergence-v1/report.md).
- [Corrected development release report](../results/reports/cross-simulator-development-v2/controls.json).
- [Frozen-source preservation](../results/reports/matched-budget-v1/frozen_source_preservation.json).

Seven new focused automated tests passed (three training/reproducibility tests
and four renderer/gating tests), in addition to the numerical controls above.
A repeated small training experiment produced identical predictions.
All 64 original frozen source files match the actual hashes recorded before
this work's new training run. The original final holdout was not rerun.

## Next work

1. Extend renderer controls across centers and native geometries, explicitly
   govern central-floor differences, and build a strict final manifest/loader.
2. Implement the declared generic/noise-only augmentation and MMD comparators
   if the paper will claim a method advantage over existing work. Keep
   source-only and target-adapted information access distinct.
3. Freeze the method set, physical prior, condition matrix (including an unseen
   PSF family), target adaptation split, new final parents, sample-size
   rationale, practical effect size, and scoring protocol.
4. Generate fresh paired A/B final systems, save predictions before the label
   join, score once, and write the paper around the observed transfer boundary.

The real-image pilot remains separately blocked for accuracy scoring; no
calibrated real-data claim is added by this work.
