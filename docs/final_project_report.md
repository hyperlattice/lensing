# Final Project Report

The final project is a completed, reproducible study of **Einstein-radius prediction from simulated strong gravitational lens images**, supported by a real-observation preprocessing pipeline, classical modeling checks, and a documented reliability evaluation.

Its strongest result is that targeted training randomization reduced average prediction error across four synthetic stress conditions by **27.4%** on an independent holdout. The evidence supports robustness within the tested simulation setup. Real-observation accuracy and calibrated predictive uncertainty remain unestablished.

Current verification also found **104 passing tests and 2 failing tests**. Those failures concern verification of training-run controls in the multi-seed experiment workflow and should be addressed before calling the current codebase fully verified.

This report is based on the repository's source, configurations, saved results, and final research package. The package was built on **September 26, 2026**. The verification reported here was performed on **October 13, 2026**.

## Scientific objective

The project predicts `theta_E_arcsec`, the angular Einstein-radius parameter, for simulated singular isothermal ellipsoid—or SIE—lens systems. This parameter describes the angular scale of gravitational lensing.

The project does not directly estimate physical lens mass. A defensible mass interpretation would additionally require the relevant lens and source distances, cosmology, and a specified mass definition. The codebase explicitly distinguishes exact simulator truth from observational radius estimates.

The central research question is whether a CNN trained under one observation setup remains reliable when pixel scale, exposure, noise, or source properties change—and whether measured weaknesses can guide a useful training intervention. The final study follows that question through sensitivity measurements, mitigation, and a new-parent holdout evaluation. See the [final manuscript](reliability_study/final/manuscript_draft.md).

## Delivered system

The delivered system covers the full experimental workflow. It contains:

- A CASTLES FITS inventory and preprocessing pipeline.
- A shared, validated dataset contract for actual and simulated observations.
- Deterministic simulation generation with recorded parameters and noise seeds.
- Splits that keep related observations together.
- Classical lens-recovery and exploratory astrometric checks.
- CNN training, checkpoint selection, evaluation, and multi-seed comparisons.
- Controlled reliability experiments and a frozen final holdout.
- Tables, figures, a failure catalog, research documentation, and checksum manifests.

The public command-line coordinator is `src.pipeline`. The implementation is organized into data processing, simulation, modeling, machine learning, and validation modules. The [codebase guide](codebase_guide.md) explains these boundaries.

## Data releases and independence

The data releases preserve an important distinction between observations, training examples, and independent systems.

| Dataset or release | Size | Role |
|---|---:|---|
| `real-castles-v2` | 137 observations from 69 lens systems | Actual-observation processing and descriptive domain comparison |
| `sim-eng1000-v1` | 1,000 independent simulated parents | Original engineering and ML benchmark |
| Original training split | 700 parents | Baseline training |
| Original validation split | 150 parents | Development and model selection |
| Original test split | 150 parents | Historical baseline evaluation |
| Randomized training release | 1,400 views from 700 parents | Final training intervention |
| Independent final holdout | 750 observations from 150 new parents | Final frozen comparison |

The randomized training release doubles the number of views, while retaining **700 independent underlying systems**. Likewise, the final holdout contains five observations per parent, so its independent statistical sample is **150 parents**.

The splitting machinery groups connected ancestry rather than treating every row as independent. This prevents related parents, templates, and derived observations from crossing split boundaries. The engineering release also records identities, seeds, checksums, rejection logs, and validation evidence. See the [engineering-release specification](production_dataset_v1.md).

## Preprocessing and dataset contract

The preprocessing and loader design make the experiments auditable. Actual images pass through metadata extraction, catalog enrichment, a shared crop/pad transform, background estimation, normalization, mask handling, variance handling, and product validation.

Missing scientific information remains explicitly missing or flagged for review. In particular, the actual release lacks empirical PSF products and uses a labeled toy variance fallback where suitable error products are unavailable. Structural validity therefore does not imply instrument calibration.

The structured loader separates:

- Image products and allowlisted observational metadata.
- Explicitly requested targets.
- Full provenance and privileged simulator information.

That separation reduces the risk of accidentally feeding truth, random seeds, identifiers, or sidecar information into a model. The full provenance view still requires careful handling by callers.

## Simulation system

The simulation system uses staged controls before generating training data. Its release ladder progresses from an analytic SIS control to SIE lenses with extended sources, then introduces lens light, point images, PSF convolution, and detector noise.

| Stage | Purpose |
|---|---|
| P10 | Analytic SIS control |
| P30 | SIE lens with extended source |
| P60 | More complete observation rendering |
| P100 | Broader stress conditions |

P100 is kept separate from ordinary training. This preserves the intended training distribution and allows broader conditions to function as stress tests.

The simulations use analytic Sersic sources, Gaussian PSFs, scalar backgrounds, independent Poisson/read noise, and hand-set priors. The backend dependency is pinned to `lenstronomy==1.14.2`. Saved products support deterministic rerender checks.

These choices make controlled experiments practical, but they do not reproduce the full complexity of astronomical instruments, source populations, or survey selection.

## Model and training

The predictor is a compact CNN ensemble. The original image-only network contains **248,409 trainable parameters** and four convolutional blocks with channel widths of 8, 16, 32, and 64. The blocks use GroupNorm and SiLU activations, followed by adaptive pooling and a small regression head.

The baseline configuration specifies:

| Setting | Value |
|---|---:|
| Maximum epochs | 20 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Huber delta | 1.0 |
| Early-stopping patience | 6 |
| Hidden features | 128 |

Training normalization is computed from training data, and checkpoints are selected using validation data.

The final predictor averages three frozen CNNs trained with seeds **12001, 12002, and 12003**. The comparator averages three CNNs trained under canonical observation conditions.

For context, the original single-model historical test MAE was **0.0876 arcsec**, and the original three-seed validation benchmark was **0.09210 ± 0.00872 arcsec**. The latter uncertainty is seed-to-seed sample standard deviation. Those results use different evaluation settings from the final holdout and should be kept distinct.

## Development findings

The development study identified specific weaknesses before choosing the mitigation. Matched-parent experiments changed observation conditions while retaining the underlying simulated lens system.

| Development intervention | Reported increase in baseline validation MAE |
|---|---:|
| Pixel scale 0.04 arcsec/pixel | +0.0707 arcsec |
| Pixel scale 0.06 arcsec/pixel | +0.0489 arcsec |
| Half exposure | +0.0463 arcsec |
| Double read noise | +0.0350 arcsec |
| Tested source-position shift of x +0.15 arcsec | Up to +0.0248 arcsec |

The tested Gaussian PSF-width effects were small or inconclusive. Fixed shear effects were also inconclusive.

Additional experiments established useful boundaries:

- Validation MAE decreased from **0.2039** at 100 training parents to **0.1163** at 300 and **0.0946 arcsec** at 700.
- A train-only Ridge baseline improved on the training mean but underperformed the CNN across the seven reported conditions.
- The observation-only point-image geometry baseline had **44 fallbacks out of 150 canonical cases**, with all-parent MAE of **0.2007 arcsec**.
- Masking a fixed image annulus increased MAE by **0.2632 arcsec**, showing that the model relies on information in that region. This does not establish that the information is an invalid shortcut.
- An alternate SIE equation/root implementation closely agreed with the original path, but shared source-light, PSF, noise, and preprocessing components.
- CNN initialization provided no consistent speed advantage for the tested classical point-image optimizer.

These findings support specific experimental conclusions. They do not establish general superiority over full classical pixel-likelihood modeling. The [claim-evidence map](reliability_study/final/claim_evidence_map.md) records these distinctions.

## Training mitigation

The adopted mitigation broadens observation conditions during training. Every training parent retains its canonical view and receives one additional randomized view.

The randomization samples:

- Pixel scales of **0.04, 0.05, or 0.06 arcsec/pixel**.
- Exposure multipliers of **0.5 or 1.0**.
- Read-noise multipliers of **1.0 or 2.0**.

Native shapes vary with pixel scale—320×320, 256×256, and 214×214—to approximately preserve angular field of view.

On development validation, this reduced the equal-weight four-stress MAE by **0.0290 arcsec**, with a parent-bootstrap 95% interval of **[−0.0351, −0.0231]**. Canonical MAE also improved by **0.0109 arcsec**. The intervention passed the predeclared selection rule before final holdout scoring. See the [mitigation record](reliability_study/week8_mitigation_status.md).

## Independent holdout results

The independent holdout provides the project's strongest performance evidence. It contains 150 new parent systems, each rendered under canonical conditions and the same four selected stress conditions.

The final comparison reports:

| Condition | Baseline MAE | Final MAE | Relative MAE reduction | Paired change, 95% interval |
|---|---:|---:|---:|---|
| Canonical | 0.0828 | 0.0655 | 20.9% | −0.0173 [−0.0253, −0.0095] |
| Pixel scale 0.04 | 0.1696 | 0.1158 | 31.7% | −0.0537 [−0.0677, −0.0403] |
| Pixel scale 0.06 | 0.1278 | 0.0862 | 32.6% | −0.0417 [−0.0507, −0.0326] |
| Half exposure | 0.1150 | 0.0936 | 18.6% | −0.0215 [−0.0318, −0.0108] |
| Double read noise | 0.1155 | 0.0879 | 23.9% | −0.0276 [−0.0406, −0.0154] |

All absolute errors and changes are in arcseconds. Relative reductions are calculated from the saved metrics.

Across the four stress conditions, equal-weight MAE decreased from approximately **0.1320 to 0.0959 arcsec**. The primary paired effect is:

**−0.0361 arcsec, with a 95% parent-bootstrap interval of [−0.0427, −0.0294].**

All five condition-specific intervals are below zero. This supports lower average error for the randomized ensemble under these tested conditions.

The bootstrap uses **10,000 resamples of independent parents**, preserving the relationship between observations of the same system. The holdout nevertheless uses the same simulator and parent prior as development, which limits the generalization claim. See the [frozen holdout report](../results/reports/reliability_v1/blind_holdout_week11/report.md).

## Error tails and uncertainty

Remaining errors are substantial and uneven. Improved averages do not mean that every prediction improved or that large errors disappeared.

| Condition | Final errors >0.10 arcsec | Baseline errors >0.10 arcsec |
|---|---:|---:|
| Canonical | 26/150 | 49/150 |
| Pixel scale 0.04 | 67/150 | 96/150 |
| Pixel scale 0.06 | 48/150 | 78/150 |
| Half exposure | 46/150 | 64/150 |
| Double read noise | 45/150 | 60/150 |

Under canonical conditions, the final median absolute error is approximately **0.0472 arcsec**, while the 95th percentile is **0.1981 arcsec**. At pixel scale 0.04, the 95th percentile reaches **0.3287 arcsec**.

The largest listed final-model error is approximately **0.7656 arcsec** under doubled read noise. That case is worse than the comparator, illustrating why aggregate improvement should not be interpreted as a per-case guarantee.

The package includes a [750-observation failure catalog](../results/reports/final_v1/tables/failure_case_catalog.csv). Three-seed prediction spread is retained as a diagnostic, but it has not been calibrated as a predictive interval.

## Classical modeling checks

The classical checks provide useful engineering evidence. The earlier recovery grid completed **13 of 13 optimization cases** across six scenarios. The noiseless control recovered Einstein radius within approximately **0.033%**; noisier conditions showed larger errors.

These results verify selected rendering and fitting behavior. Their scope is limited by small fixed scenarios, analytic source assumptions, conditioned point-image positions in some fits, and local rather than posterior identifiability checks. See the [recovery-grid report](../results/reports/fit-sie-grid-v1/recovery_grid.md).

An exploratory real-lens astrometric comparison also reproduced approximate angular scales for B0218, HE0435, and Q2237, with reported differences of **6.3%, 0.8%, and 3.4%**. These were preselected systems using a limited geometric estimator; one reference was a screening proxy. They do not validate CNN performance on real images.

## CASTLES domain comparison

The CASTLES analysis identifies material simulation-to-real gaps. It compares available metadata and image-derived proxies for 137 actual observations against the randomized synthetic training distribution.

Its main findings are:

| Coverage or metadata issue | Reported result |
|---|---:|
| Real pixel scales within synthetic p01–p99 range | 52.6% |
| Real processed angular fields within synthetic range | 6.6% |
| Missing filter metadata | 54% |
| Missing detector metadata | 46% |
| Missing calibration-stage metadata | 54% |
| Missing specified correction metadata | 73% |

Actual instruments span NICMOS, ACS, and WFPC2, while the simulator does not implement instrument-specific rendering. Empirical PSFs, calibrated noise, mask behavior, reviewed multiplicity, and realistic source/lens diversity remain unresolved.

This analysis did not score real-image model predictions or use real Einstein-radius/redshift labels. Its contribution is a coverage audit and a specification for future benchmarking. See the [domain-gap record](reliability_study/week10_domain_gap_status.md).

## Reproducibility safeguards

The final evaluation has strong procedural safeguards. The recorded workflow freezes code, configuration, checkpoint, and normalization identities before evaluation. It saves target-free predictions and hashes them before joining labels for scoring.

The documentation records:

- **117 frozen input hashes** before holdout generation.
- **4,500 per-checkpoint prediction rows** across six checkpoints.
- **1,500 ensemble prediction rows** across two model groups.
- A single scored evaluation pass.
- No historical-test or real-label access during that evaluation.
- No post-holdout tuning.

The final package builder consumes saved evidence to recreate tables, figures, and prose without rerunning inference. The recorded environment was Python **3.13.6**, PyTorch **2.8.0+cpu**, CPU execution, and six Torch threads. See the [reproducibility statement](reliability_study/final/reproducibility_statement.md).

## Current verification

Fresh verification found a sound artifact package alongside two current software failures. All 21 outputs listed in the package manifest were checked:

- **10 outputs matched their recorded SHA-256 hashes exactly.**
- **11 CSV tables matched after CRLF-to-LF normalization.**

The CSV differences are fully explained by line endings. Their local byte hashes differ, so strict byte-level verification should account for checkout newline handling.

The automated test suite was run with:

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-report-review
```

The result was **104 passed, 2 failed in 94.92 seconds**. Both failures occur in `tests/test_ml_baseline.py`, covering the multi-seed suite and smoke-run workflow. Both raise **`ValueError: run_controls changed`** when verifying a newly trained run.

This means the current experiment runner has an unresolved consistency problem. The failure does not by itself invalidate the archived holdout findings, but the current codebase cannot be described as passing all automated checks. Source code was not changed and the frozen holdout was not rerun for this review.

## Final deliverables

The final deliverables are complete for the stated synthetic study. They include a manuscript draft, model card, reproducibility statement, claim-evidence map, 11 supporting CSV tables, three figures, machine-readable summaries, and a hashed package manifest.

The most useful entry points are:

- [Final manuscript](reliability_study/final/manuscript_draft.md)
- [Research/model card](reliability_study/final/model_research_card.md)
- [Claim-evidence map](reliability_study/final/claim_evidence_map.md)
- [Package manifest](../results/reports/final_v1/package_manifest.json)
- [Final-package completion record](reliability_study/week12_final_package_status.md)

## Assessment and next work

The project's strongest contribution is the connection between measured failure modes, a targeted intervention, and an independently evaluated improvement. It also provides a reusable data and provenance framework and preserves negative findings, error tails, and domain limitations.

Further development should first resolve the two experiment-runner test failures and clarify checksum handling across operating systems. The major scientific extension is a separately governed real-observation benchmark with calibrated products, reviewed labels, and predefined scoring. Broader renderer/source diversity and uncertainty-coverage evaluation would then address the most important remaining limits.
