# Codebase guide

## What the project does

The project turns raw strong-lens images and physics simulations into a shared manifest contract. Its first supervised quantity is the angular Einstein-radius parameter `theta_E_arcsec` in arcseconds. It deliberately keeps exact simulator truth separate from observational estimates such as `einstein_radius_estimate_arcsec`, and it does not derive physical mass without the redshifts, cosmology, and aperture needed to define that mass.

The end-to-end flow is:

```text
CASTLES FITS -> metadata/catalog -> standardized products -> real release/QC
                                                         -> real astrometric check

versioned priors -> lens/source rendering -> PSF/noise -> simulated products
                                                    -> rerender validation
                                                    -> ancestry-safe splits
                                                    -> classical recovery
                                                    -> ML-safe loader/handoff
```

`src/pipeline.py` is the public command-line coordinator. `src/main.py` is a compatibility entry point that forwards to it.

## Data contract and loader

- `src/data/schema.py` is the machine-readable schema-v2 field registry and scalar validator. It defines identities, units, path rules, status values, target availability, simulation provenance, and ancestry.
- `src/data/dataset.py` is the NumPy-compatible loader. It validates array shapes/dtypes, masks, variance, geometry, and PSFs; selects a release split/domain/eligibility role; returns products and targets separately; and uses a positive allowlist for model-visible metadata.
- `docs/dataset_schema.md` is the human-readable contract.

`LensDataset.__getitem__` provides the legacy `(image, row)` view. `get_sample` is the safe structured interface:

```text
inputs.products  image, science, masks, variance, expected variance, PSF, geometry
inputs.metadata  allowlisted observational quantities only
targets          explicitly selected target values plus availability flags
metadata         full provenance, including privileged truth; never pass this to a model
```

## Real-observation processing

- `build_metadata.py` inventories FITS science HDUs and extracts instrument, unit, exposure, WCS, pixel scale, calibration, and provenance fields.
- `build_catalog.py` validates and materializes the curated lens-level source table.
- `enrich_metadata.py` joins observation metadata to the catalog without guessing missing scientific values.
- `geometry.py` implements one exact center crop/pad transform shared by images and masks.
- `background.py` estimates a robust scalar background with sigma clipping and an explicit unconstrained state.
- `normalization.py` implements `none`, percentile, and affine transformations and serializes their parameters.
- `noise.py` converts supplied errors to variance or creates the explicitly labeled toy fallback.
- `psf.py` defines PSF kernels, validation, serialization, and convolution; current real files have no empirical PSF.
- `preprocessing_config.py` loads and fingerprints the versioned preprocessing YAML.
- `preprocess.py` orchestrates these pure components and writes product arrays, masks, sidecars, a manifest, and a run record.
- `preview_dataset.py` creates per-observation previews and contact sheets for human QC. New previews are written under `results/previews/actual/<release>` or `results/previews/simulated/<release>`, outside dataset releases.
- `validate_metadata.py` separates fatal structural errors from review-level scientific/calibration issues.
- `create_splits.py` creates lens-level real splits and transitive ancestry-component simulation splits.

The rebuilt `real-castles-v2` release is structurally valid. Review states are intentional: unknown units, missing telescope labels, ambiguous redshifts, model-unsuitable systems, absent empirical PSFs, and toy variances are not silently promoted to calibrated data.

## Simulation system

- `simulation/models.py` is the adapter boundary to lenstronomy. It owns SIS/SIE and ellipticity conventions and avoids spreading backend-specific objects through the codebase.
- `simulation/priors.py` makes deterministic parameter draws from versioned YAML ranges and validates them.
- `simulation/render.py` contains the analytic SIS control and extended SIE/Sersic rendering functions.
- `simulation/observation.py` adds lens light, point images, Gaussian PSF convolution, Poisson counts, read noise, and saved expected variance.
- `simulation/generate.py` creates the P10/P30/P60/P100 stages and engineering observations. Each accepted row receives independent parameter/noise seeds, ancestry IDs, truth/diagnostic sidecars, and all array products. Exceptions go to the rejection log and never become manifest rows.
- `simulation/assemble.py` combines stage manifests without losing their roots or provenance.
- `simulation/smoke_test.py` checks the backend boundary and conventions in a minimal render.
- `testing/validate_simulations.py` rerenders saved samples and checks exact float32 reproducibility, schema, products, flux, and image multiplicity.
- `testing/report_simulation_pilot.py` summarizes staged-release metrics and limitations.

The release ladder has distinct purposes: P10 is an analytic SIS control, P30 introduces SIE plus an extended source, P60 adds lens light/point images/PSF/noise, and P100 broadens conditions for stress testing. `sim-eng1000-v1` uses the validated P60-like proposal; P100 remains separate so stress examples do not silently alter the training distribution.

## Classical fitting and real comparison

- `modeling/fit_lens.py` first checks exact SIS recovery, then fits selected SIE/SIE+shear pixel scenes with bounded least squares. It saves residual maps, objective/convergence/runtime information, truth errors, local Jacobian rank/condition, boundary flags, and failure classes. `fit_sie_recovery_grid` runs fixed noiseless, low/high-SNR, stress, PSF-mismatch, and source-mismatch conditions.
- `modeling/real_comparison.py` detects the brightest preselected lensed-image peaks, measures half-separation for a double or a best-fit monopole circle for a quad, and compares that angular scale with a compatible CASTLES reference. It is intentionally not a full lens fit.

## Machine-learning baseline

- `ml/data.py` wraps the frozen `LensDataset` splits as PyTorch tensors, computes training-only normalization, and returns no provenance or simulator nuisance parameters to the model.
- `ml/model.py` defines the 248,409-parameter four-block convolutional regressor.
- `ml/engine.py` contains deterministic seeding, device selection, data loading, training, checkpointing support, and prediction collection.
- `ml/metrics.py` computes angular, relative, bias, R², and train-mean baseline metrics without an additional metrics dependency.
- `ml/plots.py` creates noninteractive learning-curve, residual, and multi-seed diagnostics.
- `ml/train.py` trains on 700 parents, selects a checkpoint using only 150 validation parents, and records configuration, data, source, environment, and artifact hashes.
- `ml/evaluate.py` verifies hashes before evaluating a frozen split and writes overall, subgroup, and worst-case evidence.
- `ml/experiments.py` runs registered validation-only seed suites, verifies resumable artifacts, aggregates per-seed metrics, and produces the comparison report.
- `ml/provenance.py` records package, device, Git, and source-file identity for every training run.

`configs/ml_baseline_v1.yaml` is the first fixed experiment. `configs/ml/registry_v1.yaml` registers the three-seed baseline benchmark. The run procedure and interpretation guide are in `docs/ml_quickstart.md`.

## Reliability-study workflows

The reliability study adds phase-specific analyses around the baseline while
keeping model inputs, parent identity, and holdout use explicit:

- `ml/pixel_scale_audit.py`, `ml/observation_systematics_audit.py`, and
  `ml/source_physics_audit.py` estimate matched validation-parent effects for
  their named interventions.
- `ml/reliability_week6.py` compares simple baselines and audits shortcuts and
  training-set size; `ml/reliability_week7.py` evaluates the independently
  implemented SIE equation path.
- `ml/reliability_week8.py` builds and evaluates the targeted-randomization
  intervention. `modeling/hybrid_initialization.py` and
  `modeling/domain_gap_week10.py` implement the Week 9 fit-initialization and
  Week 10 domain-coverage analyses.
- `ml/reliability_week11.py` implements the new-parent blind holdout. Its
  source files and inputs are listed in
  `archive/docs/reliability_study/week11_freeze_manifest.json`; the evaluation is
  exactly-once and is not a routine rerun command.
- `archive/scripts/build_reliability_final_package.py` assembles tables and prose from
  saved evidence. It does not run model inference.

The phase protocols, status records, and final research card are indexed in
[`docs/reliability_study/README.md`](reliability_study/README.md). The temporal
script names are retained to match the frozen experiment records. New study
code should use topic-based names and a new versioned protocol/configuration.

## Configuration, artifacts, and tests

- `configs/preprocessing_v2.yaml` freezes real preprocessing.
- `configs/simulation_pilot_v1.yaml` freezes the staged controls.
- `configs/simulation_engineering_v1.yaml` freezes the 1,000-parent release and split policy.
- `configs/real_comparison_v1.yaml` freezes the real candidates before measurement.
- `data/labels/source` holds curated source tables; `data/labels/derived` holds rebuildable metadata and catalog outputs.
- `data/releases` holds actual-observation releases. Large products are ignored, while manifests and reports are tracked.
- `data/simulations` holds full generated simulations and is ignored because of size.
- `data/manifests/sim-eng1000-v1` is the compact tracked audit mirror for the engineering release.
- `results/previews/actual` and `results/previews/simulated` separate visual QC artifacts by domain and release.
- `results/reports` holds classical-fit, real-comparison, and readiness evidence.
- `tests` mirrors the modules with unit and integration checks. Tests use temporary paths and should not mutate production artifacts.

## Current scientific scope

The project has a completed reliability study through its Week 12 analysis
package. Its final evidence supports conclusions about the tested synthetic
prior, renderer, and nuisance conditions. The CASTLES comparison is descriptive
and does not establish real-image parameter accuracy. Empirical HST PSFs,
detector/error products, realistic source populations, calibrated priors,
uncertainty coverage, and a blinded real benchmark remain open research work.
See the [research card](reliability_study/final/model_research_card.md) and
[reproducibility statement](reliability_study/final/reproducibility_statement.md)
for claims and limitations.
