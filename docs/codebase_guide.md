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
- `preview_dataset.py` creates per-observation previews and contact sheets for human QC.
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

## Configuration, artifacts, and tests

- `configs/preprocessing_v2.yaml` freezes real preprocessing.
- `configs/simulation_pilot_v1.yaml` freezes the staged controls.
- `configs/simulation_engineering_v1.yaml` freezes the 1,000-parent release and split policy.
- `configs/real_comparison_v1.yaml` freezes the real candidates before measurement.
- `data/labels/source` holds curated source tables; derived label tables remain in `data/labels`.
- `data/releases` holds real releases. Large product/previews are ignored, while their manifests and reports are tracked.
- `data/simulations` holds full generated simulations and is ignored because of size.
- `data/manifests/sim-eng1000-v1` is the compact tracked audit mirror for the engineering release.
- `results/reports` holds classical-fit, real-comparison, and readiness evidence.
- `tests` mirrors the modules with unit and integration checks. Tests use temporary paths and should not mutate production artifacts.

## Where the project is now

All work through the pre-ML handoff is implemented and evidenced. The next legitimate phase is a separate ML plan using the frozen splits and target contract. Important open research work remains—empirical HST PSFs, detector/error products, realistic source morphologies, calibrated priors/selection functions, uncertainty coverage, and a blind real benchmark—but those are domain-fidelity improvements, not unfinished prerequisites hidden by the current engineering release.
