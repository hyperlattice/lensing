# C/G/R synthetic-to-HST implementation

The synthetic comparison remains `python -m scripts.study`. The new
`python -m scripts.hst_study` command implements observational preparation,
source-development preparation checks, and the paired HST comparison specified
in [the experiment plan](hst_experiment_plan.md). No HST training, fine-tuning,
new architecture, or revised R policy is introduced.

## Ownership and assets

| Component | Responsibility |
|---|---|
| `scripts/study.py` | Existing synthetic stages; separate corrected-analysis destination |
| `scripts/hst_pilot.py` | Pinned five-system development intake/audit |
| `scripts/hst_archive.py` | Cached native archive cutouts; explicit product-ledger selection |
| `scripts/hst_study.py` | Explicit observational stage dispatch |
| `src/data/hst.py` | Reviewed preparation, alias/cohort boundaries, reference metadata, confirmation freeze |
| `src/simulation/hst_bridge.py` | Matched synthetic development checks; numerical gates and descriptive prediction changes |
| `src/ml/artifacts.py` | Checkpoint identities, configuration, source normalization and completeness |
| `src/ml/predict.py` | Shared native-image inference and predictions saved before reference joining |
| `src/ml/analysis.py`, `src/ml/plots.py` | Saved-prediction analysis and figures |

The new configuration is [configs/hst_study.yaml](../configs/hst_study.yaml).
All commands run from the repository root. Observational input paths are
cwd-relative or absolute; prepared product paths are release-relative. The
synthetic bank path contract is unchanged.

## Synthetic experiment and corrected bootstrap

The checked-out bootstrap previously retained the condition axis when taking
quantiles, mixing per-condition differences instead of bootstrapping the
four-stress composite. It now averages fixed seeds and conditions within each
parent, then resamples parents. A direct-reference test covers unequal condition
effects, including an exactly zero composite with nonzero condition effects.

Historical manuscript intervals require reanalysis from the original saved
predictions; this correction does not establish which historical runner produced
them. No original checkpoint, prediction, report, or manuscript is rewritten here.
Once original artifacts and their original resolved configuration are recovered:

```powershell
python -m scripts.study analyze --report-output results/ml/equal-budget-bootstrap-v2
python -m scripts.study plot --report-output results/ml/equal-budget-bootstrap-v2
```

These commands read the configured original predictions and truth and write a
separate report/figures. They do not train or predict. Historical runs lacking a
resolved configuration still require their original runner; do not fabricate
retroactive provenance. The ordinary `train`, `predict`, `analyze`, `plot` actions
remain available for explicitly identified reproductions with all required inputs.

## Development preparation

```powershell
python -m scripts.hst_study assets
python -m scripts.hst_study prepare --initialize
python -m scripts.hst_study prepare --cohort development
```

`assets` records missing files and exits 2 until the full checkpoint matrix and
training manifest are available. Existing checkpoints must match the study's
arm, seed, architecture, optimizer updates and common source normalization.
Training inputs can be absent for inference, but they are required for reproduction
and the development bank is required for preparation validation.

Initialization creates five **pending** development observations from the pilot
audit, plus a separate reference table. It does not promote arrays to qualified
inputs and refuses to replace either ledger. Keep every development lens reserved
across aliases, bands and releases. Add confirmation candidates to the same
observation ledger with canonical physical-lens IDs and reviewed aliases.

Each observation specifies:

- `system_id`, `aliases`, `observation_id`, `cohort`, `survey`, `band`.
- `eligibility`: `pending`, `include`, or `exclude`; exclusions need a reason.
- `science`, `psf`, and optional `valid_mask` descriptors: `path`, `sha256`, and
  an NPZ `key` or FITS `hdu`. NPY, NPZ and FITS arrays are supported.
- `pixel_transform_arcsec` and `psf_pixel_transform_arcsec`: native 2x2 matrices
  mapping pixel x/y increments to local angular x/y increments. FITS declarations
  are checked against their celestial WCS.
- `input_unit`, positive `unit_multiplier`, and `background_raw`. The exact
  operation is `(science - background_raw) * unit_multiplier`, producing
  electrons/s **per native pixel**, without HST-fitted normalization. A zero
  background operation must still be explicitly reviewed. Electron-rate conversion
  alone does not establish photometric compatibility with training.
- `center`: either `method: gaussian_profile` with `[y0,y1,x0,x1]` `roi_yx`, or
  `method: reviewed_external` with native-pixel `xy`. Both require `evidence`.
  The local Gaussian-plus-constant fit is a foreground-light estimator, not a
  mass model; choose/review the ROI to avoid arcs and contaminants.
- Optional `crop_shape_yx`, explicit `center_tolerance_arcsec`, and
  `psf_center_tolerance_pixels`. Native integer crops retain subpixel offsets;
  they cannot guarantee an exactly centered arbitrary foreground position.
- Nonempty evidence strings under `review` for `centering`, `photometry`, `psf`,
  `coverage_masks`, `reference_convention`, and `morphology`.

The supplied mask must be boolean with **true = valid**. Selected fields with
invalid pixels are rejected; no implicit filling is applied to the image-only CNN.
PSFs must already match the science grid. Their positive flux is normalized and
their centroid is checked. No PSF resampling, science interpolation, padding,
automatic photometric matching, or brightest-pixel centering is performed.

Preparation saves physical float32 NPY images, normalized PSFs, native cropped
FITS/WCS when present, an inclusion/blocker ledger and preparation metadata under
`data/releases/hst-study-v1/<cohort>/`. Every included primary observation must
qualify before prediction; unqualified included rows cannot be silently dropped.
The preparation command also writes `preview.png` with the physical image,
adopted center and normalized PSF. Use it alongside the existing pilot/archive
previews and prepared FITS to review the crop. One primary observation per physical
lens is enforced.

## Synthetic preparation checks and pilot inference

Declare numerical tolerances in `bridge.tolerances` from numerical requirements
and development evidence, before reviewing CNN rankings. Null tolerances cannot
pass confirmation. The offset controls are explicitly configured.

```powershell
python -m scripts.hst_study validate-preparation --smoke
python -m scripts.hst_study validate-preparation
python -m scripts.hst_study predict --cohort development
python -m scripts.hst_study analyze --cohort development
python -m scripts.hst_study plot --cohort development
```

The bridge uses the qualified source-development bank, verifies original physical
parameter bytes and true radii, and retains matched parents across canonical,
sampled-grid, field-cropped and prepared conditions. It rerenders with the existing
adaptive quadrature. Rotated/reflected square pixels are supported explicitly;
sheared or anisotropic grids are rejected rather than approximated by their
area-equivalent scale. Sampling grids cover at least the canonical field and the
native field; field comparisons crop the same sampled observation. Canonical
versus sampled grids use seed-matched, not identical physical detector draws.

The numerical controls check injected local Gaussian-center recovery, exact
declared conversion/cropping, coordinate bookkeeping, and PSF normalization.
Actual parent-image conversion round trips are checked too. Prediction/error
changes versus canonical and field controls are saved separately. CNN sensitivity
is descriptive and never an acceptance threshold. Analytic source PSFs/noise are
retained; these checks do not validate an HST likelihood or establish unbiased
centers for complex observed galaxies. The review evidence remains necessary.

Scene cropping uses the simulator's physical foreground center, including its
nonzero offsets from the array midpoint. If the reviewed method is the local
Gaussian fit, the bridge applies that same fit with a translated ROI and tests
its recovered centers against simulator truth. External/manual methods use the
known synthetic center for cropping; their actual estimator requires separate
recovery evidence and is not automatically validated by these controls.

The smoke action uses a bounded parent subset and **cannot** satisfy the
confirmation gate. The full action uses every available development parent.
Generated views and per-parent/arm/seed results remain under the chosen output.
Neither action trains or accesses synthetic final-holdout data. The default bank
and checkpoints are currently external/missing, so no scientific bridge run is
implied by a fixture smoke test.

## Freeze and confirmation

After development, resolve every confirmation candidate to include/exclude, review
all reserved development systems, choose a primary band before examining outcomes,
and fill `eligibility_rules`, `primary_band_rule`, and `reference_strategy` in the
configuration. Reference rows need `system_id`, `median_arcsec`, positive
`error_lower`/`error_upper`, `definition`, `mass_model`, `interval_semantics`, and
`source`. Optional `alternatives` provide separately labelled model-family
sensitivity references with positive medians, definitions, mass models and sources.

```powershell
python -m scripts.hst_study prepare --cohort confirmation
python -m scripts.hst_study freeze
python -m scripts.hst_study predict --cohort confirmation
python -m scripts.hst_study analyze --cohort confirmation
python -m scripts.hst_study plot --cohort confirmation
```

Freeze binds the cohort/exclusions, preparation, references, checkpoints, settings,
full development-validation evidence, and relevant implementation files. It is
exclusive and precedes confirmation predictions. Changes require a new study
identity. Predictions are also exclusive, with finite but implausible outputs
retained and flagged. Analysis can be repeated without image/checkpoint arrays.

Primary G-minus-C and secondary descriptive R-minus-G contrasts average absolute
reference disagreement over seeds within each lens, then equally across lenses.
Confirmation uses 10,000 paired lens resamples by default, conditional on the
evaluated seed set. Seed-specific metrics, signed bias, median lens loss, survey
counts, per-lens changes and alternative-reference sensitivity are saved separately.
Development receives descriptive comparisons without confidence intervals.
Published reference intervals are shown separately in the figure, not treated as
complete reference-systematic uncertainty. The estimand is not ensemble loss.

Results live under `results/ml/hst-transfer-v1/`: model assets and validation records
at the top level; predictions, reference disagreements, per-lens losses/changes,
survey tables, sensitivity tables, analysis JSON, figures and a short overview
under `development/` or `confirmation/`. These are agreement results on the
eligible cohort, not exact physical truth or survey-wide accuracy.

## Expanding archive intake

The original `python -m scripts.hst_archive --offline` probe is unchanged. For
additional products, pass `--manifest <archive_products.json> --product-id <id>`.
The manifest schema is `hst_archive_products_v1` with a `products` list. Each row
has the probe fields (`system_id`, `observation_id`,
`product_uri`, `product_bytes`, `ra_deg`, `dec_deg`, `minimum_field_arcsec`,
`coordinate_source`, `role`). Here `observation_id` is a unique filename-safe
string; `role` is `compatibility_development` or `confirmation_candidate`.
The optional numeric MAST observation identifier can be recorded as
`archive_observation_id` for provenance.
Products get separate output subdirectories. Acquisition preserves SCI/WHT/WCS
and records source ranges, but cannot qualify units, coverage, centers or PSFs.

## Validation

```powershell
python -m ruff check .
python -m ruff format --check .
python -B -m pytest -q -p no:cacheprovider
python -m scripts.check_repository
python -m scripts.submission check
```

Tests use temporary checkpoints and images. They exercise both cohorts, prediction
sealing, arithmetic/geometry, alias exclusion, complete paired matrices, frozen
input changes, direct bootstrap comparisons and analysis without image loading.
They establish implementation behavior, not real-image performance.
