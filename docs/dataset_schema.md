# Dataset schema v2: strong gravitational lenses

This document is the contract shared by real observations, simulated
observations, validation, preprocessing, classical fitting, and future
machine-learning loaders. The machine-readable field registry is in
`src/data/schema.py`.

## 1. Contract and identity

Every manifest is a UTF-8 CSV with one row per observation. Paths use `/`
separators and are relative to the repository root for top-level manifests or
to the release root for a self-contained release manifest. A row is identified by
`sample_id`; `lens_id` identifies the physical lens system and is shared by
all of its observations. `domain` is exactly `real` or `simulated`.

The required fields for every row are:

| Field | Type | Unit | Null behavior | Validation |
| --- | --- | --- | --- | --- |
| `sample_id` | string | - | forbidden | nonempty and unique in the manifest |
| `lens_id` | string | - | forbidden | nonempty; all observations retain their system ID |
| `domain` | enum | - | forbidden | `real` or `simulated` |
| `schema_version` | string | - | forbidden | exactly `2.0` |
| `dataset_version` | string | - | forbidden | nonempty release/config identifier |
| `raw_path` | path | relative POSIX | forbidden | existing FITS artifact |
| `science_path` | path | relative POSIX | forbidden | selected science product |
| `processed_path` | path | relative POSIX | forbidden | existing standardized array |
| `image_unit` | string | declared unit | use `unknown` if unverified | never silently infer a calibrated unit |
| `calibration_status` | enum | - | use `unknown` if unverified | `pass`, `review`, `fail`, or `unknown` |
| `original_shape` | `HEIGHTxWIDTH` string | px | forbidden | agrees with the selected science array |
| `processed_shape` | `HEIGHTxWIDTH` string | px | forbidden | equals `256x256` for the current loader |
| `pixel_scale` | float | arcsec/px | forbidden | finite and greater than zero |
| `processing_config_id` | string | - | forbidden | identifies the preprocessing configuration |
| `processing_version` | string | - | forbidden | identifies the implementation/contract |
| `background_status` | enum | - | use `unknown` if unreviewed | status enum |
| `source_type` | string | - | use `unknown` if unreviewed | controlled description such as `point`, `sersic`, or `host_quasar` |
| `mass_target_available` | boolean | - | `false` when unavailable | missing targets are not replaced with zero |
| `theta_E_target_available` | boolean | - | `false` when unavailable | reflects a scientifically usable target |
| `structural_status` | enum | - | use `unknown` if unreviewed | status enum |
| `observation_status` | enum | - | use `unknown` if unreviewed | status enum |
| `science_status` | enum | - | use `unknown` if unreviewed | status enum |
| `eligible_for` | semicolon-separated enum | - | empty only when excluded | values are `preprocessing`, `simulation`, `classical_fit`, `ml`, or `real_comparison` |
| `exclusion_reason` | string | - | null for eligible rows | required explanation when a use is excluded |

Simulated rows additionally require `simulation_id`, `parent_system_id`,
`split_group_id`, `mass_model`, `theta_E_arcsec`, `simulator_name`,
`simulator_version`, `parameter_seed`, `noise_seed`, and
`parameters_path`. These fields are null for real rows unless a real
observation is explicitly being used as a simulation-derived reference.

## 2. Paths, observation, and processing products

Optional product paths are `mask_path`, `variance_path`, `psf_path`, and
`geometry_transform_path`. A missing mask means that no mask product is
available; it does not mean that every pixel is valid. A missing variance or
PSF means that the corresponding uncertainty or convolution product was not
provided. Generated observations must write their own FITS file to
`raw_path`; they must never point to a CASTLES file as a fake parent artifact.

Boolean masks use `True` to mean excluded/invalid. The preprocessing products
distinguish the invalid-pixel mask, the no-coverage/padding mask, and the
background-exclusion mask. Their crop/pad operation is serialized in the
geometry transform sidecar and must be identical across science, mask, and
variance arrays.

Observation fields include `instrument`, `detector`, `filter_name`,
`exposure_s`, `calibration_source`, and the declared `image_unit`.
Preserve unknown values explicitly. Directory names may be hints, but they are
not provenance for an instrument, filter, or calibration state.

Processing provenance includes `processing_config_id`,
`processing_version`, `normalization_path`, and `masked_fraction`.
Background subtraction, normalization, crop/pad, mask, variance, and PSF
operations must be applied consistently to all corresponding products. The
normalization transform is a saved artifact, not an undocumented side effect.

Variance products are in the square of the science/image unit. A supplied
error image is squared only where finite and positive. The current development
fallback is a positive scalar background-scale model labeled
`variance_status=toy_unprofiled`; its pixel-correlation assumption is recorded
in `noise_correlation_assumption`. This fallback is suitable for an angular
engineering pilot, not for a calibrated physical likelihood.

PSF products are shared registry kernels referenced by `psf_id` and
`psf_path`. A kernel records odd dimensions, center in pixel coordinates,
arcsec/px sampling, source, unit, and unit sum. The Gaussian control PSF is an
analytic test instrument profile; it is not a measurement of any CASTLES
observation.

## 3. Geometry and coordinate conventions

Arrays use NumPy indexing `image[y, x]`, with zero-based pixel-center
coordinates. A `256x256` image has geometric center `(127.5, 127.5)`, and
`center_x_px`/`center_y_px` may be fractional. These pixel coordinates are
distinct from `center_ra_deg`/`center_dec_deg` and from model-plane angular
coordinates.

`pixel_scale` is in arcsec/px. When the WCS is anisotropic, preserve
`pixel_scale_x`, `pixel_scale_y`, and a sidecar at
`geometry_transform_path` containing the full Jacobian/WCS and crop/pad
offsets. The transform must state the angular origin and the sign of both
axes. Display flips do not change the physical transform and must not be used
to silently redefine the model coordinates.

Right ascension and declination are in degrees. `phi_rad` is a model
orientation in radians, measured counter-clockwise in the declared model
coordinate system. Source/lens offsets are in arcseconds. Pixel bounds are
`0 <= center_x_px < width` and `0 <= center_y_px < height` when a center is
available.

## 4. Scientific quantities and provenance

Redshifts are dimensionless. When both are present they must satisfy
`0 < lens_redshift < source_redshift`; ambiguous alternatives remain null in
the primary label and are preserved in the provenance/source table. Every
redshift needs `redshift_provenance` when it is used scientifically.

The schema deliberately separates these quantities:

| Field | Meaning | Unit and null rule |
| --- | --- | --- |
| `theta_E_arcsec` | Exact simulator parameter under `mass_model` and its named convention | arcsec; required for simulated rows |
| `einstein_radius_estimate_arcsec` | Observed or literature estimate | arcsec; nullable and always paired with `einstein_radius_method` |
| `mass_enclosed_msun` | Derived projected mass inside an explicit aperture | solar masses; nullable |
| `mass_aperture_arcsec` | Aperture for the enclosed mass | arcsec; required when mass is present |

`einstein_radius_estimate_arcsec` is not simulator truth merely because it is
numeric. `einstein_radius_method` distinguishes a literature model,
astrometric estimate, CASTLES `Size/2` proxy, separation-based estimate, or
another derivation. CASTLES describes `Size/2` as an approximate monopole
Einstein-radius estimate, but its entries use mixed methods; retain the
original Size and source record in the curated catalog.

For an SIE, record `q`, `phi_rad`, and either the adapter components `e1`
and `e2` or the exact convention used to derive them. The initial lenstronomy
adapter uses the documented conversion from axis ratio and orientation; do not
copy ellipticity numbers from a paper without checking that paper's
definition. External shear is `gamma1` and `gamma2`.

The physical mass label requires `lens_redshift`, `source_redshift`, a
named `cosmology_id`, and `mass_aperture_arcsec`. Angular-only simulations
may omit redshifts and mass. Scientific target availability is represented by
boolean flags; unavailable values remain null.

The curated source table also records `source_type`, `lens_environment`,
`num_images`, `lens_confidence`, `model_suitability`,
`ring_extended_flag`, `cropped_truncated_flag`, and
`classification_evidence`. `unknown` plus a reason is a valid classification;
these fields must not be silently filled from a directory name or a visual
guess. The initial review list is in
`data/labels/source/pilot_comparison_candidates.csv`. Candidate status is
`review` until instrument products, morphology, environment, and the proposed
mass model have been checked.

## 5. Ancestry and reproducibility

`parent_system_id` is the independent synthetic parent group. All rotations,
noise realizations, crops, PSF variants, and other descendants of one parent
must share its split group. `parent_real_lens_id` is recorded when a
simulation was derived from a real lens. Source and background reuse is tracked
with `source_template_id` and `background_template_id` so asset families
can also be split without leakage.

The exact parameter dictionary is stored at `parameters_path`. It must record
the model, all generating parameters, configuration settings, and random streams.
`parameter_seed` controls physical parameter draws and `noise_seed` controls
observation/noise draws. `simulator_name` and `simulator_version` are
required for generated rows. A run-level record should also retain code
version, environment, input paths, attempted/accepted counts, and rejected
configurations.

## 6. Quality and eligibility

`structural_status` covers file existence, array shape, dtype, finiteness, and
manifest uniqueness. `observation_status` covers selected HDU, instrument,
calibration, WCS, PSF, and noise metadata. `science_status` covers the
scientific interpretation and whether the declared targets are suitable. Each
status is `pass`, `review`, `fail`, or `unknown`.

Eligibility is purpose-specific. A row may be eligible for preprocessing but
not for physical-mass fitting, or eligible for simulation development but not
for a blind real-data comparison. Validation reports must count each use and
record `exclusion_reason`; a failed mass label must not be turned into a zero
target.

## 7. Legacy migration

Older files may contain `einstein_radius` or
`einstein_radius_arcsec`. During migration:

1. Preserve the old column and its source/value in the migration report.
2. Rename a clearly angular observational value to
   `einstein_radius_estimate_arcsec` only when its method is known.
3. Store the method in `einstein_radius_method`; use `unknown` rather than
   guessing.
4. Use `theta_E_arcsec` only for an explicitly generated/model-convention
   parameter, or for a real fitted model whose model and provenance are
   recorded. It is never an automatic alias for the observational estimate.
5. Compute `mass_enclosed_msun` only with redshifts, cosmology, and aperture;
   otherwise leave it null and set `mass_target_available=false`.

## 8. Minimal valid rows

The following examples show the required interface. Empty optional values are
represented as `null` in a JSON-like row; CSV writers may use an empty cell.

### Minimal real row

```yaml
sample_id: Q2237_FIT_NIC2H_q256
lens_id: Q2237
domain: real
schema_version: '2.0'
dataset_version: real-dev-v2
raw_path: data/raw/castles/Q2237/FIT/NIC2H/q256.fits
science_path: data/raw/castles/Q2237/FIT/NIC2H/q256.fits
processed_path: data/releases/real-castles-v2/products/normalized/Q2237_FIT_NIC2H_q256.npy
image_unit: unknown
calibration_status: unknown
original_shape: 256x256
processed_shape: 256x256
pixel_scale: 0.0378
processing_config_id: baseline-v2
processing_version: '2.0'
background_status: unknown
source_type: unknown
mass_target_available: false
theta_E_target_available: false
structural_status: pass
observation_status: review
science_status: review
eligible_for: preprocessing
exclusion_reason: null
```

### Minimal simulated row

```yaml
sample_id: sim-p10-0001
lens_id: sim-p10-0001
domain: simulated
schema_version: '2.0'
dataset_version: sim-p10-v1
simulation_id: sim-p10-v1
parent_system_id: parent-0001
split_group_id: parent-0001
raw_path: raw/sim-p10-v1-0001.fits
science_path: science/sim-p10-v1-0001.npy
processed_path: normalized/sim-p10-v1-0001.npy
image_unit: relative_flux
calibration_status: pass
original_shape: 256x256
processed_shape: 256x256
pixel_scale: 0.05
processing_config_id: simulation-v1
processing_version: '2.0'
background_status: pass
source_type: sersic
mass_target_available: false
theta_E_target_available: true
structural_status: pass
observation_status: pass
science_status: pass
eligible_for: simulation;classical_fit;ml
exclusion_reason: null
mass_model: SIS
theta_E_arcsec: 1.0
simulator_name: lenstronomy
simulator_version: pinned-version
parameter_seed: 1001
noise_seed: 2001
parameters_path: parameters/sim-p10-v1-0001.json
```
