# Benchmark implementation checkpoint: foundations v1

Date: 2026-10-27. This is development infrastructure for the
[superiority study](superiority_demonstration_plan_v1.md), not a method freeze,
qualified final benchmark or new model-accuracy result.

## Implemented

`src/data/benchmark_contract_v1.py` adds a strict JSON overlay for native-grid
observations. It leaves the existing schema-v2 CSV loader and all reported
releases unchanged. Required products are image, variance, invalid mask,
coverage mask and PSF. The contract checks hashes, contained relative paths,
shapes, float32 image/variance arrays, finite values, positive valid-pixel
variance, boolean masks, and nonnegative odd-sized unit-sum PSFs. Image units
are electrons/s; variance units are their square. True masks mean excluded.

Roles are source train/development/calibration, target adaptation, numerical
development, power pilot and final. Across the supplied full ledger and
optional exclusion ledger, parent IDs, parameter/noise seeds and reused asset
IDs cannot cross roles. Paired conditions/renderers within one role may share
parent identity. A parent must retain its parameter seed and asset ancestry.
An analytic family such as Sersic is not an asset ID. This release checks
cross-role overlap; the final population audit must additionally check latent
fingerprints and uniqueness within each role, including aliases and seed
collisions. Exclusion ledgers must be assembled before new generation without
opening old final images or target values.

Only renderer A (lenstronomy) is permitted for source roles and only renderer B
(autolens) for adaptation. Moffat observations cannot enter source training,
development/calibration, or target adaptation. Engineering releases are
restricted to numerical development and require explicit opt-in. Qualified
releases require a matching, hashed qualification report bound to the exact
observation definitions. The loader verifies the report's identity and passed
status; it does not itself establish that the numerical tests were adequate.
Final access still requires the separate two-freeze/evaluation workflow, which
has not been implemented or authorized by a passed loader check.

The model-input interface returns an image array only. It never opens a target
file or exposes row metadata as a predictor input. An explicit source-only
label join checks exact observation identities and target validity. The
adaptation interface rejects label loading before opening a file. Identity
access is for auditing, not a model feature. Masks/variance/PSF are validated
products; the current image-only CNN does not consume them. No automatic
normalization is performed; the eventual trainer must use source-train-only
statistics and declare handling of excluded pixels.

`src/ml/augmentation_policy_v1.py` implements the seven proposed Gaussian-only
cells, the frozen failure-weighted probability formula, and a deterministic
paired schedule. One transformation cell is selected per batch to retain a
homogeneous native shape. Pixel-scale changes preserve the angular field to
the nearest native pixel. Parent draws, cell draws and detector seeds use
separate random streams. Changing policy weights preserves the actual parent
exposure and detector-seed schedule. Failure weights require paired errors
with seed/parent/cell axes from source development. We have not yet rendered
the complete seven-cell source-development set, scored it, or frozen empirical
weights. The renderer must explicitly apply the PSF multiplier and fresh noise
seeds; the existing v2 adapter does not automatically consume the new policy.

`src/ml/domain_adaptation_v1.py` implements differentiable Gaussian multi-kernel
squared MMD with explicit bandwidths, both biased and unbiased estimators, and
no target-label argument. The unbiased estimator may be negative. The initial
alignment site is the existing image-only CNN's pooled features; its ordinary
prediction is unchanged. This is a literature-inspired primitive, not an
author-code reproduction or a trained MMD comparison. Author implementation
pinning, kernel/weight tuning, and the integrated eight-arm trainer remain.

## Executed development release

Command:

```powershell
python -B -m scripts.materialize_benchmark_contract_development_v1
```

Output: `data/simulations/benchmark-contract-development-v1/benchmark.json`.
Tracked report:
[engineering_release.json](../results/reports/benchmark-foundations-v1/engineering_release.json).

The new copy contains 16 observations from four existing development parents,
at 256x256 and 320x320 native resolutions. All passed product/hash validation;
access through the source-training role was rejected. The importer verifies
source array hashes, copies existing variance/PSFs and writes explicitly empty
native-grid masks. It never overwrites the source or permits output reuse.

Images are the raw unnormalized rate with background retained, rather than
the old background-subtracted processed array. This keeps them consistent
with the copied analytic detector variance. They cannot replace images in a
reported historical experiment. Variance is synthetic expected variance and
does not become a model feature. Legacy noise seeds are shared across each
parent's conditions; this is not the new fresh-noise augmentation experiment.
The eligibility remains engineering-only, with no CNN inference or new
accuracy result. No old final holdout images/labels were accessed. The CSV
parser ignores target-column values; the source manifest's byte hash is
recorded for provenance.

## Verification and next gates

The initial 21 focused tests passed. An additional end-to-end importer test
checks source preservation, refusal to overwrite, hash-tamper rejection and
that an intentionally nonnumeric target column is never parsed. Regression
checks also cover the previous matched-training and simulator-v2 controls.
The final test output is recorded alongside the engineering report.

Tests cover cross-role overlap through renamed parents/seeds/assets; target
isolation; injected truth fields; path traversal; artifact mutation; invalid
variance/masks/PSFs; qualification binding; held-out-family exclusion; paired
budgets; reproducibility; analytic MMD values; autograd checks; CNN prediction
parity; and synthetic regression retaining its signal while reducing a known
nuisance shift. Synthetic recovery is a unit-level behavior check, not lens
domain-adaptation evidence.

Next, in order:

1. Create separately versioned renderer implementations that resolve the
   central radial floor without touching frozen sources. Broaden numerical
   controls over centers, caustics, native geometries and integration factors.
   Record rejected scenes and choose precision using numerical criteria.
2. Extend the manifest with reviewed observation/preprocessing provenance,
   full identity exclusion ledgers, within-role latent uniqueness, expected
   condition completeness and failure/cohort audits. Add the freeze boundary.
3. Connect G/N/R condition selection and fresh detector seeds to qualified
   source rendering. Build a source-only score artifact that binds prediction
   axes to IDs, seeds, conditions, reference checkpoints and source hashes.
4. Pin published author code/data and establish baseline reproduction checks.
   Integrate MMD with image-only target batches, source-only normalization,
   matched source schedules, histories and compute accounting.
5. Run development smoke fits and budget/tuning experiments. Neither power
   pilot nor final generation starts before its specified protocol freeze.

No methods, weights, final fidelity or success thresholds are frozen yet.
