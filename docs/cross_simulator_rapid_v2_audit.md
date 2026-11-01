# Rapid A/B renderer audit (item 2)

## Finding

The 150-parent release completed mechanically, but it does **not** provide a
simulator-induced image-domain shift.  The saved Lenstronomy and AutoLens
products are identical at the precision consumed by the CNN:

- 1,350/1,350 parent-condition image pairs are byte-identical (`max |Δ| = 0`).
- 40,500/40,500 paired predictions (3 arms × 10 seeds × 150 parents × 9
  conditions) are exactly identical.
- The backend environments are distinct (Lenstronomy 1.14.2 versus AutoLens /
  AutoGalaxy 2026.10.26.1), so package-version difference alone did not create a
  measurable input shift.

The audit is recorded in
[`renderer_identity_audit.json`](../results/ml/cross-simulator-rapid-v2/renderer_identity_audit.json)
and can be rerun with:

```text
python -B -m scripts.audit_cross_simulator_rapid_v1
```

## Why this happened

The AutoLens adapter converts the mass and Sérsic parameters to match the
Lenstronomy conventions, while the release uses shared project-owned pixel
integration, PSF convolution, detector sampling, and preprocessing.  The two
floating-point paths agree closely enough that the saved float32 images are
the same.  This is useful evidence that the convention mapping is numerically
controlled, but it removes the domain shift required to test simulator
generalization.

## What remains valid

The release is still a valid 150-parent, source-only, target-free evaluation of
the unseen Moffat PSF family.  Its primary Moffat contrasts are:

| Contrast | MAE change (arcsec) | Parent-bootstrap 95% interval |
|---|---:|---:|
| G − C | -0.001000 | [-0.009085, 0.006570] |
| R − G | +0.009602 | [+0.005284, +0.013910] |
| R − C | +0.008602 | [-0.000662, +0.017561] |

These numbers support no reliable extra benefit for the failure-weighted arm
R; they are not evidence that a model transfers between independent simulators.
The Moffat family is engineered synthetic data, not real telescope data.

## Consequence for the paper

Do not write “cross-simulator transfer” for `cross-simulator-rapid-v2`.  The
appropriate wording is “matched-renderer equivalence control plus an unseen
PSF-family robustness test.”  A genuine item-2 result requires a new release
identity in which the target renderer changes at least one image-forming
component (for example, a backend-native AutoLens rasterization/convolution
path) while preserving a documented physical target mapping.  The v2 release
must remain immutable and should not be relabeled after that change.
