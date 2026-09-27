# Week 6 shortcut audit

This report describes paired image interventions on the frozen three-seed CNN and the same 150 canonical validation parents. Each intervention was applied to processed images before the checkpoint's train-only global normalization. Targets and parent identities were held fixed.

The masks use only fixed image-centered angular radii (central disk <0.50 arcsec; annulus 0.70–2.20 arcsec). They are not truth-derived lens or source masks. Brightness and rotation tests are image interventions, not re-rendered physical observations. Effects are descriptive and not corrected for the number of interventions.

Canonical predictions reproduced the saved frozen-checkpoint outputs with maximum absolute difference 1.39e-07 arcsec.

| Intervention | Seed-averaged MAE | Paired ΔMAE [parent-bootstrap 95% interval] | Mean absolute prediction shift | Interpretation under 0.01 arcsec margin |
|---|---:|---:|---:|---|
| Rotate 90° counter-clockwise | 0.0872 | -0.0007 [-0.0074, +0.0060] | 0.0405 | practically small for this intervention |
| Brightness ×0.5 | 0.0874 | -0.0000 [-0.0000, +0.0000] | 0.0001 | practically small for this intervention |
| Brightness ×2 | 0.0874 | +0.0000 [-0.0000, +0.0000] | 0.0000 | practically small for this intervention |
| Zero central disk <0.50 arcsec | 0.0955 | +0.0074 [+0.0036, +0.0119] | 0.0134 | inconclusive under the frozen rule |
| Zero annulus 0.70–2.20 arcsec | 0.3375 | +0.2632 [+0.2246, +0.3022] | 0.3127 | practically important increase supported |

## What the interventions establish

The paired effects quantify how these fixed image transformations change predictions and validation errors. A large effect shows dependence on information changed by the intervention, but it does not identify a unique causal feature or show that the model uses an astrophysically invalid shortcut. In particular, masking removes both nuisance and potentially useful lensing information; rotation changes the orientation relative to the image axes; and brightness scaling changes the image amplitude while retaining the original normalization statistics.

Source and PSF changes were assessed through the matched rerender studies in Weeks 3–5. Attribution maps were not used as causal evidence. Historical test data were not accessed.

Machine-readable predictions, paired effects, target-bin summaries, and checkpoint hashes are stored beside this report.
