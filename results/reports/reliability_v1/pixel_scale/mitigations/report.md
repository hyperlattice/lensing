# Pixel-scale mitigation audit

Validation-only paired comparison with the frozen baseline. Checkpoint epochs were selected using the predeclared equal-weight mean across the three validation scale strata.

The utility rule requires lower shifted-scale MAE at both 0.04 and 0.06 arcsec/pixel and no more than 0.01 arcsec canonical MAE degradation. Parent-bootstrap intervals average the three seed effects within each parent before resampling.

The models were selected on these validation parents, so these figures are development evidence. The historical test split was not accessed; conclusions remain within the current synthetic simulator.

## pixel_target

- Lower MAE at both shifted scales: **False**
- Canonical degradation within 0.01 arcsec: **True**
- Utility rule passed: **False**

| Comparison | Paired MAE change (arcsec) | Parent-bootstrap 95% interval |
|---|---:|---:|
| Candidate minus baseline, canonical | -0.0141 | [-0.0224, -0.0061] |
| Candidate minus baseline, pixel-scale-004 | -0.0734 | [-0.0913, -0.0566] |
| Candidate minus baseline, pixel-scale-006 | +0.0547 | [+0.0331, +0.0764] |
| Candidate pixel-scale-004 minus candidate canonical | +0.0113 | [+0.0013, +0.0211] |
| Candidate pixel-scale-006 minus candidate canonical | +0.1177 | [+0.0962, +0.1401] |

## metadata

- Lower MAE at both shifted scales: **False**
- Canonical degradation within 0.01 arcsec: **True**
- Utility rule passed: **False**

| Comparison | Paired MAE change (arcsec) | Parent-bootstrap 95% interval |
|---|---:|---:|
| Candidate minus baseline, canonical | -0.0132 | [-0.0213, -0.0052] |
| Candidate minus baseline, pixel-scale-004 | -0.0709 | [-0.0881, -0.0541] |
| Candidate minus baseline, pixel-scale-006 | +0.0209 | [+0.0094, +0.0322] |
| Candidate pixel-scale-004 minus candidate canonical | +0.0129 | [+0.0020, +0.0244] |
| Candidate pixel-scale-006 minus candidate canonical | +0.0829 | [+0.0637, +0.1013] |

Full per-seed metrics, paired predictions, effects, and checkpoint hashes are in the neighboring CSV and JSON files.
