# Real-lens astrometric comparison

Exploratory peak astrometry reproduces the scale of preselected real lenses. This is not a calibrated pixel-level mass fit, posterior, or unbiased accuracy estimate.

| Lens | Astrometric radius | Reference | Difference | Within 15% |
|---|---:|---:|---:|:---:|
| B0218 | 0.148" | 0.158" | 6.3% | yes |
| HE0435 | 1.201" | 1.210" | 0.8% | yes |
| Q2237 | 0.879" | 0.850" | 3.4% | yes |

## Limitations

- The three systems were preselected and are not a population-level validation sample.
- Brightest-peak selection assumes the lensed images outrank lens-galaxy and field peaks.
- The diagnostic scale combines radial scatter with a half-pixel floor; it is not a confidence interval.
- HE0435 uses CASTLES Size/2 as a screening proxy rather than a like-for-like SIE critical radius.
- Real products have a toy variance model and no empirical PSF, so likelihood-based claims are deferred.
