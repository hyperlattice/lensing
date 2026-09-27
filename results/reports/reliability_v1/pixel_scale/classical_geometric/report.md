# Week 3 — classical point-image geometry baseline

Validation-only, image-derived comparator. The fitting code reads no parameter sidecars or saved analytic image-position diagnostics; labels are loaded only after predictions have been written.

This is a limited point-image SIS/SIE geometry control, not a pixel-likelihood fit or posterior analysis.

| Scale | Parents | Classical MAE, fallback included | Fit-only MAE | Fit success | CNN MAE | Classical − CNN MAE (95% parent interval) |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 150 | 0.2007 | 0.0698 | 106/150 | 0.0921 | +0.1086 [+0.0724, +0.1469] |
| high | 150 | 0.1944 | 0.0587 | 105/150 | 0.1410 | +0.0534 [+0.0216, +0.0862] |
| low | 150 | 0.2168 | 0.0688 | 100/150 | 0.1628 | +0.0540 [+0.0173, +0.0905] |

## Paired classical scale effects

| Shift | Mean MAE change versus canonical | Parent-bootstrap 95% interval |
|---|---:|---:|
| low | +0.0161 arcsec | [-0.0083, +0.0414] |
| high | -0.0063 arcsec | [-0.0348, +0.0219] |

## Limits

- This is an image-derived point-image geometry baseline, not a full pixel-likelihood SIE fit.
- The mass center is estimated from central light and the point-image astrometry is approximated with SIS for doubles and SIE for three/four detections.
- The fallback is the training-set mean; report fallback frequency and do not interpret fallback rows as successful classical fits.
- The released variance map and Gaussian control PSF are synthetic products; real-domain performance is not established.
