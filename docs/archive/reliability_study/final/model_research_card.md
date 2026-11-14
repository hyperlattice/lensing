# Research / model card — `theta_E` synthetic regression

## Model and task

- **Target:** scalar Einstein radius, `theta_E` (arcsec), for simulated SIE lens systems.
- **Model:** arithmetic mean of three fixed convolutional-network checkpoints, seeds 12001, 12002, and 12003.
- **Final training regime:** domain-randomized training on 1,400 views from 700 independent parents, retaining one canonical and adding one randomized view per parent. Normalization is training-only and checkpoint-specific.
- **Fixed comparator:** three-seed canonical-trained CNN ensemble.

## Intended use

Use for controlled synthetic benchmarking of `theta_E` regression under the recorded simulator, prior, preprocessing, and tested nuisance conditions. Use the machine-readable paired outputs and parent counts when comparing conditions.

## Unsupported use

Do not use this evidence to report Einstein radii for real CASTLES/HST observations, estimate real-data uncertainty, claim instrument-calibrated noise robustness, or assert transfer across full renderers. Do not extrapolate beyond the tested SIE/Sersic prior and fixed perturbations.

## Performance on the independent synthetic holdout

The holdout has 150 new parents and 750 observations. The randomized ensemble MAE was 0.0655 arcsec canonical; its equal-weight four-stress paired change versus the fixed comparator was -0.0361 arcsec (95% parent-bootstrap interval [-0.0427, -0.0294]).

| Condition | Randomized ensemble MAE | Fixed baseline MAE |
|---|---:|---:|
| Canonical | 0.0655 | 0.0828 |
| Pixel scale 0.04 | 0.1158 | 0.1696 |
| Pixel scale 0.06 | 0.0862 | 0.1278 |
| Half exposure | 0.0936 | 0.1150 |
| Double read noise | 0.0879 | 0.1155 |

## Calibration and uncertainty

The standard deviation across three seed predictions is retained as an ensemble-spread descriptor. It was not calibrated against target coverage and must not be interpreted as a predictive interval.

## Known failure modes and limits

- Pixel-scale 0.04 produced the largest randomized-ensemble holdout MAE (0.1158 arcsec); 67/150 errors exceeded 0.10 arcsec.
- Error distributions have substantial tails; see [`failure_case_catalog.csv`](../../../../results/reports/final_v1/tables/failure_case_catalog.csv).
- The holdout uses the same synthetic prior and renderer; the CASTLES study was descriptive and did not score labels.
- The tested cross-implementation path shared source-light, PSF, detector, and preprocessing components.
- The point-image classical comparator is a limited geometry fit with fallback behavior, not a pixel-likelihood lens model.

## Data and provenance

The frozen inputs, release hashes, exact-once marker, bootstrap seed, environment, and build procedure are recorded in [`reproducibility_statement.md`](reproducibility_statement.md) and [`package_manifest.json`](../../../../results/reports/final_v1/package_manifest.json).
