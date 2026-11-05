# Pre-ML handoff v1

## Readiness decision

The repository is ready for a separate machine-learning implementation plan and synthetic in-domain baseline. It is not ready for observational mass claims, calibrated posterior coverage, or population-level performance claims.

## Prediction contract

- Primary target: `theta_E_arcsec`, the simulator's SIE Einstein-radius parameter, in arcseconds.
- First release availability: 1,000/1,000 simulated observations; no physical-mass targets are available.
- Deferred optional targets: `q`, `phi_rad`, `e1`, and `e2`. Multi-target training remains an ML-plan decision.
- Primary input: the `256x256` float32 array at `processed_path`; for the engineering release this is an unscaled, background-subtracted electrons-per-second image.
- Optional declared inputs: science, invalid/coverage/background masks, variance, expected detector variance, PSF kernel, geometry, and allowlisted observational metadata. The model input choice remains to be benchmarked.
- Forbidden features: target values, source/lens generating parameters, seeds, ancestry identifiers, sample/lens IDs, truth/diagnostic paths, and full provenance metadata.

## Frozen datasets

| Role | Release | Parents/systems | Observations | Split |
|---|---|---:|---:|---|
| matched synthetic engineering | `sim-eng1000-v1` | 1,000 | 1,000 | 700 train / 150 validation / 150 test |
| synthetic stress | `sim-p100-v1` | 40 in the stress stage; 100 cumulative | 40 stress observations | separate evaluation only |
| real preprocessing/QC | `real-castles-v2` | 69 | 137 | 94 / 24 / 19 observations grouped by lens |
| exploratory real comparison | `real-comparison-v1` | 3 | 3 selected F160W observations | no training use |

Engineering manifest SHA-256: `189c1351e768c6a6563c2af3243419df92d39f14a9670ff6385b693405e06a44`.

Real manifest SHA-256: `1849ab21013ce282ec6fe98af7f24b767925e8045beec30c155d5c319c903d2e`.

The engineering split seed is `4201`. Splits are connected components over parent, source-template, background-template, split-group, and real-seed ancestry where present. All 1,000 current components are independent; zero identifiers cross splits.

## Engineering distribution

- `theta_E_arcsec`: range 0.5007–1.9995, mean 1.2536, standard deviation 0.4280.
- Axis ratio `q`: range 0.5013–0.9500, mean 0.7220.
- Pixel scale: exactly 0.05 arcsec/pixel.
- Gaussian PSF FWHM: 0.1001–0.1999 arcsec.
- Exposure time: 80.02–119.99 seconds.
- Physical point-image multiplicity: 5 singles, 778 doubles, 3 triples, and 214 quads as returned by the frozen solver configuration.
- Storage: 2,573,791,870 bytes across 21,009 checksummed artifacts.
- Build plus full rerender/checksum workflow: 773.6 seconds on the recorded local environment.
- Rejections and validator errors: zero.

## Baselines and gates

The analytic SIS control recovered all ten P10 radii to numerical precision. The six-condition SIE recovery grid completed 13/13 optimizer fits with 13 locally identifiable solutions and no optimizer/non-identifiability failures. The P30 noiseless control had maximum `theta_E` relative error 0.033%. Noisy cases reached a maximum error of 11.71%; this is retained as an experimental result rather than hidden by the analytic gate.

The exploratory real check measured 0.148″ for B0218, 1.201″ for HE0435, and 0.879″ for Q2237. All are within 6.4% of the configured compatible reference scale. These three systems were preselected and the method is peak astrometry, not a calibrated pixel likelihood or posterior.

## Required evaluation discipline

1. Tune only on train/validation. Open the frozen synthetic test once the ML plan, metrics, and ablations are fixed.
2. Report the constant/mean baseline and the classical recovery results beside any ML metric.
3. Evaluate P100 separately as distribution shift; never merge it into training without a new release ID and split audit.
4. Keep real observations exploratory until empirical PSFs/variance, model suitability, and reference conventions are reviewed system by system.
5. Report synthetic in-domain error separately from uncertainty calibration and real-domain transfer.

## Unresolved domain gaps

- No empirical CASTLES PSF, ERR, or DQ products are available locally.
- Real variance uses a `toy_unprofiled` fallback and cannot support a calibrated likelihood.
- Sources and lens light are analytic Sersic profiles; real morphology and line-of-sight complexity are absent.
- Priors are engineering ranges, not a population/selection-function fit.
- Local Jacobian identifiability is not posterior coverage.
- The real demonstration is small, preselected, and not blind.

No neural architecture, optimizer, loss, augmentation, batch size, learning rate, or training schedule is selected by this handoff.
