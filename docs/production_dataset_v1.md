# Engineering simulation release proposal

`sim-eng1000-v1` is the first frozen, model-ready engineering release. It contains 1,000 independent SIE parent systems with one noisy observation per parent. Its primary supervised target is `theta_E_arcsec`; shape targets are retained for later experiments but are not required for the first baseline.

## Selection

- Reuse the P60 parameter ranges only after the P10/P30/P60/P100 staged checks and the SIE recovery grid pass.
- Keep P100 separate as a stress and out-of-distribution suite. Do not silently mix those broader shear and morphology priors into training.
- Give every parent a unique source-template ID, background-template ID, parameter seed, and noise seed.
- Split connected ancestry components, not rows, using the fixed seed `4201` and fractions 70/15/15.

## Acceptance gates

1. Exactly 1,000 requested and accepted parents; no silent rejections.
2. Unique sample IDs, parent IDs, parameter seeds, and noise seeds.
3. Every saved product passes schema, shape, dtype, finite-value, variance, PSF, and deterministic rerender checks.
4. Train, validation, and test contain no shared parent, source template, background template, or other connected ancestry component.
5. The strict loader opens all products with simulator truth separated from model inputs.
6. The release records the configuration hash, artifact checksums, runtime, storage size, split report, rejection log, and validator results.

## Known boundary

This is an engineering benchmark, not an observationally calibrated population model. It uses analytic Sersic sources, analytic Gaussian PSFs, scalar backgrounds, independent Poisson/read noise, and uniform hand-set priors. Real HST PSFs, detector systematics, correlated noise, selection effects, and population-calibrated priors remain future work. Any machine-learning result must therefore be reported as in-domain synthetic performance first.
