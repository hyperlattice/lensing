# Reliability study protocol v1.7 — Week 7 independent SIE lens-equation study

**Status:** frozen before Week 7 alternate-renderer predictions  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior completed amendments:** [`protocol_v1.1.md`](protocol_v1.1.md) through [`protocol_v1.6.md`](protocol_v1.6.md)

## Scope and implementation

The installed simulator is Lenstronomy 1.14.2. No second full renderer package is installed. Week 7 will therefore use an independently implemented SIS/SIE deflection and image-root path, evaluated against Lenstronomy as the reference implementation. The independent equation is in `src/simulation/independent_sie.py` and uses the closed-form singular isothermal ellipsoid deflection. The mass convention is `theta_E` circularized and `q` minor/major; coordinates are sky x/y in arcseconds about the stated lens center. The circular limit `q=1` is SIS. The convention follows the isothermal ellipsoid formulation of [Kormann, Schneider & Bartelmann](https://arxiv.org/abs/astro-ph/9311011) and the [Lenstronomy SIE definition](https://lenstronomy.readthedocs.io/en/latest/_modules/lenstronomy/LensModel/Profiles/sie.html).

This is a controlled lens-equation implementation comparison, not a fully independent software stack. Both paths use Lenstronomy's Sersic surface-brightness model, the project's Gaussian PSF, detector-noise draw, and preprocessing. Those shared components are held fixed to isolate the lens deflection and point-image root path. The renderer must be described with this limitation in all reports.

## Matched parent set and conditions

- Use all 150 validation parents from `sim-eng1000-v1`. Do not use train or historical test parents.
- For each parent, use the same frozen mass and source parameters, source/lens light, 256×256 grid, 0.05 arcsec pixel scale, supersampling factor 2, Gaussian PSF FWHM, background rate, read noise, and noise seed as its canonical simulator record.
- Render two independent-equation conditions: `independent-canonical` and `independent-exposure-half`. For the latter, change exposure time by exactly 0.5 while holding the other observation parameters fixed.
- Use the same 150 parent IDs, all three frozen CNN checkpoints (12001, 12002, 12003), and their saved training-only normalization. Do not retrain, tune, or select a model using Week 7 predictions.
- Save predictions before joining validation targets. Join targets only for the predeclared paired scoring.

## Control gates before CNN scoring

1. Compare independent and Lenstronomy SIE deflections over a fixed grid for `q` values 1.0, 0.95, 0.70, and 0.50; orientations 0, 0.3, 1.2, and 2.6 radians; `theta_E=1.15` arcsec; and lens center (0.10, −0.08) arcsec. Exclude the singular center. Maximum component difference must be at most 1e-6 arcsec.
2. Independently solve image positions for all 150 validation parents. At least 95% must have the same root count as the saved Lenstronomy solution; for matching counts, Hungarian-matched maximum separation must be at most 0.02 arcsec. If this gate fails, stop before CNN scoring and report the failure.
3. Validate the 256×256 sampling, image-center convention, PSF, and preprocessing with a four-parent renderer smoke before generating the full matched release.

## Primary metrics and analysis

- Primary comparison: paired change in absolute theta_E error and paired prediction shift for each independent-render condition relative to the same parent's original Lenstronomy rendering.
- Report per-seed metrics and the metric from seed-averaged predictions separately. Average seed-specific effects within parent before bootstrap.
- Use 10,000 parent bootstrap resamples with seed 6031; report 95% intervals. Use the 0.01 arcsec practical-effect margin from v1.0.
- Report root-count agreement, matched point-position separations, per-image pixel correlation/RMSE, and renderer-level flux differences as implementation diagnostics, not as parameter-accuracy metrics.
- Compare canonical and exposure-half effects with the matched Week 4 results as contextual reference. Do not use them to change Week 7 model or preprocessing choices.

## Release, outputs, and claim limits

The matched release is `data/simulations/reliability-independent-sie-v1`. Save conventions, root-control records, images, manifests, before-scoring predictions, paired metrics, and hashes under `results/reports/reliability_v1/cross_simulator_week7/`. The report is `docs/reliability_study/week7_cross_simulator_status.md`.

The result can establish sensitivity to this independently coded SIE deflection/rasterization path. Because source-light evaluation, PSF, noise, and preprocessing are shared, it cannot establish general simulator independence or robustness to another source, PSF, or detector implementation. Historical test samples and real-data labels remain prohibited.
