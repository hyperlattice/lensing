# Week 7 — cross-simulator implementation comparison

## Result

Week 7 completed as a controlled SIE equation and rasterization comparison. The environment had Lenstronomy 1.14.2 but no second full renderer, so the alternate path independently implements the closed-form SIS/SIE deflection, point-image root search, finite-difference magnification, and supersampled ray mapping. It shares Lenstronomy's Sersic surface-brightness evaluator and the project's Gaussian PSF, detector noise, and preprocessing.

The independent deflection differed from Lenstronomy by at most **7.63×10⁻⁹ arcsec** on the frozen 41×41 grid. Independent root counts agreed for **150/150** validation parents; the largest Hungarian-matched root separation was below the frozen **0.02 arcsec** limit. The four-parent smoke reproduced the stored processed image exactly. The `0.05²` pixel-area conversion was required to convert the subpixel surface brightness into native-pixel flux; without it, the renderer was brighter by a factor of 400.

The matched release contains 150 parents in each of two conditions (300 records). It passed manifest schema checks and strict product loading. The three frozen CNNs were evaluated without loading validation targets; predictions were saved before target joining. Relative to the paired Lenstronomy images, the mean absolute prediction shifts were **3.02×10⁻⁷ arcsec** for canonical images and **2.86×10⁻⁷ arcsec** for half exposure. Seed-averaged MAE was 0.08735 and 0.13234 arcsec, respectively. The paired MAE changes were effectively zero and far inside the 0.01-arcsec practical margin.

These results show agreement between two implementations of the tested SIE equation and rasterization conventions. They do not establish robustness to a different source-light family, PSF, detector model, preprocessing stack, or real observational domain.

## Artifacts

- Frozen design: [`protocol_v1.7.md`](protocol_v1.7.md) and [`reliability_cross_simulator_v1.yaml`](../../configs/reliability_cross_simulator_v1.yaml)
- Reproducible audit: [`reliability_week7.py`](../../src/ml/reliability_week7.py)
- Matched release: `data/simulations/reliability-independent-sie-v1/`
- Controls, before-scoring predictions, paired effects, hashes, and figure: `results/reports/reliability_v1/cross_simulator_week7/`
- Machine-readable summary: [`summary.json`](../../results/reports/reliability_v1/cross_simulator_week7/summary.json)

Historical synthetic test data and real-data labels were not accessed.

## Next phase

Proceed to Week 8 using the predeclared high-impact failures: pixel scale (0.04/0.06 arcsec per pixel), half exposure, and doubled read noise. Freeze the domain-randomization assignment before generating augmented training images or fitting new checkpoints. Keep validation parents and the three established seeds fixed.
