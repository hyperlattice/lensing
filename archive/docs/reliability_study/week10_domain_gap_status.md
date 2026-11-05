# Week 10 status — CASTLES/synthetic domain gap

Week 10 completed using all 137 processed CASTLES images from 69 lens systems and the 1,400-view Week 8 training distribution from 700 independent synthetic parents. All eligible images loaded; no rows were excluded. The review used only instrument/observation metadata and image-derived proxies. Einstein-radius/redshift labels and real-image model predictions were not read or scored.

The direct pixel-scale comparison is weakly represented: 52.6% of CASTLES observations fall inside the synthetic training distribution's 1st–99th percentile range (0.04–0.06 arcsec/pixel). The retained processed-image angular field is unsupported under this simulator coverage rule: only 6.6% of real products fall within the synthetic training range. Raw/native shapes and processed shapes were both recorded to distinguish original FOV from crop/pad and model-input FOV.

Image intensity range and central-concentration proxies are weakly represented because the real and simulated product normalization semantics differ, and the profiles use the fixed cutout center rather than a measured lens center. They are descriptive shape proxies, not photometric or physical morphology comparisons. CASTLES filter metadata is missing in 54% of rows, detector in 46%, calibration stage in 54%, and correction metadata for cosmic-ray/photometric correction in 73%. Instrument categories are available for all images and span NICMOS, ACS, and WFPC2, while the simulator has no instrument-specific rendering.

The simulator's analytic Gaussian PSF, toy variance/read-noise, SIE mass, and Sersic-plus-point-source light do not cover calibrated real PSF/noise, invalid/coverage masks, reviewed image multiplicity, or full real source/lens diversity. These are explicit gaps, not zero-valued image measurements. The existing three-system astrometric comparison remains exploratory context only.

The machine-readable feature distributions, metadata quality, coverage map, figures, hashes, and benchmark requirements are under [`results/reports/domain_gap_v1/`](../../results/reports/domain_gap_v1/). No conclusion from this phase supports real-image parameter accuracy.

## Next phase

Proceed to Week 11 with the frozen Week 8 randomized three-checkpoint ensemble as the final method, and the original three-checkpoint CNN ensemble as a fixed comparator. Generate 150 entirely new independent parent systems and evaluate canonical plus the same four selected Week 8 stress conditions once. Freeze code, configs, checkpoint hashes, scoring, and the no-retuning rule before creating the holdout release.
