# Reliability study protocol v1.10 — Week 10 CASTLES domain gap

**Status:** frozen before Week 10 image-statistic extraction and domain comparisons  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior amendments:** [`protocol_v1.1.md`](protocol_v1.1.md) through [`protocol_v1.9.md`](protocol_v1.9.md)

## Question and eligible observations

Describe how the final synthetic training distribution differs from available CASTLES/HST development observations on observable metadata and image proxies. This is not an accuracy benchmark and will not score a lens parameter on real data.

- Use the 137 rows in `data/labels/derived/metadata.csv` that have a corresponding `data/labels/derived/processed_metadata.csv` row and normalized `.npy` product under `data/derived/actual/castles-development-v1/products/normalized/`. Keep all eligible images regardless of redshift or catalog `theta_E` availability; group repeated observations by their 69 lens identifiers.
- Use only `data/simulations/reliability-domain-randomized-v1/splits/train.csv`, which represents the final Week 8 model-training observation distribution (1,400 views from 700 independent parents). Do not read validation or historical test labels for this phase.
- Read only image paths/shapes, pixel scale, instrument/filter/exposure/calibration metadata, and physical model fields needed to characterize the simulator. Do not read `metadata_enriched.csv` or `lens_catalog.csv` parameter labels, including Einstein radii and redshifts. Do not access model predictions on real images.
- Exclude a row only when its listed normalized image is missing, unreadable, not two-dimensional, or has no finite pixels; log the exact reason. Do not filter on astrophysical labels.

## Frozen measurable features

Report per-image rows and domain-level distributions for:

1. Pixel scale; original/native center-to-center field of view; and processed-image center-to-center field of view (`(processed_width-1) × pixel_scale` and `(processed_height-1) × pixel_scale`). The coverage map uses the processed-image field actually presented by the retained image product; raw/native dimensions are reported separately to describe crop/pad history.
2. Native and processed image shapes, padding fraction when derivable from recorded original/processed dimensions, and finite-pixel fraction.
3. Within-image robust intensity summaries (1st/50th/99th percentiles and 99–1 range), explicitly marked as not photometrically comparable because real and synthetic products have different calibration/normalization semantics.
4. Fixed-cutout-center radial absolute-intensity profiles in the frozen physical annuli `[0,0.5)`, `[0.5,1.0)`, `[1.0,1.5)`, `[1.5,2.5)`, and `[2.5,4.0)` arcsec, divided by each image's 99th percentile absolute intensity. These are image-shape proxies, not lens-centered source profiles.
5. Central concentration (absolute intensity sum within 0.5 arcsec divided by that within 2.5 arcsec) and an outer-region brightness proxy (median and MAD beyond 2.5 arcsec), with all valid-pixel and radius counts recorded. The outer proxy is not a calibrated background or detector-noise estimate.
6. CASTLES metadata completeness and category counts for telescope, instrument, detector, filter, exposure, units, calibration stage, and available correction flags. Keep missing categories explicit; do not infer them.
7. Simulator PSF family/FWHM and supported source/mass families from the run manifests only. Do not treat its variance or read-noise parameters as calibrated HST detector values.

Do not report a mask-fraction comparison: the retained CASTLES normalized products do not include an invalid-pixel or coverage-mask product. Do not report real PSF widths or multiplicity labels: no eligible PSF product or reviewed multiplicity field exists in this release. Report these gaps as unavailable, not zero.

## Summary and coverage rules

- Give row counts, unique lens/parent counts, missingness, median, IQR, and 5th/95th percentiles. Treat results as descriptive; do not use image rows as independent inferential units.
- For a directly comparable numeric feature, report the fraction of real observations within the synthetic 1st–99th percentile and full observed ranges. Mark **well represented** only if at least 90% fall in that robust synthetic interval and measurement semantics match; mark **weakly represented** for partial overlap or a proxy/normalization mismatch; mark **unsupported** when either domain lacks a usable measurement or the simulator lacks that mechanism. When semantics differ, retain a numeric overlap for context but rate the feature weakly represented rather than using the overlap threshold as a direct transfer claim.
- For categories and physical mechanisms, use **well represented** only when the same relevant category/mechanism is explicitly simulated; otherwise use **weakly represented** for a limited analogous treatment or **unsupported** when absent.
- Freeze the coverage rationale in a table with feature, synthetic range/mechanism, CASTLES availability, result, rating, and limitation.
- Mention the existing three-system astrometric comparison only as exploratory context; it is not a general accuracy benchmark.

## Deliverables and integrity

Write machine-readable comparisons, metadata-quality table, coverage map, figures, source/config hashes, and a limitations report under `results/reports/domain_gap_v1/`, plus a phase status note under `docs/reliability_study/`.

The report must specify requirements for a genuine future blinded real benchmark: reviewed and uncertainty-qualified lens parameters, independent lens-system holdout, instrument/filter-aware calibrated PSF and noise products, masks/crop/normalization provenance, a frozen model, and a scorer that joins real labels only after predictions are irreversibly saved. No claim from this phase may state real-data model accuracy or calibrated noise transfer.
