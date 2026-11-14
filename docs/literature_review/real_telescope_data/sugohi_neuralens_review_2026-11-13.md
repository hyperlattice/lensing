# SuGOHI, the lens-finding challenge, and NeuraLens: sample-size review

Checked 13 November 2026. Targeted single-reviewer data-access and compatibility
audit, not a systematic literature review. No training, predictions, final-cohort
selection, or changes to the experiment implementation were performed.

## Decision

SuGOHI is a promising route to a larger **separate HSC observational cohort**.
Its catalogue size is not the number of independently measured Einstein-radius
references. Prefer the published classical-model subsets over catalogue-only
radii or another network's predictions. Keep the HST and HSC results separate,
deduplicate physical lenses across surveys, and exclude development systems in
every band/release. The final additional eligible count is not yet known.

For a smaller extension within the existing HST experiment, investigate the
NeuraLens DESI Strong Lens Foundry models first. The lens-finding challenge is
an external synthetic resource, not a real-observation sample-size increase.

## SuGOHI: downloaded catalogue and actual screening

Source: [SuGOHI catalogue](https://www-utap.phys.s.u-tokyo.ac.jp/~oguri/sugohi/).
The public CSV was downloaded, bounded to 2 MB, and retained without modification.
It is 423,170 bytes, has no header, and contains 14 columns. Column meanings were
checked against the public query form and a small query response.

| Screen | Catalogue rows |
|---|---:|
| Complete snapshot | 3,961 |
| Grade A | 325 |
| Grade B | 1,266 |
| Grade C | 1,764 |
| Ungraded | 606 |
| Grade A/B, galaxy lens and galaxy source (`GG`) | 1,182 |
| Same subset with a positive catalogue radius | 530 |
| Same subset with catalogue radius 0.5–2.0 arcsec | 493 |

The last group contains 115 grade-A and 378 grade-B rows, with 493 distinct
catalogue IDs. It is a candidate pool only: radius conventions, uncertainties,
model quality, science-image access, and overlap with our HST data remain to be
qualified. The full catalogue has 3,960 distinct IDs: `092647+050003` occurs twice
with conflicting positions/types. Neither row enters the 493-row range screen.
The website reports positional merging within three arcseconds; that does not
replace our independent cross-survey identity audit.

The catalogue has no radius-uncertainty column. Its radius entries also need
individual provenance checks. For example, HSCJ015618-010747 has catalogue radius
1.47 arcsec, whereas Gawade et al.'s printed YattaLens reference is 0.87 +/- 0.18
arcsec. This is not a determination of which estimate is correct; it demonstrates
why catalogue and classical-table values cannot be interchanged silently.

There are also 106 grade-A/B galaxy–quasar (`GQ`) rows. That may be a closer
source-morphology route to our point-source simulations, but their numerical
reference and image eligibility were not screened in this review.

## The stronger SuGOHI reference route

[Gawade et al. (2025), MNRAS 540, 3384](https://doi.org/10.1093/mnras/staf935),
[arXiv v3](https://arxiv.org/html/2404.18897v3), models 182 grade-A/B systems
with YattaLens. Appendix D distinguishes 122 visually GOOD fits from 60 POOR fits.
These quality classes concern both foreground-light subtraction and mass modelling;
they are not guaranteed exact parameter truth or universal reference validation.

The grade-A table contains 25 systems, with separate CNN and YattaLens rows.
We extracted **only the 25 YattaLens radii and quoted symmetric errors**. Twenty-three
printed point estimates lie within 0.5–2.0 arcsec, before conversion to our target
convention. The other 157 grade-B references are reported in supplementary material.
The publisher's supplementary-data route returned HTTP 403, so those rows and
the per-system GOOD/POOR flags were not retrieved here. Thus neither 182 nor 122
is a verified final sample count for this repository.

Appendix A explicitly distinguishes radius conventions between GRAVLENS, YattaLens,
Sonnenfeld et al. and Schuldt et al.; some mappings depend on axis ratio. Verify
the printed table convention against our lenstronomy SIE target using matched
synthetic deflection/critical-curve controls. Obtain underlying shape parameters
or posterior samples where needed; marginal errors alone do not establish a
valid nonlinear uncertainty conversion.

The paper uses 101x101 three-band HSC images spanning about 17 arcsec, approximately
0.168 arcsec/pixel. Our model is single-band, canonically 0.05 arcsec/pixel, with
0.04/0.06 scale stresses. A representative HSC resolution/band/PSF/calibration
check is therefore necessary. Resampling onto 0.05 pixels does not recover lost
resolution and is not implemented by the current native-crop preparation path.

Use original images with foreground light consistent with our trained inputs.
The paper's lens-light-subtracted inputs for some networks should not be copied
into our frozen C/G/R pipeline as an unrecorded change. YattaLens can still supply
classical references while its image-preparation choices are reviewed separately.

The small catalogue query linked PNG cutout previews. No science FITS, PSF,
coverage or variance arrays were downloaded. HSC public-release acquisition must
be checked separately, including release identities and any account requirements.
The public YattaLens code is a modelling route, not a substitute for a vetted
reference catalogue.

## First supplied paper: the 2019 challenge

The linked paper is [Metcalf et al., The Strong Gravitational Lens Finding
Challenge](https://doi.org/10.1051/0004-6361/201832797). The publisher PDF returned
403; relevant methods were read in the [author preprint](https://arxiv.org/pdf/1802.03609).
It describes a lens/non-lens classification challenge with 20,000 training images
per dataset and 100,000 test candidates, using 101x101 images. Space-like images
use 0.1 arcsec pixels, ground-like images 0.2 arcsec pixels. Ground-like data
include simulated arcs injected into some real KiDS galaxy cutouts.

It does not add confirmed real radius-labelled lenses to Experiment 2. It could
support a separately designed external synthetic test. Its radius is defined
from critical-curve enclosed area, and its mass construction includes cosmological
halos and stellar mass maps, so it is not automatically our SIE parameter target.
The paper also describes reuse of physical structures/critical curves; image
count cannot establish independent-parent count. Archive accessibility, numeric
label files and shared ancestry were not verified here.

## NeuraLens: smaller but closer HST route

The [supplied publications page](https://sites.google.com/usfca.edu/neuralens/publications)
has moved to [Berkeley Lab](https://sites.google.com/lbl.gov/neuralens). It is a
publication/catalogue hub, not one uniform radius-labelled dataset. Large discovery
and spectroscopic counts do not imply classical mass-model reference coverage.

[DESI Strong Lens Foundry I](https://arxiv.org/abs/2502.03455) describes 51 successful
HST observations and a demonstration model. The newer [Foundry V, v2](https://arxiv.org/html/2512.07823v2)
includes seven distinct modelled systems in Table 1, counting the earlier model
once, even though its abstract highlights six new systems. The table's raw
point-estimate screen retains three in 0.5–2.0 arcsec:

| System | Table-1 radius (arcsec) |
|---|---:|
| DESI J234.4783+14.7232 | 1.548 |
| DESI J257.4348+31.9046 | 1.9899 |
| DESI J238.5690+04.7276 | 1.475 |

These are provisional reference candidates. Foundry V uses EPL plus shear,
WFC3/F140W data drizzled to 0.065 arcsec/pixel, and some small modelling crops.
Target-convention, full-field coverage, PSF, masks, photometry, overlap, and
reference-summary consistency still require review. Neither imaging/posterior
intake nor a joint numerical eligibility screen was performed here. Optimizer or
sampling convergence does not establish absence of model-family systematics.

## Recommended bounded next work

1. Keep the present HST qualification work and audit the three in-range Foundry
   references as a small possible same-telescope extension.
2. Obtain the 157-row SuGOHI supplement and per-system GOOD/POOR identities. Check
   published YattaLens rows and conventions; keep CNN predictions outside reference
   labels. Quality rules must precede our CNN outcomes.
3. Retrieve science images, PSFs and coverage for 3–5 development-only HSC lenses
   with classical references. Resolve band choice, flux scale, native geometry and
   support before extending the acquisition to the whole reference subset.
4. Design HSC as a separate observational domain. Reuse checkpoint inference and
   lens-paired analysis after explicit preparation validation, using a separate
   config/output identity. Do not simply append it to the HST manifest or silently
   retrain the existing arms at HSC resolution.
5. Coordinate/alias-match against all HST candidates, all development exclusions,
   and overlapping SuGOHI papers. Count physical lenses, not bands, catalogues,
   alternative fits or sources behind the same deflector.

If a calibrated HSC-specific C/G/R training comparison is desired later, specify
that as a new experiment with matched budgets and an untouched HSC cohort.

## Saved evidence

- `sources/sugohi_catalog_2026-11-13.csv`: original public CSV bytes.
- `sources/sugohi_index_2026-11-13.html`: query documentation.
- `sources/sugohi_small_query_2026-11-13.html`: column/PNG-link check.
- `sources/gawade_2404_18897v3.html`: inspected article/table source.
- `sugohi_candidate_pool_2026-11-13.csv`: 493-row catalogue-only screen.
- `sugohi_gawade_grade_a_references_2026-11-13.csv`: 25 classical table rows.
- `sugohi_screening_2026-11-13.json`: counts, schema, hash, access and scope limits.
- `sugohi_review_access_2026-11-13.json`: source identities and targeted searches.

No qualified observational sample increase has yet been established.
