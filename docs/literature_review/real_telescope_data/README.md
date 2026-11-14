# Real telescope data for C/G/R evaluation

Checked 12 November 2026. This is a targeted data-access and compatibility review,
not an exhaustive systematic review. No CNN was trained or scored.

**13 November update:** [SuGOHI and NeuraLens sample-size review](sugohi_neuralens_review_2026-11-13.md)
screens the downloaded SuGOHI catalogue, extracts 25 published classical reference
rows, and identifies a newer HST modelling route. Candidate counts are not final
eligible sample sizes; HSC is a separate observational domain.

**Implementation follow-up:** the [five-system HST compatibility pilot](../../hst_compatibility_pilot.md)
has now acquired and audited native images and PSFs. All five pass structural
checks and remain unqualified for scoring; field size, photometry, centering,
and reference/source compatibility still need resolution.

**Recommendation:** use Dinos-I HST data for a first compatibility pilot; use Euclid
Q1 as the larger external test after reconciling target definitions and image
preprocessing. These are verified public image-and-reference sources, not yet
final evaluation cohorts. The labels are classical-model estimates with
model-conditioned uncertainty, not known physical truth.

**Closer morphology match found in the final search:** TDCOSMO XXVI provides eight
HST lensed quasars. This is a promising small pilot for our lensed point sources,
but remains provisional pending the published correction and direct image intake.

## What was verified

| Resource | Actual access checked | Candidate support | Decision |
|---|---|---|---|
| [Dinos-I](https://github.com/Project-Dinos/dinos-i) | Downloaded three reference tables and one real HST image/PSF pair; inspected HDF5 contents; enumerated repository image files | 77 reference systems; 76 have median radius in 0.5–2.0 arcsec, positive reported errors, and matching image/PSF files; these 76 have 143 image files | Best first pilot because the inspected HST sampling is close to the simulation |
| [Euclid Q1 discovery engine](https://zenodo.org/records/15025832) | Downloaded mass/catalogue tables; read ZIP directory remotely; extracted one real multi-extension FITS file | 335 mass-reference rows; 283 grade A/B pass the parameter-radius screen, all with archive matches; 277 pass the separate effective-radius screen | Best larger external cohort, conditional on radius mapping |
| [LOS in SLACS](https://arxiv.org/html/2512.05050v2) | Read paper/data statement and Zenodo API metadata | Paper reports 45 successful fits; archive is about 21.5 GB | Useful alternative classical fits and systematic checks; overlaps SLACS, so not 45 additional independent lenses |
| [HSC/SuGOHI, Gawade et al.](https://arxiv.org/html/2404.18897v3) | Read methods/results sections | Paper models 182 real systems with YattaLens | Backup route; raw-image/reference joining and complete classical-label intake remain unverified |
| [TDCOSMO XXVI](https://zenodo.org/records/19421439) | Read selected paper sections, repository README, and Zenodo API metadata | Eight HST doubly imaged quasars; lenstronomy SIE + shear models | Closest point-source morphology; small pilot only; published correction not retrieved |
| Existing CASTLES products | Consulted repository's archived intake report | Earlier local processing lacked validated radius references and calibrated variance | Not currently a scored benchmark; would need a separate label and calibration audit |

The first two counts above were computed from downloaded tables, not copied from
paper abstracts. Multiple observations of one system are not independent samples.

## 1. Dinos-I: accessible HST observations plus classical fits

[Tan et al. (2024)](https://academic.oup.com/mnras/article/530/2/1474/7636508)
release HST modelling inputs and inferred parameters. The repository contains
more imaging directories than successfully reported reference systems; do not
treat all 154 directories as labelled evaluation examples.

The downloaded reference tables contain 34 SLACS, 24 SL2S, and 19 BELLS systems.
Our range screen retains 34 + 23 + 19 = **76 unique systems**. Every retained
system has image and PSF files in the repository tree. The
[candidate manifest](dinos_candidate_references.csv) records individual IDs,
radius estimates, asymmetric errors, observation counts, and commit-pinned image
URLs. The screened set has 143 image files, not 143 independent systems.

The source extraction notebook establishes that `R_Ein` is the posterior median
of `theta_E_lens0`; `R_Ein_e1` is median minus the 16th percentile and
`R_Ein_e2` is the 84th percentile minus median. These fits use an elliptical
power-law mass model and external shear. Their relationship to our SIE target
must be checked; setting the power-law slope to two is a useful synthetic
mapping control, not a justification to treat all observed lenses as SIE.

The downloaded SDSSJ0037-0942 F606W example contains:

- A finite 140 × 140 science array and exposure array.
- A pixel-to-angle transform giving area-equivalent sampling of
  **0.049759 arcsec/pixel**, close to our canonical 0.05.
- Background RMS and a 61 × 61 PSF kernel. Its sum is about 0.99197; an intake
  routine must explicitly handle PSF normalization.

This example spans about 6.97 arcsec, versus our canonical 12.8 arcsec field.
Simply resizing it to 256 × 256 would change the apparent angular scale. Padding
cannot restore unobserved sky. Use sufficient original HST coverage or a
predeclared common field with matched synthetic controls. Image units and full
noise treatment still need tracing through the preparation notebooks/original
FITS headers; the HDF5 root has no unit annotation. One inspected image does not
establish those properties for every observation.

Repository snapshot: `6810ea6d8b7f97e8a7c4699d2b81b5da311c64cb`.
Raw calibrated observations can be traced back through the HST preparation
notebooks and MAST; no bulk MAST retrieval was performed here.

## 2. Euclid Q1: larger cohort with science, RMS, and PSF products

The [public release](https://zenodo.org/records/15025832), version 0.0.3,
provides candidate grades, classical mass fits, and image cutouts. Its `lens.zip`
is 3,051,362,622 bytes. Only the directory and one 2.95 MB decompressed FITS
example were retrieved, not the entire archive.

The checked example has `VIS_FLUX`, `VIS_RMS`, and `VIS_PSF` extensions, alongside
Y/J/H counterparts. VIS science and RMS arrays are 300 × 300; the PSF is 21 × 21.
The WCS gives **0.1 arcsec/pixel**, so the full science cutout spans 30 arcsec.
This establishes that usable numerical images exist, beyond preview PNGs.

Our screen joins `id_str`, requires grade A/B, finite ordered one-sigma bounds,
and a median in 0.5–2.0 arcsec. It retains:

- **283** systems using `einstein_radius` (172 A, 111 B).
- **277** using `einstein_radius_effective` (171 A, 106 B).

All IDs in both screens have matching archive directories. The two lists are
alternative target screens, not additional cohorts to combine. No complete
image-quality or model-fit review has been performed.

The release explicitly distinguishes a scaled PyAutoLens SIE parameter from
the effective critical-curve radius. **Neither column may be assumed identical
to our lenstronomy SIE target.** Verify the conversion using identical synthetic
lenses, ellipticities, and shear conventions before choosing the target and
final range screen. Marginal percentile bounds alone may not propagate a
nonlinear conversion correctly. Original posterior samples or refitting may be
needed for transformed uncertainty.

Euclid also introduces a larger resolution and bandpass shift. Resampling 0.1
arcsec pixels onto a 0.05 grid does not add angular resolution; preserve the
effective PSF and flux convention. The FITS headers inspected here do not supply
a simple `BUNIT` field, so flux calibration must be resolved before applying the
current training normalization. The release discusses limits of its PSF model.

Use the classical mass-fit table as the reference. Outputs from LEMON or another
CNN are a model comparison, not independent radius truth. Discovery and fit
success select this cohort; report that scope rather than claiming survey-wide
performance. Paper sample sizes and the current deposited 335-row table differ;
freeze and cite the actual release used.

## A defensible experiment for this project

### Optional closer-match pilot: TDCOSMO XXVI

[Brady et al. (2026)](https://arxiv.org/html/2604.24908v3) model eight doubly
imaged quasars using HST imaging and lenstronomy SIE plus external shear. This
matches our simulated point-source-plus-host morphology and mass family more
closely than the large galaxy–galaxy samples. A nearby perturber in one system
and additional light complexity in others still require explicit eligibility.

The [public archive](https://zenodo.org/records/19421439) is
21,252,745,419 bytes. Its description includes cutouts, PSFs, and full-image
modelling outputs, with code in
[doubles_modeling](https://github.com/brady-ryan/doubles_modeling).
Archive metadata and the repository tree were downloaded; its individual images
and reference posteriors were not extracted in this review. Eight systems are
useful for diagnostic case studies, not a precise ranking of C/G/R.

The [journal landing page](https://astro.theoj.org/article/166553-tdcosmo-xxvi-uniform-lens-modeling-of-eight-doubly-imaged-quasars)
links a [correction notice](https://doi.org/10.33232/001c.171886).
Retrieval of the correction failed here. Do not adopt its numerical labels until
the correction and the relationship between the April archive and current paper
are resolved. Consequently this is a promising source, not a verified benchmark.

### Main comparison

1. **Resolve compatibility on 5–10 HST systems.** Check centering, units, angular
   field, masks, normalization, reference conventions, and source morphology.
   Do not use these systems later as untouched confirmation data.
2. **Freeze the intake rules and C/G/R models.** Select systems using declared
   quality/reference rules, without inspecting which CNN performs best. Our
   simulations include lensed point sources as well as extended light; establish
   whether the selected galaxy–galaxy lenses answer the intended transfer
   question or require a separately declared simulation change.
3. **Evaluate all three models on identical remaining systems.** Record signed
   residuals, absolute errors, model-reference differences, exclusions, and
   per-system predictions. Use paired differences for G−C and R−G, with
   uncertainty resampled by lens system. Keep training-seed variability separate.
4. **Add Euclid as a separate transfer domain.** Keep HST and Euclid results
   separate initially. Report observational resolution and morphology shifts;
   do not pool systems until overlaps and target conventions are reconciled.
5. **Check reference dependence.** Where alternative SLACS fits are available,
   examine whether conclusions survive plausible classical-model differences.
   Do not count the same lens twice because it appears in two releases.

This answers: **Do improvements from synthetic observing-condition diversity
persist on selected real lenses, relative to independently fitted references?**
It can strengthen external validation even if G or R fails to improve. It cannot
establish exact physical truth, universal telescope generalization, or that
synthetic realism alone caused a difference when morphology and mass families
also change. The final eligible sample sizes remain unknown.

## Evidence and reproducibility

- [Resource evidence matrix](dataset_evidence_matrix.csv).
- [Screening counts](screening_summary.json), [HST candidate references](dinos_candidate_references.csv),
  and [Euclid candidate IDs](euclid_candidate_ids.json).
- [HST contents](hst_example_hdf5_inspection.json), [HST numerical checks](hst_example_validation.json),
  and [Euclid FITS headers](euclid_example_fits_headers.json).
- [Source manifest with hashes](source_manifest.json) and [access/scope record](access_log.json).

The saved source files are intake evidence, not a complete benchmark. Public
access was verified; record the applicable archive/software reuse terms and
required citations when assembling a redistributed release. No project runtime
dependencies, manuscript, simulation, or training code were changed.
