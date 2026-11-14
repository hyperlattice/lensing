# HST compatibility development pilot

This pilot asks whether public HST images can be used faithfully with the current
C/G/R Einstein-radius estimators. It audits inputs; it does not measure prediction
accuracy or change the existing synthetic experiment.

## Fixed sample and independence

[The manifest](../configs/hst_compatibility_pilot.json) fixes five Dinos-I systems:
three SLACS, one SL2S, and one BELLS. Selection uses catalogue order and optical
band availability, includes the previously inspected example, and precedes any
CNN prediction. This is a convenience sample for diagnosing compatibility, not a
representative population or a powered model comparison.

Each lens system is one independent unit. These five IDs are reserved for
compatibility development and must be excluded from a subsequent untouched HST
confirmation cohort, including other bands or releases of the same lenses.
The manifest records this exclusion; a future observational evaluation loader
must enforce it. The existing synthetic final-data loader is unchanged.

## Ownership and invariants

`scripts/hst_pilot.py` owns acquisition, structural inspection, and diagnostic
outputs. It reuses `src.data.schema.contained_path` for bounded local paths.
The fixed JSON manifest owns sample IDs, source commit, Git blob identities,
reference values and table hashes, and comparison geometry. `h5py` is an optional
`hst` dependency; no new training dependency or preprocessing framework is needed.

- Preserve native science values, shapes, and coordinate transforms. Do not
  resize, pad, recenter, background-subtract again, or fit normalization on HST.
- Verify both downloaded and cached bytes against the pinned Git blob identity.
- Normalize a separate PSF array by its sum and record the original sum. Keep
  source files unchanged.
- Keep reference labels in metadata, outside the exported image arrays.
- Preserve supplied masks without assigning unverified mask semantics.
- Keep array validation separate from scientific qualification for inference.
- Do not produce predictions, choose CNN checkpoints, or rank C/G/R.

The raw files are ignored under `data/raw/external/hst_compatibility_pilot/`.
Derived native arrays and the preview live under the report's `products/` and
`preview.png`; compact JSON/Markdown reports remain reviewable.

## Run and regenerate

Use the project environment, with the HST extra if it is missing:

```powershell
python -m pip install -e ".[hst,dev]"
$env:NUMBA_DISABLE_JIT = "1"
python -m scripts.hst_pilot
```

The current `.venv` already provided h5py and the other required packages; no
installation was necessary for this pilot. Run without network after acquisition:

```powershell
python -m scripts.hst_pilot --offline
python -m pytest tests/test_hst_pilot.py -q
```

The default output is `results/reports/hst_compatibility_pilot/`. Explicit
`--manifest`, `--data-root`, and `--output` arguments select alternate locations.
The script reads at most 5 MB per asset and never downloads the complete Dinos
repository. It records input, manifest, implementation, and native-product
hashes, package versions, and the numerical convention control. Failed source
hashes or invalid arrays raise errors; unresolved scientific compatibility is
reported with `model_input_ready: false`.

## Executed pilot: 12 November 2026

Five systems and eleven source assets were acquired and audited successfully.
The preview was visually inspected. Native arrays are finite, exposure metadata
is positive, transforms are nonsingular, and every PSF has positive total flux.

| System | Band | Native shape | Approximate field edges |
|---|---|---|---|
| SDSSJ0008-0004 | F606W | 130 × 130 | 6.47 × 6.47 arcsec |
| SDSSJ0029-0055 | F606W | 140 × 140 | 6.96 × 6.97 arcsec |
| SDSSJ0037-0942 | F606W | 140 × 140 | 6.96 × 6.97 arcsec |
| SL2SJ0214-0405 | F606W | 120 × 120 | 6.00 × 6.00 arcsec |
| SDSSJ0151+0049 | F814W | 110 × 110 | 5.50 × 5.50 arcsec |

Pixel scales are approximately 0.04976–0.05000 arcsec. Original PSF sums range
from 0.97468 to 0.99211. SL2SJ0214-0405 supplies a mask with 146 nonzero pixels;
it has not been applied. No mask is supplied for the selected bands of the other
four systems; absence of a supplied mask does not establish clean coverage.

The convention control compared lenstronomy SIE and EPL deflections at slope two
over nine radius/ellipticity combinations and four positions. Maximum component
difference was approximately 3.45e-10 arcsec, below the declared 1e-6 tolerance.
This supports matching the parameter convention in that limit. It does not
establish that observed EPL fits or their errors are interchangeable with SIE
references.

Inspect the generated [audit](../results/reports/hst_compatibility_pilot/audit.md)
and [machine-readable report](../results/reports/hst_compatibility_pilot/audit.json).
All systems remain unqualified for CNN scoring.

## Wider-image follow-up: one executed archive probe

The fixed manifest now includes a separate archive probe for the already reserved
SDSSJ0037-0942 system. This does not add a final-evaluation system or replace the
original Dinos cutout. MAST observation `208003580` identifies the F606W WFPC2/PC
data from proposal 10798. The product is
`hst_10798_89_wfpc2_pc_f606w_u9qc89_drz.fits`.

`scripts/hst_archive.py` retrieved only the FITS headers and native SCI/WHT row
ranges needed for a wider cutout. About 11.5 MB was retrieved, including header
probes, from a 166.7 MB product. Exact byte ranges, their SHA256 identities, ETags,
and modification times are recorded and cached; this is not a whole-file hash.
Offline replay verifies the cache. Source SCI/WHT grids must have matching WCS.

The result is a **282 × 282 native cutout at 0.045536 arcsec/pixel**, covering
**12.841152 × 12.841152 arcsec**. All science pixels are finite and all weight
pixels are positive. No interpolation, padding, unit conversion, background
subtraction, or recentering was applied. FITS checksums and the cropped WCS were
read back successfully, and the preview was visually inspected.

This demonstrates a solution to the field-size problem for one system. Positive
weight coverage does not prove that pixels are free of artifacts, and WHT is not
assumed to be an inverse variance map. The separately processed Dinos PSF was
not transferred to this different archive grid.

The header's `BUNIT=counts` is incomplete on its own: the drizzle output cards
specify `cps`, the instrument is WFPC2, and `ATODGAIN=7` records electrons per DN.
The [HST data handbook](https://hst-docs.stsci.edu/hstdhb/4-hst-data-analysis/4-6-analyzing-hst-images)
distinguishes WFPC2 drizzled DN/s from electron-rate data. Together these support
DN/s for this archive product, with a candidate factor of seven to electron/s.
No conversion was applied, and these headers do not establish the units of the
separately drizzled Dinos HDF5 files. Matching photometric scales and background
to training remains a separate requirement.

The catalogue coordinate falls near pixel (140.62, 140.95), whereas the brightest
supported pixel is at (127, 132), approximately 0.74 arcsec away. This is a
diagnostic, not a lens-center estimator. Catalog astrometry, foreground-center
identification, and the adopted centering rule still need review.

To regenerate this probe without network:

```powershell
python -m scripts.hst_archive --offline
```

The [archive audit](../results/reports/hst_compatibility_pilot/archive/audit.json)
records the source, coverage, unit evidence, centering diagnostic, and remaining
qualification gates. Its native FITS and preview are generated local products.
Next work is to obtain a compatible PSF and resolve centering/photometry for this
one wider image before extending the route to the other four systems.

## What must be resolved before inference

1. **Field of view:** these cutouts cover less than the canonical 12.8 arcsec
   training field. Native-shape support in the CNN does not establish invariance
   to this difference. Obtain wider HST cutouts, or explicitly design and validate
   a matched-field synthetic control before interpreting observational errors.
2. **Photometry and background:** the pinned Dinos preprocessing notebook saves
   `data_cutout - mean`, plus background RMS and an exposure map. Avoid repeating
   background subtraction. Trace original FITS units, gain, drizzle processing,
   and photometric calibration before applying training-only normalization.
   An exposure map alone does not prove electron-per-second units or a calibrated
   full noise model.
3. **Centering, masks, and coverage:** review coordinate origins, foreground-lens
   location, contaminating objects, supplied mask meaning, and unobserved pixels.
   Do not assume the brightest pixel is the correct lens center.
4. **Reference and source mismatch:** Dinos references use EPL plus shear and
   galaxy–galaxy sources. Our training includes SIE and lensed point sources.
   Keep these as declared transfer differences or adopt a separately specified
   compatible cohort/control; do not silently revise the original C/G/R study.

The [real-data evidence review](literature_review/real_telescope_data/README.md)
also identifies a small lensed-quasar alternative, currently provisional because
its correction and archive-version alignment remain unresolved.
