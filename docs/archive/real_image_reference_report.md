# Real-image references and simulation compatibility

Research report prepared October 26, 2026, before implementation of the new
simulation layer. This is a dataset-selection and compatibility report, not a
real-image accuracy result. Downloaded evidence and measured implementation
status are distinguished from recommendations below.

## Recommendation

Use three tiers rather than treating all new images as a single benchmark:

1. **SLACS/HST engineering pilot:** start with SDSS J1430+4105, for which a
   small public science/noise/PSF bundle and a catalog radius can be matched.
   Its missing radius uncertainty and unresolved target conversion prevent
   confirmatory accuracy scoring until independently reviewed.
2. **SLACS/BELLS-GALLERY reference benchmark:** acquire reviewed individual
   model results and original/reduced science FITS; keep effective radii
   separate from SIE parameters and retain per-system uncertainty and flags.
3. **Euclid VIS external-telescope extension:** prioritize confidently modeled
   galaxy-scale lenses with public Q1 imaging. Audit reference uncertainties
   and radius conventions before inclusion. SL2S/CFHT is another useful
   ground-based route, especially for lenses with higher-resolution references.

The best documented reference is not automatically the easiest complete
download. A public model-results archive may contain plots and summaries
without the science images or derived-radius posterior needed for scoring.

## Candidate assessment

### SLACS catalog and a concrete pilot image

Sonnenfeld's 2024 analysis provides a public `SLACS_table.cat` for 59 lenses.
The paper identifies its angular radii as based on SIE fits to HST imaging
and discusses few-percent measurement precision, while neglecting radius
uncertainties for its own population analysis. The downloaded table itself
has no individual Einstein-radius uncertainty column. Do not reinterpret
that omission as zero uncertainty or assign a blanket uncertainty as though
it were measured for each system.

Sources: [paper](https://doi.org/10.1051/0004-6361/202451341),
[data deposit](https://zenodo.org/records/13366012).

The actual downloaded row for **SDSSJ1430+4105** contains:

| Field | Value |
|---|---:|
| Right ascension | 217.517083 degrees |
| Declination | 41.099194 degrees |
| Lens redshift | 0.285 |
| Source redshift | 0.575 |
| Catalog angular Einstein radius | 1.52 arcsec |
| Individual radius uncertainty | Not supplied in this table |

The PyAutoLens workspace exposes `data.fits`, `noise_map.fits`, and `psf.fits`
in `dataset/imaging/slacs1430+4105/`. Its accompanying script specifies 0.05
arcsec/pixel. The repository tree was inspected at commit
`dd9c54ed8d6680d04d0d2404b5bb68180c791fce`. File availability is verified;
instrument, filter, units, centering, processing history, and lens-light
subtraction still need header/provenance inspection. The companion script
contains generic text referring to simulated data; the directory name alone
is insufficient to resolve that provenance ambiguity.

Source: [workspace image bundle](https://github.com/PyAutoLabs/autolens_workspace/tree/main/dataset/imaging/slacs1430%2B4105).

Treat this as an **ingestion and preprocessing pilot**, not a fully qualified
accuracy benchmark. Use unmodified science data where possible; do not use
the fit's model image or its PNG visualization as an observation. Its radius
falls inside our 0.5--2.0 arcsec training range, but the source morphology and
photometric calibration still differ from our synthetic data.

Further reference search located a second measurement for the same object:
Etherington et al. (2023), Table 1, reports **1.481 +0.002/-0.002 arcsec**,
alongside the older 1.52-arcsec literature value. This is an effective-radius
reference from their more flexible mass modeling, not an uncertainty to attach
to the older SIE number. The implementation preserves it as a separate
comparison reference. Its quoted precision does not establish target
equivalence or include every systematic uncertainty.

Sources: [published table](https://academic.oup.com/view-large/401449642),
[paper](https://doi.org/10.1093/mnras/stad582).

### SLACS/BELLS-GALLERY detailed modeling references

Etherington et al. analyze 59 HST lenses with power-law mass models plus
external shear and derive effective Einstein radii with posterior-based
uncertainties. Their reported mean uncertainty is approximately 1%, but varies
substantially by lens; it excludes all possible mass-model systematics. Their
comparison with older SIE measurements has materially larger scatter than
that formal precision. Keep both the reference definition and individual
quality information.

Source: [published analysis](https://doi.org/10.1093/mnras/stac2639).

The associated 2.7-GB deposit explicitly lists model priors, parameter estimates
and errors, and diagnostic PNGs; full sampling chains are excluded. The
metadata does not establish that all required science/noise/PSF FITS or
effective-radius posterior samples are available. Avoid downloading the full
archive under the assumption that it is a ready-to-score imaging benchmark.

Source: [model-results deposit](https://zenodo.org/records/6104823).

Recommended use: per-system cross-checks against the older SIE reference,
followed by a selected, reviewed benchmark once science products and derived
reference errors have actually been retrieved. HST data outside CASTLES are
an independent sample, but not automatically a different telescope.

### SL2S: CFHT images with published lens models

Sonnenfeld et al. provide galaxy-scale SL2S lens models using HST and CFHT
data. The paper covers 56 candidates rather than 56 automatically eligible
single-lens references. VizieR exposes a lens-model table, making this a
promising route to cross-match ground-based imaging with published radii.
Candidate grades, multiple components, model success, and image provenance
must be checked per object.

Sources: [original lens-model paper](https://arxiv.org/abs/1307.4764),
[VizieR catalog](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/777/97).

For an external-telescope experiment, favor CFHT observations of a lens with
an independently reviewed higher-resolution model. Do not pool HST and CFHT
views as independent systems. A smaller labeled cross-match is more useful
than hundreds of candidates with no compatible reference.

### Euclid Q1: a genuine different-telescope target

Rojas et al. describe automated isothermal lens modeling of visually selected
Euclid candidates and report radii for successfully modeled systems judged to
be lenses. This provides a better starting point than an unmodeled discovery
catalog. Successful model fitting alone is not confirmation, and candidate
grades and quality review remain essential.

Source: [Euclid modeling paper, section 5](https://arxiv.org/html/2503.15325v1#S5).

IRSA provides programmatic Q1 FITS access and cutout services. This makes
Euclid VIS an actionable imaging source, but a cutout endpoint is not a
reference-radius table. Obtain the exact model-table row, per-object error
information, and science/noise/PSF metadata before scoring.

Sources: [Q1 archive overview](https://irsa.ipac.caltech.edu/data/Euclid/docs/overview_q1.html),
[FITS cutout service](https://irsa.ipac.caltech.edu/ibe/cutouts.html).

The initial implementation should register this source and its outstanding
requirements without inventing per-system measurements or claiming an
already assembled Euclid benchmark.

## Compatibility requirements

The current CNN predicts a Lenstronomy SIE `theta_E` parameter. References
can instead report an SIE parameter under another normalization, a radius
derived from the area inside a tangential critical curve, an aperture-based
mass radius, or an arc-separation proxy. Even similar numeric values do not
justify treating them as the same supervised quantity.

Require a documented convention mapping and validate it on synthetic lenses
before using real-reference residuals. For external shear, nonisothermal
profiles, or multiple deflectors, a single global conversion may not exist.
Keep the reference radius in a separate field; never fill simulated
`theta_E_arcsec` truth for real observations.

The following gates should be explicit in configuration and code:

| Gate | Evidence required |
|---|---|
| Identity | Stable lens ID and coordinate cross-match to the reference. |
| Image provenance | Real science FITS, source URL/checksum, HDU, instrument/filter, calibration and units. |
| Geometry | Pixel scale, lens center, crop transform, and adequate angular coverage. |
| Noise and PSF | Supplied uncertainty semantics and kernel provenance; missing products remain missing. |
| Reference | Published model family, radius definition, uncertainty and quality assessment. |
| Target compatibility | Reviewed conversion to the existing target or a separately defined estimand. |
| Independence | One statistical group per lens, including alternate filters/telescopes and reused templates. |

Our training scenes contain smooth host light plus lensed point sources.
Many candidate real benchmarks contain extended galaxy arcs without a bright
point source. That is an explicit morphology shift, not merely a telescope
change. Flux calibration and the training-only normalization also require
attention: rescaling an arbitrary real image until predictions look plausible
would be evaluation-driven tuning.

## What the reference requirement means

Real-image inference is technically possible without labels. Accuracy scoring
is not. If the CNN predicts 1.10 arcsec and a compatible independent analysis
reports 1.00 +/- 0.05 arcsec, we can report a 0.10-arcsec residual against an
uncertain measurement. We cannot claim an exactly known physical error.
Reference posteriors and systematic differences should inform interpretation.
Do not divide by seed-ensemble spread and call the result calibrated coverage.

Unlabeled real images remain useful for data-quality, observable-coverage,
and repeated-observation consistency studies. Those are distinct outcomes.

## Configuration and implementation decision

After this report, implement a real-image intake layer that can download and
inspect an explicit pilot bundle, preserve checksums and references, and
produce a readiness report. Missing uncertainty or unreviewed target mapping
must prevent an accuracy-ready status. A missing FITS unit must not be filled
with a simulated electron-rate assumption.

For the simulation extension, use an optional pinned PyAutoLens environment.
Build a zero-shot development renderer with explicit physical conventions and
independent light/PSF/noise code paths where supported. Start with analytic
SIS controls and SIE deflection comparisons. Keep the source-only/point-source
distinction explicit, and do not label an extended-only scene an exact match
to the original point-plus-host release.

Do not score either CNN on the new real or synthetic target domain during
integration. A new protocol and freeze must precede confirmatory evaluation.
The first implemented release is a development smoke/control dataset, not
the new blind test. Dependencies, actual downloads, passed controls, and
remaining blockers will be recorded separately after implementation.
