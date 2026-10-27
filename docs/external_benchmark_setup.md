# External reference intake and alternate simulator development

Implementation follows the [real-image reference report](real_image_reference_report.md),
which was written before the new simulation code. Both workflows are isolated
from the original frozen holdout. Neither runs CNN inference or trains models.

## Real-image layer

Configuration: [`real_reference_pilot_v1.yaml`](../configs/real_reference_pilot_v1.yaml).
Implementation: [`external_reference.py`](../src/data/external_reference.py).

```powershell
python -m src.data.external_reference fetch
python -m src.data.external_reference prepare
```

The fetch command verifies the catalog's SHA-256 and the image bundle's Git
blob hashes against a pinned source commit. Existing files are rechecked,
not silently replaced. The prepare command inspects native FITS arrays,
preserves supplied science, squares a declared standard-deviation map into
variance, and writes a normalized copy of the supplied PSF. It records the
original PSF sum and leaves the source files unchanged.

Products stay at native geometry and flux scaling. A later versioned
preprocessing policy must define lens centering, field coverage, instrument
units and any angular resampling before model evaluation. This pilot does
not crop/pad and scale the input until the CNN output looks plausible.

The downloaded J1430+4105 bundle has 151-by-151 science/noise arrays and a
nominal 7.55-arcsec pixel-extent field. Its headers supply no science units,
instrument, filter, exposure, or object metadata. The input PSF sum is about
0.95341, so the staged kernel is explicitly normalized. These are important
provenance limitations even though the arrays pass structural checks.

The primary catalog radius is 1.52 arcsec, without an individual uncertainty
in that catalog. A separately cited effective-radius estimate is 1.481
+0.002/-0.002 arcsec. They are retained as separate measurements; no radius
is written as exact simulator truth. The sample currently has zero
accuracy-ready observations because unit/provenance/geometry/reference
reviews and target mapping remain incomplete.

Inspect:

- `data/derived/actual/real-reference-pilot-v1/reference_intake.csv`
- `results/reports/real-reference-pilot-v1/readiness.json`
- `data/raw/external/real-reference-pilot-v1/download_manifest.json`

The small intake table and readiness report are versionable; downloaded FITS
and staged arrays remain ignored. Euclid and SL2S access/reference candidates
are registered in configuration, but no complete labeled sample from those
surveys has been downloaded or declared ready.

## Alternate simulation layer

Configuration: [`cross_simulator_development_v1.yaml`](../configs/cross_simulator_development_v1.yaml).
Implementation: [`autolens_backend.py`](../src/simulation/autolens_backend.py)
and [`cross_simulator.py`](../src/simulation/cross_simulator.py).

Use a separate **Python 3.12** environment on this Windows machine:

```powershell
# With a Python 3.12 interpreter available:
python3.12 -m venv .venv-autolens
.venv-autolens\Scripts\python.exe -m pip install -r requirements-cross-simulator.txt
.venv-autolens\Scripts\python.exe -m src.simulation.cross_simulator controls
.venv-autolens\Scripts\python.exe -m src.simulation.cross_simulator generate
```

The installation attempt under system Python 3.13 failed because the backend's
`xxhash<=3.4.1` dependency needed unavailable Microsoft C++ Build Tools.
A project-local Python 3.12 runtime was downloaded under `.python-crosssim/`.
Its environment is independent of the existing Python 3.13 research packages.
The environment manager's version-link step failed on this filesystem; the
downloaded interpreter can still be addressed by its full path.

The adapter uses PyAutoLens mass and light profiles, with explicit project
code for supersampled pixel integration, point-root solving, point-image
depositing, PSF construction/convolution, and independent detector draws.
It does **not** claim to use PyAutoLens `SimulatorImaging`, and does not call
the frozen Lenstronomy renderer. Shared components include the original
parameter proposal, independent analytic SIE control equations, and common
preprocessing. The code records this boundary.

The mass-profile gate checks four axis ratios and three orientations at
offset centers, comparing external deflections to analytic SIE equations.
The pinned backend's `PowerLawIntermediate` source documents that at slope
two, `theta_Lenstronomy = 2*sqrt(q)/(1+q) * theta_PyAutoLens`. The adapter
therefore converts the radius before constructing `Isothermal`. Identical
radius numbers would otherwise represent different elliptical mass models.
Only after this gate passes can generation begin. Each generated parent is
rendered under canonical, finer-pixel, half-exposure, and Moffat-PSF conditions.
The Moffat family tests a new PSF shape at the same FWHM; it is not an empirical
HST or Euclid instrument model. Point images are solved from external
deflections rather than copied from stored Lenstronomy roots.

Generation writes a **development manifest**, parameters, raw/processed
images, expected variance, PSF, point-image diagnostics, hashes, failures,
and environment versions. It refuses nonempty release output directories.
Rejected observations are retained in a failure ledger. Every row explicitly
has `accuracy_evaluation_eligible: false`: it is not advertised as a validated
drop-in release for the existing strict ML loader.

## Remaining scientific gates

Before a confirmatory cross-simulator experiment:

1. Validate mass normalization and coordinate conventions against the frozen
   Lenstronomy implementation as well as the analytic control.
2. Establish Sersic amplitude/effective-radius mappings using noiseless light
   profiles and integrated-flux controls; copying parameters is insufficient.
3. Compare independent point-image counts/positions, pixel integration and
   boundary behavior; test repeated-noise moments against expectation.
4. Add a strict manifest adapter and validate parent/template isolation against
   development data. The four-parent seed here is only an engineering choice.
5. Resolve the earlier recorded training-run verification failures before
   additional training; implement matched-update training controls separately.
6. Freeze models, normalization, condition matrix, sampling, metrics and final
   scoring protocol before generating a new independent final sample.

For real-image scoring, independently review source provenance, calibrated
units, lens center and field coverage, the reference measurement and its
uncertainty, and its relationship to the CNN's SIE target. Preserve missing
information rather than substituting simulator quantities.
