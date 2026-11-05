# Lensing

This repository builds an auditable workflow for strong-gravitational-lens
images. It prepares actual CASTLES observations, generates controlled
simulations, validates both against one manifest contract, performs classical
recovery checks, and trains an initial image-only model for the simulated
Einstein radius `theta_E_arcsec`.

The project currently supports reproducible synthetic experiments. It does not
yet support calibrated mass measurements or validated performance on real HST
observations.

## Repository map

```text
configs/       Versioned preprocessing, simulation, comparison, and ML settings
data/          Raw inputs, curated labels, releases, and compact audit manifests
docs/          Dataset contract, codebase guide, research protocols, and results guidance
notebooks/     Notebook navigation; historical notebooks are archived
results/       Derived reports, ML artifacts, and domain-separated previews
src/            Reproducible pipeline implementation
tests/          Automated unit and integration tests
CONTRIBUTING.md Coding, documentation, and research reproducibility conventions
pyproject.toml  Shared pytest configuration
.editorconfig   Shared text-file and indentation conventions
```

Each working area has its own guide: [source code](src/README.md),
[data](data/README.md), [configuration](configs/README.md),
[results](results/README.md), [documentation](docs/README.md), and
[notebooks](notebooks/README.md). Contributor conventions are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Data and previews

Actual observations and simulations are deliberately separate.

| Purpose | Canonical location |
|---|---|
| Raw actual FITS observations | `data/raw/castles/` |
| Curated actual-observation labels | `data/labels/source/` |
| Derived actual metadata and catalogs | `data/labels/derived/` |
| Frozen actual preprocessing release | `data/releases/real-castles-v2/` |
| Generated simulation releases | `data/simulations/<release>/` |
| Tracked engineering-release audit mirror | `data/manifests/sim-eng1000-v1/` |
| Actual-observation review previews | `results/previews/actual/<release>/` |
| Simulated review previews | `results/previews/simulated/<release>/` |

Dataset releases keep their products beside their manifests because those
manifests contain release-relative paths. New review images go in
`results/previews/` and never become model inputs. Large raw data, arrays,
previews, and model checkpoints are ignored by Git; manifests, reports,
configurations, and source tables are versioned.

## Current project state

- `real-castles-v2`: 137 observations from 69 lens systems after structural
  preprocessing, with calibration and variance limitations recorded.
- `sim-eng1000-v1`: 1,000 independent simulated parents split into 700 train,
  150 validation, and 150 test parents without ancestry leakage.
- P10, P30, P60, and P100 are staged controls. P100 is a separate stress suite
  and is never training data.
- Classical recovery completed 13 of 13 fixed optimization cases. The
  noiseless control recovered `theta_E` within 0.033%.
- The initial CNN achieved 0.0876 arcsec MAE on the frozen synthetic test
  split. Its three-seed validation benchmark measured 0.09210 +/- 0.00872
  arcsec MAE; the second value is seed-to-seed sample standard deviation.

Read [the codebase guide](docs/codebase_guide.md) for the architecture and
[the ML quickstart](docs/ml_quickstart.md) for model execution and results.
Historical ML planning material is retained in the
[archive](archive/README.md); the active workflow is documented in the
[ML quickstart](docs/ml_quickstart.md).
The completed reliability study and its frozen holdout evidence are indexed in
[the reliability-study guide](docs/reliability_study/README.md).

## Setup

Use Python 3.12 on Windows for the unified dependency set, including the
pinned PyAutoLens backend. Earlier releases used Python 3.13.6 without that
backend; its recorded Python 3.13 installation required C++ Build Tools.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` contains preprocessing, simulation, ML, and alternate
simulator dependencies. The actual-data workflow also needs `data/raw/castles/`.

The dependency files were consolidated on 2026-11-04 without changing their
version constraints. Archived reports and the Week 11 freeze manifest retain
the original filenames and hashes as historical evidence. This maintenance
checkout does not match those frozen dependency hashes, and installation
messages in backend adapters now refer to the unified file. Reproducing the
frozen source and environment requires its original checkout. The unified
environment has not been validated by a fresh installation.

## Main commands

Show all project commands:

```powershell
python -m src.pipeline --help
```

Build a new actual-observation release and its domain-separated previews:

```powershell
python -m src.pipeline prepare-real --release-id real-castles-v3 --output data/releases/real-castles-v3
```

Generate a small simulation and validate its release:

```powershell
python -m src.pipeline simulate --stage observation --output data/simulations/example-v1 --simulation-id example-v1 --count 10 --seed 6001 --config configs/simulation_pilot_v1.yaml
python -m src.pipeline validate data/simulations/example-v1/manifest.csv --root data/simulations/example-v1
python -m src.pipeline make-splits data/simulations/example-v1/manifest.csv --output data/simulations/example-v1/splits --ancestry
```

Use `--preview-root some/path` to put review images elsewhere, or
`--no-preview` when previews are not needed. Release directories are protected
against accidental replacement; use a new ID or pass `--overwrite` only when
you intentionally rebuild a release.

Reproduce the recorded engineering evidence:

```powershell
python -m src.pipeline fit-grid
python -m src.pipeline real-comparison
python -m src.pipeline engineering-release
```

## Load a split safely

The loader exposes arrays and explicitly allowlisted observational metadata.
Simulator truth, random seeds, ancestry, identifiers, and sidecar paths remain
outside model inputs.

```python
from src.data.dataset import LensDataset

dataset = LensDataset(
    "data/simulations/sim-eng1000-v1",
    root="data/simulations/sim-eng1000-v1",
    split="train",
    domain="simulated",
    eligible_for="ml",
    target_fields={"theta_E_arcsec"},
    strict_products=True,
)
sample = dataset.get_sample(0)
image = sample["inputs"]["products"]["image"]
theta_e = sample["targets"]["theta_E_arcsec"]
```

See [the dataset schema](docs/dataset_schema.md) for the complete contract.

## Machine learning

Train the original image-only baseline using training and validation data:

```powershell
python -m src.ml.train --config configs/ml_baseline_v1.yaml
```

After accepting the checkpoint using validation only, evaluate it once on the
locked test split:

```powershell
python -m src.ml.evaluate --checkpoint results/ml/theta-e-cnn-baseline-v1/best.pt --split test
```

Measure seed variation before comparing new inputs, losses, or architectures:

```powershell
python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1
```

The report is written to `results/ml/theta-e-cnn-seeds-v1/report.md`. Use
`--smoke` for a wiring check, `--output` for a new repeat, and `--resume` to
verify and reuse completed runs. The full procedure is in
[the ML quickstart](docs/ml_quickstart.md). Historical planning material is in
the [archive](archive/README.md).

## Verification

Run all automated tests:

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-work
```

Validate the local engineering release when its generated products are
available:

```powershell
python -B -m src.testing.validate_simulations data/simulations/sim-eng1000-v1/manifest.csv
```

The engineering checksum ledger is in
`data/simulations/sim-eng1000-v1/checksums.sha256`; its tracked audit copy is
under `data/manifests/sim-eng1000-v1/`.

## Scientific boundary

The engineering simulations use analytic Sersic sources, Gaussian control
PSFs, scalar backgrounds, independent Poisson/read noise, and hand-set uniform
priors. The actual release lacks empirical PSF, ERR, and DQ products and uses
a labeled toy variance fallback. Results demonstrate in-domain synthetic
learning and reproducible engineering behavior; they do not establish
observational mass accuracy, calibrated uncertainty, or population-level
performance.
