# Lensing

This repository builds an auditable strong-gravitational-lens dataset for angular mass-model inference. The pre-ML pipeline is complete: it preprocesses the local CASTLES observations, generates deterministic SIE simulations, validates classical recovery, creates ancestry-safe splits, and exposes a truth-isolated loader. No machine-learning architecture or training policy is selected here.

Current frozen artifacts:

- `real-castles-v2`: 137 observations from 69 lens systems; structural preprocessing passed, with calibration limitations retained as review states.
- `sim-eng1000-v1`: 1,000 independent simulated parents; 700/150/150 train/validation/test; zero rejected rows and zero validation errors.
- P10/P30/P60/P100: staged analytic, extended-source, noisy-observation, and stress controls.
- Classical recovery: 13/13 optimizer successes across six conditions; noiseless `theta_E` error below 0.033%, with low-SNR degradation reported separately.
- Real demonstration: three preselected systems agree with compatible astrometric/model radii within 6.4%, explicitly not a calibrated posterior or blind population test.

See [the codebase guide](docs/codebase_guide.md), [dataset contract](docs/dataset_schema.md), and [ML handoff](docs/ml_handoff_v1.md) for the full design and current boundary.

## Setup

Use Python 3.11 or newer in a virtual environment. The recorded release was built with Python 3.13.6.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-simulation.txt
```

The real-data build requires the local CASTLES FITS tree at `data/raw/castles`. Large generated products under `data/simulations` and release product arrays are intentionally ignored by Git; their compact manifests and evidence are tracked.

## Unified commands

Show every command:

```powershell
python -m src.pipeline --help
```

Build the versioned real release:

```powershell
python -m src.pipeline prepare-real --release-id real-castles-v2 --output data/releases/real-castles-v2
```

Generate a small deterministic simulation for development:

```powershell
python -m src.pipeline simulate --stage observation --output data/simulations/example-v1 --simulation-id example-v1 --count 10 --seed 6001 --config configs/simulation_pilot_v1.yaml --no-preview
python -m src.pipeline validate data/simulations/example-v1/manifest.csv --root data/simulations/example-v1
python -m src.pipeline make-splits data/simulations/example-v1/manifest.csv --output data/simulations/example-v1/splits --ancestry
```

Reproduce the pre-ML evidence:

```powershell
python -m src.pipeline fit-grid
python -m src.pipeline real-comparison
python -m src.pipeline engineering-release
```

Release directories are protected against accidental replacement. Use a new versioned ID for a new release; `--overwrite` must be explicit when intentionally rebuilding an existing location.

## Load a frozen split

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

`inputs.metadata` uses a positive allowlist. IDs, seeds, ancestry, source/lens truth, target parameters, and truth-sidecar paths remain only in top-level provenance and cannot silently become model features.

## Verification

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-work
python -B -m src.testing.validate_simulations data/simulations/sim-eng1000-v1/manifest.csv
```

The release checksum ledger is `data/simulations/sim-eng1000-v1/checksums.sha256`; its tracked audit copy is under `data/manifests/sim-eng1000-v1`.

## First machine-learning baseline

Install PyTorch and run the frozen image-only Einstein-radius experiment:

```powershell
python -m pip install -r requirements-ml.txt
python -m src.ml.train --config configs/ml_baseline_v1.yaml
python -m src.ml.evaluate --checkpoint results/ml/theta-e-cnn-baseline-v1/best.pt --split test
```

The first CPU run reached test MAE `0.0876` arcsec, median relative error `4.93%`, and R² `0.9127` on the in-domain synthetic test split. See the [ML quickstart](docs/ml_quickstart.md) to reproduce the run and interpret every output. These metrics do not establish accuracy on real observations.

## Scientific boundary

The engineering simulations use analytic Sersic sources, Gaussian PSFs, scalar backgrounds, independent Poisson/read noise, and hand-set uniform priors. The real release lacks empirical PSF, ERR, and DQ products and currently uses a toy variance fallback. The project is ready for an ML experiment plan and synthetic in-domain baselines, but not for observational mass claims or population-level validation.
