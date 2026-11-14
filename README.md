# Synthetic lens robustness study

This repository studies whether varying observing conditions improves an image-only
CNN's Einstein-radius estimates on new synthetic lens systems. The maintained
comparison uses canonical (C), uniform variation (G), and source-development
failure-weighted variation (R), with paired seeds and identical optimizer budgets.
The secondary experiment compares paired Lenstronomy/AutoLens image products.
These experiments do not establish real-telescope parameter accuracy.

## Setup

Use Python 3.12 for the pinned optional AutoLens stack. The primary CPU workflow
is also tested locally on Python 3.10. Install from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
# Only when running the secondary physical backend:
python -m pip install -e ".[autolens]"
```

## Run the study

```powershell
python -m scripts.study develop reference
python -m scripts.study develop weights
python -m scripts.study train
python -m scripts.study predict
python -m scripts.study analyze
python -m scripts.study plot
```

Every action is explicit: training never chains into holdout inference or scoring.
`develop tune` and `develop review` are optional source-only development actions;
they do not modify the final study configuration or reopen the holdout.

**All existing data and result destinations are preserved.** Primary results remain
under `results/ml/equal-budget-final-v1/`; policy development remains under
`results/ml/robustness-development-v4/`; the secondary comparison remains under
`results/ml/cross-simulator-rapid-v2/`. Existing artifacts are not moved or rewritten.
Required banks, private image arrays and checkpoints are not necessarily available
in a code-only checkout. See [reproduction](docs/reproduction.md).

## Layout

```text
configs/             Shared synthetic-study profiles, simulation and HST settings
src/data/            Stored image contracts, bank loading and qualification
src/simulation/      Scene sampling, one renderer, backend adapters and generation
src/ml/              Model, batches, policies, training, prediction, analysis and plots
scripts/             Study command, repository check and submission export
tests/              Small temporary fixtures and pre-refactor regression controls
data/               Existing datasets, source labels and audit manifests; unchanged
results/            Existing output hierarchy; unchanged
docs/               Current design/reproduction guides and preserved historical records
submission/         Canonical manuscript/reviewer bundle; unchanged
archive/            Historical research records; use their original Git checkout
```

Current documentation: [study](docs/study.md), [data contract](docs/data_contract.md),
[rendering](docs/rendering.md), and [migration](docs/migration.md).
The proposed HST transfer workflow is documented in [HST implementation](docs/hst_study.md).
The manuscript is [paper.tex](submission/aaai27/reviewer_bundle/paper.tex).

## Checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -B -m pytest -q -p no:cacheprovider
python -m scripts.check_repository
python -m scripts.submission check
```

Tests use operating-system temporary storage. They never train on production
banks, score production holdouts, or accumulate `.pytest-work*` folders.
See [contributor guidance](CONTRIBUTING.md) before changing scientific behavior.
