# Source-code map

`src` is organized by responsibility. `pipeline.py` is the public command-line
entry point; `main.py` is a compatibility wrapper for the same CLI. The code
uses the repository root as its working directory, so paths in commands and
configs are repository-relative.

| Directory or file | Responsibility |
|---|---|
| `layout.py` | Safe, shared locations for generated review previews. |
| `data/` | Manifest schema and the truth-isolated `LensDataset` loader. |
| `processing/` | FITS metadata, catalog joins, geometry, masks, backgrounds, noise, normalization, preprocessing, previews, and split creation. |
| `simulation/` | Priors, lens-model adapters, rendering, observation effects, release generation, and stage assembly. |
| `modeling/` | Classical recovery fits and the bounded real-data angular comparison. |
| `ml/` | PyTorch datasets, deterministic training, evaluation, metrics, plots, provenance, and registered experiments. |
| `testing/` | Command-line validation helpers for datasets and simulations. Unit and integration tests live in top-level `tests/`. |

The dependency direction is deliberate: `data` defines the contract;
`processing` and `simulation` create contract-compatible products;
`modeling` and `ml` consume them. `pipeline.py` coordinates these modules but
does not contain scientific calculations itself.

Use these entry points from the repository root:

```powershell
python -m src.pipeline --help
python -m src.ml.train --config configs/ml_baseline_v1.yaml
python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1
```

When adding code, place reusable logic in the owning package, add a focused
test under `tests/`, and keep command parsing at a module `main()` or in
`src/pipeline.py`. Do not put large generated arrays, raw observations, model
checkpoints, or interactive scratch work in `src`.
