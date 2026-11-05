# Reproducibility statement

## Frozen final evaluation

- **Current protocol/config locations:** `archive/docs/reliability_study/protocol_v1.11.md` and `configs/reliability/blind_holdout.yaml`. These relocated maintenance files are distinguished from the original frozen paths by the [migration record](../../maintenance/cleanup_v1.md).
- **Freeze manifest:** [`week11_freeze_manifest.json`](../../../archive/docs/reliability_study/week11_freeze_manifest.json), SHA-256 `01a2c64e81cd9813f22988293f6aa992ade8fa8c67bdd652e8da218c3706da79`.
- **Holdout:** 150 independent parents; seed 11031; 750 rows in canonical, pixel-scale 0.04, pixel-scale 0.06, half-exposure, and double-read-noise conditions.
- **Models:** fixed checkpoint seeds 12001/12002/12003 for both ensembles; checkpoint and normalization hashes are recorded in the freeze manifest and prediction files.
- **Bootstrap:** 10,000 parent resamples, seed 6031.
- **Environment:** Python 3.13.6, PyTorch 2.8.0+cpu, CPU, 6 Torch threads.
- **Prediction provenance:** 4,500 per-checkpoint rows and 1,500 ensemble rows were saved target-free and hashed before the single label join. The completion marker certifies one scored pass. Do not rerun holdout inference.
- **Quarantine:** no historical test data or real-parameter labels were accessed; no post-holdout model selection or tuning occurred.

## Regeneration

The historical protocols are under `archive/docs/reliability_study/`. Exact source/environment reproduction requires the recorded original checkout; this maintenance checkout has changed paths and hashes. The original Week 11 sequence was:

```text
python -m src.ml.reliability_week11 generate
python -m src.ml.reliability_week11 preflight
python -m src.ml.reliability_week11 evaluate
```

The `evaluate` command is exactly-once for this release. Regenerate tables, figures, and documents from the saved result files with:

```text
python archive/scripts/build_reliability_final_package.py
```

The freeze and package manifests record the relevant code, protocol, config, release, checkpoint, normalization, prediction, and output hashes. The package does not rerun model inference.

The archived package builder retains historical path assumptions. Use its
matching checkout and saved artifacts; maintenance has not regenerated the
published analysis package.
