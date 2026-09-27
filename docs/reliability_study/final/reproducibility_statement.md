# Reproducibility statement

## Frozen final evaluation

- **Protocol/config:** `docs/reliability_study/protocol_v1.11.md` and `configs/reliability_blind_holdout_v1.yaml`.
- **Freeze manifest:** [`week11_freeze_manifest.json`](../week11_freeze_manifest.json), SHA-256 `01a2c64e81cd9813f22988293f6aa992ade8fa8c67bdd652e8da218c3706da79`.
- **Holdout:** 150 independent parents; seed 11031; 750 rows in canonical, pixel-scale 0.04, pixel-scale 0.06, half-exposure, and double-read-noise conditions.
- **Models:** fixed checkpoint seeds 12001/12002/12003 for both ensembles; checkpoint and normalization hashes are recorded in the freeze manifest and prediction files.
- **Bootstrap:** 10,000 parent resamples, seed 6031.
- **Environment:** Python 3.13.6, PyTorch 2.8.0+cpu, CPU, 6 Torch threads.
- **Prediction provenance:** 4,500 per-checkpoint rows and 1,500 ensemble rows were saved target-free and hashed before the single label join. The completion marker certifies one scored pass. Do not rerun holdout inference.
- **Quarantine:** no historical test data or real-parameter labels were accessed; no post-holdout model selection or tuning occurred.

## Regeneration

The synthetic study workflows and phase-specific commands are recorded under `docs/reliability_study/` and their frozen protocols/configs. The Week 11 sequence was:

```text
python -m src.ml.reliability_week11 generate
python -m src.ml.reliability_week11 preflight
python -m src.ml.reliability_week11 evaluate
```

The `evaluate` command is exactly-once for this release. Regenerate tables, figures, and documents from the saved result files with:

```text
python scripts/build_reliability_final_package.py
```

The freeze and package manifests record the relevant code, protocol, config, release, checkpoint, normalization, prediction, and output hashes. The package does not rerun model inference.
