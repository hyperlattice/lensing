# Workshop submission reproducibility checklist

Status: packaging complete for the narrow equal-budget synthetic-robustness
claim. This checklist does not certify simulator transfer or real-data
accuracy.

## Scientific identity

- [x] Primary question fixed: does source-only observation variation help when
  CNN/update/source-exposure budgets are matched?
- [x] Arms fixed: C (canonical), G (uniform variation), R (failure-weighted).
- [x] Ten paired training seeds: 730001–730010.
- [x] 880 optimizer updates per fit; batch size 16; 14,080 source exposures
  per arm/seed.
- [x] Source train/development and holdout parent IDs are disjoint.
- [x] Holdout predictions were sealed before the target join.
- [x] Primary endpoint is parent-averaged four-stress MAE.
- [x] Bootstrap resamples independent parents, not correlated condition views.
- [x] R-versus-G is reported as inconclusive rather than selectively omitted.
- [x] The older 27.4% unequal-budget result is labeled historical context.
- [x] Cross-simulator and real-data claims are explicitly excluded.

## Reproduction commands

Run from `D:\lensing` in the source Python environment:

```powershell
python -B -m scripts.run_equal_budget_final_v1 fit
python -B -m scripts.run_equal_budget_final_v1 predict
python -B -m scripts.run_equal_budget_final_v1 score
python -B -m scripts.plot_equal_budget_final_v1
python -B -m pytest -q tests/test_paired_evaluation_images_v1.py tests/test_study_freeze_v1.py tests/test_cross_simulator_v2.py
```

The fit and predict actions are target-free. The score action performs the
one-way target join after the prediction seal exists. Existing output
identities are immutable; a rerun that changes results must use a new study
identity rather than overwrite these artifacts.

## Environment recorded for the equal-budget fits

- Windows AMD64, Python 3.13.6.
- CPU execution; CUDA unavailable; six Torch threads.
- PyTorch 2.8.0, NumPy 2.3.5, pandas 2.3.3, matplotlib 3.10.8 and PyYAML
  6.0.2.
- CNN configuration: channels [8, 16, 32, 64], pooled size 4, hidden size
  128, dropout 0.1, AdamW learning rate 0.001, weight decay 0.0001, Huber
  delta 1.0.

## Key artifact hashes

SHA-256 values below are hashes of the files used for the submission package.

| Artifact | SHA-256 |
|---|---|
| `results/ml/equal-budget-final-v1/fit_complete.json` | `ab42ea531e42017677313c741d3405f2f83ea50b7cb68dc054e3d99ee4a8b14c` |
| `results/ml/equal-budget-final-v1/predictions_before_scoring.csv` | `ee7af715e5bcbe70aee93bd661a984091d5375bb3831d8b34b8d4a30d33e9f0e` |
| `results/ml/equal-budget-final-v1/predictions_complete.json` | `c58e51cea928cd392fbcc8db352b9a2e00e3175dd0bb88f52bbff56f1334b9b1` |
| `results/ml/equal-budget-final-v1/final_report.json` | `486303c0210eb0ecbf81a57f479bd7a79ebc3b0f8b2c499006dc3e35c64630df` |
| `results/ml/cross-simulator-rapid-v2/renderer_identity_audit.json` | `7b1b4f3fed54104d38d363bb9504962998e047abf7fdf5c8f92042c7ec40e47e` |
| `docs/reliability_study/final/manuscript_submission_v2.md` | `609fd37516dae2d2dfb9154e471ccda52c1375d8b30050388d36a9d780369d59` |
| `results/ml/equal-budget-final-v1/figures/equal_budget_mae_by_condition.png` | `f29c35a8ee128f1e41872cb85f066529eb64bc65f4e3e60e71bb9c63396e0d56` |
| `results/ml/equal-budget-final-v1/figures/equal_budget_paired_contrasts.png` | `47581dace9c3788490a63fbe020dad7ae0017a8f8919bf601f7c87bf6ad2dcf2` |

## Claim guardrails for reviewers

The main result is a comparison among our own source-only training arms on one
synthetic benchmark. It is not an exact reproduction of Hezaveh, Pearson,
Swierc, Agarwal or Gawade, and their reported metrics are not numerically
ranked against ours. AutoLens/Lenstronomy package versions differ, but the
rapid-v2 paired images are identical at saved precision; therefore the release
is a renderer-equivalence audit plus a Moffat observation-family check, not an
independent simulator-transfer experiment.
