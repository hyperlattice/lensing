# Machine-learning baseline quickstart

This guide trains and evaluates the first image-only model for `theta_E_arcsec`. It assumes the frozen `sim-eng1000-v1` release already exists under `data/simulations`.

## 1. Install the ML dependency

From the repository root and your activated virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-simulation.txt
python -m pip install -r requirements-ml.txt
```

Confirm the device PyTorch can see:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The first recorded run used PyTorch 2.8.0 on CPU. A CUDA installation can be selected automatically when available.

## 2. Train without touching the test split

```powershell
python -m src.ml.train --config configs/ml_baseline_v1.yaml
```

The configuration fixes the release, target, seed, optimizer, loss, model size, batch size, epoch limit, and early-stopping patience. Training computes image and target normalization from the 700 training parents only. Each epoch prints:

- `train_loss`: Huber loss on normalized targets;
- `validation_loss`: the same loss on 150 validation parents;
- `validation_mae_arcsec`: the most intuitive angular error;
- `validation_r2`: improvement over predicting no target variation;
- `runtime_seconds`: epoch runtime.

The best checkpoint is chosen by validation MAE. The command records that zero test samples were accessed.

For a fast wiring check, use a new output directory:

```powershell
python -m src.ml.train --config configs/ml_baseline_v1.yaml --output results/ml/smoke --epochs 2 --max-train-samples 32 --max-validation-samples 16
```

The two `--max-...` options are hidden development controls; a smoke result is not a scientific result.

## 3. Inspect validation before opening the test split

Open these files under `results/ml/theta-e-cnn-baseline-v1`:

- `training_curves.png`: training/validation loss and validation MAE by epoch;
- `validation_predictions.png`: predicted versus true values, residual trend, and residual histogram;
- `training_summary.json`: exact metrics, train-mean baseline, normalization, runtime, device, and sample counts;
- `validation_predictions.csv`: every validation prediction.

Interpret the curves as follows:

- Both curves improve: the model is learning a pattern that generalizes to validation.
- Training improves while validation worsens: overfitting has started; rely on the earlier saved checkpoint.
- Both remain poor: the model, input representation, or data signal is insufficient.
- Predictions collapse near one value: the network is behaving like the mean baseline.
- Residuals bend with true radius: the model has target-dependent bias.
- A few isolated residuals dominate RMSE: inspect their image multiplicity, PSF, and detectability before changing the model.

Do not select a checkpoint or architecture from the test split.

## 4. Evaluate the fixed checkpoint once

After accepting the model and checkpoint based on validation:

```powershell
python -m src.ml.evaluate --checkpoint results/ml/theta-e-cnn-baseline-v1/best.pt --split test
```

This verifies both the release and split hashes before inference, then writes:

- `evaluation-test/report.md`: short human interpretation;
- `evaluation-test/metrics.json`: model, baseline, subgroup, and worst-case metrics;
- `evaluation-test/predictions.csv`: sample-level truth, prediction, residual, relative error, image counts, and PSF width;
- `evaluation-test/predictions.png`: the three primary diagnostic plots.

MAE is the average angular miss. RMSE weights large misses more strongly. Median relative error describes a typical fractional miss. Bias indicates systematic over/underprediction. R² near one means the model explains most target variation; zero is comparable to predicting a constant mean, and a negative value is worse than that constant.

## 5. Recorded baseline result

The first full CPU run used 248,409 trainable parameters and selected epoch 19 of 20.

| Metric | Validation | Locked test |
|---|---:|---:|
| MAE | 0.0884″ | 0.0876″ |
| RMSE | 0.1163″ | 0.1292″ |
| Median relative error | 5.71% | 4.93% |
| Bias | -0.0031″ | -0.0090″ |
| R² | 0.9285 | 0.9127 |

The test MAE is 77.2% below the train-mean baseline MAE of 0.3835″. Validation and test MAE differ by only -0.0007″, which is reassuring for this frozen synthetic distribution.

The error is not uniform. Test doubles have 0.0951″ MAE, while quads have 0.0608″ MAE. Two low-radius doubles are large positive-residual outliers; one has only one image above the simulator's detectability threshold. This suggests that ambiguous/weak configurations deserve an explicit robustness experiment in the next model iteration.

## 6. What this result means

It shows that a compact CNN can recover much of the Einstein-radius signal in held-out simulations drawn from the same engineering proposal. It does not show that the model is calibrated, causal, robust outside the proposal, or accurate on real HST data.

Do not apply this checkpoint to `real-castles-v2` as a scientific measurement. The real observations have different PSFs, variance quality, source morphology, calibration, and selection effects. The next safe work is synthetic ablation and robustness testing, followed by an explicitly designed simulation-to-real study.
