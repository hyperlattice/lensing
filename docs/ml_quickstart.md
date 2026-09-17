# Machine-learning quickstart

This guide trains and evaluates the first image-only model for `theta_E_arcsec`. It assumes the frozen `sim-eng1000-v1` release already exists under `data/simulations`.

For ongoing development, use the [multi-seed workflow](#7-run-the-development-benchmark-across-seeds) below. The original test result is already recorded; future model selection uses validation and the planning notebook calls for a new independent final holdout.

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

## 7. Run the development benchmark across seeds

Phase 1 of [the planning notebook](../notebooks/ml_implementation_plan_v1.ipynb) is implemented. Run all commands from the repository root. The versioned registry is `configs/ml/registry_v1.yaml`; `configs/ml/baseline_seeds_v1.yaml` fixes seeds `12001`, `12002`, and `12003`, reuses the original CNN/Huber training configuration, and declares radius-bin edges before training.

A quick check runs two epochs with up to 32 training and 16 validation examples per seed:

```powershell
python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1 --smoke
```

This writes to `results/ml/theta-e-cnn-seeds-v1-smoke` and labels the report `smoke_only`. It verifies the workflow but does not measure model quality.

The full three-seed benchmark uses 700 training and 150 validation examples for each seed:

```powershell
python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1
```

For a fresh repeat, choose a new output directory:

```powershell
python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1 --output results/ml/theta-e-cnn-seeds-v1-repeat
```

The runner never loads test images or calculates test metrics. It hashes the release and all split manifests to record their identity. The model still receives only normalized images; normalization uses the training split, and validation MAE selects the checkpoint independently for each seed. All seeds are reported; a particularly lucky seed is not the benchmark result.

## 8. Read the seed comparison

Open these files in `results/ml/theta-e-cnn-seeds-v1`:

| File | What it tells you |
|---|---|
| `report.md` | Per-seed results, averages, seed variation, radius-bin bias, and interpretation |
| `seed_comparison.png` | MAE/RMSE/P95 for each seed and systematic errors versus true radius |
| `per_seed_metrics.csv` | Exact validation metrics, best epochs, and runtimes |
| `validation_target_bins.csv` | Counts and errors for every seed and fixed radius bin |
| `aggregate_validation.json` | Machine-readable summary, run records, hashes, and environments |
| `resolved_suite.json` | Frozen suite settings, base config, source hashes, and release identity |
| `seed-12001/`, `seed-12002/`, `seed-12003/` | Individual checkpoints, normalization, resolved configs, training histories, predictions, and plots |

P90 and P95 are absolute-error thresholds that contain 90% and 95% of the validation examples. A good mean MAE can coexist with a poor P95, so inspect both. Radius-bin bias shows whether small or large lenses are systematically overestimated or underestimated. The default bins are `[0.5, 1.0)`, `[1.0, 1.5)`, and `[1.5, 2.0]` arcsec; out-of-range and empty groups remain visible in the CSV/JSON.

The headline is the arithmetic mean of the individual seeds' validation MAEs with their **sample standard deviation** (`ddof=1`). This spread measures sensitivity to initialization, dropout, and training order on one fixed data split. It is not a confidence interval, an ensemble score, or variation over different data splits. Compare future models on the same seeds and examine paired changes; a gain smaller than the baseline seed variation is inconclusive. Three seeds alone do not establish significance.

Every training run now records the seed, resolved-config hash, release/split hashes, checkpoint and prediction hashes, Python/package versions, device and CPU-thread details, Git revision, and hashes of the Python source. Checkpoints remain ignored by Git.

The first full three-seed benchmark completed with validation MAEs of `0.08836`, `0.08587`, and `0.10206` arcsec: a mean of `0.09210` with sample standard deviation `0.00872` arcsec. Mean P95 was `0.23782` arcsec. It took about seven minutes on CPU, and seed 12001 exactly reproduced the original validation predictions. All three seeds overestimated small radii and underestimated large radii; mean bin biases were `+0.0466` arcsec for `[0.5, 1.0)` and `-0.0507` arcsec for `[1.5, 2.0]`. These are the reference findings for future comparisons. See [the recorded report](../results/ml/theta-e-cnn-seeds-v1/report.md).

## 9. Resume and preserve experiments

An occupied output directory is protected. To verify and reuse completed seeds:

```powershell
python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1 --resume
```

Add the same `--output`, `--device`, and `--smoke` options if they were used originally. Resume requires matching configurations, environment, source, data identities, and checkpoint/prediction hashes. Missing seed directories are trained; completed seeds are reused. An interrupted seed with incomplete artifacts is rejected: preserve it in another directory and resume, or start the suite with a new `--output`. Resume works at completed-seed boundaries and does not restore partially trained optimizer state.

The first implementation milestone covers the registry, seed runner, aggregation, tail errors, and radius-bin diagnostics. Input-channel adapters, loss/architecture comparisons, robustness studies, and uncertainty remain the subsequent notebook phases.
