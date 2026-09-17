# Einstein-radius ML experiment plan

This is the implementation plan for improving the image-only machine-learning
model. It applies only to the simulated Einstein-radius target
`theta_E_arcsec`. It does not authorize real-observation training, physical
mass inference, or multi-parameter prediction.

## 1. Starting point

The recorded reference is `theta-e-cnn-seeds-v1`:

| Item | Reference value |
|---|---|
| Target | `theta_E_arcsec` |
| Inputs | One normalized 256 x 256 image |
| Model | Four-block CNN; 248,409 trainable parameters |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Loss | Huber, delta `1.0` on normalized targets |
| Data | `sim-eng1000-v1`: 700 train / 150 validation parents |
| Seeds | `12001`, `12002`, `12003` |
| Validation MAE | `0.09210 +/- 0.00872` arcsec across seeds |
| Validation P95 | `0.23782` arcsec mean across seeds |

The primary finding is systematic, not merely random: all three seeds tend to
overestimate the smallest radius bin and underestimate the largest. The
overall mean bias is therefore insufficient for ranking models.

The full baseline evidence is in
`results/ml/theta-e-cnn-seeds-v1/report.md`.

## 2. Non-negotiable scope and evaluation rules

1. Keep exactly one output variable: `theta_E_arcsec`.
2. Use the frozen `sim-eng1000-v1` train and validation splits for all model
   selection.
3. Use the same three seeds for every substantive candidate.
4. Keep P100 separate as a stress evaluation; do not merge it into training.
5. Do not use the existing test split to choose a loss, model, input, epoch,
   augmentation, or hyperparameter.
6. Make one controlled experimental change per candidate relative to its
   declared parent configuration.
7. Record resolved configuration, Git revision, source/data/split hashes,
   environment, parameter count, runtime, checkpoint hash, predictions,
   plots, and metrics for every completed run.
8. Run a small smoke test only to verify wiring. Do not rank models from smoke
   metrics.

The existing test result remains a historical baseline. A new, independently
seeded final holdout release is required before a final challenger claim.

## 3. First implementation milestone

Before comparing models, make the experiment system support controlled
candidates without changing the baseline result.

### Code changes

| Area | Change |
|---|---|
| `src/ml/model.py` | Add a model factory with the existing `cnn` and a future `residual_cnn` option. Preserve the current CNN exactly. |
| `src/ml/losses.py` | Add a loss factory for Huber, MAE/L1, MSE, and log-cosh. |
| `src/ml/train.py` | Read model, loss, optimizer, scheduler, dropout, and input settings from a fully resolved config. |
| `src/ml/metrics.py` | Add threshold accuracy, calibration slope/intercept, prediction-range, and subgroup metrics. |
| `src/ml/experiments.py` | Compare a candidate with its declared parent using the same seeds and write paired deltas. |
| `configs/ml/` | Store one versioned config per candidate and a registry entry for every three-seed suite. |
| `tests/` | Verify scalar output shape, loss selection, config validation, provenance, and metric calculations. |

Each candidate must contain a `parent_experiment_id` and differ from the
parent in exactly one declared factor. An example structure is:

```yaml
experiment_id: theta-e-huber-delta-025-v1
parent_experiment_id: theta-e-cnn-seeds-v1
target_field: theta_E_arcsec
model:
  kind: cnn
loss:
  kind: huber
  delta: 0.25
```

All other resolved fields must match the parent. The runner should reject a
candidate whose declared change and resolved config do not agree.

## 4. Experiment sequence

Run the phases in order. Advance only when the prior phase has a clearly
reported winner or no useful effect.

### Phase A: training controls

Keep the image-only baseline CNN and Huber loss fixed. Test one setting at a
time:

| Factor | Candidate values |
|---|---|
| Epoch budget | `40` versus baseline `20` |
| Learning rate | `0.0003`, `0.001`, `0.002` |
| Weight decay | `0`, `0.00001`, `0.0001`, `0.001` |
| Dropout | `0.0`, `0.1`, `0.2` |
| Batch size | `8`, `16`, `32` |
| Scheduler | none versus validation-driven plateau reduction |

The first epoch-budget check is important because one baseline seed selected
its best checkpoint near the current 20-epoch limit. A validation-driven
learning-rate scheduler is a later single-factor comparison; it reduces the
rate after a monitored metric stops improving. See the official
[AdamW documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)
and [ReduceLROnPlateau documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

### Phase B: loss ablations

Keep the selected Phase-A training controls, image-only input, CNN, seeds, and
checkpoint-selection rule fixed.

| ID | Loss | Reason |
|---|---|---|
| LO-00 | Huber, delta `1.0` | Recorded reference |
| LO-01 | Huber, delta `0.5` | More robust beyond about `0.214` arcsec raw error |
| LO-02 | Huber, delta `0.25` | More robust beyond about `0.107` arcsec raw error |
| LO-03 | MAE/L1 | Directly emphasizes typical absolute error |
| LO-04 | MSE | Emphasizes large failures |
| LO-05 | log-cosh | Smooth robust alternative |

Huber delta is applied after target normalization. With target standard
deviation near `0.428` arcsec, the baseline delta of `1.0` transitions near
`0.428` arcsec—larger than most current errors—so it is close to an MSE-like
objective for typical cases. Tune delta before assuming a different loss is
needed. Huber and Smooth L1 use quadratic behavior near zero and a linear
outlier regime; their transition parameters are not interchangeable. See the
[PyTorch loss reference](https://docs.pytorch.org/docs/main/nn.html) and
[Smooth L1 documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.loss.SmoothL1Loss.html).

Do not introduce target-bin weighting until unweighted loss and architecture
studies are complete. Any future weighting must be fixed from training-data
rules, not tuned from test results.

### Phase C: compact architecture ablations

Keep the winning training controls and loss fixed.

1. Current CNN: immutable reference.
2. Wider CNN: channels `[12, 24, 48, 96]`.
3. Wider CNN: channels `[16, 32, 64, 128]`, only if the smaller wider CNN is
   promising.
4. Small residual CNN with a comparable parameter budget.
5. Residual CNN with explicit `x` and `y` coordinate channels.

Do not begin with pretrained ResNet, EfficientNet, or vision-transformer
models. The training set contains only 700 parents and the current compact
model has about 248 thousand parameters, whereas stock ResNet-18 has about
11.7 million parameters and standard pretrained models expect natural-image
preprocessing. Compact, transparent models provide a more informative first
comparison. [TorchVision ResNet](https://docs.pytorch.org/vision/stable/models/resnet)
and [TorchVision model guidance](https://docs.pytorch.org/vision/main/models)
describe those model families and their preprocessing requirements.

### Phase D: physical input ablations

Keep one scalar output and the selected model/loss/training controls. Add one
input representation at a time:

1. Image only: reference.
2. Image plus invalid-pixel mask.
3. Image plus coverage mask.
4. Image plus background-exclusion mask.
5. Image plus variance map.
6. Image plus a PSF representation.

Combine inputs only after their isolated experiments show a useful effect.
Additional channels are model inputs, not additional inferred parameters.

### Phase E: robustness and failure analysis

Evaluate the selected candidate without retraining on:

- P100 shear/source-complexity stress scenes;
- controlled PSF mismatch;
- controlled noise and signal-to-noise changes;
- source-morphology changes;
- doubles versus quads;
- physical versus detectable image multiplicity; and
- weak or detectability-limited configurations.

The purpose is to expose where the model is reliable and where it is not.
Never hide hard subgroups inside a favorable overall mean.

## 5. Metrics and model-selection gate

Validation MAE in arcseconds remains the primary model-selection metric.
Every report must also contain:

| Metric | Role |
|---|---|
| MAE and median absolute error | Typical angular error |
| RMSE | Sensitivity to large misses |
| P90 and P95 absolute error | Tail risk |
| Mean and median relative error | Scale-aware error over the nonzero target range |
| Overall signed bias | Global over/underprediction |
| Radius-bin MAE and bias | Detects the existing edge-radius pattern |
| Accuracy within 0.05 and 0.10 arcsec | Predeclared, interpretable thresholds |
| Calibration slope and intercept | Detects regression to the mean |
| Out-of-range prediction fraction | Identifies invalid predictions outside the simulated prior |
| Metrics by multiplicity, PSF, SNR, and detectability | Physical failure modes |
| Parameter count and runtime | Practical cost |

Compare candidates with paired seed-wise changes against the same baseline
seeds. A proposed acceptance gate is:

1. Mean validation MAE improves by at least `0.01` arcsec, approximately the
   observed baseline seed variation.
2. Mean P95 does not materially worsen.
3. The worst absolute radius-bin bias does not materially worsen; the goal is
   to reduce the current roughly `0.05` arcsec edge-bin bias.
4. The result is consistent across seeds rather than driven by one lucky run.

Three seeds quantify initialization/training-order sensitivity, not a
confidence interval. Treat borderline gains as inconclusive.

## 6. Final challenger evaluation

After Phase A through E:

1. Freeze the winning configuration and all acceptance rules.
2. Generate an independent, newly seeded final holdout simulation release.
3. Evaluate the frozen challenger once on the new holdout.
4. Write a model card with data, target, inputs, architecture, metrics,
   subgroup/tail results, robustness, compute, and limitations.
5. Keep synthetic in-domain performance distinct from real-data transfer.

Do not report a new model as better based on the already-observed baseline test
split.

## 7. Future uncertainty and multi-parameter roadmap

Uncertainty is a later phase. First establish deterministic scalar accuracy and
robustness. A future two-head network can predict a mean and a positive
variance with Gaussian negative log likelihood, then must report negative log
likelihood, interval coverage, calibration, and sharpness in addition to point
error. [PyTorch GaussianNLLLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.GaussianNLLLoss.html)

Do not expand directly to all simulator fields. Use this sequence:

1. Keep `theta_E_arcsec` as the stable scalar reference.
2. Validate a separate scalar `q` model.
3. Infer orientation through `sin(2 * phi_rad)` and `cos(2 * phi_rad)`, not
   raw angle regression, so the 180-degree lensing symmetry is represented.
4. Validate `e1` and `e2` separately.
5. Only then implement a multi-task head with missing-target masks,
   per-target normalization, scientifically chosen loss weights, and
   per-target/subgroup metrics.

Not every simulation nuisance parameter should become an inference target. A
future target must have a scientific definition, expected observational
availability, a leakage-safe label, and a metric appropriate to its units and
symmetry.

## 8. Ready-to-start checklist

The first implementation can begin with no new scientific data or labels. It
uses the frozen engineering release, three known seeds, and one target.

- [ ] Add configuration validation and model/loss factories.
- [ ] Add the new metrics and paired-candidate report.
- [ ] Preserve exact compatibility with `theta-e-cnn-seeds-v1`.
- [ ] Add tests and run a smoke suite.
- [ ] Run the first full Huber-delta suite with unchanged image-only CNN.

The recorded three-seed benchmark took roughly seven CPU minutes. A CUDA device
is optional but useful for the larger ablation sequence; it is not required.
