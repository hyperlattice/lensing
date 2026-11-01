# Matched training protocol v1

Declared before launching `theta-e-matched-budget-v1` on 2026-10-27.

This is source-only development. It does not reopen the frozen final holdout.
The 880-update budget equals 20 times the original baseline's 44 batches per
epoch. It was chosen from training structure, not new validation results.

Both arms use batch size 16, seeds 12001/12002/12003, constant AdamW learning
rate 0.001, weight decay 0.0001, Huber delta 1.0, the original CNN architecture,
and shared normalization from canonical training parents only. No early
stopping is used. The primary checkpoint is update 880; validation-selected
checkpoints are secondary. Canonical source validation is checked every 44
updates using the same 150 systems. No final holdout or PyAutoLens image is
read. No validation outcome may change this experiment's training controls.

The randomized release contains two views for each of the 700 parents. A
seeded schedule chooses shape buckets proportional to view counts and draws
16 views with replacement per batch. Each view therefore has equal marginal
probability. The canonical arm substitutes the canonical image for precisely
the same parent at every exposure. This makes realized parent exposures,
number of examples and optimizer updates identical, while retaining native
shapes and the original varied view distribution. Parent frequencies are
uniform in expectation; actual frequencies are recorded and paired exactly.

Per-arm initial weight digests, normalization, schedule, parent exposures,
input-array hashes, config/source hashes, environment, pixel counts, runtime,
final checkpoint, predictions and validation history are retained. Unequal
pixel counts and runtime are reported. Existing fixed noise realizations are
reused, so this is not an independent-noise augmentation comparison.

The new runner refuses existing outputs and does not implement interrupted
training resume. Failed partial runs must be retained and a new versioned
output chosen. The separately versioned `src.ml.experiments_v2` corrects the
legacy suite's incomplete control plan; the frozen original modules remain
unchanged. Focused verification established suite creation/resume, rejection
of changed control records, fixed batch geometry, and paired training controls.

Primary output: `results/ml/theta-e-matched-budget-v1/report.json`. Canonical
validation performance is descriptive development evidence, not an unbiased
generalization estimate. A new frozen A/B-renderer final evaluation remains
necessary, including familiar observation stresses and a separate unseen PSF
condition, with parent-level paired scoring.

## Descriptive source-stress evaluation

After all fixed-update runs finish, evaluate their final checkpoints on the
existing 150 source-validation parents under canonical, pixel-scale-004,
pixel-scale-006, exposure-half and read-noise-double conditions. Configuration:
`configs/ml/matched_budget_validation_v1.yaml`; command:
`python -B -m scripts.evaluate_matched_validation`. This evaluates familiar
development data without changing training controls or selecting checkpoints.
Its four-stress parent-averaged difference and bootstrap interval are
descriptive; they cannot replace a fresh final test. Save target-free
predictions and their hashes before the scoring join. No existing final
holdout or target-simulator image may be loaded.
