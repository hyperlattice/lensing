# Experiment index

This index distinguishes maintained commands, development workflows and
completed research records. An implemented runner or saved report does not
establish that its private datasets, checkpoints or environment are present.
Run commands as modules from repository root.

## Current paper workflow

| Workflow | Configuration / command | Inputs → outputs |
|---|---|---|
| Equal-budget C/G/R comparison; completed result documented | `configs/ml/equal_budget.yaml`; `python -m scripts.run_equal_budget_final_v1 fit --config configs/ml/equal_budget.yaml` | Qualified v4 source banks + allocation policy → `results/ml/equal-budget-final-v1/` |
| Prediction and scoring stages | Same runner with `predict` or `score` | Completed fits + existing 150-parent holdout → sealed predictions and paired contrasts |
| Figures | `python -m scripts.plot_equal_budget_final_v1` | Saved `final_report.json` → figures under the same result root |
| Submission checks | `python -m scripts.check_reviewer_boundary_v1`; `python -m scripts.validate_ai2ase_submission_v1` | Canonical reviewer bundle → read-only validation |
| Overleaf export | `python -m scripts.export_submission` | Reviewer bundle → ignored, byte-verified ZIP |

See [equal-budget interpretation](equal_budget_final_v1.md). This comparison
reuses a synthetic holdout previously scored by older models. It does not
establish real-data or full cross-simulator superiority. The v4 banks and
equal-budget report are absent in the cleanup checkout; maintenance does not
launch training to fill those gaps.

New equal-budget fits save `resolved_config.json`; prediction/scoring require
that exact configuration. Legacy runs without it belong to the historical
runner. Do not retrofit new provenance onto old completed runs.

## Foundation and development

| Workflow/status | Configuration / entry point | Inputs → outputs |
|---|---|---|
| Real preprocessing; raw inputs required | `configs/preprocessing/default.yaml`; `python -m src.pipeline prepare-real --help` | CASTLES FITS + labels → `data/releases/<release>/` |
| Staged controls | `configs/simulation/pilot.yaml`; `python -m src.pipeline simulate --help` | Fixed priors/seeds → a new simulation release |
| Engineering simulation | `configs/simulation/engineering.yaml`; `python -m src.pipeline --help` | 1,000-parent proposal → release and compact manifests |
| Original CNN; recorded | `configs/ml/baseline.yaml`; `python -m src.ml.train --config configs/ml/baseline.yaml` | `sim-eng1000-v1` → baseline model artifacts |
| Seed/loss suites; recorded | `configs/ml/registry.yaml`; `python -m src.ml.experiments --experiment theta-e-cnn-seeds-v1` | Registered suite → per-seed artifacts and aggregates |
| Matched budgets; development only | `configs/ml/matched_budget.yaml`, `configs/ml/matched_budget_validation.yaml`; `python -m src.ml.matched_training --help` | Canonical/randomized releases → fits and validation summaries |
| Policy/MMD search; development only | `configs/ml/robustness_development.yaml`; `python -m scripts.run_robustness_development_v1 --help` | Original v3 banks → `results/ml/robustness-development-v1/` |
| Later v4 development snapshot | Explicit `--config archive/configs/ml/robustness_development_v4.yaml` | v4 banks → `results/ml/robustness-development-v4/` |
| Search review/budget curves | Same chosen config; `python -m scripts.review_robustness_development_v1 --help` | Verified tuning records → source-only review and curves |
| Alternate renderer; development only | `configs/simulation/cross_simulator.yaml`; `python -m src.simulation.cross_simulator_v2 --help` | Qualified controls → development images |
| Benchmark contract intake; engineering only | `python -m scripts.materialize_benchmark_contract_development_v1` | Existing development release → image-only overlay |
| Real reference intake; engineering only | `configs/observations/reference_intake.yaml`; `python -m src.data.external_reference --help` | Explicit external assets → readiness report |
| Real-image geometry; exploratory | `configs/observations/comparison.yaml`; `python -m src.pipeline real-comparison --help` | Selected CASTLES products → comparison report |

## Original reliability study: completed phases

Configurations below are under `configs/reliability/`. Most diagnostics use
the original 150 validation parents and three frozen CNN checkpoints. Extra
views are repeated measurements of those parents.

| Configuration | Entry point | Purpose |
|---|---|---|
| `counterfactual.yaml` | `src.simulation.counterfactual` | Small canonical/lens-light generation example. |
| `pixel_scale.yaml` | `src.ml.pixel_scale_audit` | Frozen-model scale sensitivity. |
| `pixel_scale_training.yaml` | `src.ml.scale_aware_train`, `src.ml.scale_aware_audit` | Train/evaluate two scale-aware variants. |
| `observation_systematics.yaml` | `src.ml.observation_systematics_audit` | PSF, detector, light and normalization audit. |
| `source_physics.yaml` | `src.ml.source_physics_audit` | Source profile/size/position and shear. |
| `cross_simulator.yaml` | `src.ml.reliability_week7` | Independent-equation control with shared light/observation components. |
| `mitigation.yaml` | `src.ml.reliability_week8` | Original nuisance-randomized training. |
| `hybrid_initialization.yaml` | `src.modeling.hybrid_initialization` | CNN versus classical fitting starts. |
| `domain_gap.yaml` | `src.modeling.domain_gap_week10` | Descriptive CASTLES/synthetic coverage; no real-label accuracy. |
| `blind_holdout.yaml` | `src.ml.reliability_week11` | Original 150-parent final evaluation; **not a maintenance rerun**. |
| `classical_pixel_scale.yaml` | `src.modeling.pixel_scale_geometric_baseline` | Classical scale comparator. |
| `classical_observation_systematics.yaml` | `src.modeling.observation_systematics_classical_audit` | Classical observation comparator. |

Principal outputs remain in `results/reports/reliability_v1/`,
`results/reports/domain_gap_v1/`, and `results/ml/` as specified in each YAML.
Protocols are retained under `archive/docs/reliability_study/`. Read the
[research card](reliability_study/final/model_research_card.md),
[reproducibility statement](reliability_study/final/reproducibility_statement.md)
and [migration record](maintenance/cleanup_v1.md) together.

## Implementation inventory for the later refactor

- `experiments.py` is the historical suite runner; `experiments_v2.py` adds
  stricter controls. The original runner has a `run_controls` verification
  mismatch with current training summaries (two pre-existing test failures).
  Compare their output/resume contracts before merging; see the
  [validation record](maintenance/cleanup_validation.md).
- `src/ml/observation_bank_v1.py` is a compatibility re-export of the loader
  maintained in `src/data/observation_bank_v1.py`.
- ML and processing observation-condition definitions coexist to keep data
  generation independent of Torch; existing parity tests cover them.
- Original, precise, low-memory and adaptive renderers have different
  numerical assumptions. Review qualification and callers before retirement.
- `simulation/cross_simulator_legacy.yaml` preserves the original development
  adapter; the current cross-simulator configuration uses the v2 convention.
- `archive/` holds historical protocols and generation/qualification runners.
  Their path strings remain historical and may require the original checkout.

## Maintenance checks

```powershell
python -m scripts.check_repository
python -m src.simulation.smoke_test
python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-work-tests
```

The smoke performs deterministic analytic checks without private datasets and
reports optional backend availability separately. Full checks require the
dependencies in `requirements.txt`; they do not open the production holdout.
