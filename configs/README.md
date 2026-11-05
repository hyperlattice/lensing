# Configuration guide

Start with the [experiment index](../docs/experiments.md). Current filenames
describe purpose; schema versions and experiment/release IDs remain inside
the YAML and recorded outputs. Commands use repository-relative paths.

| Directory | Purpose |
|---|---|
| `preprocessing/` | Actual-image preprocessing policy. |
| `simulation/` | Pilot/engineering generation and alternate-renderer controls. |
| `observations/` | Real-image comparison and external-reference intake. |
| `ml/` | Baseline, seed/loss suites, matched budgets, robustness and equal-budget training. |
| `reliability/` | Diagnostics, mitigation, classical comparisons, domain gap and original holdout. |

`ml/robustness_development.yaml` retains its original v3-bank settings. Select
the later v4 snapshot explicitly with
`--config archive/configs/ml/robustness_development_v4.yaml`. The equal-budget
configuration `ml/equal_budget.yaml` uses v4 banks as in the recorded runner.

The [migration record](../docs/maintenance/cleanup_v1.md) and
[path/hash map](../docs/maintenance/cleanup_v1.json) identify historical paths.
Historical freeze manifests remain unchanged. This maintenance checkout has
different source/config hashes and is not the identical frozen study.

## Configuration inventory

Configurations are recorded inputs to reproducible commands. Give changed
scientific settings a new descriptive configuration and experiment ID;
the current navigation does not require numeric suffixes on every filename.

| File or directory | Purpose |
|---|---|
| `preprocessing/default.yaml` | Actual-observation preprocessing policy. |
| `simulation/pilot.yaml` | P10, P30, P60, and P100 staged simulation controls. |
| `simulation/engineering.yaml` | Frozen 1,000-parent engineering simulation proposal. |
| `archive/configs/observation_profiles/` | Historical instrument/PSF control profiles retained for provenance. |
| `observations/comparison.yaml` | Preselected exploratory actual-observation comparison. |
| `observations/reference_intake.yaml` | Pinned external science/noise/PSF intake and separately governed observational references. |
| `simulation/cross_simulator_legacy.yaml` | Optional PyAutoLens controls and development rendering; no CNN scoring. |
| `simulation/cross_simulator.yaml` | Corrected Sersic mapping and gated alternate-renderer development release. |
| `ml/baseline.yaml` | First CNN training configuration. |
| `ml/registry.yaml` | Registry of multi-run ML experiment suites. |
| `ml/baseline_seeds.yaml` | Fixed three-seed validation benchmark. |
| `ml/matched_budget.yaml` | Source-only training with paired parent schedules and equal optimizer updates. |
| `ml/matched_budget_validation.yaml` | Descriptive source-validation stress scoring of the fixed-update checkpoints. |
| `reliability/classical_*.yaml` | Pixel-scale and observation-systematics classical controls. |
| `reliability/*.yaml` | Versioned counterfactual, audit, mitigation, cross-simulator, hybrid, domain-gap, and blind-holdout study settings. |

Do not overwrite a configuration after reporting results from it. Make a new
file, assign a new experiment or release ID, and record the parent experiment
in the output report.

The reliability configurations correspond to phase records in
[`docs/reliability_study/`](../docs/reliability_study/README.md). The original
blind-holdout configuration is part of the Week 11 freeze manifest. Its
relocated maintenance copy does not replace the original frozen identity.

Earlier maintenance archived six configurations with no maintained callers
under [`archive/configs/`](../archive/README.md).
