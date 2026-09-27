# Configuration map

Configuration files are versioned inputs to reproducible commands. Copy a
configuration to a new versioned name before changing a scientific setting.

| File or directory | Purpose |
|---|---|
| `preprocessing_v2.yaml` | Actual-observation preprocessing policy. |
| `simulation_pilot_v1.yaml` | P10, P30, P60, and P100 staged simulation controls. |
| `simulation_engineering_v1.yaml` | Frozen 1,000-parent engineering simulation proposal. |
| `observation_profiles/` | Instrument/PSF control profiles used by simulations. |
| `real_comparison_v1.yaml` | Preselected exploratory actual-observation comparison. |
| `ml_baseline_v1.yaml` | First CNN training configuration. |
| `ml/registry_v1.yaml` | Registry of multi-run ML experiment suites. |
| `ml/baseline_seeds_v1.yaml` | Fixed three-seed validation benchmark. |
| `classical_*.yaml` | Frozen pixel-scale, observation-systematics, and source-physics classical controls. |
| `reliability_*.yaml` | Versioned counterfactual, audit, mitigation, cross-simulator, hybrid, domain-gap, and blind-holdout study settings. |

Do not overwrite a configuration after reporting results from it. Make a new
file, assign a new experiment or release ID, and record the parent experiment
in the output report.

The reliability configurations are inputs to the phase records in
[`docs/reliability_study/`](../docs/reliability_study/README.md). In particular,
the blind-holdout configuration is part of the Week 11 freeze manifest; do not
edit it when reproducing or describing that frozen evaluation.
