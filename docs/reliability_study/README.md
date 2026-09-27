# Reliability study guide

This directory records the preregistered development phases and final
synthetic holdout analysis. Read the [base protocol](protocol_v1.md) for the
research question and study-wide rules, then follow the numbered amendments
for each phase. The [decision log](decision_log.md),
[experiment registry](experiment_registry.csv), and
[baseline state](baseline_state.md) provide the project-level audit trail.

## Phase map

| Phase | Question | Main record |
|---|---|---|
| Weeks 1–2 | Freeze the baseline and build matched, ancestry-safe counterfactuals. | [Week 2 counterfactual report](week2_counterfactual.md) |
| Weeks 3–5 | Measure sensitivity to pixel scale, observation conditions, and source/physics mismatch. | [Week 3](week3_pixel_scale_status.md), [Week 4](week4_observation_systematics_status.md), [Week 5](week5_source_physics_status.md) |
| Week 6 | Compare against simple baselines and audit shortcuts and data scaling. | [Week 6 report](week6_baselines_shortcuts_status.md) |
| Week 7 | Compare the frozen model under an independently implemented SIE equation path. | [Week 7 report](week7_cross_simulator_status.md) |
| Week 8 | Test targeted nuisance randomization against the predeclared mitigation rule. | [Week 8 report](week8_mitigation_status.md) |
| Week 9 | Compare classical fit initializations under a common fitting procedure. | [Week 9 report](week9_hybrid_status.md) |
| Week 10 | Describe measurable gaps between the synthetic and CASTLES domains. | [Week 10 report](week10_domain_gap_status.md) |
| Week 11 | Evaluate the selected models once on a new synthetic holdout. | [Frozen protocol](protocol_v1.11.md), [freeze manifest](week11_freeze_manifest.json) |
| Week 12 | Assemble the recorded results into a final analysis package. | [Week 12 package record](week12_final_package_status.md) |

The phase-specific programs are under `src/ml/` and `src/modeling/`; their
configs are under `configs/`, and machine-readable outputs are under
`results/reports/reliability_v1/`. The temporal module names preserve links to
the phase records. New workflows should use descriptive scientific names and
receive a new protocol/configuration version.

## Final package

The [research card](final/model_research_card.md),
[manuscript draft](final/manuscript_draft.md),
[claim-evidence map](final/claim_evidence_map.md), and
[reproducibility statement](final/reproducibility_statement.md) summarize the
study with its scope and limitations. Supporting tables, figures, and hashes
are in [`results/reports/final_v1/`](../../results/reports/final_v1/).

The Week 11 freeze manifest identifies the exact source, data, model, and
protocol inputs used in its evaluation. Its `evaluate` command is exactly-once;
do not rerun it to recreate presentation material. The final package builder
uses saved outputs and does not run inference.
