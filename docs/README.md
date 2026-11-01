# Documentation map

| File | Use it for |
|---|---|
| `codebase_guide.md` | Scientific and software architecture. |
| `dataset_schema.md` | Manifest fields, paths, units, and domain rules. |
| `production_dataset_v1.md` | Engineering simulation release gates and limits. |
| `simulation_pipeline.md` | Current simulation backend, physical proposals, rendering, noise, products, and reliability evaluation. |
| `cross_simulator_generalization_plan.md` | Proposed independent-renderer benchmark, training controls, and real-data reference requirements. |
| `real_image_reference_report.md` | Research on published SLACS, SL2S, and Euclid reference measurements and compatible images. |
| `external_benchmark_setup.md` | Real-reference intake and optional PyAutoLens development commands, validation, and current limits. |
| `private/external_benchmark_handoff.md` | **Internal:** current status and ordered next steps for the real-data and cross-simulator work. |
| `private/aaai27_conversion_status.md` | **Internal:** AAAI-27 AI2ASE framing, source-package contents, validation and final PDF-build gate. |
| `private/aaai27_real_data_benchmark_report_2026-11-01.md` | **Internal:** accuracy-readiness report for the SLACS pilot and proposed HST/CFHT/Euclid benchmark tiers. |
| `research_submission_priorities.md` | Matched training budgets, cross-simulator validation gates, and comparison to related research. |
| `matched_training_protocol_v1.md` | Declared equal-update, paired-parent source-only training experiment. |
| `cross_simulator_validation_protocol_v2.md` | Declared light-profile, image, point-source, PSF and noise controls for the next renderer version. |
| `prior_work_and_novelty_analysis.md` | Primary-source research, closest comparators, and the evidence needed for novelty and superiority claims. |
| `superiority_demonstration_plan_v1.md` | Complete proposed comparator, data, power, statistics, reproduction, and publication plan for testing superiority. |
| `benchmark_foundations_v1.md` | Implemented strict benchmark overlay, paired augmentation policies, MMD primitive, executed engineering copy, and remaining qualification gates. |
| `renderer_qualification_protocol_v1.md` | Declared 106-parent, three-geometry central-floor and image-integration qualification protocol. |
| `prior_baseline_code_audit_v1.md` | Pinned author-code audit, verified MMD gradient discrepancy, kernel parity, and implications for fair comparisons. |
| `research_execution_2026-10-27.md` | Active end-to-end study handoff, numerical failures/fixes, running qualification phases, and remaining experiment gates. |
| `augmentation_prior_work_addendum.md` | Primary-source active/random augmentation precedents and limits on algorithmic novelty and superiority claims. |
| `recent_prior_work_addendum_2026-10-27.md` | Additional 2024-2026 posterior, multi-survey, contamination and generative-simulation work; precise limits on our novelty claim. |
| `robustness_development_protocol_v1.md` | Declared source reference fits, policy score, ten comparison experiments, shared HPO criterion, and remaining freeze requirements. |
| `confirmatory_analysis_protocol_v1.md` | Pre-outcome contrasts, alpha allocation, canonical/tail guardrails, variance-only pilot and final scoring rules. |
| `confirmatory_workflow_v1.md` | Executable ordered development, review, two-freeze, pilot, final-generation, prediction, analysis and figure commands. |
| `statistical_prior_work_addendum.md` | Crossed-bootstrap references, recent seed/item run-count research and limits of finite-pilot power claims. |
| `research_progress_2026-10-27.md` | Completed matched-budget results, corrected renderer controls, and the remaining submission work. |
| `cross_simulator_convergence_protocol_v1.md` | Additional predeclared integration-refinement check on four development systems. |
| `ml_handoff_v1.md` | Contract and scientific boundary for ML work. |
| `ml_quickstart.md` | Training, evaluation, multi-seed benchmark commands, and result interpretation. |
| `reliability_study/README.md` | Phase protocols, frozen inputs, final holdout, and study evidence. |
| `ml_experiment_plan/README.md` | Controlled one-target model, loss, parameter, metric, robustness, and future multi-parameter implementation plan. |
| `ml_model_landscape/README.md` | Comparison and implementation order for non-neural, convolutional, equivariant, physics-constrained, and probabilistic one-target models. |
| `archive/README.md` | Dated repository snapshots retained for historical context. |

Private working notes belong in `docs/private/`; that directory is ignored by
Git. Public claims should be supported by tracked manifests and reports.
Contributor conventions are in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

For workshop submission, the reviewer boundary is narrower than the project
workspace: upload only the curated files in
`../submission/aaai27/reviewer_bundle/`. Do not upload `docs/`, `results/`,
`data/`, or the private working tree.
