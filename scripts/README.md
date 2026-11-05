# Workflow entry points

Run from repository root using `python -m scripts.<name>`. See the
[experiment index](../docs/experiments.md) for scope, inputs and outputs.

| Script | Action |
|---|---|
| `run_equal_budget_final_v1.py` | `fit`, `predict`, `score`; `--config configs/ml/equal_budget.yaml`. |
| `plot_equal_budget_final_v1.py` | Plot saved evidence; no inference. |
| `run_robustness_development_v1.py` | Reference fits, allocation weights, tuning; `--config`. |
| `review_robustness_development_v1.py` | Source-only tuning review and budget curves; `--config`. |
| `materialize_benchmark_contract_development_v1.py` | Engineering-only image contract intake. |
| `score_paired_evaluation_v1.py` | Frozen paired scoring, not a maintenance smoke. |
| `create_reviewer_simulation_pdf_v1.py` | Reviewer PDF from saved examples/results. |
| `check_reviewer_boundary_v1.py` | Reviewer file allowlist and private-path check. |
| `validate_ai2ase_submission_v1.py` | Static manuscript/assets/citation checks. |
| `export_submission.py` | Export and verify the canonical reviewer bundle. |
| `check_repository.py` | Check current config references and documentation links. |

Versioned runner names remain stable for provenance. Archived generation and
qualification runners may require their historical checkout. Implementation
consolidation belongs to the later refactor.
