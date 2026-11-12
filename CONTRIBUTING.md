# Contributing

This repository is a research workflow for strong gravitational lens images.
Run commands from the repository root: the `src` package, configurations,
datasets, and reports use repository-relative paths.

## Project map

- `src/data/` defines the manifest contract and the truth-isolated dataset
  loader.
- `src/processing/` prepares observations, masks, variance, metadata, and
  splits.
- `src/simulation/` renders controlled lens scenes and materializes releases.
- `src/modeling/` contains classical fits and the bounded real-image
  comparison.
- `src/ml/` contains model training, evaluation, metrics, and experiment
  workflows.
- `src/testing/` contains command-line release validators. Automated tests
  live in the top-level `tests/` directory.
- `configs/` holds versioned scientific inputs; `docs/` describes contracts,
  protocols, and interpretation; `results/` holds derived evidence.

See [the codebase guide](docs/codebase_guide.md) for module responsibilities
and [the reliability-study guide](docs/reliability_study/README.md) for the
multi-phase research record.

## Scientific changes

Current workflow/configuration locations are indexed in
[`docs/experiments.md`](docs/experiments.md). Historical path/hash migrations
are recorded in [`docs/maintenance/cleanup_v1.md`](docs/maintenance/cleanup_v1.md).

Treat a reported experiment as an immutable record. Give changed scientific
settings a new configuration and experiment or release ID. Record the parent
experiment, random seeds, software environment, Git revision and dirty status, and output
locations. Do not use test-split or blind-holdout results to choose later model
settings. Keep simulator truth separate from model inputs and actual-observation
labels separate from synthetic targets.

Completed-result analysis may be repeated from saved predictions and truth.
Keep their identity checks; current renderer source and training checkpoints
are unnecessary for this analysis. A dirty Git revision alone does not capture
uncommitted source edits, so commit the code used for a reported experiment.

The original Week 11 record retains historical checksums. Its source paths, configurations,
protocol, and exactly-once evaluation record are part of the evidence. Do not
rename or edit those inputs while presenting the existing result as the same
frozen run. Maintenance work that changes them needs a separately versioned
study record.

## Code and comments

Place reusable logic in the package that owns the responsibility; keep command
parsing at the CLI boundary. Prefer descriptive module and function names for
new code. The existing `reliability_week*.py` modules retain their names
because study records identify those phase workflows.

Document scientific interfaces with docstrings. State the purpose, units,
coordinate convention, assumptions, and meaningful failure conditions when
they are not clear from the signature. Use inline comments to explain why a
non-obvious numerical step or guard is necessary; avoid comments that merely
repeat the next line. Keep comments next to the calculation or decision they
explain, and update them when the behavior changes.

Prefer deterministic random-number generators passed from the experiment
boundary. Keep units in field names or documentation, validate external data
at module boundaries, and return explicit status or provenance rather than
silently filling missing scientific values.

## Data and generated files

Do not commit raw observations, generated array products, preview images, local
environment files, or model checkpoint binaries. `.gitignore` covers common
local outputs. Version the compact manifests, source labels, configurations,
and reports needed to interpret an experiment. A report should state sample
and independent-parent counts, the comparison being made, uncertainty method,
and scientific limitations.

See [`docs/artifact_policy.md`](docs/artifact_policy.md) for compact-evidence
exceptions and use explicit paths when staging. Only
`submission/aaai27/reviewer_bundle/` is editable submission source; generate
delivery ZIPs with `python -m scripts.export_submission`.

## Local checks

The test suite is configured in `pyproject.toml`. From the repository root:

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-work
python -m src.pipeline --help
```

Tests should use temporary directories and must not rewrite frozen releases or
reported results.
