# Commands

Run from the repository root:

- `python -m scripts.study --help`: maintained generation/development/training/analysis.
  Use `--profile seed_replication` on `train`, `predict`, `analyze`, or `plot` for
  the additional five-seed run; source-only development settings are in the same
  `configs/study.yaml` file.
- `python -m scripts.hst_pilot`: acquire and audit five reserved HST development systems;
  see the [pilot protocol](../docs/hst_compatibility_pilot.md). No CNN scoring.
- `python -m scripts.hst_archive`: retrieve a wider native SCI/WHT cutout for the
  fixed archive probe using bounded, cached byte ranges; no CNN scoring.
- `python -m scripts.hst_study --help`: explicit asset/preparation/validation/freeze/
  prediction/analysis/plot stages; see [the implementation guide](../docs/hst_study.md).
- `python -m scripts.check_repository`: maintained references and imports.
- `python -m scripts.submission check`: reviewer boundary and static manuscript checks.
- `python -m scripts.submission export --output <zip>`: deterministic verified export.
- `python -m scripts.run_architecture_ablation_v1 all`: run the exploratory
  compact-versus-reference capacity check when the qualified local banks and
  primary reference checkpoints are available.
- `python -m scripts.validate_cgr_seed_replicates_v1`: reanalyze the saved
  primary ten-seed checkpoints on the frozen holdout; it writes the separate
  seed-consistency report and does not train models.

See [reproduction](../docs/reproduction.md) for inputs, stage ordering and unchanged
output destinations. Historical script commands require their original Git checkout.
