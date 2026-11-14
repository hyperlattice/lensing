# Commands

Run from the repository root:

- `python -m scripts.study --help`: maintained generation/development/training/analysis.
- `python -m scripts.check_repository`: maintained references and imports.
- `python -m scripts.submission check`: reviewer boundary and static manuscript checks.
- `python -m scripts.submission export --output <zip>`: deterministic verified export.
- `python -m scripts.run_architecture_ablation_v1 all`: run the exploratory
  compact-versus-reference capacity check when the qualified local banks and
  primary reference checkpoints are available.

See [reproduction](../docs/reproduction.md) for inputs, stage ordering and unchanged
output destinations. Historical script commands require their original Git checkout.
