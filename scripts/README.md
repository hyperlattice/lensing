# Commands

Run from the repository root:

- `python -m scripts.study --help`: maintained generation/development/training/analysis.
- `python -m scripts.hst_pilot`: acquire and audit five reserved HST development systems;
  see the [pilot protocol](../docs/hst_compatibility_pilot.md). No CNN scoring.
- `python -m scripts.hst_archive`: retrieve a wider native SCI/WHT cutout for the
  fixed archive probe using bounded, cached byte ranges; no CNN scoring.
- `python -m scripts.hst_study --help`: explicit asset/preparation/validation/freeze/
  prediction/analysis/plot stages; see [the implementation guide](../docs/hst_study.md).
- `python -m scripts.check_repository`: maintained references and imports.
- `python -m scripts.submission check`: reviewer boundary and static manuscript checks.
- `python -m scripts.submission export --output <zip>`: deterministic verified export.

See [reproduction](../docs/reproduction.md) for inputs, stage ordering and unchanged
output destinations. Historical script commands require their original Git checkout.
