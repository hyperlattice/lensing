# Contributing

Prioritize scientific correctness, readability, reproducibility, maintainability,
and then measured performance. Use descriptive names, explicit control flow,
four-space indentation, public interface annotations and explanatory docstrings.
Ruff is configured for a 100-character target line length in `pyproject.toml`.

Before editing, identify the scientific invariants and inspect current tests.
Keep refactoring separate from numerical or methodological changes. Preserve
parent-level independence, source-only normalization, paired schedules and
final-image/label access boundaries. Record resolved settings and provenance
for new runs; do not retrofit them onto historical evidence.

The canonical dependency specification is `pyproject.toml`. Add an abstraction,
file or dependency only when a present responsibility justifies it. Keep command
parsing in `scripts/` and computations in `src/`.

Existing dataset and result paths are fixed contracts. Do not move, rename or
regenerate historical evidence as cleanup. New experiments must use an explicitly
chosen output identity if an existing run is complete. Historical source hashes
cannot be made current by editing old qualification or freeze manifests.

Run Ruff, pytest and the repository checker using the commands in the README.
Use pytest's `tmp_path` for fixtures; no repository-local test work folders.
Tests for retired experiment families live in the historical revision documented
in `docs/migration.md`; they are not replacements for current regression checks.
