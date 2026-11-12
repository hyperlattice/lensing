# Cleanup validation

This records the earlier cleanup. The later simplification consolidated the
seed runners and fixed the control-schema mismatch described below; it also
retired the standalone simulation smoke in favor of the existing tests.
Historical test results here are preserved as recorded.

Validation ran on `richard_changes`, relative to `e76b72d`. No production
training, holdout inference/scoring or scientific result regeneration was
performed. Tests used small temporary fixtures.

## Environment

Final checks use this repository's `.venv/Scripts/python.exe`: Python 3.10.11,
CPU PyTorch 2.9.1, Lenstronomy 1.14.2, Astropy 6.1.7, PyYAML 6.0.3 and pytest
9.2.0. The venv uses base/system site packages, not another project environment.
Python 3.12 with pinned PyAutoLens remains the documented production setup;
this is not validation of that complete stack. `pip check` passes.

An early subset used the `science2` interpreter, with PyYAML available from
another existing environment. Those projects were read-only; outputs stayed
in this workspace. After the user's question, dependencies were installed in
this repository's venv and the final full suite was rerun locally.

## Executed checks

| Check | Result |
|---|---|
| Local full suite: `python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-work-suite-final --tb=short` | **174 passed, 2 failed, 1 skipped**. Both failures are historical; optional PyAutoLens test skipped. |
| New configuration/sealing/export regressions | All 11 pass in the full suite. |
| `python -m scripts.check_repository` | 28 YAML configs load; no broken maintained references. Two unavailable optional result-figure links reported separately. |
| Pipeline and equal-budget/development/review runner `--help` | Pass, including explicit `--config`. |
| `python -m src.simulation.smoke_test --require-backend` | Pass: analytic controls, Lenstronomy conventions and 32×32 render. |
| Reviewer boundary and static submission checks | Pass; no TeX compile or visual manuscript check claimed. |
| Export verification | Member allowlist and canonical bytes verified; deterministic export and overwrite protection tested. |
| `git diff --check` | Pass. |

## Existing seed-runner failure

`test_seed_suite_trains_validation_only_resumes_and_rejects_changed_artifacts`
and `test_smoke_run_label_and_controls` fail with `ValueError: run_controls
changed`. The training summary includes newer run-control fields absent from
the historical runner's expected schema. Both failures were reproduced from
`e76b72d` source/tests extracted under `.pytest-work-precleanup/`, using the
same local interpreter. Missing catalog fixtures in that partial extraction
were supplied from Git; all four catalog tests then pass.

These failures remain visible; tests were not marked xfail or redirected.
Resolving the historical/v2 suite interface belongs to the follow-up refactor.

## Integrity audit

- All 27 relocated YAML configurations are semantically equivalent after
  replacing configuration/protocol paths.
- Extracted equal-budget training settings equal the pre-cleanup runner's
  literal settings, including model and optimizer budget.
- Modified `src/` modules differ only in path-string references; their
  normalized syntax trees otherwise match the pre-cleanup source.
- 2,627 tracked data/result/archive/reviewer evidence files retain exact
  pre-cleanup bytes. Original freeze/result manifests were not rewritten.
- Fifteen Git ignore probes verify evidence visibility, privacy/product
  exclusions and source visibility.
- Canonical reviewer files match the removed expanded export byte-for-byte.
  The previous ZIP differs only in line endings and was preserved locally
  under ignored `archive/private/submission_aaai27/`.

Mappings and hashes are in [cleanup_v1.json](cleanup_v1.json). Temporary
test/baseline/export files are under ignored `.pytest-work*` directories.
No files were staged, committed or pushed.
