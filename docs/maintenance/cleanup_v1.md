# Repository cleanup v1

Maintenance migration on `richard_changes` from `e76b72d`; the full revision
is recorded in [cleanup_v1.json](cleanup_v1.json). This groups configurations,
updates current references, externalizes existing equal-budget settings,
documents experiments/artifact policy, and makes submission exports
reproducible. The implementation refactor is a separate follow-up.

## Frozen evidence

The JSON record maps every relocated configuration to its original path and
byte hash. Configuration/protocol references change; scientific settings,
experiment IDs, dataset IDs and existing output destinations are preserved.
Source defaults consequently also have different hashes.

`requirements.txt` now explicitly includes `tqdm`, imported by the pinned
Lenstronomy fitting workflow but missing from the previous dependency list.

Historical outputs and the Week 11 freeze manifest remain unchanged. This
maintenance checkout must not be presented as the identical frozen study.
Use the recorded original checkout for exact reproduction. `e76b72d` restores
the pre-cleanup maintenance state, which already contains earlier archive and
dependency maintenance and does not guarantee every original freeze hash.
Do not rewrite historical manifests to conceal maintenance differences.

The equal-budget YAML preserves the runner's seeds, 880 updates, batch size
16, v4 banks and bootstrap settings. New fits save resolved configuration;
downstream stages require it. Legacy completed runs without that record
remain historical and must not be retroactively given new provenance.

## Artifacts and exports

No existing data/result evidence is deleted or untracked. Compact text
evidence is made visible for explicit Git review; large products stay
ignored. See [artifact policy](../artifact_policy.md).

`submission/aaai27/reviewer_bundle/` is the only editable submission source.
The duplicate expanded Overleaf tree and tracked ZIP are replaced by a
validated, byte-verified exporter. Generated exports are ignored and never
implicitly overwritten. No source manuscript or canonical figure is removed.

## Validation

Baseline analytic smoke passed. The initial test subset had 60 passes, one
optional-backend skip, and one failure caused by missing Astropy. Final
checks and environment limitations are in
[cleanup_validation.md](cleanup_validation.md). Maintenance does not train
models, regenerate scientific results, or reopen the production holdout.
