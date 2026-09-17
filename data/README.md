# Data layout

`data` holds source material, versioned releases, and compact audit records.
Large raw observations and generated arrays are local artifacts and are
ignored by Git. The tracked CSV/JSON manifests make those local artifacts
auditable and reproducible.

| Location | Contents | Domain |
|---|---|---|
| `raw/castles/` | Local CASTLES FITS tree supplied by the researcher. | Actual observations |
| `labels/source/` | Curated, human-maintained input tables. | Actual observations |
| `labels/derived/` | Rebuildable metadata, catalog, and enriched-table outputs. | Actual observations |
| `releases/real-castles-v2/` | Frozen real preprocessing release, its manifest, splits, validation, and local products. | Actual observations |
| `simulations/<release>/` | Self-contained generated simulation releases and their split manifests. | Simulated observations |
| `manifests/sim-eng1000-v1/` | Tracked audit mirror for the ignored engineering simulation release. | Simulated observations |
| `manifests/legacy/` | Older development split artifacts retained for traceability. | Actual observations |

The current canonical training split is inside
`data/simulations/sim-eng1000-v1/splits/`. It has 700 train, 150 validation,
and 150 test parents. Do not combine it with `sim-p100-v1`, which is a separate
stress suite.

Release manifests use paths relative to their own release root. Keep a frozen
release directory intact: moving its products can invalidate manifest hashes,
checksums, and reproducibility records. The older release-local `previews/`
folders therefore remain frozen in place; their review copies are organized
under `results/previews/`. New derived visual previews belong only there.
