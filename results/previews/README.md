# Preview artifacts

`previews` is the single review-image area. It is deliberately outside `data`
because PNGs, contact sheets, and their indexes are for human quality control,
not model inputs.

| Location | Contents |
|---|---|
| `actual/<release>/` | Previews created from real observations. |
| `simulated/<release>/` | Previews created from generated simulation releases. |

Each release has its own `preview_index.csv` when an index is produced.
Preview images are reproducible and ignored by Git. Frozen releases may retain
an older local preview copy because removing it would invalidate their recorded
artifact layout; use this directory for all new review work.
