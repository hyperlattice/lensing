# Data and result tracking

| Location | Policy for new files |
|---|---|
| `data/labels/` | Reviewed CSV source/derived tables and Markdown documentation. |
| `data/manifests/` | Reviewed CSV, JSON, Markdown and checksum records. |
| `data/releases/` | Reviewed CSV/JSON/Markdown/checksums; `products/` and `previews/` stay ignored. |
| Raw/simulated data and other generated roots | Ignored; publish a reviewed compact manifest separately when needed. |
| `results/ml/`, `results/reports/` | Reviewed CSV, JSON, Markdown and checksum evidence. |
| Model binaries, arrays, previews and other result roots | Ignored by default. |
| `submission/aaai27/reviewer_bundle/` | Canonical curated manuscript source and figures. |
| Overleaf ZIP/project exports | Generated and ignored. |

Eligibility is not approval to stage every file. Review size, provenance,
permissions and content. Prediction CSVs can be large; observational metadata
can need additional review. Use explicit paths when staging, not `git add .`.

Ignore rules affect untracked files only. Existing tracked evidence remains;
this cleanup neither removes it nor rewrites Git history. Data/result
tracking does not place those files in the workshop upload: only the curated
reviewer bundle defines that boundary.

Retain resolved settings, independent-parent/split identity, source/environment
metadata, analysis inputs and linking hashes for new reported experiments.
Missing external data/checkpoints must remain explicit reproduction limits.
