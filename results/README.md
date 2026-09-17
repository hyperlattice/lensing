# Results layout

`results` contains derived evidence and review artifacts. It is separate from
`data` so plots and reports never become model inputs by accident.

| Location | Contents |
|---|---|
| `previews/actual/<release>/` | FITS previews and contact sheets for actual observations. |
| `previews/simulated/<release>/` | FITS previews and contact sheets for simulated releases. |
| `reports/` | Classical recovery, validation, and bounded real-comparison evidence. |
| `ml/<experiment>/` | ML training histories, predictions, reports, and plots. |

New preview commands write to `results/previews/actual` or
`results/previews/simulated` automatically. These images are generated review
artifacts and are ignored by Git. The historical `results/preview` directory
has been migrated to `previews/actual/castles-development-v1`.

ML checkpoints (`best.pt`) are also ignored because they are binary artifacts;
their configs, hashes, metrics, prediction tables, and plots remain available
in the experiment directory.
