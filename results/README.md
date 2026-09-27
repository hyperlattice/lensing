# Results layout

`results` contains derived evidence and review artifacts. It is separate from
`data` so plots and reports never become model inputs by accident.

| Location | Contents |
|---|---|
| `previews/actual/<release>/` | FITS previews and contact sheets for actual observations. |
| `previews/simulated/<release>/` | FITS previews and contact sheets for simulated releases. |
| `reports/` | Classical recovery, validation, and bounded real-comparison evidence. |
| `ml/<experiment>/` | ML training histories, predictions, reports, and plots. |
| `reports/reliability_v1/` | Phase-level paired effects, metrics, and reports for the reliability study. |
| `reports/final_v1/` | Final study tables, figures, analysis summary, and input/output manifests. |

New preview commands write to `results/previews/actual` or
`results/previews/simulated` automatically. These images are generated review
artifacts and are ignored by Git. The historical `results/preview` directory
has been migrated to `previews/actual/castles-development-v1`.

Model checkpoint files (`.pt`, `.pth`, `.ckpt`) are ignored because they are
binary artifacts. Their configs, hashes, metrics, prediction tables, and plots
remain eligible for version control. Keep the exact-once marker and prediction
provenance for the blind holdout with its report; do not regenerate those files
by rerunning holdout inference.
