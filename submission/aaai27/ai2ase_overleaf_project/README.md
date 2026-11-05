# Reviewer-facing AI2ASE-27 source bundle

This directory contains only the source assets intended for a workshop
submission build:

- `paper.tex` and `paper.bib`;
- the official `aaai2027` style and bibliography files;
- the two local result figures referenced by the paper.

It intentionally excludes project notebooks, development logs, raw or derived
data, intermediate checkpoints, private benchmark reports, author-kit archives,
and local machine paths. Build the anonymous paper from this directory with the
official AAAI-27 toolchain, then upload the resulting PDF to CMT. Do not upload
the repository root or the parent `submission/aaai27` directory as a substitute
for this bundle.

## Results

The controlled equal-budget experiment uses a frozen 150-parent synthetic
holdout and ten paired training seeds. Four-stress MAE is:

| Arm | Four-stress MAE (arcsec) |
|---|---:|
| C: canonical | 0.141251 |
| G: uniform variation | 0.111507 |
| R: difficult-condition sampling | 0.107385 |

The primary G-minus-C contrast is `-0.029744` arcsec with a parent-bootstrap
95% interval of `[-0.045725, -0.012141]`. The R-minus-G contrast is
`-0.004122` arcsec with interval `[-0.026208, 0.009341]`, so the paper does
not claim a reliable extra benefit from failure weighting.

The result applies only to the stated synthetic generator, target definition,
preprocessing, and stress conditions. It is not a real-data score, a test of
transfer to an independent simulator, or a numerical comparison with published
models.

## Simulated-lens PDF

[simulated_lens_examples_and_results.pdf](figures/simulated_lens_examples_and_results.pdf)
shows one synthetic lens under five observation conditions, five independent
canonical simulated lenses, the aggregate MAE/contrast figures, and a concise
methodology/claim-boundary page. The images are normalized synthetic inputs;
they are not telescope observations.

## Overleaf

Create a new Overleaf project using **New Project -> Upload Project** and upload
the contents of this directory as a ZIP. Set `paper.tex` as the main document
and use the `pdfLaTeX` compiler. Overleaf should find `paper.bib`, the local
AAAI style files, and both figures without any path changes.
