# Maintained study

The question is whether source-only variation in observing conditions improves
Einstein-radius point regression on new synthetic parent systems under equal
CNN, optimizer-update and source-exposure budgets.

C uses canonical images, G samples uniformly from seven observation cells, and R
uses probabilities estimated from source-development errors. All use the same
four-block CNN, AdamW, Huber loss, source-only normalization, initialization seed
and parent draw stream. The recorded study uses 880 updates, batch size 16 and
ten seeds. The seven cells cover scale, exposure, read noise, Gaussian PSF width
and fresh noise. The final manifest has 150 parents under canonical and four
stress conditions. Parent systems are the independent units; their views and
training seeds do not increase the number of independent lenses.

The primary endpoint averages the four stress errors within each parent, then
across parents and fixed seed runs. Bootstrap draws resample parents jointly
across views and seeds. Canonical performance and seed variation are reported
separately. Preserve the manuscript's contrast ordering and interpretation;
R versus G is not automatically significant because R has a lower point estimate.

The secondary paired-renderer comparison uses fixed primary checkpoints and a
new paired cohort, including engineered Moffat conditions. Identical image
products establish an equivalence control, not independent-simulator transfer.
Neither experiment validates real HST parameter estimates or uncertainty coverage.
