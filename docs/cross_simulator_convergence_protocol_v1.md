# Alternate-renderer integration convergence protocol v1

Declared after the 50 development controls passed and before running this
additional check. The earlier criteria and outputs are unchanged.

Use the same four development parameter draws and canonical geometry. Render
each with supersampling factors 2, 4 and 8 in the corrected alternate adapter.
For clean and Gaussian-convolved expected images, compute normalized L1
differences between successive factors. On these development systems only,
require the 4-to-8 difference to be no larger than the 2-to-4 difference and
no larger than 0.005 (0.5% normalized L1). This is an engineering tolerance,
not an inferred astronomical precision requirement.

Do not score models, alter the frozen simulator, or overwrite previous arrays.
Store every measured difference and pass/fail decision, with adapter/reference
hashes. A failed case blocks a general convergence claim; do not relax the
tolerance after seeing results. Passing does not cover arbitrary centers,
parameters or PSFs, nor validate the absent central-floor correspondence.
