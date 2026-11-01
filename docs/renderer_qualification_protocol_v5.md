# Low-memory root-search qualification

Declared before low-memory root qualification or scientific model fitting.

The 8-GB host exhausted committed memory when the first bank attempt used
eight rendering workers. Dense A root searches allocate 2560x2560 search
grids; the partial development bank was rejected (149/150 source-development
proposals reported memory allocation failures). Other in-progress bank jobs
were interrupted, their workers confirmed exited, and partial outputs retained.
No bank became eligible and no scientific model was fitted.

Use `LowMemorySceneEngine` in a new bank version. A starts from Lenstronomy's
specialized analytical one-dimensional solver, then refines every position
against the actual original SIE deflections with a local root solve. Require
residual <=1e-8 arcsec and 2/4 distinct roots. Magnifications use the same
qualified fourth-order stencil. B remains unchanged. Read-only file-backed
training arrays and at most two active rendering workers avoid recreating the
resource failure. No parents, seeds, conditions, metrics or tolerances change.

Compare all 105 multiple-image numerical parents' root counts, matched
positions and amplitudes against v3, and regenerate all 315 expected scenes
using the SAME immutable clean arrays. Require the previous 1e-5 image L1,
1e-5 arcsec position and 1e-3 amplitude gates. The v3 fine-integration,
cross-engine, profile/PSF/detector evidence remains applicable because those
components are unchanged. The additional root report and parent qualification
hash must be bound in a new qualification-v4 report. All failures are retained.

The failed v1 bank attempts are infrastructure failures, not grounds for
discarding physical parents. Rerender the same fixed 700/150 source cohorts
and the same ordered 570-proposal adaptation pool in v2 bank directories.
Only already prescribed independently confirmed one-image target proposals
may be excluded. Numerical/cohort failures still block completion.
