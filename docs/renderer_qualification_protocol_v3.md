# Point-Jacobian refinement for adaptive qualification

Date 2026-10-27, before adaptive B validation. Preserve v1/v2 records.

V1 B validation passed all 36 central-floor profile checks (worst relative
error 1.89e-13), all PSF and detector checks, and 303/315 cross-engine image
comparisons. Twelve expected-image comparisons exceeded 1e-5 even though
clean images agreed near 1e-9 and root positions near 1e-10 arcsec. Their
point-amplitude differences implicate numerical magnification; the old B
adapter uses a two-point derivative at step 1e-5 arcsec.

Use the new `PreciseSceneEngine` for B: independently solved positions with
five-point fourth-order derivatives of external deflections at step 1e-4.
Use its inherited, unchanged A branch. Reuse v2's immutable A adaptive arrays
and fine-uniform reference comparisons because no A quantity changes. Record
both the inherited A-source hashes and the new B-source hashes; do not modify
v1 or v2 source/configuration files to make a stale report appear current.

Before full B validation, test magnification step stability at 1e-4 versus
5e-5 for the four previously failing physical parents and require relative
agreement <=1e-6. Compare all 315 adaptive B views to A using the original
1e-5 clean/expected L1, 1e-5 arcsec position and 1e-3 amplitude tolerances.
Retain and report every failure. The single-image proposal still requires the
independent classification prescribed by v2. No change to sample selection,
integration accuracy thresholds, observing conditions or target metrics.

The new final qualification report must combine v2 fine-reference controls,
this B comparison, unchanged profile/PSF/detector controls and unconditional
proposal accounting. It qualifies only the stated multiple-image population
on this finite numerical-development grid, with shared project integration,
PSF and detector. It is not model-performance evidence.
