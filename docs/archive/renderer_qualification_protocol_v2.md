# Adaptive renderer qualification v2, declared before execution

Date 2026-10-27. This extends the failed uniform-quadrature development check;
it does not relax its 0.005 clean/expected normalized-L1 accuracy threshold.
No model-accuracy data have been generated, scored or used to choose this fix.

V1 source export attempted 106 parents at three geometries. One random
proposal returned only one point image and was retained as three failed view
records. Of 315 otherwise renderable views, 36 failed factor-4-to-8 clean-image
convergence, with worst L1 0.0856617; two also failed expected-image convergence.
Original v1 reference arrays, code and configuration remain intact.

The new `adaptive_integration_v1.py` starts with uniform factors 2 and 4,
then refines pixels contributing 90% of the estimated remaining L1 error.
It doubles selected pixel factors to at most 128 until the sum of successive
grid differences is at most 1e-4 of clean integrated flux. These are empirical
error estimates, not guaranteed bounds. Never select pixels from CNN errors,
Einstein-radius labels or final observations. Integration failure rejects the
scene explicitly; it cannot silently return an unconverged image.

Qualification controls:

1. Run the adaptive method on all 315 renderable v1 A views and independently
   on the corresponding B views. Keep original IDs, parameters, geometries and
   point checks. Numerical A/B agreement tolerances remain 1e-5 image L1,
   1e-5 arcsec position and 1e-3 relative point amplitude.
2. For every one of the 36 views that failed the old clean gate, all six edge
   parents, and 12 additional v1 views selected using seed 720301 without
   replacement, compare A adaptive clean/expected scenes with uniform factors
   16 and 32. If 16->32 fails 0.005 for either product, compute factor 64;
   if 32->64 still fails, the qualification gate fails rather than extrapolating
   or relaxing the tolerance. Adaptive versus the finest accepted reference
   must also pass 0.005 for both products. Report every attempted comparison.
3. Reuse v1's 36 center/floor profile cases, 12 PSF numerical cases and detector
   moment check only if its B validation passes them and its source hashes
   remain unchanged. The adaptive method changes integration only.
4. Investigate the single-image parent with both solvers and the independent
   analytic SIE solver. The scientific benchmark population requires 2 or 4
   images, as did the original accepted release. Only an agreed physical
   single-image classification permits a declared population rejection;
   disagreement remains a solver failure. Do not replace the rejected parent.

Record unconditional proposal/rejection counts separately from conditional
numerical agreement on renderable multiple-image systems. A passing
qualification applies to this declared multiple-image population and finite
development grid, not all prior draws or all singular caustics. Report solver
failures separately in future generated splits. The one-image proposal is not
an accuracy result and does not justify removing difficult multi-image cases.

Both phases refuse overwrite and save starts, source snapshots/hashes,
configuration identities, all case outcomes and runtime. Qualification must
finish before scientific bank eligibility is assigned. Any failure requires a
new protocol/version and cannot be handled by target-label tuning.

Pre-execution engineering amendment: the worst-v1-case prototype exhausted
factor 128 at an estimated 3.62e-4, above the new internal 1e-4 goal (though
below the unchanged external 0.005 gate). Before executing the v2 qualification,
raise the refinement cap to 512, which samples .06/512 arcsec at the coarsest
geometry, close to the explicit 1e-4 light-profile floor. Keep all tolerances,
sample identities and reference comparisons unchanged. Also correct the
implementation to snapshot each round's starting pixel factors so a pixel
cannot unintentionally advance several levels within one round. Preserve the
prototype as an engineering attempt; it has no scientific eligibility.
