# Renderer qualification v1, declared before execution

Date 2026-10-27. Configuration: `configs/renderer_qualification_v1.yaml`.
This is numerical development only. No CNN predictions, target-domain method
selection or old final-holdout access is permitted.

The new `benchmark_renderer_v1.py` leaves frozen simulation files unchanged.
It uses each package's mass/light/solver implementation and project-owned
midpoint integration, bilinear point deposition, finite PSF convolution and
Poisson/read detector. The alternate light evaluator floors the package's
circularized elliptical radius at 1e-4 arcsec and matches the original 1000
effective-radius cutoff. Existing b_n, light-radius/intensity and mass-radius
mappings are retained. Thus this is a controlled external-profile renderer,
not an entirely independent end-to-end telescope simulator.

Use 100 fresh physical proposals from seed 720101 under the existing source
prior, plus six deterministic edge cases derived from separate indices:
nearly aligned, small source offset, lens center on pixel center, lens center
on a subpixel boundary, compact high-index source, and a low-q elongated case.
Exact source/lens alignment is singular for point roots and is not proposed.
Numerical cases may fail and must remain in the failure ledger; they cannot
be replaced with favorable cases.

For every parent, test all three native geometries (256/.05, 320/.04,
213/.06). Export A clean images at factors 2, 4, 8, its factor-8 expected
scene, positions and amplitudes. Compare factor 4 versus factor 8 normalized
L1 for clean and Gaussian-convolved scenes at 0.005. Factor 8 is only a
candidate reference: if 4->8 fails, do not declare factor 8 converged; extend
to 16 or revise the numerical protocol with a new experiment ID.

Compare B at the same factor to A with normalized L1 <=1e-5 for clean and
expected scenes, matched point multiplicity, max matched position error
<=1e-5 arcsec and relative amplitude error <=1e-3. Report every rejection and
the unconditional case count. A failed gate blocks scientific qualification;
equivalent images do not establish domain-shift robustness.

Also compare 36 light configurations (n=.5,1,2,4; q=1,.7,.4; phi=0,.3,1.1)
on radii including zero, 1e-8, 1e-5, 1e-4, 3e-4 and extended wings at an
off-center location. Relative error tolerance is 1e-9 with denominator floor
1e-100. Include an independent analytic central-floor value. Test native
Gaussian and Moffat kernels for normalization/finite values and detector
first/second moments; these are physical numerical controls, not Moffat
model-development scores.

Save source/config/package hashes, parameter identity ledger, all attempted
cases, failures and runtimes. Refuse overwrite; a run-start marker prevents
implicit restart. Separate A export and B validation use isolated installed
environments. A generated-array checksum failure blocks validation.

This stage cannot certify all possible caustics, solver completeness or
arbitrary physical priors. These limits must be carried into any benchmark.
