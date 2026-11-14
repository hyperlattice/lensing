# Cross-implementation validation protocol v2

Declared before measuring the new light/image controls on 2026-10-27.
These controls use development systems only and do not score CNNs.

The v1 development images remain immutable. A corrected adapter and new
output/config version will be used for any light-mapping change. This protocol
does not authorize revisiting the original frozen holdout.

## Identified conventions

Installed Lenstronomy 1.14.2 uses the circularized Sersic radial coordinate by
default and `b_L(n) = max(1.9992*n - 0.3271, 1e-5)`. Installed PyAutoGalaxy
2026.10.26.1 also uses the circularized elliptical coordinate, but its Sersic
constant `b_A(n)` is a higher-order expansion. Copying intensity and effective
radius directly therefore changes the brightness function even at fixed n.

To preserve the original radial brightness function outside the numerical
central floor, set:

- `R_A = R_L * (b_A / b_L)**n`
- `I_A = I_L * exp(b_L - b_A)`

This follows by equating the constant term and the coefficient of `R**(1/n)`
in the exponential. It is a conversion between implementation parameters;
the converted backend radius is not an assertion that the original numerical
radius was an exact physical half-light aperture. The original evaluator
floors radii at 1e-4 arcsec. Explicitly disclose central-floor differences.

The Gaussian control also needs the same finite kernel support: the frozen
pipeline uses the next odd size at least `max(5, ceil(10*sigma_pixels))`.
Declare that support in the new implementation instead of treating different
truncations as equivalent PSFs. Moffat remains an intentional new family.

## Controls and acceptance criteria

1. Export original Lenstronomy evaluations on offset noncentral coordinate
   grids for 4 Sersic indices (0.5/1/2/4), 3 axis ratios (1/0.7/0.4), and 3
   orientations (0/0.3/1.1 radians). Test both unchanged direct parameters and
   converted parameters; only converted parameters are candidates for release.
   Require maximum relative brightness error <=1e-10 on points at elliptical
   radius >=1e-4 arcsec. Compare analytic integrated flux under the two
   parameter conventions with relative error <=1e-10. Keep central-floor
   diagnostics separately from this noncentral equivalence gate.
2. For the four existing development parameter draws, export original clean
   and Gaussian-convolved images, independently solved positions and
   magnifications. Compare the new path with the same native geometry and
   supersampling. Require clean and expected-image normalized L1 errors <=1e-5,
   matched positions <=1e-5 arcsec and point amplitudes relative error <=1e-3.
   Match points by assignment rather than order. Do not transplant original
   positions into the alternate solver. Rejections fail the relevant gate.
3. Check nonnegative unit-sum Gaussian/Moffat kernels and impulse flux with
   relative tolerance 1e-6. Show boundary truncation separately. Compare
   original and alternate Gaussian support and values explicitly.
4. Exercise the actual new detector helper on fixed flat fields with exposure
   50/100 seconds, read noise 0/2 electrons, and signal rates 0.1/1.0
   electrons/s/pixel, with background 0.03 electrons/s/pixel. For 32 draws of
   128x128 pixels per setting, require mean residual within 5 standard errors
   and measured variance within 2% of `(mean_counts + read_noise**2)/t**2`.
   Use fixed independent seeds; these are numerical controls, not telescope
   calibration.

Store input hashes and both environments, reference arrays, per-case errors,
and all pass/fail decisions. If a predeclared control fails, record it; do not
quietly relax its tolerance. Changed algorithms/settings need a new record.

Passing these gates establishes the tested implementation equivalence and
noise moments. Pixel-integration convergence beyond matching the frozen
discretization, broader parameter coverage, empirical instrument effects and
a strict final dataset/loader contract remain separate gates. Native
PyAutoLens simulator components should be adopted where feasible and every
shared/project-owned component disclosed. A passed numerical equivalence
check alone is not a novel distribution shift or real-data validation.
