# Reliability study protocol v1.3 — truth-free classical pixel-scale control

**Status:** frozen before classical validation predictions  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Pixel-scale design:** [`protocol_v1.1.md`](protocol_v1.1.md)  
**Mitigation design:** [`protocol_v1.2.md`](protocol_v1.2.md)

This amendment completes the Week 3 classical-method specification without changing the CNN results, pixel-scale intervention, parent set, support rules, or timeline. It defines a limited image-derived point-image geometry control because the existing pixel fitter initializes from hidden truth and cannot serve as a blind baseline.

## Permitted inputs

For validation descendants, read only the observed science image, released variance map, invalid and coverage masks, released PSF kernel, pixel scale, sample/parent IDs, split, and shape/PSF metadata. Do not load parameter sidecars, image-position diagnostics, labels, or source/lens truth for fitting. Use parent truth only after estimates are written, to score results. The fallback estimate is the mean `theta_E_arcsec` from the 700-parent training split, computed without validation labels.

## Frozen estimator

1. Form a PSF-matched signal-to-noise map by convolving the observed science image with the released PSF and dividing by the square root of the released variance convolved with PSF squared. Invalid and uncovered pixels are excluded.
2. Estimate the lens center as the maximum of a Gaussian-smoothed science image within 0.35 arcsec of the image field center. Use smoothing sigma 0.15 arcsec.
3. Retain local maxima with matched-filter S/N at least 5, outside a 0.25 arcsec central exclusion and inside 2.25 arcsec of the estimated center. Enforce at least 0.30 arcsec between peaks and retain at most the four highest-S/N peaks.
4. With exactly two detected images, solve the SIS lens equation by least squares for `theta_E` and the two unknown source coordinates. With three or four detections, fit an SIE point-image lens equation for `theta_E`, axis ratio, orientation, and source coordinates, holding the image-estimated lens center fixed. Use bounds `theta_E in [0.25, 2.25]` arcsec, `q in [0.3, 1]`, `phi in [0, pi]`, and source coordinates in `[-0.5, 0.5]` arcsec.
5. If fewer than two peaks are found, the optimizer fails, or the estimate is outside its bounds, return the training-mean fallback and mark the estimate as fallback. Report detection and fit success separately from all-sample error.

Run this fixed estimator on the same 150 validation parents at 0.04, 0.05, and 0.06 arcsec/pixel. Compare with the frozen CNN on those parents; average CNN seed errors within each parent before parent bootstrap. Do not use the historical test split. Report failures and fallback use; do not present this as a full pixel-likelihood fit or posterior uncertainty analysis. The SIS/SIE point-image approximation and image-center assumptions are limitations of this control.
