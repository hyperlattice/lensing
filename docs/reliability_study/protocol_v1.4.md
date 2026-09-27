# Reliability study protocol v1.4 — Week 4 observation systematics

**Status:** frozen before Week 4 predictions  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior completed amendments:** [`protocol_v1.1.md`](protocol_v1.1.md), [`protocol_v1.2.md`](protocol_v1.2.md), [`protocol_v1.3.md`](protocol_v1.3.md)

This amendment predeclares the Week 4 observation-systematics benchmark. It preserves the 150-parent validation set, frozen three-seed CNN, the historical-test quarantine, the 0.01 arcsec practical-effect threshold, and the 10,000-resample parent bootstrap (seed 6031).

## Matched conditions

Use all 150 independent validation parents from `sim-eng1000-v1`. Generate each condition by rerendering its unchanged canonical parent state. Keep `theta_E_arcsec`, mass model, lens/source state, source position, detector geometry, pixel scale (0.05 arcsec/pixel), and parent split fixed except for the named change.

| Family | Frozen intervention | Interpretation |
|---|---|---|
| Reference | `canonical` | Exact unchanged parent rerender; identity gate required. |
| PSF width | `psf-width-075`, `psf-width-125` | Gaussian PSF FWHM multiplied by 0.75 and 1.25, respectively. PSF family remains Gaussian. |
| Noise repeat | `noise-repeat-1`, `noise-repeat-2` | Same expected scene and detector settings; new deterministic noise seeds derived from the parent seed with offsets 1 and 2. Repeats remain clustered by parent. |
| Observation quality | `exposure-half`, `read-noise-double` | Exposure multiplied by 0.5 or read noise multiplied by 2, one factor at a time. |
| Lens light | `lens-light-zero`, `lens-light-double` | Lens-light amplitude set to zero or multiplied by 2; all other parent state fixed. |
| Input normalization | `normalization-affine`, `normalization-percentile` | Reprocess the unchanged detector observation with robust affine (median/MAD) or mask-aware 1st–99th percentile normalization. The frozen CNN's train-only global normalization statistics are then applied exactly as before. |

The renderer provides only a validated analytic Gaussian PSF. Do not simulate or claim a PSF-family comparison in this phase. Exposure/read-noise settings and preprocessing variance are simulator controls, not calibrated instrument noise.

## Evaluation and reporting

- Evaluate the three frozen `theta-e-cnn-seeds-v1` checkpoints on all 150 parents in all 11 conditions. Do not retrain or select on Week 4 outputs.
- Compare each intervention with its parent's canonical image. Average paired error differences across CNN seeds within parent before the 10,000-resample parent bootstrap (seed 6031). Report seed-specific metrics separately.
- Report MAE, RMSE, median absolute error, median/mean relative error, signed bias, P90/P95 absolute error, prediction shift, fixed `theta_E` bins `[0.5,1.0)`, `[1.0,1.5)`, `[1.5,2.0]`, and physical/detectable image-count strata when available.
- Report multiplicity counts and a simulated integrated-scene SNR proxy descriptively. These simulator-derived descriptors are never model inputs and are not calibrated astronomical SNR.
- Run the frozen truth-free SIS/SIE point-image geometry control from v1.3 on the same conditions. It reads only observed science, released variance/masks, PSF, and pixel geometry while fitting; labels are joined only after predictions are written. Report fallback and detection rates. This remains a point-image control, not pixel-likelihood inference.
- Report paired CNN-versus-classical effects where both predictions exist. Do not treat repeated noise descendants as independent systems.
- Preserve predictions, paired effects, seed metrics, summary metadata, and plots under `results/reports/reliability_v1/observation_systematics/`.
- Do not open the historical test split. No result from this validation audit supports real-instrument accuracy.

**Support rule:** a practically important penalty is supported when paired MAE increases by at least 0.01 arcsec and the 95% parent-bootstrap interval is wholly above zero. Call an effect practically small only when the whole interval lies in `[-0.01,+0.01]` arcsec; otherwise call it inconclusive. Treat the multi-condition family scan as a set of predeclared comparisons and disclose the number of comparisons; do not promote an unadjusted subgroup result as confirmatory.
