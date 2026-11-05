# Week 4 — PSF and observation-systematics status

**Status:** complete for the Week 4 timeline scope  
**Date:** 2026-09-26  
**Frozen design:** [`protocol_v1.4.md`](protocol_v1.4.md)  
**Decision:** D-013 in [`decision_log.md`](decision_log.md)

## Work completed

- Rendered 11 matched conditions for the same 150 independent validation parents: canonical; two Gaussian PSF widths; two repeat noise realizations; half exposure; double read noise; zero/double lens-light amplitude; and affine/1st–99th-percentile preprocessing.
- Generated 1,661 descendants total (1 train parent retained for strict-loader checks plus 150 validation parents in each condition). The release had zero rejected renders, exact canonical identity, invariant `theta_E_arcsec`, no ancestry crossing, and successful strict loading of all 1,650 validation descendants.
- Evaluated each condition with the three frozen CNN seeds. Paired effects average seed-specific error changes within each parent before 10,000 parent bootstrap resamples (seed 6031). The historical test split was not read.
- Evaluated the v1.3 truth-free SIS/SIE point-image control on all nine observation-changing conditions. Fitting predictions were saved before validation truth was joined.
- Saved machine-readable predictions, seed metrics, paired predictions/effects, radius/multiplicity summaries, simulated SNR descriptors, fit outcomes, plots, and experiment/release hashes under [`results/reports/reliability_v1/observation_systematics/`](../../results/reports/reliability_v1/observation_systematics/).

## Frozen CNN results

The three-seed mean canonical MAE was **0.0921 arcsec**. Across the ten predeclared intervention comparisons:

| Intervention | Paired MAE change (arcsec) | 95% parent-bootstrap interval | Finding |
|---|---:|---:|---|
| Gaussian PSF width ×0.75 | +0.0009 | [−0.0077, +0.0104] | Inconclusive; close to the practical-small boundary. |
| Gaussian PSF width ×1.25 | +0.0011 | [−0.0060, +0.0079] | Practically small for this tested width range. |
| Noise repeat 1 | −0.0025 | [−0.0125, +0.0083] | Inconclusive; mean prediction shift was 0.0468 arcsec. |
| Noise repeat 2 | +0.0013 | [−0.0090, +0.0126] | Inconclusive; mean prediction shift was 0.0473 arcsec. |
| Exposure ×0.5 | **+0.0463** | **[+0.0297, +0.0649]** | Practically important increase supported. |
| Read noise ×2 | **+0.0350** | **[+0.0254, +0.0451]** | Practically important increase supported. |
| Lens-light amplitude 0 | +0.0058 | [−0.0032, +0.0159] | Inconclusive. |
| Lens-light amplitude ×2 | −0.0052 | [−0.0111, +0.0002] | Inconclusive. |
| Affine normalization | +0.000007 | [−0.000005, +0.000019] | Practically small. |
| 1st–99th percentile normalization | +0.0059 | [−0.0023, +0.0139] | Inconclusive. |

The exposure and read-noise changes are the supported Week 4 CNN failure mechanisms. On average across seeds, MAE rose from 0.0921 arcsec canonical to 0.1384 with half exposure and 0.1271 with doubled read noise. Repeat realizations produced noticeable prediction changes even though neither repeat showed a supported average absolute-error penalty; they remain paired descendants of the same parents.

The PSF results measure width sensitivity only. The validated renderer has a Gaussian PSF control and no validated alternate family, so no Gaussian-to-Moffat or empirical-PSF claim is made. The ten condition intervals are reported without a familywise multiple-comparison adjustment; cross-condition rankings and subgroup results are descriptive.

## Classical point-image control

At canonical conditions the fixed geometric control fit 106/150 systems and fell back to the training-mean estimate on 44; all-parent MAE was 0.2007 arcsec versus the CNN’s seed-mean MAE of 0.0921. Across the conditions, the classical fallback fraction ranged from **24.0% to 43.3%**. The exposure-half condition increased its MAE by **0.0346 arcsec** relative to canonical (95% interval **[+0.0053, +0.0639]**), while doubled read noise was inconclusive (**+0.0159 [−0.0042, +0.0366]**). Removing lens light improved this estimator (**−0.0211 [−0.0427, −0.0017]**), consistent with foreground-light contamination of its peak detector in this synthetic setup.

The classical pipeline remained less accurate than the CNN on all nine tested observation conditions when scored with the prescribed fallback-inclusive all-parent metric. It is an SIS/SIE point-image approximation with an estimated center and substantial fallback use, not a full pixel-likelihood fit or a general comparison of Bayesian lens modelling.

## Interpretation and limitations

Within the validated simulator’s Gaussian PSF and observation-noise controls, the CNN’s average Einstein-radius error was sensitive to lower exposure and higher read noise. The tested Gaussian width changes were small or inconclusive, and lens-light/percentile-normalization effects did not meet the support rule. This does not establish that PSF-family mismatch or lens-light mismatch is unimportant for HST or survey images. Noise and variance are simulator abstractions; the integrated-scene SNR field is a simulator-defined descriptor, not calibrated astronomical SNR. Physical/detectable image counts refer to the renderer’s point-source solutions and amplitude threshold.

## Week 4 decision and ordered next work

Close the Week 4 milestone. Carry the source-position detectability component of H4 into the source/physics phase, where the reference timeline places source-position and multiplicity analysis.

**Week 5, next:**

1. Freeze matched source morphology/profile/size and source-position interventions before predictions.
2. Measure physical versus detectable multiplicity separately, with parent-clustered comparisons.
3. Run a controlled external-shear study with lens truth held fixed except for the declared shear intervention.
4. Attempt a mass-family mismatch only if the renderer, parameter conventions, and truth labels can be independently validated; otherwise record it as unsupported.
5. Rank failure mechanisms using the completed pixel-scale and Week 4 evidence together with Week 5 results.

The historical test split remains quarantined. Week 5 will continue on development parents only; the new blind holdout remains deferred until the Week 11 freeze.

## Detailed artifacts

- CNN report and paired-effects plot: [`results/reports/reliability_v1/observation_systematics/report.md`](../../results/reports/reliability_v1/observation_systematics/report.md), [`paired_effects.png`](../../results/reports/reliability_v1/observation_systematics/paired_effects.png)
- CNN paired effects and predictions: [`cnn_paired_effects.csv`](../../results/reports/reliability_v1/observation_systematics/cnn_paired_effects.csv), [`cnn_predictions.csv`](../../results/reports/reliability_v1/observation_systematics/cnn_predictions.csv)
- Classical report and comparisons: [`classical/report.md`](../../results/reports/reliability_v1/observation_systematics/classical/report.md), [`classical_metrics.csv`](../../results/reports/reliability_v1/observation_systematics/classical/classical_metrics.csv), [`classical_paired_effects.csv`](../../results/reports/reliability_v1/observation_systematics/classical/classical_paired_effects.csv)
- Counterfactual release gate: [`run_manifest.json`](../../data/simulations/reliability-observation-systematics-v1/run_manifest.json)
