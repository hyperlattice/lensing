# Week 5 — Source and physics mismatch status

**Status:** complete for the validated source/position/shear scope  
**Date:** 2026-09-26  
**Frozen design:** [`protocol_v1.5.md`](protocol_v1.5.md)  
**Decision:** D-015 in [`decision_log.md`](decision_log.md)

## Work completed

- Generated 11 matched conditions for the same 150 H1 validation parents: canonical; two source Sersic indices; two effective radii; four coordinate shifts; and two external-shear components.
- The release contained 1,661 descendants (1 train parent for loader checks plus 150 validation parents in each condition). It had zero rejected renders, exact canonical identity, unchanged `theta_E_arcsec`, isolated ancestry/splits, and successful strict loading.
- Scored all conditions with the three frozen CNN seeds. Parent-paired effects average seed-level errors within each independent system before the 10,000-resample bootstrap (seed 6031).
- Measured three separate multiplicity quantities: renderer image roots, renderer amplitude-threshold count, and peak count from an observation-only PSF-matched S/N detector (threshold 5). The detector table was saved before physical multiplicity labels were joined. All 1,650 parent-condition rows passed a target-consistency check after the ID-based join.
- Ran the frozen v1.3 point-image classical control on all 11 conditions and retained detection, fit, fallback, residual, and objective-count outcomes.
- Built a first descriptive failure-mechanism ranking across Weeks 3–5, including paired effects and fixed `theta_E` bin results.
- Preserved outputs and hashes under [`results/reports/reliability_v1/source_physics/`](../../results/reports/reliability_v1/source_physics/).

## CNN paired results

The canonical three-seed mean MAE was **0.0921 arcsec**. Results below use 150 paired parents and unadjusted 95% parent-bootstrap intervals. Ten intervention comparisons were predeclared.

| Intervention | Paired MAE change (arcsec) | 95% interval | Interpretation under the 0.01 arcsec rule |
|---|---:|---:|---|
| Sersic index 0.5 | +0.0097 | [+0.0002, +0.0207] | Positive, but mean is just below the practical-effect threshold. |
| Sersic index 5.0 | −0.0103 | [−0.0186, −0.0024] | Error decreased for this tested profile condition. |
| Effective radius ×0.5 | **+0.0129** | **[+0.0041, +0.0222]** | Practically important increase supported. |
| Effective radius ×2 | −0.0156 | [−0.0260, −0.0059] | Error decreased for this tested size condition. |
| Source x −0.15 arcsec | **+0.0185** | **[+0.0008, +0.0374]** | Practically important increase supported. |
| Source x +0.15 arcsec | **+0.0248** | **[+0.0093, +0.0398]** | Practically important increase supported. |
| Source y −0.15 arcsec | **+0.0155** | **[+0.0013, +0.0306]** | Practically important increase supported. |
| Source y +0.15 arcsec | +0.0108 | [−0.0038, +0.0255] | Inconclusive. |
| External shear, `gamma1=0.05` | +0.0063 | [−0.0059, +0.0183] | Inconclusive. |
| External shear, `gamma2=0.05` | +0.0075 | [−0.0015, +0.0164] | Inconclusive. |

The response is not monotonic across the tested source profiles: a smaller effective radius increased error, while a doubled radius and index 5 reduced it. Source-position shifts produced the clearest source-related instability. Their mean absolute prediction shifts ranged from **0.086 to 0.101 arcsec**. Shear changed predictions by **0.057–0.071 arcsec** on average, but the paired error increases did not meet the support rule.

The fixed radius-bin analysis shows source-position penalties concentrated mainly in the `[0.5,1.0)` and `[1.5,2.0]` arcsec bins; estimates in the middle bin were close to zero. These subgroup intervals are unadjusted and are preserved in [`radius_bin_effects.csv`](../../results/reports/reliability_v1/source_physics/radius_bin_effects.csv).

## Physical and detectable multiplicity

At canonical conditions, the renderer produced a mean of **2.52 physical point-image roots** per parent. Its 0.05-amplitude threshold count agreed with physical roots in **100%** of canonical cases and **99.3–100%** across the Week 5 conditions. The observation-only S/N≥5 peak detector matched the physical count in only **46.7–58.0%** of parents, depending on condition. Its mean canonical count was 2.22 peaks.

Source-coordinate shifts changed the physical root distribution: mean counts fell to **2.37** for x+0.15, **2.39** for x−0.15, and **2.39** for y+0.15; y−0.15 averaged 2.48. The `gamma2=0.05` shear condition averaged 2.61 roots. These are renderer outcomes for the fixed intervention levels, not a calibrated detection/completeness model. The observation-based count comes from one fixed peak detector, not universal “detectability.”

## Classical comparison and failure ranking

The truth-free point-image control fit **106/150** canonical systems and used its training-mean fallback on **44/150**. Its fallback rate ranged from **22.7% to 42.7%** across Week 5 conditions. Its all-parent MAE exceeded the CNN seed-mean MAE in every condition; for source x+0.15, for example, the point-image control had MAE **0.2419 arcsec** with 64 fallbacks versus **0.1169 arcsec** for the CNN. This remains a limited analytic SIS/SIE point-image control, not a full pixel-likelihood method.

The first failure ranking is led by the Week 3 native pixel-scale penalties, then Week 4 observation quality, followed by Week 5 source-position and size effects:

| Rank signal | Paired MAE change (arcsec) | 95% interval |
|---|---:|---:|
| Pixel scale 0.04 arcsec/pixel | +0.0707 | [+0.0519, +0.0900] |
| Pixel scale 0.06 arcsec/pixel | +0.0489 | [+0.0326, +0.0647] |
| Half exposure | +0.0463 | [+0.0297, +0.0649] |
| Double read noise | +0.0350 | [+0.0254, +0.0451] |
| Source x +0.15 arcsec | +0.0248 | [+0.0093, +0.0398] |
| Source x −0.15 arcsec | +0.0185 | [+0.0008, +0.0374] |
| Source y −0.15 arcsec | +0.0155 | [+0.0013, +0.0306] |
| Source effective radius ×0.5 | +0.0129 | [+0.0041, +0.0222] |

The complete 22-comparison ranking, including negative and inconclusive effects, is in [`failure_mechanism_ranking.csv`](../../results/reports/reliability_v1/source_physics/failure_mechanism_ranking.csv). The ranking is descriptive; intervals are not corrected for the number of conditions.

## Scope and limitations

The validated renderer has one elliptical Sersic source-light family and an SIE lens, optionally with external shear. It does not support a validated alternative source family or EPL/other mass-profile intervention, so neither was tested. The ±0.15 arcsec coordinate shifts probe source-position sensitivity but do not force every parent across its caustic; the released root counts show which systems changed multiplicity. The fixed shear value and extended source-profile settings are synthetic stresses, not population-calibrated priors. None of these results establishes real-observation accuracy.

## Week 5 decision and ordered next work

Close the Week 5 milestone for the supported interventions. Source position and source size show measurable sensitivity; shear effects remain inconclusive. Keep the source-family and mass-family gaps explicit in later claims.

**Week 6, next:**

1. Validate simple baselines: training-mean, an auditable geometric/radial estimate where applicable, and a regularized model on fixed image features.
2. Produce a parent-count learning curve with a fixed validation set and the established seeds.
3. Complete the broader classical-versus-CNN comparison, preserving failures and fallbacks.
4. Run the predeclared image interventions/shortcut checks from the source plan, paired against unchanged controls.
5. Decide whether the dominant limitation is sample size, representation, or distribution only after those comparisons.

The historical test remains quarantined. Do not begin cross-simulator work or mitigation until Week 6 closes.

## Detailed artifacts

- CNN paired-effects report and plots: [`report.md`](../../results/reports/reliability_v1/source_physics/report.md), [`paired_effects.png`](../../results/reports/reliability_v1/source_physics/paired_effects.png), [`multiplicity_comparison.png`](../../results/reports/reliability_v1/source_physics/multiplicity_comparison.png)
- Parent-level outcomes: [`cnn_paired_effects.csv`](../../results/reports/reliability_v1/source_physics/cnn_paired_effects.csv), [`physical_detectable_observed_multiplicity.csv`](../../results/reports/reliability_v1/source_physics/physical_detectable_observed_multiplicity.csv), [`multiplicity_scored.csv`](../../results/reports/reliability_v1/source_physics/multiplicity_scored.csv)
- Classical outcomes: [`classical/report.md`](../../results/reports/reliability_v1/source_physics/classical/report.md), [`classical_metrics.csv`](../../results/reports/reliability_v1/source_physics/classical/classical_metrics.csv)
- Release gate: [`run_manifest.json`](../../data/simulations/reliability-source-physics-v1/run_manifest.json)
