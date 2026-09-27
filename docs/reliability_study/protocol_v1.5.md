# Reliability study protocol v1.5 — Week 5 source and physics interventions

**Status:** frozen before Week 5 predictions  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Completed amendments:** [`protocol_v1.1.md`](protocol_v1.1.md), [`protocol_v1.2.md`](protocol_v1.2.md), [`protocol_v1.3.md`](protocol_v1.3.md), [`protocol_v1.4.md`](protocol_v1.4.md)

This amendment predeclares source-profile, source-position, and external-shear experiments. It preserves the 150-parent H1 validation set, frozen three-seed CNN, target convention, 0.01 arcsec practical-effect threshold, 10,000-resample parent bootstrap (seed 6031), and the historical-test quarantine.

## Matched conditions

Use the same 150 independent validation parents as Weeks 3–4. Rerender the same canonical parent state at 0.05 arcsec/pixel. Keep the SIE Einstein radius, ellipticity, lens center, lens-light state, PSF, detector parameters, preprocessing, and parent noise seed fixed unless a row names the single changed physical factor.

| Family | Frozen intervention | Isolated change |
|---|---|---|
| Reference | `canonical` | Unchanged parent rerender with identity checks. |
| Source profile | `source-n-050`, `source-n-500` | Sersic index set to 0.5 or 5.0, respectively. The high-index condition modestly extends beyond the training range. |
| Source size | `source-re-half`, `source-re-double` | Source effective radius multiplied by 0.5 or 2.0. |
| Source position | `source-x-minus`, `source-x-plus`, `source-y-minus`, `source-y-plus` | Add or subtract 0.15 arcsec from one source-center coordinate. Point-image roots are recomputed. |
| External shear | `shear-gamma1-005`, `shear-gamma2-005` | Add an external shear component of 0.05 along the two predeclared component orientations while holding SIE `theta_E` and ellipticity fixed. |

The renderer exposes a single elliptical Sersic extended-source family. It has no validated alternative source family. The observation renderer exposes SIE, optionally with external shear; the maintained rendering path does not provide a validated EPL/other mass-profile intervention. Therefore source-family and mass-family mismatch are not claimed or improvised in this phase. The controlled shear study proceeds as its own physical perturbation.

## Multiplicity and comparison rules

- Keep three counts distinct: renderer physical point-image roots; renderer amplitude-threshold count at its frozen 0.05 threshold; and observation-only peak detections using the v1.3 PSF-matched S/N detector at threshold 5. The latter uses observed science, variance, masks, PSF, and geometry only.
- Evaluate the three frozen CNN seeds on every condition. Pair each intervention with its parent's canonical result, average seed deltas within parent, and bootstrap independent parents.
- Run the v1.3 truth-free SIS/SIE point-image control on the same conditions. Save fits/detection results before joining validation targets and report fallback/convergence outcomes.
- Report aggregate paired effects and effects within the frozen Einstein-radius bins `[0.5,1.0)`, `[1.0,1.5)`, and `[1.5,2.0]` arcsec. Also report physical, amplitude-threshold, and observed-peak multiplicity strata with counts.
- Preserve effects, per-seed metrics, predictions, multiplicity diagnostics, classical outcomes, and hashes under `results/reports/reliability_v1/source_physics/`.
- Build the first failure-mechanism map by joining Week 3 native pixel-scale and Week 4 observation-quality results with Week 5 effects. Rank positive paired MAE increases and display intervals/support status; treat the cross-family ranking as descriptive and disclose the comparison count.

**Support rule:** a practically important error penalty requires mean paired MAE increase of at least 0.01 arcsec and a 95% parent-bootstrap interval wholly above zero. Call an effect practically small only if the entire interval lies within `[-0.01,+0.01]` arcsec; otherwise classify it as inconclusive. Do not treat descendants as independent parents. No model selection may use the historical test split.
