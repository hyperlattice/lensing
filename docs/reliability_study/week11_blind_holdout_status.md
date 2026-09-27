# Week 11 status — independent synthetic blind holdout

Week 11 is complete under frozen protocol v1.11. The three-seed Week 8 domain-randomized ensemble was compared once with the fixed canonical-trained ensemble on 150 fresh parent systems and 750 observations. The holdout generator used seed 11031 and preserved each parent's physical state across five conditions.

## Integrity

- The pre-generation freeze manifest records 117 file hashes, including all source modules, frozen configs and protocols, development split manifests, six checkpoints, and six normalization artifacts.
- The release contains 150 unique parents, 750 observations, and exactly one final `splits/holdout.csv`. All generated observation inputs and products are covered by 18,159 hash records.
- Parent IDs, observation IDs, parameter seeds, and noise seeds are disjoint from development train and validation. Ancestry, canonical identity, strict product loading, and preflight checks passed.
- Both target-free prediction files were written and hashed before scoring: 4,500 per-checkpoint rows and 1,500 ensemble rows. The evaluation completion marker records one target join. Its output hashes were independently checked against the saved files.
- The historical test set and real-data labels were not accessed. No tuning or checkpoint selection followed the result.

## Paired ensemble results

The effect is randomized ensemble absolute error minus canonical-baseline absolute error. Negative values favor the randomized ensemble.

| Condition | Baseline MAE | Randomized MAE | Paired MAE change | Parent-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Canonical | 0.0828 | 0.0655 | -0.0173 | [-0.0253, -0.0095] |
| Pixel scale 0.04 | 0.1696 | 0.1158 | -0.0537 | [-0.0677, -0.0403] |
| Pixel scale 0.06 | 0.1278 | 0.0862 | -0.0417 | [-0.0507, -0.0326] |
| Half exposure | 0.1150 | 0.0936 | -0.0215 | [-0.0318, -0.0108] |
| Double read noise | 0.1155 | 0.0879 | -0.0276 | [-0.0406, -0.0154] |
| Equal-weight four-stress composite | — | — | -0.0361 | [-0.0427, -0.0294] |

All five condition-specific paired intervals are below zero. This is independent synthetic evidence for the tested prior, renderer, preprocessing, and nuisance conditions. The synthetic holdout does not establish performance on CASTLES or other real observations, calibrated HST noise behavior, or agreement across a second full rendering stack.

## Frozen artifacts

- [Protocol v1.11](protocol_v1.11.md)
- [Pre-generation freeze manifest](week11_freeze_manifest.json)
- [Machine-readable result, figures, and tables](../../results/reports/reliability_v1/blind_holdout_week11/)
- [Holdout release and product hashes](../../data/simulations/reliability-blind-holdout-v1/)
- [Exactly-once completion marker](../../results/reports/reliability_v1/blind_holdout_week11/holdout_evaluation_complete.json)

## Next phase

Proceed to Week 12 using these frozen results and prior phase artifacts. Assemble the paired analysis, final tables and figures, failure-case catalog, model/research card, Methods and Results, claim-evidence map, Limitations before Discussion, manuscript draft, and reproducibility statement. Do not select or tune a model from the holdout.
