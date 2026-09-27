# Week 12 status — final analysis and manuscript package

Week 12 is complete. The package is built from the recorded Week 1–11 protocols, phase reports, machine-readable effects, the Week 11 exactly-once predictions/score, and the frozen release manifest. It does not rerun model inference or select a model from holdout results.

## Delivered

- Harmonized paired-effect evidence ledger covering Weeks 3–11.
- Final holdout ensemble/seed metrics, paired effects, target-range coverage, and fixed-threshold error counts.
- Full 750-observation final-model failure catalog with comparator error, seed spread, condition, error tier, and available observation diagnostics; top-error table and error-count summary.
- Three reproducible figures: parent-paired effects, ensemble MAE, and error distributions.
- Research/model card, manuscript draft, claim-evidence map, and reproducibility statement.
- Input/output package manifest with SHA-256 hashes.

The paper states Limitations before Discussion and distinguishes synthetic in-domain, synthetic stress, the limited Week 7 equation path, and descriptive CASTLES evidence. It makes no real-image accuracy claim.

## Primary final result

The randomized ensemble's equal-weight four-stress paired MAE change versus the fixed canonical-trained ensemble was **-0.0361 arcsec** (95% parent-bootstrap interval [-0.0427, -0.0294]) on 150 independent synthetic parents. This supports a result for the tested simulator/prior and conditions only.

## Package files

- [Manuscript draft](final/manuscript_draft.md)
- [Research/model card](final/model_research_card.md)
- [Claim-evidence map](final/claim_evidence_map.md)
- [Reproducibility statement](final/reproducibility_statement.md)
- [Tables and figures](../../results/reports/final_v1/)
- [Package manifest](../../results/reports/final_v1/package_manifest.json)
