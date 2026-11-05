# Local provenance and verification

Prepared in `C:/Users/richa/Desktop/charlie_lensing`, branch `richard_changes`, base commit `6362f2cc228ec2343dd36f347177adcab80bfa1a`. Review cutoff: 4 November 2026.

The manuscript, existing bibliography, experiment runner and training/simulation sources below are unchanged against HEAD. Their contents informed the local-project appraisal, not the external literature findings. Numerical study results were read from the manuscript and existing report; no new training, scoring or raw-output reproduction was performed.

## Source fingerprints (SHA-256)

- `submission/aaai27/reviewer_bundle/paper.tex`: `ace12f26ff9c968312a4932a271231fa9cdab6a7a951913264f7ddef4c250cd0`
- `submission/aaai27/reviewer_bundle/paper.bib`: `afae0350cd77ebd8e26e57030145434a1718592ec76bc22d13e0f10030178fef`
- `docs/equal_budget_final_v1.md`: `de3ec19cdc37eacf951162f74c21ef4e71faa6090e3298b446a740744dd74bc3`
- `scripts/run_equal_budget_final_v1.py`: `29e3647c7dbb20bfc9ad39e8cb9ad6ee7dc729b92ef89b226deb9d3ad556215c`
- `src/ml/robustness_training_v1.py`: `96c1112c6b83155405714a1cbcd8baf836a17510678799f2226e88eda977287a`
- `src/simulation/autolens_backend_v2.py`: `b2ec2a1822c523c8b28a2c280eca1e4e3777cdf8cf19208c56f3904eb4474689`

## Review construction

- Public arXiv records and publisher/Crossref metadata verified bibliographic identity and linked journal/preprint reports. The CSV contains reviewer paraphrases rather than copied abstracts.
- Direct Crossref requests initially encountered one rate-limit response; that citation was retrieved successfully on the later check. HOLISMOKES correction identity is verified, but its full text remains unavailable after publisher HTTP 403 responses.
- Ten works received selected full-text-section assessment. The 60-work count includes reviews, software and general methodology, not solely independent lens-inference experiments.
- The exact retained web-discovery queries and their date are in `search_log.json`. Early broad orientation searches are not retained as exact queries; completeness is not claimed.
- The PDF was created using ReportLab in this repository's `.venv`, checked with pypdf, rendered with Poppler and visually inspected. The source is the review plus readable bibliography and citation audit.
- No manuscript or scientific source changes, repository refactors, external messages, paid research calls or experiment launches were made for this review.

## Computational checks

The validation checks 60 unique topical records; DOI and ID uniqueness; agreement between the evidence CSV and metadata; claim-source IDs and local source paths; BibTeX entry count; JSON readability; missing metadata; encoding/replacement characters; accidental tool references or unresolved drafting markers; and unchanged local source files. These checks establish packet consistency, not independent replication of the scientific literature.
