# Existing manuscript bibliography audit

Checked 4 November 2026 against publisher/Crossref, arXiv and workshop records. These are proposed corrections; `submission/aaai27/reviewer_bundle/paper.bib` has not been changed. Use the corresponding verified records in `references.json` or `references.bib` when revising it.

| Existing key | Finding | Verified identity / action |
| --- | --- | --- |
| `hezaveh2017fast` | Identity, venue, DOI and pages agree with the record. | Hezaveh, Perreault Levasseur and Marshall, *Fast automated analysis of strong gravitational lenses with convolutional neural networks*, Nature 548, 555-557 (2017), [DOI](https://doi.org/10.1038/nature23463). |
| `pearson2019effect` | Title is a description of the study rather than its actual title; page range is wrong. | *The use of convolutional neural networks for modelling large optically-selected strong galaxy-lens samples*, MNRAS 488, 991-1004 (2019), [DOI](https://doi.org/10.1093/mnras/stz1750). |
| `swierc2023domain` | Title and first two author initials are wrong. | Paxson Swierc, Megan Zhao, Aleksandra Ciprijanovic and Brian Nord, *Domain Adaptation for Measurements of Strong Gravitational Lenses*, [arXiv](https://arxiv.org/abs/2311.17238), [2023 workshop PDF](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_121.pdf). Preserve the source's accented surname in the final citation. |
| `agarwal2024domain` | Title is inaccurate. | Shrihan Agarwal, Aleksandra Ciprijanovic and Brian D. Nord, *Neural Network Prediction of Strong Lensing Systems with Domain Adaptation and Uncertainty Quantification*, [arXiv](https://arxiv.org/abs/2411.03334), [2024 workshop PDF](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_58.pdf). |
| `gawade2025cnn` | Title and journal are wrong; the arXiv identifier points to the relevant work. | *Neural network prediction of model parameters for strong lensing samples from Hyper Suprime-Cam Survey*, MNRAS 540, 3384-3399 (2025), [DOI](https://doi.org/10.1093/mnras/staf935). |
| `galan2024comparison` | Title is inaccurate; DOI, journal and article number identify the intended paper. | *Exploiting the diversity of modeling methods to probe systematic biases in strong lensing analyses*, A&A 692, A87 (2024), [DOI](https://doi.org/10.1051/0004-6361/202451095). |
| `mehta2020active` | Title, proceedings year, volume and pages agree with PMLR. Author list can be completed. | Bhairav Mehta, Manfred Diaz, Florian Golemo, Christopher J. Pal and Liam Paull, *Active Domain Randomization*, PMLR 100, 1162-1176 (2020), [proceedings](https://proceedings.mlr.press/v100/mehta20a.html). |
| `cubuk2020randaugment` | The paper exists in NeurIPS 2020, but the current BibTeX type is `article` with a `booktitle` field. | Use an `inproceedings` entry for *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*, NeurIPS 33 (2020), [proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html). |

Five of eight entries have inaccurate titles; Swierc also has inaccurate initials, Gawade the wrong journal, and Pearson the wrong page range. These are bibliographic defects, not evidence that the intended studies do not exist. A citation's title should be copied accurately, while its relevance belongs in the surrounding prose.

## Additional corrections and report families

- HOLISMOKES IV has a [2022 corrigendum](https://doi.org/10.1051/0004-6361/202039574e), verified in publisher metadata. Retrieval of the correction text was blocked by HTTP 403; this review does not quote the original numerical results or guess what was corrected.
- Euclid Q1 LEMON is now a [2026 journal article](https://doi.org/10.1051/0004-6361/202554538). Its 2025 arXiv submission is linked to the same topical record rather than counted twice.
- Filipp's [2024 workshop report](https://openreview.net/pdf?id=1nQIQg3Uiw) is a related precursor to the expanded [2025 robustness paper](https://doi.org/10.3847/1538-4357/adee20). It is not added as an independent item to inflate the evidence count.
- Poh's earlier workshop report and the later [2025 journal paper](https://doi.org/10.1088/1475-7516/2025/05/053) are not counted as independent evidence in this packet. The latter's Appendix A documents the noise conversion issue.
- Legin 2025, Holloway 2026 and Jarugula 2026 remain labelled as preprints because no journal version was verified in the retrieved records. Their findings are not presented as settled consensus.

## Claim checks that matter more than bibliographic formatting

Do not equate R-squared, signed residuals, 68% residual scatter, posterior precision, real-fit agreement and MAE. Do not describe simulated DES-like target data in the two domain-adaptation papers as real DES validation. Do not describe the Galan benchmark as a comprehensive real-data comparison. A correct DOI cannot rescue an incorrect interpretation of a result.
