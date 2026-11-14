# CNN feature reliance in synthetic strong lensing: focused review

Search and assessment date: **12 November 2026**. Status: literature synthesis and proposed directions; no new experiments, training or scoring.

## Scope and evidence status

This extends the existing 4 November review around the supplied novelty-assessment Markdown: Einstein-radius image regression, feature reliance under synthetic distribution changes, controlled nuisance interventions, and failure mechanisms. The [master evidence matrix](../../../literature_review/strong_lens_inference/evidence_matrix.csv) now contains **78 works: the original 60 plus 18 additions**. These are works, not independent experiments or 78 lens-regression studies. Existing Pearson 2019, Gawade 2025 and Herle 2024 rows remain relevant; they were not duplicated.

The [extension records](../../../literature_review/strong_lens_inference/feature_reliance_extension_2026-11-12.json) contain the new rows and assessment date. The old references.json, references.bib, references.md, coverage.json, search_log.json, verification.json and exported PDF remain **November 4 snapshots of 60 works**, not synchronized exports of the expanded matrix. Use the extension source URLs and this review for new evidence. The [extension search log](../../../literature_review/strong_lens_inference/feature_reliance_search_log_2026-11-12.json) records queries, access limitations and selection decisions.

This is a bounded single-reviewer narrative/scoping extension, not a systematic review. Discovery used web-index searches targeting arXiv, proceedings, publisher pages and citation chaining. No direct ADS/Scopus/Web of Science exhaustive search, independent duplicate screening, meta-analysis or scientific replication was performed. Ranked search results cannot establish a complete absence of prior work. Four additions received selected full-text-section assessment; the other fourteen received abstract/publisher-summary assessment. Search snippets alone were not the basis for the principal new regression/invariance conclusions. Detailed claims about Krell's sample sizes remain unverified because full-text retrieval failed. Publication and correction checks are not exhaustive.

## What is established, and what remains open?

### 1. Lensing sensitivity and interpretability already have substantial precedent

[Wilde et al. 2022](https://arxiv.org/abs/2202.12776) interpret lens-classification CNNs using visualization and kernel analysis. [LenNet 2025](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2025.1656917/full), Section 4.3, visualizes internal feature maps. Neither supports novelty from producing another lensing heatmap. Their classification/localization endpoints differ from continuous radius prediction.

[Sheng et al. 2022](https://proceedings.mlr.press/v151/sheng22a.html) explicitly identify simulation artifacts learned by lens classifiers and investigate a combined mitigation pipeline. Thus, the generic statement that synthetic lens images can teach unreliable cues is established prior art.

[Pearson et al. 2019](https://doi.org/10.1093/mnras/stz1750) and [Gawade et al. 2025](https://arxiv.org/abs/2404.18897) are more direct regression comparators: foreground light and simulation/observation assumptions affect parameter inference. Gawade also compares against observed HSC lens models. The proposal cannot rely on a blanket claim that earlier work addresses classification only. This extension does not freshly verify Pearson's quoted 34% result or its exact denominator; omit that number until the original methods/results have been checked.

[Herle et al. 2024](https://arxiv.org/abs/2307.10355) show physical selection biases in lens finding. [LUMA 2026](https://arxiv.org/html/2609.36857v1), Section 4.1, uses physical metadata to predict CNN classification correctness. These support failure stratification, but a parameter that predicts errors is not necessarily a cue used by the CNN. An explanation of error association and a mechanism of input dependence are different estimands.

### 2. Continuous regression does not itself supply the novelty

[Hermann et al. 2024](https://arxiv.org/abs/2310.16228) distinguish cue predictivity from availability in controlled shortcut experiments. Their framework suggests varying both nuisance-target correlation and visibility of genuine lens information, rather than treating dataset size as the sole driver.

The important additional comparator is [Xu et al. 2026, Shortcut to Nowhere](https://arxiv.org/html/2606.01723v1). Its setup addresses attribute-target confounding in continuous regression. Appendix C.1 includes angle regression with correlated background colors and test coverage across combinations. Therefore, an artificial nuisance correlated with a continuous target, followed by shifted testing, already exists outside astronomy. A lensing study must add physical mechanism controls or a scientifically useful diagnostic beyond that design.

[Scimeca et al. 2025](https://arxiv.org/abs/2311.16176) also precede counterfactual synthesis for shortcut mitigation. This motivates a physically rendered approach, but does not make counterfactual augmentation itself novel. Its arXiv overlap notice is recorded in the matrix rather than counted as another independent study.

### 3. Invariance can discard information needed for the scientific target

[LenSiam, Chang et al. 2023](https://arxiv.org/abs/2311.10100) already construct pairs by holding the lens model fixed and changing source galaxies. Matched physical parents and lens-preserving augmentation therefore need explicit acknowledgment as existing methods.

[Lens-LeJEPA, Batra et al. 2026](https://representations-physical-sciences.github.io/workshop-2026/assets/papers/paper-54.pdf) is especially relevant to the mechanism framing. It studies lensing-specific representation constraints and discusses benefits for classification alongside losses for axion-mass regression and reconstruction. This is not Einstein-radius regression, but it narrows claims about being first to connect lensing feature constraints to downstream regression failure. The workshop landing page and PDF differ on a headline classification gain; this review deliberately does not repeat that unresolved number.

Physical angular radius and its pixel representation must stay distinct: resampling the same physical scene changes sampling, whereas scaling an image on a fixed angular grid changes apparent angular size. A fixed angular target with unrestricted pixel-scale variation can become ambiguous for an image-only estimator. Treat missing calibration as an identifiability issue before labeling its errors a shortcut.

### 4. Attribution agreement is not mechanism validation

[Krell et al. 2025](https://repository.library.noaa.gov/view/noaa/74544) examine correlated inputs and attribution variability. [Fisher, Rudin and Dominici 2019](https://www.jmlr.org/papers/volume20/18-760/18-760.pdf) study reliance across well-performing models. Together these motivate asking whether equally accurate CNN seeds learn different dependencies. A finite collection of CNN seeds is not a formal bound on model-class reliance.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) and [TCAV](https://proceedings.mlr.press/v80/kim18d.html) supply supporting tools. Their scores do not establish causal necessity. [Adebayo et al. 2018](https://arxiv.org/abs/1810.03292), [Hooker et al. 2019](https://arxiv.org/abs/1806.10758), and [Hedstrom et al. 2024](https://arxiv.org/abs/2405.02383) motivate randomization checks, predictive validation, and attention to the evaluation metric. Removal-and-retraining evaluates recoverable information after fitting a new model; it is not equivalent to changing an input to a frozen model.

## Assessment of the supplied Markdown

| Proposed claim or component | Evidence judgment | Consequence |
|---|---|---|
| Lensing CNN feature visualization | Established | Supporting illustration only |
| Synthetic artifacts cause unreliable lens classifiers | Established | Require a more specific mechanism |
| Regression across simulation realism | Established | Compare directly with Pearson and Gawade |
| Continuous-target shortcut/reversal experiment | General-ML precedent | Xu is a required comparator |
| Physically matched lens/source views | Established augmentation precedent | Cite LenSiam |
| Representation constraints can hurt lensing regression | Close adjacent precedent | Compare Lens-LeJEPA's target and intervention explicitly |
| Training size changes measured reliance | Plausible question | No monotonic concentration trend is established by this review |
| Concentrated sensitivity causes failure | Unsupported as currently formulated | Separate magnitude, direction, visibility and concentration |
| A rendered diagnostic predicts independent shift errors and guides a targeted repair | Candidate contribution | Must outperform declared simple diagnostics and matched-budget randomization |

The proposed sensitivity S_k is useful as a response to a declared intervention. It is not a percentage of prediction explained. Its normalized maximum depends on intervention strength, the list and grouping of families, and the denominator. Tiny total sensitivity can produce high concentration. Nuisance sensitivity can also improve a prediction, while blur or foreground shot noise may remove genuine information.

Report per-family sensitivities, signed prediction shifts, paired changes in absolute error and total sensitivity. Mark concentration undefined below a predeclared negligible-sensitivity threshold. Compare response curves at several meaningful strengths. A model-level concentration score must not be correlated with individual-image errors as if it varied across parents. If a per-parent score is used, predeclare it and avoid testing it against the same interventions used to construct it. Use independent shifts and simple visibility/SNR/error-bin comparators.

## Ranked potential directions

These rankings are judgments about relevance, feasibility and potential contribution, not verified priority claims.

| Priority | Question | Minimum discriminating design | Contribution and limit |
|---|---|---|---|
| 1 | When does foreground-light dependence represent a shortcut rather than obscuration? | Matched lens/source parents; correlated versus decorrelated foreground assignments with equal marginals; aligned/absent/reversed test correlations; high-visibility and obscured regimes | Physically separates learned correlation from information loss; synthetic mechanism only |
| 2 | Do equally accurate models learn different physical dependencies? | Paired seeds and fixed optimizer budget; predeclared accuracy-equivalence band; rendered sensitivity profiles; forecast independent stress errors | Goes beyond attribution disagreement; seed sampling cannot characterize all possible models |
| 3 | Can a diagnosed mechanism improve data allocation beyond generic variation? | Freeze one source-development diagnosis; equal rendering/update budgets; targeted decorrelation versus uniform variation and error-only targeting; fresh parents for final evaluation | Tests the value of explanation, not merely extra data; overlaps the current failure-weighted study and needs a distinct comparator |
| 4 | Which nuisance invariances preserve Einstein-radius information? | One justified invariance at a time; preserve correct angular conventions; test nuisance robustness and response to target-changing scenes | Addresses robustness-information tradeoffs; LenSiam and Lens-LeJEPA are close prior art |
| 5 | Does independent-parent training size alter reliance? | Nested parent subsets; matched training updates and a convergence/exposure sensitivity analysis; multiple subset draws and paired seeds | Useful secondary result; 100/300/700 alone may be incremental and size is not diversity |

### Recommended first study

Start with Direction 1 and use Direction 2 as a secondary analysis. The question is: **Under what arc-visibility conditions does an imposed foreground-light/Einstein-radius correlation change CNN predictions, and does decorrelated training remove the resulting reversal bias?**

Keep mass, source and angular grid fixed within each evaluation family. Manipulate one foreground property initially. Preserve target and nuisance marginals across training arms through a recorded assignment mechanism; verify the resulting joint distributions. Use a high-SNR control to distinguish visible added foreground from signal-dependent photon-noise degradation. Paired random streams can reduce variance, but a shared seed does not make Poisson noise identical when intensities differ. Keep diagnostic combinations distinct from claims about naturally occurring lens populations.

Use correlated and decorrelated training arms first, with equal parent counts, optimizer updates and preprocessing. A uniformly varied arm becomes necessary when claiming targeted mitigation superiority. Measure signed bias, MAE, parent-level paired error changes and nuisance-response curves on fresh test parents. Require the repair to retain sensitivity to true radius changes; a constant predictor can appear perfectly invariant. Compare a mean predictor and a simple arc-geometry baseline, documenting where geometry extraction fails.

For learning curves, fixed updates estimate the effect of data access at fixed compute; fixed epochs change optimizer exposure. Neither alone estimates every notion of a sample-size effect. Keep that question secondary until the mechanism is reproducible.

Bootstrap independent parents with repeated observations kept together. Report paired training-seed variation separately. Freeze development choices before final testing. Use an independently generated final set; neither observational validation nor causal effects in real galaxies follow from simulator interventions alone.

## Workflow attribution

The literature-review skill guided evidence separation and search logging. Its attribution is methodological, outside the 78 topical works: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). Current arXiv identity was checked during this extension.
