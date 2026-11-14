# Strong-lens inference literature packet

Assembled 13 November 2026 from the literature Markdown previously split across this folder.
This is a scoping and planning packet; the maintained experiment scope is in the [study guide](../../study.md) and [eight-page paper plan](../../eight_page_paper_plan.md). The proposal sections below are historical options and do not override the current C/G/R synthetic comparison or the proposed HST transfer study; feature analysis is deferred from the current paper.

## Current evidence state

The current [`evidence_matrix.csv`](evidence_matrix.csv) contains **92 work records**. The counts of 60, 78, and 87 below are dated review stages: the 4 November bibliography and narrative review covered 60 works; the 12 November feature-reliance extension brought the matrix to 78; novelty verification brought it to 87; the later candidate-directions/data pass brought it to 92. The older readable bibliography, BibTeX, metadata, coverage, verification, and exported PDF remain 60-work snapshots.

These are counts of works, not independent empirical lens-inference studies. The review is single-reviewer and targeted; it does not claim exhaustive database coverage or prove novelty. The HST/Euclid readiness work is summarized in the separate [real-telescope data packet](../real_telescope_data/README.md); no observational CNN scoring is reported here.

## Packet contents

- [Candidate directions and data](#candidate-directions-after-the-novelty-check)
- [Novelty verification](#novelty-verification-foreground-dependence-model-variation-and-targeted-repair)
- [CNN feature-reliance review](#cnn-feature-reliance-in-synthetic-strong-lensing-focused-review)
- [Manuscript bibliography audit](#existing-manuscript-bibliography-audit)
- [Original 4 November narrative review](#reliable-automated-inference-of-galaxy-scale-strong-gravitational-lenses)
- [Original 60-work bibliography snapshot](#bibliography)
- [Historical provenance and verification](#local-provenance-and-verification)
- Supporting machine-readable records: [matrix](evidence_matrix.csv), [verified references](references.json), [BibTeX](references.bib), [claim-source map](claim_source_map.json), [search log](search_log.json), [coverage](coverage.json), [verification](verification.json), and the dated [78-work](feature_reliance_extension_2026-11-12.json), [87-work](novelty_verification_2026-11-12.json), and [92-work](directions_search_2026-11-12.json) follow-up records.
- The historical review PDF is recorded at `output/pdf/strong_lens_inference_review.pdf`.

## Consolidated source material

## Candidate directions after the novelty check

Date: 12 November 2026. Status: proposed questions and experiments, not executed findings or verified priority claims. Decision owner: user; no direction selected. Constraints: approximately 700 original training parents (697 qualified in the maintained manuscript), compact image-only Einstein-radius regression, 256x256 canonical images at 0.05 arcsec/pixel, simulator radius range 0.5-2 arcsec, limited development parents and no new observational validation. Missing production arrays/checkpoints remain an execution dependency.

### Recommendation and evidence boundary

The strongest next question is **whether robustness to individual observing changes survives their combinations, and which extra synthetic examples restore it**. A complementary question is whether a small development sample can reliably identify the repair at all. A third is how to spend a fixed image budget between independent physical scenes and repeated observing conditions.

None is a verified first-of-kind method. These are narrower candidate contributions than another shortcut injection/removal study. Generic compound-shift robustness, experimental interactions and simulation-budget allocation are established concepts. The proposed value is an interpretable lensing result with explicit physical controls and a small-data decision problem.

The [master matrix](evidence_matrix.csv) now has **92 works**, after five additions. The [search/extraction record](directions_search_2026-11-12.json) records assessed sources, search strings and access limits. This is a bounded AI-assisted, single-reviewer scoping exercise, not exhaustive ADS/Scopus screening. No independent second reviewer was used. Query strings were transcribed during the session rather than exported from a bibliographic service; the log identifies missing exact variants. Earlier bibliography/PDF exports remain dated snapshots.

### What changed after reading the limitations?

| Located evidence | Limitation or implication | Candidate response |
|---|---|---|
| [Filipp et al. 2025](https://arxiv.org/html/2411.05905v2), Sections IV-V | Tests individual deviations while acknowledging simultaneous effects in observations | Quantify interactions and dependence shifts; do not extrapolate its subhalo-posterior results to radius error |
| [Jacobs et al., arXiv 2021](https://arxiv.org/abs/2112.02479) | Sensitivity probes already expose lens-classifier weaknesses | A response sweep alone is insufficient; test whether diagnostic decisions reproduce and transfer |
| [Olischlager et al. 2026](https://arxiv.org/html/2609.39525v1), Sections 5.4/A.8 | Already diagnose representation gaps and use auxiliary supervision in lensing | Avoid claiming novelty for latent probes or predicting simulator parameters from features |
| [Hughes et al. 2024](https://arxiv.org/abs/2403.04349) | Source-complexity experiments already exist for subhalo detection | Study allocation or transfer, not merely more Sersic clumps |
| [Galan et al. 2024](https://arxiv.org/html/2406.08484v2) | Classical source-model and PSF assumptions interact in reconstruction | Physical interactions are known; the open candidate is learned-regressor robustness and repair under matched budgets |
| Existing local policy and secondary conditions | Seven separate training cells; secondary suite already includes Moffat plus half exposure | Do not claim the first combined-condition test even within this project; require a systematic factorial comparison |

### A. Robustness to the pieces versus robustness to their combination

**Plain-language question:** Does practising blurry images and noisy images separately prepare the CNN for an image that is both blurry and noisy?

**Design.** Start with two factors at a fixed pixel scale: Gaussian PSF width and exposure time. For every parent, render a 2x2 family: canonical, wider PSF only, shorter exposure only, both. Add one intermediate level or an independent PSF family later. Hold mass/source/foreground identities fixed; repeat detector draws to distinguish systematic effects from random realization. Exposure and PSF both alter information, so failure is not automatically a shortcut.

Compare canonical training, single-factor variation, and joint-factor variation with the same 697 parent identities, initialization schedule, updates and exposure count. A particularly clean contrast compares two sampling policies with the same frequency of each PSF level and each exposure level but different pairings. Each parent should receive those assignments through a balanced recorded schedule. The target must remain independent of the assigned observation policy. Test new combinations and strengths on fresh parents.

**Feature analysis.** Measure signed prediction response and error in all four cells. A signed second difference, prediction(both)-prediction(PSF)-prediction(exposure)+prediction(canonical), detects non-additive model response to the declared changes. Apply a corresponding contrast to absolute error. Neither is a unique decomposition of learned features: image formation and the absolute-value loss also introduce nonlinearities. Supplement with interpretable arc visibility/contrast descriptors and component renders, without feeding truth-side descriptors to the predictor.

**Potential contribution.** Establish where single-factor training fails despite matched marginal coverage, and whether allocating examples to missing combinations is more efficient than oversampling individually difficult conditions. This changes the question from which condition is hardest to which combinations are missing.

**Feasibility.** High relative to source-family changes: existing PSF/noise controls and CNN can be reused, but generation/scheduling needs new support. Four evaluation cells preserve 150 independent test units; they do not create 600 independent lenses. Begin with a few paired seeds for development, expanding only after the pilot is informative.

**Adversarial check.** If joint training wins only because it sees more severe images, marginal matching failed. If combined error is fully explained by diminished visibility and every policy behaves alike, report that boundary rather than a learned shortcut. Null interaction is useful evidence against an unnecessarily complicated policy.

### B. How many independent scenes are needed to trust a feature diagnosis?

**Plain-language question:** Would a different small set of lenses, or a fresh noise draw, make us choose a different repair?

**Design.** Freeze a small candidate set of repairs and a source-development population. Compare diagnostics based on error alone, rendered prediction responses, and (optionally) one validated attribution method. Allocate the same diagnostic render budget differently: for example, 25 parents x 12 noise draws, 75 x 4, or 150 x 2, giving 300 views per intervention. These numbers are proposed budget cells, not power calculations. Repeat parent-subset selection within development and preserve that repeated-use structure in analysis.

Each diagnostic selects a repair from the same declared candidates. Candidate models can be trained once per paired seed and kept fixed while evaluating many diagnostic allocations; there is no need to retrain for every subset. Final selection policies must be frozen before evaluating fresh parents and fresh noise. Compare selection stability and downstream error, not just visual similarity of explanations. An evaluation-set best-candidate reference is descriptive and uncertain, not a known population oracle.

**Potential contribution.** A practical reliability curve for mechanism-guided data decisions under a finite diagnostic budget. Attribution instability is known from Krell, and seed-dependent shortcuts from D'Amour; the distinct candidate estimand is how those uncertainties affect an actual synthesis decision and its independent outcome.

**Feasibility.** High once a few candidate models/renders exist: much work is repeated inference. Use parent-level inference, not thousands of correlated noise images as independent samples. Ten training seeds are ten model realizations, insufficient for flexible model-level failure prediction.

**Adversarial check.** Stable diagnosis may be stably wrong. Require independent repair benefit. Conversely, different rankings may choose equally effective repairs; ranking disagreement alone is not scientific harm. This is an empirical benchmark contribution, not a new general explanation method.

### C. More physical systems or more observing-condition variants?

**Plain-language question:** With a fixed number of synthetic training images, is it better to show more distinct lenses or show fewer lenses under many conditions?

**Design.** Cross independent-parent count (e.g., 100, 300, 697) with narrow versus broader observing variation while fixing total generated-image count and optimizer updates. Use stratified, repeated parent draws and a common target distribution; do not let a small set accidentally exclude part of the radius range. Separate the rendering budget from optimizer exposures and wall-clock cost. A noiseless scene reused for fresh detector noise is cheaper than a new physical render; report both image-count and actual-cost comparisons.

Evaluate ordinary conditions, held-out combinations from A, and a source-family shift if implemented. Track whether feature-response profiles stabilize with additional independent parents or merely with more noisy views. This is stronger than a standard learning curve because it compares opportunity costs, but it is still related to longstanding augmentation and sample-efficiency research.

**External extension.** Add PSF/noise-aware empirical source templates from COSMOS through the common renderer, or initially use a validated procedural multi-component family. Keep mass and target distributions matched. Empirical-template rendering is not already supported merely because the library exists. Split source-template identities before generating lenses; repeated transforms of one template are not new galaxies. Cluster or use a crossed design if source templates and lens masses are reused.

**Feasibility.** Moderate. The in-family allocation study fits the present scale; empirical morphology adds meaningful implementation and validation work. Do not make a high novelty claim for 100/300/697 alone.

**Adversarial check.** If differences disappear once training convergence or target coverage is controlled, the original interpretation was wrong. If all arms use the same Sersic family, conclusions concern parameter diversity within that family, not realism.

### D. Calibration and tensor-shape cues: prerequisite audit, possible smaller methods study

The current renderer changes array dimensions to preserve field of view when pixel scale changes: nominally 320 pixels at 0.04, 256 at 0.05 and 213 at 0.06 arcsec/pixel. Input geometry is therefore associated with the observing condition even when no metadata vector is supplied. This is a code-level possibility, not evidence the CNN actually exploits it.

Compare explicit pixel-scale conditioning, physically calibrated common-grid preprocessing, and native-shape inputs with carefully controlled cropping/padding. Do not blindly resize and retain an inconsistent angular label. Separate array-shape cues, PSF-in-pixel cues and prior-based size cues. Exact scale-ambiguous synthetic pairs can provide an identifiability control if all angular components and flux conventions are transformed consistently.

The need for angular calibration is established physics, not a novel theorem. This audit could overturn an interpretation of robustness or reveal a hidden information channel. Native shape itself legitimately carries calibration under a fixed-field-of-view contract; it is not automatically leakage. Audit that contract before making cross-instrument claims.

### Dataset compatibility: direct answer

**No external resource checked here is verified as a drop-in expansion of our full image/target contract.** A resource can be useful without being directly poolable.

| Resource | Radius labels and population | Match to our images? | Access checked and recommended role |
|---|---|---|---|
| [DeepSkies/Agarwal release](https://zenodo.org/records/13647416), [paper](https://arxiv.org/html/2411.03334v3) | Radius regression; paper reports 100000 objects in each of two domains, not verified distinct physical parents across domains | SIE, but three bands, 40x40, 0.263 arcsec/pixel, radius prior 1-3 arcsec versus ours 0.5-2 | Public two ZIPs listed, 7.6 GB total. Not downloaded/validated here. Best located ready-made external regression domain, after label/ancestry checks; retrain on its native contract rather than treating upsampling as new angular information |
| [Ancona-Flores et al. 2026](https://arxiv.org/html/2603.06339v2) | Paper reports 76396 SIE images with radius and shape labels | One band and 0.06 arcsec/pixel are relatively close; 100x100 means 6 arcsec field, versus our 12.8 | Public image-label download not located in this check. Promising access lead, not an available data expansion |
| [Galan public release](https://github.com/aymgal/LensSourceDegeneracy_public), [paper](https://arxiv.org/html/2406.08484v2) | Modeling products and truth-bearing synthetic challenge | Different mass/source/PSF assumptions; repeated modeling methods do not create independent parent systems | Repository exposes COOLEST lens models. Useful targeted challenge/intake, not a large independent training population |
| [GalSim COSMOS](https://galsim-developers.github.io/GalSim/_build/html/real_gal.html) | Unlensed galaxies, PSFs and noise information; no complete-lens radius targets | Can generate compatible outputs only through validated source preparation plus our lens/instrument renderer | Public library documented. Best source-diversity route, not a labelled lens dataset. Original PSF/noise and template reuse must be handled |
| [Duplinskii et al. 2025](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_317.pdf) | Paper describes 20000 paired high/low-quality lens images; per-image radius-label release unverified | Single-channel 48x48 with different instrument setup | Paper retrieved; download and target sidecars not verified. Restoration benchmark lead only |
| [Chang/ViT regression repository](https://github.com/kuanweih/strong_lensing_vit_resnet) | Physical-parameter generation notebooks | Configurable simulator code, not a verified matching saved dataset | Public generation code linked. Useful regeneration reference, not independent external image evidence |
| [Hugging Face gravitational-lenses catalog](https://huggingface.co/datasets/juliensimon/gravitational-lenses) | Viewer shows coordinates, redshift, type, grade and references | Tabular catalog, no image-plus-radius target contract shown | Not suitable for supervised radius expansion as supplied |

Before pooling any external lenses, verify mass-model radius convention, units, PSF, band, angular sampling and field of view, noise/normalization, target distribution within overlapping support, licenses, and parent/template ancestry. Treat overlapping source/target domains as paired views until identity checks establish independence. CASTLES remains observational data with reference-label work required, not automatic ground-truth regression expansion.

### Ranking and decision

Criteria were relevance to synthetic-data decisions, distinct contribution beyond the checked matrix, feasibility at current independent-parent counts, and interpretability of a null result. Ratings are qualitative judgments, not measurements or acceptance forecasts.

| Rank | Direction | Novelty potential | Feasibility | Evidence that would justify proceeding |
|---|---|---|---|---|
| 1 | Compound shifts plus interaction-aware synthesis | Moderate, with direct limitation-based motivation | High for two factors | Reproducible non-additive error or a training-policy difference under matched marginals |
| 2 | Reliability of feature-based repair decisions | Moderate; small-data decision endpoint is the distinction | High after candidate models exist | Parent/noise allocation changes independent repair performance, beyond ranking instability |
| 3 | Independent-parent diversity versus condition coverage | Moderate-low alone; stronger with held-out morphology | Moderate | A reproducible budget frontier after target and optimization controls |
| Supporting | Calibration/shape-cue audit | Uncertain, potentially useful negative methods result | High for initial audit | Demonstrated dependence on a representation convention rather than presumed leakage |

Recommended coherent project: **Do synthetic robustness interventions compose, and can a small diagnostic set identify which missing combinations to simulate?** A tests the failure and repair; B establishes whether that repair choice can be trusted at our sample size. C is a later budget extension. Run D before interpreting scale-related results. No recommendation requires a new architecture, a learned image generator or a large external dataset.

Workflow attribution: literature-review and scientific-brainstorming skills guided source separation, alternative generation and adversarial checks. Their library description is cited outside topical evidence: Kassis et al. (2026), [Scientific Agent Skills](https://arxiv.org/abs/2609.00065), current identity checked. No experiment, manuscript or memory file was changed.

---

## Novelty verification: foreground dependence, model variation and targeted repair

Checked 12 November 2026. This assessment **revises downward** the earlier moderate-high novelty ratings. The project remains a plausible empirical lensing study, but the broad diagnose-compare-repair loop is established methodology. No new experiment was run.

### Search and artifact scope

The [search and extraction log](novelty_verification_2026-11-12.json) records 15 exact follow-up queries, nine new works and a deeper assessment of Pearson 2019. The [master matrix](evidence_matrix.csv) now contains **87 works**, not independent studies. The earlier 78-work extension and original 60-work bibliography/PDF remain dated snapshots. Primary sources were retrieved from arXiv, JMLR, proceedings and publisher records. This is single-reviewer targeted verification, not exhaustive ADS/Scopus coverage or proof of priority. Four additions received selected full-text assessment; five are abstract/publisher-summary extractions. Pearson's existing row was upgraded from abstract to selected full-text assessment. Previous-source entries such as Xu remain relevant and were not duplicated.

### Closest overlaps

| Proposed element | Closest checked prior work | What this rules out |
|---|---|---|
| Foreground correlations alter lens inference | [Pearson, Li and Dye 2019](https://arxiv.org/pdf/1904.06199), Section 3.3, Figures 10-12 | First study of learned lens-light dependence under train/test correlation mismatch |
| Decorrelating lens mass and light during retraining | [Venkatraman et al. 2025](https://arxiv.org/html/2510.20778v1), Section IV.4 | First lensing decorrelated-training experiment |
| Similar accuracy, different seed-dependent shortcuts | [D'Amour et al. 2022](https://jmlr.org/papers/v23/20-1335.html), full-text protocol and shortcut passages | New general phenomenon or general stress-testing protocol |
| Diagnosis followed by targeted image generation | [ASPIRE, Ghosh et al. 2024](https://arxiv.org/html/2308.10103v3), Section 2; [SCGS, Li et al. 2024](https://arxiv.org/html/2412.19457v1), Sections II-III | First explanation-guided synthetic-data repair pipeline |
| Interpretable concept discovery followed by data intervention | [DISC, Wu et al. 2023](https://proceedings.mlr.press/v202/wu23w.html) | First discover-and-intervene approach |
| Causal augmentation selection | [Ilse et al. 2021](https://proceedings.mlr.press/v139/ilse21a.html) | New general idea of using augmentation as an intervention |
| Error-only repair baseline | [JTT, Liu et al. 2021](https://proceedings.mlr.press/v139/liu21f.html) | Novelty from identifying errors and retraining on them |
| Perturbation-based explanation and correction | [Hagos et al. 2022](https://arxiv.org/abs/2211.08285) | Novelty from perturbation diagnosis plus explanation-based correction alone |
| Astrophysical counterfactual signal/instrument separation | [Mercader-Perez et al. 2026](https://arxiv.org/abs/2604.09787), current abstract | First astrophysical counterfactual or measurement-artifact separation claim |

Pearson manipulates mass-light shape/alignment scatter during training and testing. It explicitly interprets foreground dependence and finds smaller correlation-mismatch effects on radius than on shape/orientation. This is substantially closer than merely adding or removing lens light. Its experiments are not the exact proposed matched brightness-radius/visibility design, but changing the cue alone is weak differentiation.

Venkatraman retrains NPE without mass-light ellipticity correlations and evaluates the original correlated test population. Removing those correlations can worsen inference. This is not the proposed correlation-reversal experiment, but it demonstrates why a correlation is not automatically harmful: it can provide informative population structure. The paper's model/task differs from scalar CNN regression; that distinction does not erase the methodological overlap.

ASPIRE identifies cues through edited images and then generates training data. SCGS combines errors and attribution masks for image synthesis, and includes comparisons involving JTT. Therefore, even demonstrating improvement beyond error targeting is not a new general-ML claim. The possible contribution is a stronger physical validation and a domain-specific result under declared controls.

Mercader's latest arXiv abstract reports NeurIPS 2026 acceptance; an earlier search-indexed OpenReview report described ICLR. Use the current arXiv status, without assuming all report versions are identical. The current extraction concerns multi-instrument galaxy representations, not an Einstein-radius shortcut-repair benchmark.

### Revised judgment for directions 1-3

| Direction | Standalone novelty assessment | Remaining possible contribution |
|---|---|---|
| 1: shortcut versus obscuration | Low for correlation injection/removal alone; moderate if physically discriminating | Quantify when correlation dependence, obscuration and photon-noise loss produce distinct failure signatures |
| 2: model variation | Low as a generic observation; potentially moderate in this domain | Physically defined diagnostics that prospectively distinguish independent vulnerabilities among accuracy-equivalent models |
| 3: explanation-guided synthesis | Low for the loop itself; potentially moderate with decisive evaluation | Whether validated physical diagnosis selects a more effective repair than matched-budget broad variation and error targeting |
| Combined study | Plausible focused empirical contribution; high novelty is not verified | Evidence about a nontrivial conditional mechanism and the practical value/limits of diagnosis |

Combining three established components is not automatically a new method. A publishable contribution could instead be a carefully demonstrated scientific result. The search did not identify an exact match to all the proposed controls, but the absence of a retrieved match is not evidence of first-of-kind status.

### What would make the next experiment more distinctive?

1. **Discriminate rival mechanisms.** Factor foreground-target association separately from arc visibility and signal-dependent noise. Record matched-parent rendering and marginal-distribution checks. A single foreground sweep cannot isolate all three.
2. **Test a prediction before confirmation.** Freeze the diagnosis on development parents and predict which independent shifts and visibility regimes will produce signed bias. Test it on new parents and unselected interventions, rather than explaining the same errors used for diagnosis.
3. **Measure specificity of repair.** Compare mechanism-targeted data against uniform and error-only targeting. If budgets permit, include a deliberately mismatched repair with comparable rendering strength and a diagnostic ablation. This checks whether correct diagnosis, rather than any extra varied data, matters.
4. **Preserve useful information.** Test ordinary accuracy and response to actual radius changes. A reduction in nuisance sensitivity alone could result from prediction collapse.
5. **Show a boundary, not just a win.** Include aligned and shifted populations. Decorrelation may trade ordinary accuracy for shift protection; report when it helps and hurts.

An imposed brightness-radius shortcut is a positive control with a known answer. It validates that a diagnostic can detect the planted dependency; it does not demonstrate discovery of an unknown failure mechanism. Calling the synthesis explanation-guided is only useful if the diagnosis affects the chosen repair in a testable way. Selecting foreground decorrelation because the experimenter planted that exact correlation does not by itself establish the value of explanation.

Suggested question: **Can physically controlled diagnostics distinguish foreground-correlation reliance from obscuration and predict which equal-budget training-data intervention will reduce independent Einstein-radius bias?**

This is a defensible proposed contribution, not a verified novel result. Require a pilot showing meaningful competing mechanisms before committing to the larger repair study. If every injected correlation is simply removed by the obvious decorrelation arm, frame the result as a controlled domain validation rather than a new explanation method.

### Implication for the current manuscript

The existing equal-budget C/G/R robustness comparison supplies experimental infrastructure and a baseline. It does not supply the proposed mechanism evidence. A stronger successor would center a discriminating physical result and use cross-seed analysis as support. The existing equal-budget design remains valuable, but matching budgets alone does not establish methodological novelty.

Workflow attribution remains as recorded in the earlier review: Scientific Agent Skills, arXiv:2609.00065, outside topical evidence counts. No manuscript, experiment code, results or memory files were edited by this check.

---

## CNN feature reliance in synthetic strong lensing: focused review

Search and assessment date: **12 November 2026**. Status: literature synthesis and proposed directions; no new experiments, training or scoring.

### Scope and evidence status

This extends the existing 4 November review around the supplied novelty-assessment Markdown: Einstein-radius image regression, feature reliance under synthetic distribution changes, controlled nuisance interventions, and failure mechanisms. The [master evidence matrix](evidence_matrix.csv) now contains **78 works: the original 60 plus 18 additions**. These are works, not independent experiments or 78 lens-regression studies. Existing Pearson 2019, Gawade 2025 and Herle 2024 rows remain relevant; they were not duplicated.

The [extension records](feature_reliance_extension_2026-11-12.json) contain the new rows and assessment date. The old references.json, references.bib, references.md, coverage.json, search_log.json, verification.json and exported PDF remain **November 4 snapshots of 60 works**, not synchronized exports of the expanded matrix. Use the extension source URLs and this review for new evidence. The [extension search log](feature_reliance_search_log_2026-11-12.json) records queries, access limitations and selection decisions.

This is a bounded single-reviewer narrative/scoping extension, not a systematic review. Discovery used web-index searches targeting arXiv, proceedings, publisher pages and citation chaining. No direct ADS/Scopus/Web of Science exhaustive search, independent duplicate screening, meta-analysis or scientific replication was performed. Ranked search results cannot establish a complete absence of prior work. Four additions received selected full-text-section assessment; the other fourteen received abstract/publisher-summary assessment. Search snippets alone were not the basis for the principal new regression/invariance conclusions. Detailed claims about Krell's sample sizes remain unverified because full-text retrieval failed. Publication and correction checks are not exhaustive.

### What is established, and what remains open?

#### 1. Lensing sensitivity and interpretability already have substantial precedent

[Wilde et al. 2022](https://arxiv.org/abs/2202.12776) interpret lens-classification CNNs using visualization and kernel analysis. [LenNet 2025](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2025.1656917/full), Section 4.3, visualizes internal feature maps. Neither supports novelty from producing another lensing heatmap. Their classification/localization endpoints differ from continuous radius prediction.

[Sheng et al. 2022](https://proceedings.mlr.press/v151/sheng22a.html) explicitly identify simulation artifacts learned by lens classifiers and investigate a combined mitigation pipeline. Thus, the generic statement that synthetic lens images can teach unreliable cues is established prior art.

[Pearson et al. 2019](https://doi.org/10.1093/mnras/stz1750) and [Gawade et al. 2025](https://arxiv.org/abs/2404.18897) are more direct regression comparators: foreground light and simulation/observation assumptions affect parameter inference. Gawade also compares against observed HSC lens models. The proposal cannot rely on a blanket claim that earlier work addresses classification only. This extension does not freshly verify Pearson's quoted 34% result or its exact denominator; omit that number until the original methods/results have been checked.

[Herle et al. 2024](https://arxiv.org/abs/2307.10355) show physical selection biases in lens finding. [LUMA 2026](https://arxiv.org/html/2609.36857v1), Section 4.1, uses physical metadata to predict CNN classification correctness. These support failure stratification, but a parameter that predicts errors is not necessarily a cue used by the CNN. An explanation of error association and a mechanism of input dependence are different estimands.

#### 2. Continuous regression does not itself supply the novelty

[Hermann et al. 2024](https://arxiv.org/abs/2310.16228) distinguish cue predictivity from availability in controlled shortcut experiments. Their framework suggests varying both nuisance-target correlation and visibility of genuine lens information, rather than treating dataset size as the sole driver.

The important additional comparator is [Xu et al. 2026, Shortcut to Nowhere](https://arxiv.org/html/2606.01723v1). Its setup addresses attribute-target confounding in continuous regression. Appendix C.1 includes angle regression with correlated background colors and test coverage across combinations. Therefore, an artificial nuisance correlated with a continuous target, followed by shifted testing, already exists outside astronomy. A lensing study must add physical mechanism controls or a scientifically useful diagnostic beyond that design.

[Scimeca et al. 2025](https://arxiv.org/abs/2311.16176) also precede counterfactual synthesis for shortcut mitigation. This motivates a physically rendered approach, but does not make counterfactual augmentation itself novel. Its arXiv overlap notice is recorded in the matrix rather than counted as another independent study.

#### 3. Invariance can discard information needed for the scientific target

[LenSiam, Chang et al. 2023](https://arxiv.org/abs/2311.10100) already construct pairs by holding the lens model fixed and changing source galaxies. Matched physical parents and lens-preserving augmentation therefore need explicit acknowledgment as existing methods.

[Lens-LeJEPA, Batra et al. 2026](https://representations-physical-sciences.github.io/workshop-2026/assets/papers/paper-54.pdf) is especially relevant to the mechanism framing. It studies lensing-specific representation constraints and discusses benefits for classification alongside losses for axion-mass regression and reconstruction. This is not Einstein-radius regression, but it narrows claims about being first to connect lensing feature constraints to downstream regression failure. The workshop landing page and PDF differ on a headline classification gain; this review deliberately does not repeat that unresolved number.

Physical angular radius and its pixel representation must stay distinct: resampling the same physical scene changes sampling, whereas scaling an image on a fixed angular grid changes apparent angular size. A fixed angular target with unrestricted pixel-scale variation can become ambiguous for an image-only estimator. Treat missing calibration as an identifiability issue before labeling its errors a shortcut.

#### 4. Attribution agreement is not mechanism validation

[Krell et al. 2025](https://repository.library.noaa.gov/view/noaa/74544) examine correlated inputs and attribution variability. [Fisher, Rudin and Dominici 2019](https://www.jmlr.org/papers/volume20/18-760/18-760.pdf) study reliance across well-performing models. Together these motivate asking whether equally accurate CNN seeds learn different dependencies. A finite collection of CNN seeds is not a formal bound on model-class reliance.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) and [TCAV](https://proceedings.mlr.press/v80/kim18d.html) supply supporting tools. Their scores do not establish causal necessity. [Adebayo et al. 2018](https://arxiv.org/abs/1810.03292), [Hooker et al. 2019](https://arxiv.org/abs/1806.10758), and [Hedstrom et al. 2024](https://arxiv.org/abs/2405.02383) motivate randomization checks, predictive validation, and attention to the evaluation metric. Removal-and-retraining evaluates recoverable information after fitting a new model; it is not equivalent to changing an input to a frozen model.

### Assessment of the supplied Markdown

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

### Ranked potential directions

These rankings are judgments about relevance, feasibility and potential contribution, not verified priority claims.

| Priority | Question | Minimum discriminating design | Contribution and limit |
|---|---|---|---|
| 1 | When does foreground-light dependence represent a shortcut rather than obscuration? | Matched lens/source parents; correlated versus decorrelated foreground assignments with equal marginals; aligned/absent/reversed test correlations; high-visibility and obscured regimes | Physically separates learned correlation from information loss; synthetic mechanism only |
| 2 | Do equally accurate models learn different physical dependencies? | Paired seeds and fixed optimizer budget; predeclared accuracy-equivalence band; rendered sensitivity profiles; forecast independent stress errors | Goes beyond attribution disagreement; seed sampling cannot characterize all possible models |
| 3 | Can a diagnosed mechanism improve data allocation beyond generic variation? | Freeze one source-development diagnosis; equal rendering/update budgets; targeted decorrelation versus uniform variation and error-only targeting; fresh parents for final evaluation | Tests the value of explanation, not merely extra data; overlaps the current failure-weighted study and needs a distinct comparator |
| 4 | Which nuisance invariances preserve Einstein-radius information? | One justified invariance at a time; preserve correct angular conventions; test nuisance robustness and response to target-changing scenes | Addresses robustness-information tradeoffs; LenSiam and Lens-LeJEPA are close prior art |
| 5 | Does independent-parent training size alter reliance? | Nested parent subsets; matched training updates and a convergence/exposure sensitivity analysis; multiple subset draws and paired seeds | Useful secondary result; 100/300/700 alone may be incremental and size is not diversity |

#### Recommended first study

Start with Direction 1 and use Direction 2 as a secondary analysis. The question is: **Under what arc-visibility conditions does an imposed foreground-light/Einstein-radius correlation change CNN predictions, and does decorrelated training remove the resulting reversal bias?**

Keep mass, source and angular grid fixed within each evaluation family. Manipulate one foreground property initially. Preserve target and nuisance marginals across training arms through a recorded assignment mechanism; verify the resulting joint distributions. Use a high-SNR control to distinguish visible added foreground from signal-dependent photon-noise degradation. Paired random streams can reduce variance, but a shared seed does not make Poisson noise identical when intensities differ. Keep diagnostic combinations distinct from claims about naturally occurring lens populations.

Use correlated and decorrelated training arms first, with equal parent counts, optimizer updates and preprocessing. A uniformly varied arm becomes necessary when claiming targeted mitigation superiority. Measure signed bias, MAE, parent-level paired error changes and nuisance-response curves on fresh test parents. Require the repair to retain sensitivity to true radius changes; a constant predictor can appear perfectly invariant. Compare a mean predictor and a simple arc-geometry baseline, documenting where geometry extraction fails.

For learning curves, fixed updates estimate the effect of data access at fixed compute; fixed epochs change optimizer exposure. Neither alone estimates every notion of a sample-size effect. Keep that question secondary until the mechanism is reproducible.

Bootstrap independent parents with repeated observations kept together. Report paired training-seed variation separately. Freeze development choices before final testing. Use an independently generated final set; neither observational validation nor causal effects in real galaxies follow from simulator interventions alone.

### Workflow attribution

The literature-review skill guided evidence separation and search logging. Its attribution is methodological, outside the 78 topical works: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). Current arXiv identity was checked during this extension.

---

> Historical audit snapshot. Its proposed corrections were not rechecked against the current manuscript bibliography during this consolidation.

## Existing manuscript bibliography audit

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

### Additional corrections and report families

- HOLISMOKES IV has a [2022 corrigendum](https://doi.org/10.1051/0004-6361/202039574e), verified in publisher metadata. Retrieval of the correction text was blocked by HTTP 403; this review does not quote the original numerical results or guess what was corrected.
- Euclid Q1 LEMON is now a [2026 journal article](https://doi.org/10.1051/0004-6361/202554538). Its 2025 arXiv submission is linked to the same topical record rather than counted twice.
- Filipp's [2024 workshop report](https://openreview.net/pdf?id=1nQIQg3Uiw) is a related precursor to the expanded [2025 robustness paper](https://doi.org/10.3847/1538-4357/adee20). It is not added as an independent item to inflate the evidence count.
- Poh's earlier workshop report and the later [2025 journal paper](https://doi.org/10.1088/1475-7516/2025/05/053) are not counted as independent evidence in this packet. The latter's Appendix A documents the noise conversion issue.
- Legin 2025, Holloway 2026 and Jarugula 2026 remain labelled as preprints because no journal version was verified in the retrieved records. Their findings are not presented as settled consensus.

### Claim checks that matter more than bibliographic formatting

Do not equate R-squared, signed residuals, 68% residual scatter, posterior precision, real-fit agreement and MAE. Do not describe simulated DES-like target data in the two domain-adaptation papers as real DES validation. Do not describe the Galan benchmark as a comprehensive real-data comparison. A correct DOI cannot rescue an incorrect interpretation of a result.

---

> Historical narrative review, cutoff 4 November 2026. The later evidence additions and novelty follow-ups are summarized above.

## Reliable automated inference of galaxy-scale strong gravitational lenses

Comprehensive scoping and critical narrative review for `charlie_lensing`\
Search cutoff: 4 November 2026. Prepared as a research decision document, not a manuscript revision.

### Assessment

The scientific application is the recovery of the projected mass distribution of a foreground galaxy from the distorted images of a background galaxy or quasar. A fast Einstein-radius estimator addresses one useful component of that inverse problem. Its scientific value depends on whether it recovers the angular mass scale reliably across the observing conditions, galaxy morphologies and selected lens populations encountered in observations.

The literature supports the concern that the current framing is too narrow. Fast CNN regression, observational augmentation, uncertainty prediction, domain adaptation and neural initialization of physical fitting all have substantial precedent. A new paper needs to explain a remaining scientific or methodological uncertainty and resolve it with discriminating experiments. Adding a more elaborate network is not, by itself, the missing contribution.

My recommended direction is a controlled study of **which sources of simulation mismatch cause Einstein-radius bias, which inexpensive interventions remove that bias, and when the remaining errors make an estimate unsuitable for use**. Angular resampling, fresh-noise augmentation, observation-condition randomization and calibrated uncertainty are candidate interventions. Their relative value must be measured rather than assumed. This is a proposed direction, not a claim of established novelty.

The evidence base contains 60 unique scholarly works, plus one linked corrigendum, two official detector/noise documents and one workflow attribution. These are not 60 independent experiments or 60 complete full-text reviews. The evidence table records access depth and task relevance. Preprint/journal versions are linked in one record. No pooled performance estimate is justified.

### 1. Scope and review method

The question is: what is established about fast, reliable galaxy-scale strong-lens parameter inference, particularly Einstein radius, and what evidence would make an observation-robustness study scientifically useful?

Included literature covers classical image inversion; CNN regression; Bayesian neural networks; neural posterior, likelihood and ratio estimation; simulation design; detector and PSF effects; prior mismatch; source morphology; uncertainty validation; real-lens applications; failure analysis; selection effects; and relevant downstream mass/population science. Foundational studies before 2017 are included. Recent work is searched through the stated cutoff. General machine-learning papers are explicitly marked as methodological context rather than evidence of lensing performance.

Discovery used recorded web searches targeting arXiv, journal/publisher records, proceedings and citation chains. Crossref records and arXiv citation metadata were retrieved directly to verify identities and link journal versions. Parallel was unavailable because its CLI and authentication were not configured. Searches were refined as negative evidence and recent observational applications emerged. The exact executed discovery strings are in `search_log.json`; metadata provenance and access limitations are in `references.json` and `coverage.json`.

This is a single-reviewer scoping/narrative review using ranked discovery, not a registered systematic review. There were no complete ADS, Scopus or Web of Science exports, no independent duplicate screening, and no defensible denominator for all search hits screened. Consequently, there is no invented PRISMA flow. Some records were assessed only at abstract level; selected full-text methods, results and limitations were inspected for the closest comparisons. Citation verification establishes identity and retrieved support, not that every result was independently reproduced. Correction checks are bounded to retrieved publisher/arXiv records; absence of a flag is not a comprehensive retraction audit.

### 2. The scientific problem and what Einstein radius measures

Light from a background source is deflected by foreground mass. In the usual thin-lens description, the lens equation maps image-plane angle to source-plane angle through a mass-dependent deflection. Observed pixels then combine the lensed source, foreground light, optical blurring, detector integration and noise. Recovering mass therefore involves disentangling astrophysics from measurement effects.

For a circular lens, the Einstein radius defines an aperture whose mean projected surface density equals the critical lensing density. The corresponding projected mass is

`M_E = pi * (D_l * theta_E)^2 * Sigma_crit`,

where the angle is in radians and the critical density depends on lens/source distances. An SIE model's radius convention must be stated explicitly; equivalent-radius and profile-normalization conventions should not be silently interchanged. An angular radius is not a total galaxy mass, a dark-matter fraction, or a direct measurement of the Hubble constant. Mass conversion additionally needs geometry and assumptions. The SLACS analysis of 73 real lenses illustrates the richer science obtained by combining lensing with stellar masses and velocity dispersions. [Auger et al. 2010](https://doi.org/10.1088/0004-637X/724/1/511)

The application has three connected motivations. First, mass modelling helps study galaxy structure and evolution. Second, detailed image perturbations can reveal dark substructure; a scalar radius alone does not make that measurement. Third, time-delay cosmography needs lens models alongside measured delays, environmental information and other constraints. Observational dark-satellite inference and synthetic neural cosmography demonstrate these distinct uses. [Vegetti et al. 2012](https://doi.org/10.1038/nature10669), [Park et al. 2021](https://doi.org/10.3847/1538-4357/abdfc4)

There is also a fundamental identifiability limit. Mass-sheet and source-position transformations allow different mass/source descriptions to preserve important imaging observables while changing other inferred quantities. Even a visually excellent reconstruction cannot establish that every physical parameter is correct. Accurate radius recovery is valuable but cannot certify a complete mass profile or a cosmological analysis. [Schneider and Sluse 2013](https://doi.org/10.1051/0004-6361/201321882), [2014](https://doi.org/10.1051/0004-6361/201322106)

For orientation, the broader application and cosmography are synthesized in [Shajib et al. 2024](https://doi.org/10.1007/s11214-024-01105-x) and [Birrer et al. 2024](https://doi.org/10.1007/s11214-024-01079-w). These reviews supply context; specific empirical claims here are tied to original studies.

### 3. Why automation matters, and what the comparator should be

Survey forecasts motivated automated modelling well before the present project. Collett's population-based forecasts also show that discoverable lens counts depend on source populations, subtraction assumptions and detection thresholds. Such numbers are forecasts, not observed counts or guarantees about the eventual analysis sample. [Collett 2015](https://doi.org/10.1088/0004-637X/811/1/20)

Classical modelling is a serious, evolving comparator. Semilinear inversion solves pixelized source brightness linearly for a fixed lens model; Bayesian extensions address regularization and model comparison. AutoLens subsequently automated joint lens light, mass and source modelling. These methods do not require optimizing every source pixel as an unrelated nonlinear parameter. [Warren and Dye 2003](https://doi.org/10.1086/375132), [Suyu et al. 2006](https://doi.org/10.1111/j.1365-2966.2006.10733.x), [Nightingale et al. 2018](https://doi.org/10.1093/mnras/sty1264)

More recent work accelerates physical inference with automatic differentiation and GPUs. GIGA-Lens combines gradient-based optimization, variational approximation and Hamiltonian Monte Carlo. TinyLensGPU tests simulated and real SLACS lenses and reports catastrophic failures as well as speed. Wall-clock comparisons need matched hardware, fit complexity, initialization, posterior quality and accounting for simulation/training costs. Comparing a forward pass only against historical expert-intensive fitting would overstate the present computational gap. [Gu et al. 2022](https://doi.org/10.3847/1538-4357/ac6de4), [Cao et al. 2025](https://doi.org/10.1093/mnras/staf891)

Software papers establish useful infrastructure: [lenstronomy](https://doi.org/10.1016/j.dark.2018.11.002), [lenstronomy II](https://doi.org/10.21105/joss.03283), [PyAutoLens](https://doi.org/10.21105/joss.02825), [deeplenstronomy](https://doi.org/10.21105/joss.02854), and [Caustics](https://doi.org/10.21105/joss.07081). A simulator's availability, test suite or throughput does not establish that a particular configuration reproduces a survey. Independent software is useful only if the actual rendered experiment exercises independent implementations and compatible conventions.

### 4. From point regression to calibrated inference

#### 4.1 Fast prediction is established

Hezaveh and colleagues established rapid CNN estimation of SIE parameters in 2017, using observational augmentation and demonstrating analysis of observed systems. This is the central historical precedent: neither fast CNN lens modelling nor adding noise/PSF variation is a new research premise today. [Hezaveh et al. 2017](https://doi.org/10.1038/nature23463)

Pearson and colleagues then examined practical dependencies on foreground lens light, colour, SNR and mass-light scatter. HOLISMOKES studied training distributions and lens configurations before extending to parameter uncertainties. These works make training-data assumptions part of the scientific method, not a minor implementation detail. The original HOLISMOKES IV has a publisher-listed corrigendum; its original numerical results are deliberately not used here because the correction text could not be retrieved. [Pearson et al. 2019](https://doi.org/10.1093/mnras/stz1750), [Schuldt et al. 2021](https://doi.org/10.1051/0004-6361/202039574), [corrigendum](https://doi.org/10.1051/0004-6361/202039574e), [Schuldt et al. 2023](https://doi.org/10.1051/0004-6361/202244325)

#### 4.2 Confidence intervals on a score are different from uncertainty for a lens

Perreault Levasseur and colleagues evaluated marginal coverage for neural parameter uncertainties in 2017. LEMON likewise predicts uncertainties, rather than only radius point estimates. Thus uncertainty is not an optional decorative addition to a mature inference study. [Perreault Levasseur et al. 2017](https://doi.org/10.3847/2041-8213/aa9704), [Gentile et al. 2023](https://doi.org/10.1093/mnras/stad1325)

A bootstrap interval around mean test MAE describes uncertainty in an aggregate performance estimate under the resampling scheme. It does not provide an interval for an individual lens. Seed-to-seed variation measures training variability; it is also not automatically a calibrated per-lens posterior. A model can have low average error while making confident mistakes on a scientifically important subset.

For this project, an initial uncertainty baseline could be a mean-variance head or a small ensemble, followed by calibration assessment on held-out parents. Neither deserves the label 'reliable' solely because it returns a variance. A Gaussian head can miss asymmetric or multimodal uncertainty; an ensemble can agree on a shared simulator-induced bias. Coverage must be assessed with interval width and conditional error.

#### 4.3 Posterior and likelihood learning change the output, not the need for validation

Simulation-based inference can learn a posterior, likelihood or likelihood ratio. A scalar network trained with a regression loss is a point estimator; it is not already a Bayesian inference procedure. Legin and colleagues show how regression outputs can instead be treated as compressed statistics whose likelihood is estimated, permitting explicit inference priors. This offers one way to reuse a clean research core without pretending a point predictor contains complete uncertainty information. [Legin et al. 2023](https://doi.org/10.3847/1538-4357/aca7c2), [Cranmer et al. 2020](https://doi.org/10.1073/pnas.1912789117)

Poh and colleagues compare NPE and BNNs on DES-like simulations with one, five and twelve inferred parameters. Their evaluation includes coverage, rank statistics and posterior predictive checks. Their appendix also documents a sky-magnitude conversion bug that reduced simulated background RMS. This makes the paper a useful diagnostic precedent while limiting any attempt to transfer its precision directly to another simulator or real data. [Poh et al. 2025](https://doi.org/10.1088/1475-7516/2025/05/053)

NPE is not a universal fix. General SBI benchmarking finds that performance depends on task, diagnostic and simulation budget; posterior approximation methods can be overconfident. Autoregressive ratio methods offer additional flexibility but introduce training, sampling and validation choices. Complexity should follow a demonstrated limitation of simpler inference. [Lueckmann et al. 2021](https://proceedings.mlr.press/v130/lueckmann21a.html), [Hermans et al. 2022](https://arxiv.org/abs/2110.06581), [Anau Montel et al. 2024](https://doi.org/10.1093/mnras/stae1130)

### 5. Simulation realism has several independent dimensions

#### 5.1 Detector statistics and preprocessing

A basic detector model combines source/background Poisson counts with read noise. Exposure time changes the relation between signal and noise; arbitrary additive noise in a normalized image need not correspond to a physically plausible observation. One must state units, gain, background treatment, exposure combination and the point at which normalization is applied. Official HST exposure-time documentation is a suitable authority for instrument-level noise accounting. [STScI ETC SNR documentation](https://etc.stsci.edu/etcstatic/users_guide/1_2_1_snr.html)

Processed images require additional care. Drizzling/resampling can correlate neighbouring pixels, so a per-pixel variance map alone need not characterize aperture or residual statistics. An independent Gaussian perturbation is not an empirical model of all HST products. A simulator should be matched to a named image product, rather than described generically as 'HST-like'. [STScI DrizzlePac correlated-noise documentation](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise)

Erickson and colleagues provide a close methodological example: empirical HST PSF maps and dither/drizzle treatment enter their simulations. Shajib and colleagues' independent analysis of a real quasar lens shows that PSF treatment can explain important differences between modelling programs. Changing Gaussian to Moffat is a useful controlled stress, but cannot substitute for validating empirical instrument PSFs and image processing. [Erickson et al. 2025](https://doi.org/10.3847/1538-3881/add99f), [Shajib et al. 2022](https://doi.org/10.1051/0004-6361/202243401)

#### 5.2 Astrophysical scene realism

Detector realism does not resolve missing source clumps, mass-profile flexibility, foreground-light residuals, contaminants or correlations between mass and light. A clean Sérsic source behind a smooth isothermal lens may be useful for unit-controlled recovery while being an inadequate stress test of a model intended for observations.

Gawade and colleagues compare increasing simulation realism and benchmark against models of 182 real HSC lenses. Their results favour a particular background-injection/subtraction strategy for that setting, with remaining outliers and ellipticity systematics. This is evidence that scene construction and nuisance handling matter; it is not proof that one universal realism ranking exists. [Gawade et al. 2025](https://doi.org/10.1093/mnras/staf935)

Flexible physical models address different assumptions. Chianese and colleagues couple learned source structure to differentiable lensing; Galan and colleagues use wavelets for mass perturbations; Biggio and colleagues represent lens potentials with neural fields, with known source brightness in their demonstration. Diffusion-based source work supplies learned priors, while a later SLACS preprint models source, lens light and PSF jointly. These methods expand the model family; they do not automatically establish real-data calibration. [Chianese et al. 2020](https://doi.org/10.1093/mnras/staa1477), [Galan et al. 2022](https://doi.org/10.1051/0004-6361/202244464), [Biggio et al. 2023](https://doi.org/10.1051/0004-6361/202245126), [Karchev et al. 2022](https://arxiv.org/abs/2211.04365), [Legin et al. 2025, preprint](https://arxiv.org/abs/2511.19595)

#### 5.3 Population realism and selection

Training priors can influence inferred lens parameters even if images look plausible. Wagner-Carena and colleagues show that hierarchical inference can correct an interim training prior when the training distribution covers the relevant parameter space. This is a specific statistical correction, not a way to repair arbitrary misspecified source morphologies or missing image artefacts. [Wagner-Carena et al. 2021](https://doi.org/10.3847/1538-4357/abdf59)

The lens sample is also selected. Herle and colleagues demonstrate selection dependencies on radius and source morphology in controlled lens-finder experiments. An unweighted average over a convenient synthetic prior need not describe either the selected observed sample or the underlying galaxy population. Selection by successful conventional fits can additionally omit the hardest cases. [Herle et al. 2024](https://doi.org/10.1093/mnras/stae2106)

These distinctions suggest an explicit test taxonomy:

| Test | What changes | What success supports |
| --- | --- | --- |
| New parent systems, same joint distribution | Identity | Generalization within the simulator population |
| Known condition values on new parents | Identity; familiar nuisance support | Robustness within the trained condition range |
| Held-out nuisance values or combinations | Observation conditions | Interpolation/compositional or extrapolation robustness, as specified |
| New PSF, source or mass family | Generative assumptions | Robustness to that specific model mismatch |
| Independent rendering implementation | Numerical implementation | Robustness to measured implementation differences |
| Real observations | Population and observation process jointly | Observational applicability, with imperfect reference models |

These are proposed reporting categories. None should be inferred merely from a directory name or a different simulator label.

### 6. Randomization and domain adaptation: relevant, but different evidence

Domain randomization varies simulated training conditions without necessarily observing a target domain. Domain adaptation can exploit unlabeled target images during training. A method with target-domain access has an information advantage over a source-only method and should not be compared as if the settings were identical.

Swierc and colleagues test DANN and MMD for Einstein-radius regression from a cleaner synthetic source to a DES-like noisy synthetic target. Agarwal and colleagues combine mean-variance estimation with adaptation and assess uncertainty. Both are particularly close precedents, but neither establishes transfer to arbitrary real observations. Their reported metrics also differ from this project's MAE. [Swierc et al. 2023, workshop](https://arxiv.org/abs/2311.17238), [Agarwal et al. 2024, workshop](https://arxiv.org/abs/2411.03334)

Generic work on domain randomization, active sampling and reduced augmentation search provides methodological context. Robotics and image-classification gains cannot establish effectiveness or novelty in astronomical inference. Physically meaningful image transformations and label conventions must be preserved; a geometric resize can change the pixel-to-angle relation even when it leaves an image visually recognizable. [Tobin et al. 2017](https://arxiv.org/abs/1703.06907), [Mehta et al. 2020](https://proceedings.mlr.press/v100/mehta20a.html), [Cubuk et al. 2020](https://arxiv.org/abs/1909.13719)

For the present project, the relevant question is not simply whether variation helps. It is whether the gain survives controls for fresh noise, image scale, source exposure count and policy-selection effort, and whether it extends beyond the nuisance support optimized during development. Failure-weighted sampling also needs a sufficiently precise comparison with uniform sampling before its additional machinery is justified.

### 7. Failure evidence and uncertainty under mismatch

Filipp and colleagues directly challenge a reassuring matched-distribution evaluation. Controlled changes in source morphology, lens parameters, halo descriptions and noise can bias neural ratio and posterior population inference. This concerns subhalo population inference, so its numerical bias is not a forecast of this project's radius errors. It nevertheless establishes that calibration under an assumed simulator and robustness to modest simulator mismatch are separate questions. [Filipp et al. 2025](https://doi.org/10.3847/1538-4357/adee20)

A blind comparison of six physical modelling methods finds systematic differences even with independently simulated, relatively controlled HST data. No method is uniformly best across properties. Source and PSF choices contribute to discrepancies. The experiment's idealizations restrict generalization of its favourable model-combination result. [Galan et al. 2024](https://doi.org/10.1051/0004-6361/202451095)

Failure analysis also appears in observational automation. Etherington and colleagues examine 59 HST lenses and identify foreground-light subtraction and initial model construction as difficult steps; complete recovery includes pipeline adjustments. A paper should distinguish first-pass success from the final result after rescue. [Etherington et al. 2022](https://doi.org/10.1093/mnras/stac2639)

General uncertainty research finds that calibration can deteriorate under distribution shift. Simulation-based calibration checks computational consistency under an assumed generative model, rather than proving that this model describes the telescope. Modrak and colleagues further show why the choice of SBC test quantities affects sensitivity to failure. Coverage/rank diagnostics should therefore be combined with informativeness, conditional performance and predictive checks. [Ovadia et al. 2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [Talts et al. 2018](https://arxiv.org/abs/1804.06788), [Modrak et al. 2025](https://doi.org/10.1214/23-BA1404)

For scalar radius estimation, an informative failure analysis would ask whether large errors concentrate at low radius relative to PSF width, low arc SNR, strong foreground contamination, crop boundaries, poor normalization, or source/mass families excluded from training. Inspect both worst failures and matched successful examples. A saliency map alone does not identify a causal mechanism; remove or perturb the suspected cue while controlling the rest of the scene.

### 8. What real observations now show

The observational evidence is stronger than a narrative in which all lens inference remains synthetic. LEMON has a pilot SLACS application; Gawade and colleagues analyse real HSC systems; Erickson and colleagues apply posterior inference to 14 real STRIDES lenses. These are different populations, instruments and reference standards. Their performance is not a shared benchmark leaderboard.

The published Euclid Q1 LEMON study is especially relevant. Its 354 selected Q1 comparisons give encouraging Einstein-radius agreement but poor mass-ellipticity agreement. Its appendix discusses uncertainty underestimation on real data after calibration on simulations. Agreement with successful conventional fits is useful cross-validation, while selection and shared model assumptions limit what it proves. [Euclid Collaboration / Busillo et al. 2026](https://doi.org/10.1051/0004-6361/202554538)

Adjacent lens-detection evidence reinforces the distinction between simulated and observed validation. Euclid mock-image studies address faint-arc difficulty, and a later Q1 paper measures a substantial detection-domain gap and benefits from real training examples. These are classification studies: their purity/completeness numbers are not regression accuracy or posterior coverage. [Euclid Collaboration 2024](https://doi.org/10.1051/0004-6361/202347244), [2026](https://doi.org/10.1051/0004-6361/202558456)

A credible observational radius check should document usable bands, pixel scale, PSF, weight maps, masking, foreground subtraction, lens confirmation and reference model conventions. Reference-fit disagreement must be reported, not converted into a fictitious ground-truth label. Candidate samples and already well-modelled lenses answer different applicability questions.

### 9. Population science makes small biases more consequential

Neural inference is being developed for dark-matter populations, not just individual scalar parameters. Wagner-Carena and colleagues combine simulated lens posteriors hierarchically and investigate sequential simulation efficiency. These works are adjacent scientific motivation, not direct evidence that radius augmentation detects substructure. [Wagner-Carena et al. 2023](https://doi.org/10.3847/1538-4357/aca525), [2024](https://doi.org/10.3847/1538-4357/ad6e70)

A 2026 LSST sensitivity analysis explicitly conditions its dark-matter forecast on knowledge of the data-generating process. That is an appropriate boundary for useful synthetic science. Two recent preprints develop contamination-aware cosmology and population-calibrated neural ratio inference with spectroscopic information. They connect automated modelling to scientific endpoints while adding assumptions and observables absent from image-only radius prediction. [Filipp et al. 2026](https://doi.org/10.3847/1538-4357/ae8fd0), [Holloway et al. 2026, preprint](https://arxiv.org/abs/2609.21699), [Jarugula et al. 2026, preprint](https://arxiv.org/abs/2608.23534)

The methodological implication is that small shared biases do not disappear when more lenses are analysed. Increasing the number of examples can reduce statistical scatter while leaving a systematic error dominant. A study aimed at reusable survey inference should quantify bias and failure frequency, alongside average precision.

### 10. What this means for the present manuscript

The current local manuscript describes a compact CNN predicting a simulator-defined Einstein radius, trained on 697 qualified parents and evaluated on 150 held-out parents. It compares fixed-condition, uniformly varied and failure-weighted training with matched optimizer updates and ten paired seeds. Parent-level independence and paired budgets are strengths. The four main stress conditions are within the varied-training support, which narrows the generalization claim. The repository's equal-budget report also records that older models had already scored this frozen holdout; an independent refit is not the same as a never-opened dataset. After the present redesign decisions, a fresh confirmatory test population would provide a stronger evaluation boundary.

The manuscript reports four-stress MAE of 0.141251, 0.111507 and 0.107385 arcsec, respectively. Uniform-minus-fixed is -0.029744 arcsec, with parent-bootstrap 95% interval [-0.045725, -0.012141]. Failure-weighted-minus-uniform is -0.004122, with interval [-0.026208, 0.009341]. These are manuscript-reported results, not newly executed or independently regenerated analyses. The evidence supports an improvement from the uniform-training package under this setup; it does not establish an advantage from failure weighting.

The implementation comparison reportedly produces identical float32 images and predictions. This is an engineering consistency result, not an independent simulation-domain transfer result. The Moffat cases instead probe a specific held-out PSF family. Neither result validates the complete simulation-to-real gap.

Several issues need experimental resolution before stronger wording:

1. Fixed-condition and varied training also differ in noise-realization exposure. The implemented noise-only condition is needed to separate that effect from observing-condition variation.
2. Pixel-scale changes may mix angular representation with robustness. Compare native pixels, common-angular-grid preprocessing, and explicitly supplied pixel-scale information before attributing gains to a learned physical invariance.
3. Four known conditions do not establish robustness to unknown survey conditions. Hold out values, combinations and model families explicitly.
4. Aggregate MAE intervals do not characterize individual uncertainty or conditional catastrophic errors.
5. Additional architectures do not resolve a simulator whose PSF, noise or source priors are unvalidated.

These points are an appraisal of the local design against the literature. They are not external published findings about this repository. The manuscript and code have not been edited in this review.

### 11. A defensible next experimental programme

#### First: define the intended use and tolerance

Choose a named image product and a concrete use: fast angular-radius screening, initialization of detailed fitting, or calibrated radius inference. These uses have different acceptable error and failure costs. Choose a scientifically justified tolerance before ranking models. For a mass proxy, state the geometry and radius convention; for initialization, measure downstream fitting success and time.

#### Second: resolve the present mechanism with a small controlled design

Use the same independent parents, paired seeds and optimization budgets. Compare fixed-condition/fixed-noise, fixed-condition/fresh-noise, uniformly varied/fresh-noise, and failure-weighted/fresh-noise training. Cross the most important comparisons with angular-standardized preprocessing. If the complete cross is expensive, run a documented pilot and retain the comparisons that discriminate the competing explanations. Include policy-development computation in the budget report.

The estimand is the paired difference in held-out-parent loss, conditional on a declared test population. Repeated views of a parent are repeated measurements, not independent lenses. Report sampling uncertainty over parents separately from variation across training seeds; a two-way or hierarchical uncertainty analysis is an option if both populations are part of the intended generalization claim. Do not multiply the independent sample size by views or seeds.

#### Third: build physically interpretable stress axes

Select a small set before final evaluation: resolution relative to radius, arc SNR, foreground contrast, PSF family/error, source morphology, mass-family/shear mismatch and correlated/background noise. Start with one-factor tests to localize failure, then add selected physically plausible interactions. Varying all factors arbitrarily at once may expose failure but cannot explain it. Keep development diagnostics and the final holdout separate.

For detector validation, compare simulated blank sky, variance-versus-flux, spatial noise correlation and PSF encircled-energy profiles with the named observational product. Verify units and flux conservation through pixel integration and resampling. These checks test the simulator; predictive accuracy alone does not.

#### Fourth: evaluate accuracy, uncertainty and operational failure

Keep one prespecified primary endpoint and use secondary diagnostics to explain it. A useful set is angular MAE, signed bias, median and upper-tail absolute error, fractional error with explicit handling of small radii, and catastrophic-error fraction at a justified tolerance. Report these by the prespecified stress axes. A larger metric list without an interpretation is not greater scientific depth.

For uncertainty models, evaluate 68%, 90% and 95% coverage with interval widths, appropriate proper scoring rules, and conditional calibration. For flexible posteriors, add rank/SBC diagnostics and posterior predictive residual checks. Fit any calibration mapping on a separate development set. Report shift performance after freezing that mapping.

Choose enough independent test parents to resolve the intended differences. For illustration, with 150 independent Bernoulli coverage outcomes at true coverage 0.95, the binomial standard error is about 1.8 percentage points before accounting for subgrouping. Dividing those parents into small strata makes conditional coverage much less precise. More rendered views do not repair a shortage of independent systems.

#### Fifth: add an external validation level

Use a truly independent rendering path or a clearly different source/mass family first, with convention and image-difference checks. Then evaluate a curated real sample with documented preprocessing and conventional-model comparisons. Restrict claims if the required raw products are unavailable. Do not optimize on the same real lenses later presented as final validation.

A hybrid comparator is particularly attractive: use neural estimates to initialize physical fitting, then measure convergence, residual structure, catastrophic fits, manual interventions and runtime. Pearson's combined modelling and Euclid's initialization experiments provide direct precedent; the new contribution would need to lie in a better reliability characterization or a demonstrated improvement under a defined mismatch. [Pearson et al. 2021](https://doi.org/10.1093/mnras/stab1547)

### 12. Candidate research questions and contribution boundary

The following are proposed questions, not confirmed literature gaps:

- How much of the observed improvement comes from fresh noise, angular standardization and variation of observation conditions, respectively?
- Which instrument and scene mismatches produce systematic radius bias or catastrophic errors, and can simple interventions reduce them without impairing nominal-condition accuracy?
- Can uncertainty estimates or physical-fit diagnostics identify unreliable cases at a useful retained-sample size?
- Do the conclusions survive independent rendering and observational comparison under a specified survey pipeline?

The first two questions offer a focused controlled study. The latter two extend it towards practical inference reliability. A full cosmology or dark-matter-population paper would require a different scientific target and substantially more modelling; it should not be implied by an Einstein-radius-only experiment.

The most supportable current positioning is a **controlled reliability study of a compact Einstein-radius estimator**, with explicit mechanism tests and bounded validation levels. If source realism, observational calibration and external validation are unavailable, a narrower synthetic result remains legitimate, but its contribution must be argued against the existing controlled-simulation literature. The present evidence does not justify a claim that randomization itself is novel, that failure weighting is superior, or that the system is ready for survey science.

### 13. Reading order and evidence limits

For the immediate redesign, start with Pearson 2019, Pearson 2021, Gawade 2025, Poh 2025, Euclid Q1 LEMON 2026, Filipp 2025 and Etherington 2022. Together they cover nuisance assumptions, hybrid baselines, real-image comparisons, posterior diagnostics, negative shift evidence and operational failures. Then read the foundational Hezaveh/Perreault Levasseur papers and Wagner-Carena's prior-correction framework. The full bibliography and 60-row evidence matrix accompany this document.

The literature does not establish that a particular refactor, architecture or adaptive policy will improve this repository. Nor does the search prove that the proposed mechanism study is unprecedented. A novelty check should be repeated once the exact scientific question, stress family and comparator are fixed. The strongest immediate conclusion is that the application remains worthwhile, but average synthetic MAE after augmentation is too limited an endpoint to carry broad claims about reliable astronomical inference.

### Workflow attribution

The local literature-review and research-lookup skills guided search logging, bibliographic checks and evidence separation. Their library description was verified at the current arXiv record and is cited separately from scientific evidence: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). This attribution is not included in the 60 topical scholarly works.

---

> Historical 60-work bibliography snapshot from 4 November 2026; it is not synchronized with the current 92-row matrix.

## Bibliography

60 unique topical scholarly works. Journal year is preferred where verified. Preprints and workshop reports are labelled. Full author metadata is preserved in references.json; long author lists are abbreviated here.

1. **hezaveh2017**. Hezaveh, Yashar D.; Perreault Levasseur, Laurence; Marshall, Philip J. (2017). [Fast automated analysis of strong gravitational lenses with convolutional neural networks](https://doi.org/10.1038/nature23463). Nature, 548, 555-557. [journal article; abstract and bibliographic metadata]

2. **levasseur2017**. Perreault Levasseur, Laurence; Hezaveh, Yashar D.; Wechsler, Risa H. (2017). [Uncertainties in Parameters Estimated with Neural Networks: Application to Strong Gravitational Lensing](https://doi.org/10.3847/2041-8213/aa9704). The Astrophysical Journal Letters, 850, L7. [journal article; abstract and bibliographic metadata]

3. **pearson2019**. Pearson, James; Li, Nan; Dye, Simon (2019). [The use of convolutional neural networks for modelling large optically-selected strong galaxy-lens samples](https://doi.org/10.1093/mnras/stz1750). Monthly Notices of the Royal Astronomical Society, 488, 991-1004. [journal article; abstract and bibliographic metadata]

4. **pearson2021**. Pearson, James; Maresca, Jacob; Li, Nan; Dye, Simon (2021). [Strong lens modelling: comparing and combining Bayesian neural networks and parametric profile fitting](https://doi.org/10.1093/mnras/stab1547). Monthly Notices of the Royal Astronomical Society, 505, 4362-4382. [journal article; selected full-text sections]

5. **schuldt2021**. Schuldt, S.; Suyu, S. H.; Meinhardt, T.; Leal-Taixé, L.; Cañameras, R.; Taubenberger, S.; et al. (2021). [HOLISMOKES. IV. Efficient mass modeling of strong lenses through deep learning](https://doi.org/10.1051/0004-6361/202039574). Astronomy & Astrophysics, 646, A126. [journal article; abstract and bibliographic metadata]

6. **schuldt2023**. Schuldt, S.; Cañameras, R.; Shu, Y.; Suyu, S. H.; Taubenberger, S.; Meinhardt, T.; et al. (2023). [HOLISMOKES. IX. Neural network inference of strong-lens parameters and uncertainties from ground-based images](https://doi.org/10.1051/0004-6361/202244325). Astronomy & Astrophysics, 671, A147. [journal article; abstract and bibliographic metadata]

7. **wagner2021**. Wagner-Carena, Sebastian; Park, Ji Won; Birrer, Simon; Marshall, Philip J.; Roodman, Aaron; Wechsler, Risa H.; et al. (2021). [Hierarchical Inference with Bayesian Neural Networks: An Application to Strong Gravitational Lensing](https://doi.org/10.3847/1538-4357/abdf59). The Astrophysical Journal, 909, 187. [journal article; abstract and bibliographic metadata]

8. **park2021**. Park, Ji Won; Wagner-Carena, Sebastian; Birrer, Simon; Marshall, Philip J.; Lin, Joshua Yao-Yu; Roodman, Aaron; et al. (2021). [Large-scale Gravitational Lens Modeling with Bayesian Neural Networks for Accurate and Precise Inference of the Hubble Constant](https://doi.org/10.3847/1538-4357/abdfc4). The Astrophysical Journal, 910, 39. [journal article; abstract and bibliographic metadata]

9. **legin2023**. Legin, Ronan; Hezaveh, Yashar; Perreault-Levasseur, Laurence; Wandelt, Benjamin (2023). [A Framework for Obtaining Accurate Posteriors of Strong Gravitational Lensing Parameters with Flexible Priors and Implicit Likelihoods Using Density Estimation](https://doi.org/10.3847/1538-4357/aca7c2). The Astrophysical Journal, 943, 4. [journal article; abstract and bibliographic metadata]

10. **gentile2023**. Gentile, Fabrizio; Tortora, Crescenzo; Covone, Giovanni; Koopmans, Léon V E; Li, Rui; Leuzzi, Laura; et al. (2023). [lemon: LEns MOdelling with Neural networks – I. Automated modelling of strong gravitational lenses with Bayesian Neural Networks](https://doi.org/10.1093/mnras/stad1325). Monthly Notices of the Royal Astronomical Society, 522, 5442-5455. [journal article; abstract and bibliographic metadata]

11. **poh2025**. Poh, Jason; Samudre, Ashwin; Ćiprijanović, Aleksandra; Frieman, Joshua; Khullar, Gourav; Nord, Brian D. (2025). [Deep inference of simulated strong lenses in ground-based surveys](https://doi.org/10.1088/1475-7516/2025/05/053). Journal of Cosmology and Astroparticle Physics, 2025, 053. [journal article; selected full-text sections]

12. **gawade2025**. Gawade, Priyanka; More, Anupreeta; More, Surhud; Kimura, Akisato; Sonnenfeld, Alessandro; Oguri, Masamune; et al. (2025). [Neural network prediction of model parameters for strong lensing samples from Hyper Suprime-Cam Survey](https://doi.org/10.1093/mnras/staf935). Monthly Notices of the Royal Astronomical Society, 540, 3384-3399. [journal article; selected full-text sections]

13. **erickson2025**. Erickson, Sydney; Wagner-Carena, Sebastian; Marshall, Phil; Millon, Martin; Birrer, Simon; Roodman, Aaron; et al. (2025). [Lens Modeling of STRIDES Strongly Lensed Quasars Using Neural Posterior Estimation](https://doi.org/10.3847/1538-3881/add99f). The Astronomical Journal, 170, 44. [journal article; selected full-text sections]

14. **euclid_lemon**. Euclid Collaboration; Busillo, V.; Tortora, C.; Metcalf, R. B.; Nightingale, J. W.; Meneghetti, M.; et al. (2026). [Euclid Quick Data Release (Q1). XXXI. LEMON – LEns MOdelling with Neural networks. Automated and fast modelling of Euclid gravitational lenses with singular isothermal ellipsoid mass profile](https://doi.org/10.1051/0004-6361/202554538). Astronomy & Astrophysics, 711, A31. [journal article; selected full-text sections]

15. **swierc2023**. Swierc, Paxson; Zhao, Megan; Ćiprijanović, Aleksandra; Nord, Brian (2023). [Domain Adaptation for Measurements of Strong Gravitational Lenses](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_121.pdf). NeurIPS Workshop on Machine Learning for the Physical Sciences, . [workshop paper; selected full-text sections]

16. **agarwal2024**. Agarwal, Shrihan; Ćiprijanović, Aleksandra; Nord, Brian D. (2024). [Neural Network Prediction of Strong Lensing Systems with Domain Adaptation and Uncertainty Quantification](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_58.pdf). NeurIPS Workshop on Machine Learning for the Physical Sciences, . [workshop paper; selected full-text sections]

17. **holloway2026**. Holloway, Philip; Verma, Aprajita; Marshall, Philip J.; Venkatraman, Padmavathi; Erickson, Sydney; Li, Tian; et al. (2026). [Lens Modeling and Cosmological Inference from an Impure Sample of Galaxy-Galaxy Strong Lenses](https://arxiv.org/abs/2609.21699). arXiv preprint, . [preprint; no journal version verified; abstract and bibliographic metadata]

18. **jarugula2026**. Jarugula, Sreevani; Nord, Brian D.; Ćiprijanović, Aleksandra; Trivedi, Shubhendu (2026). [Strong Lensing Cosmology with Population-level Calibrated Neural Ratio Estimation](https://arxiv.org/abs/2608.23534). arXiv preprint, . [preprint; no journal version verified; abstract and bibliographic metadata]

19. **chianese2020**. Chianese, Marco; Coogan, Adam; Hofma, Paul; Otten, Sydney; Weniger, Christoph (2020). [Differentiable strong lensing: uniting gravity and neural nets through differentiable probabilistic programming](https://doi.org/10.1093/mnras/staa1477). Monthly Notices of the Royal Astronomical Society, 496, 381-393. [journal article; abstract and bibliographic metadata]

20. **biggio2023**. Biggio, L.; Vernardos, G.; Galan, A.; Peel, A.; Courbin, F. (2023). [Modeling lens potentials with continuous neural fields in galaxy-scale strong lenses](https://doi.org/10.1051/0004-6361/202245126). Astronomy & Astrophysics, 675, A125. [journal article; abstract and bibliographic metadata]

21. **galan2022**. Galan, A.; Vernardos, G.; Peel, A.; Courbin, F.; Starck, J.-L. (2022). [Using wavelets to capture deviations from smoothness in galaxy-scale strong lenses](https://doi.org/10.1051/0004-6361/202244464). Astronomy & Astrophysics, 668, A155. [journal article; abstract and bibliographic metadata]

22. **legin2025**. Legin, Ronan; Stone, Connor; Adam, Alexandre; Barco, Gabriel Missael; Coogan, Adam; Malkin, Nikolay; et al. (2025). [Mind the Information Gap: Unveiling Detailed Morphologies of z 0.5-1.0 Galaxies with SLACS Strong Lenses and Data-Driven Analysis](https://arxiv.org/abs/2511.19595). arXiv preprint, . [preprint; no journal version verified; abstract and bibliographic metadata]

23. **wagner2023**. Wagner-Carena, Sebastian; Aalbers, Jelle; Birrer, Simon; Nadler, Ethan O.; Darragh-Ford, Elise; Marshall, Philip J.; et al. (2023). [From Images to Dark Matter: End-to-end Inference of Substructure from Hundreds of Strong Gravitational Lenses](https://doi.org/10.3847/1538-4357/aca525). The Astrophysical Journal, 942, 75. [journal article; abstract and bibliographic metadata]

24. **wagner2024**. Wagner-Carena, Sebastian; Lee, Jaehoon; Pennington, Jeffrey; Aalbers, Jelle; Birrer, Simon; Wechsler, Risa H. (2024). [A Strong Gravitational Lens Is Worth a Thousand Dark Matter Halos: Inference on Small-scale Structure Using Sequential Methods](https://doi.org/10.3847/1538-4357/ad6e70). The Astrophysical Journal, 975, 297. [journal article; abstract and bibliographic metadata]

25. **warren2003**. Warren, S. J.; Dye, S. (2003). [Semilinear Gravitational Lens Inversion](https://doi.org/10.1086/375132). The Astrophysical Journal, 590, 673-682. [journal article; abstract and bibliographic metadata]

26. **suyu2006**. Suyu, S. H.; Marshall, P. J.; Hobson, M. P.; Blandford, R. D. (2006). [A Bayesian analysis of regularized source inversions in gravitational lensing](https://doi.org/10.1111/j.1365-2966.2006.10733.x). Monthly Notices of the Royal Astronomical Society, 371, 983-998. [journal article; abstract and bibliographic metadata]

27. **nightingale2018**. Nightingale, J W; Dye, S; Massey, Richard J (2018). [AutoLens: automated modeling of a strong lens’s light, mass, and source](https://doi.org/10.1093/mnras/sty1264). Monthly Notices of the Royal Astronomical Society, 478, 4738-4784. [journal article; abstract and bibliographic metadata]

28. **etherington2022**. Etherington, Amy; Nightingale, James W; Massey, Richard; Cao, XiaoYue; Robertson, Andrew; Amorisco, Nicola C; et al. (2022). [Automated galaxy–galaxy strong lens modelling: No lens left behind](https://doi.org/10.1093/mnras/stac2639). Monthly Notices of the Royal Astronomical Society, 517, 3275-3302. [journal article; abstract and bibliographic metadata]

29. **birrer2018**. Birrer, Simon; Amara, Adam (2018). [lenstronomy: Multi-purpose gravitational lens modelling software package](https://doi.org/10.1016/j.dark.2018.11.002). Physics of the Dark Universe, 22, 189-201. [journal article; abstract and bibliographic metadata]

30. **birrer2021**. Birrer, Simon; Shajib, Anowar; Gilman, Daniel; Galan, Aymeric; Aalbers, Jelle; Millon, Martin; et al. (2021). [lenstronomy II: A gravitational lensing software ecosystem](https://doi.org/10.21105/joss.03283). Journal of Open Source Software, 6, 3283. [journal article; publisher software-paper landing page and bibliographic metadata]

31. **nightingale2021**. Nightingale, James.; Hayes, Richard; Kelly, Ashley; Amvrosiadis, Aristeidis; Etherington, Amy; He, Qiuhan; et al. (2021). [PyAutoLens: Open-Source Strong Gravitational Lensing](https://doi.org/10.21105/joss.02825). Journal of Open Source Software, 6, 2825. [journal article; publisher software-paper landing page and bibliographic metadata]

32. **morgan2021**. Morgan, Robert; Nord, Brian; Birrer, Simon; Lin, Joshua; Poh, Jason (2021). [deeplenstronomy: A dataset simulation package for strong gravitational lensing](https://doi.org/10.21105/joss.02854). Journal of Open Source Software, 6, 2854. [journal article; abstract and bibliographic metadata]

33. **stone2024**. Stone, Connor; Adam, Alexandre; Coogan, Adam; Yantovski-Barth, M. J.; Filipp, Andreas; Setiawan, Landung; et al. (2024). [Caustics: A Python Package for Accelerated Strong Gravitational Lensing Simulations](https://doi.org/10.21105/joss.07081). Journal of Open Source Software, 9, 7081. [journal article; abstract and bibliographic metadata]

34. **gu2022**. Gu, A.; Huang, X.; Sheu, W.; Aldering, G.; Bolton, A. S.; Boone, K.; et al. (2022). [GIGA-Lens: Fast Bayesian Inference for Strong Gravitational Lens Modeling](https://doi.org/10.3847/1538-4357/ac6de4). The Astrophysical Journal, 935, 49. [journal article; abstract and bibliographic metadata]

35. **galan2024**. Galan, A.; Vernardos, G.; Minor, Q.; Sluse, D.; Van de Vyvere, L.; Gomer, M. (2024). [Exploiting the diversity of modeling methods to probe systematic biases in strong lensing analyses](https://doi.org/10.1051/0004-6361/202451095). Astronomy & Astrophysics, 692, A87. [journal article; selected full-text sections]

36. **schneider2013**. Schneider, Peter; Sluse, Dominique (2013). [Mass-sheet degeneracy, power-law models and external convergence: Impact on the determination of the Hubble constant from gravitational lensing](https://doi.org/10.1051/0004-6361/201321882). Astronomy & Astrophysics, 559, A37. [journal article; abstract and bibliographic metadata]

37. **schneider2014**. Schneider, Peter; Sluse, Dominique (2014). [Source-position transformation: an approximate invariance in strong gravitational lensing](https://doi.org/10.1051/0004-6361/201322106). Astronomy & Astrophysics, 564, A103. [journal article; abstract and bibliographic metadata]

38. **collett2015**. Collett, Thomas E. (2015). [THE POPULATION OF GALAXY–GALAXY STRONG LENSES IN FORTHCOMING OPTICAL IMAGING SURVEYS](https://doi.org/10.1088/0004-637X/811/1/20). The Astrophysical Journal, 811, 20. [journal article; abstract and bibliographic metadata]

39. **birrer2024**. Birrer, S.; Millon, M.; Sluse, D.; Shajib, A. J.; Courbin, F.; Erickson, S.; et al. (2024). [Time-Delay Cosmography: Measuring the Hubble Constant and Other Cosmological Parameters with Strong Gravitational Lensing](https://doi.org/10.1007/s11214-024-01079-w). Space Science Reviews, 220, 48. [journal article; abstract and bibliographic metadata]

40. **tinylens2025**. Cao, Xiaoyue; Li, Ran; Li, Nan; Chen, Yun; Li, Rui; Shan, Huanyuan; et al. (2025). [CSST strong lensing preparation: fast modelling of galaxy–galaxy strong lenses in the big data era](https://doi.org/10.1093/mnras/staf891). Monthly Notices of the Royal Astronomical Society, 540, 3121-3134. [journal article; abstract and bibliographic metadata]

41. **herle2024**. Herle, A; O’Riordan, C M; Vegetti, S (2024). [Selection functions of strong lens finding neural networks](https://doi.org/10.1093/mnras/stae2106). Monthly Notices of the Royal Astronomical Society, 534, 1093-1106. [journal article; abstract and bibliographic metadata]

42. **euclid2024**. Euclid Collaboration; Leuzzi, L.; Meneghetti, M.; Angora, G.; Metcalf, R. B.; Moscardini, L.; et al. (2024). [Euclidpreparation. XXXIII. Characterization of convolutional neural networks for the identification of galaxy-galaxy strong-lensing events](https://doi.org/10.1051/0004-6361/202347244). Astronomy & Astrophysics, 681, A68. [journal article; abstract and bibliographic metadata]

43. **euclid2026**. Euclid Collaboration; Lines, N. E. P.; Collett, T. E.; Holloway, P.; Rojas, K.; Schuldt, S.; et al. (2026). [Euclid Quick Data Release (Q1). XXXII. From simulations to sky: Advancing machine-learning lens detection with real Euclid data](https://doi.org/10.1051/0004-6361/202558456). Astronomy & Astrophysics, 711, A32. [journal article; abstract and bibliographic metadata]

44. **cranmer2020**. Cranmer, Kyle; Brehmer, Johann; Louppe, Gilles (2020). [The frontier of simulation-based inference](https://doi.org/10.1073/pnas.1912789117). Proceedings of the National Academy of Sciences, 117, 30055-30062. [journal article; abstract and bibliographic metadata]

45. **lueckmann2021**. Lueckmann, Jan-Matthis; Boelts, Jan; Greenberg, David S.; Gonçalves, Pedro J.; Macke, Jakob H. (2021). [Benchmarking Simulation-Based Inference](https://proceedings.mlr.press/v130/lueckmann21a.html). Proceedings of AISTATS, PMLR 130, , 343-351. [conference paper; abstract and bibliographic metadata]

46. **hermans2022**. Hermans, Joeri; Delaunoy, Arnaud; Rozet, François; Wehenkel, Antoine; Begy, Volodimir; Louppe, Gilles (2022). [A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful](https://arxiv.org/abs/2110.06581). Transactions on Machine Learning Research (TMLR version identified in arXiv record), . [journal version; arXiv access; abstract and bibliographic metadata]

47. **talts2018**. Talts, Sean; Betancourt, Michael; Simpson, Daniel; Vehtari, Aki; Gelman, Andrew (2018). [Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://arxiv.org/abs/1804.06788). arXiv preprint, . [preprint; no journal version verified; abstract and bibliographic metadata]

48. **ovadia2019**. Ovadia, Yaniv; Fertig, Emily; Ren, Jie; Nado, Zachary; Sculley, D; Nowozin, Sebastian; et al. (2019). [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html). Advances in Neural Information Processing Systems 32, . [conference paper; abstract and bibliographic metadata]

49. **cubuk2020**. Cubuk, Ekin D.; Zoph, Barret; Shlens, Jonathon; Le, Quoc V. (2020). [RandAugment: Practical automated data augmentation with a reduced search space](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html). Advances in Neural Information Processing Systems 33, . [conference paper; abstract and bibliographic metadata]

50. **tobin2017**. Tobin, Josh; Fong, Rachel; Ray, Alex; Schneider, Jonas; Zaremba, Wojciech; Abbeel, Pieter (2017). [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907). arXiv (conference report; journal DOI not used here), . [conference report; arXiv access; abstract and bibliographic metadata]

51. **mehta2020**. Mehta, Bhairav; Diaz, Manfred; Golemo, Florian; Pal, Christopher J.; Paull, Liam (2020). [Active Domain Randomization](https://proceedings.mlr.press/v100/mehta20a.html). Proceedings of the Conference on Robot Learning, PMLR 100, , 1162-1176. [conference paper; abstract and bibliographic metadata]

52. **shajib2022**. Shajib, A. J.; Wong, K. C.; Birrer, S.; Suyu, S. H.; Treu, T.; Buckley-Geer, E. J.; et al. (2022). [TDCOSMO. IX. Systematic comparison between lens modelling software programs: Time-delay prediction for WGD 2038−4008](https://doi.org/10.1051/0004-6361/202243401). Astronomy & Astrophysics, 667, A123. [journal article; selected full-text sections]

53. **karchev2022**. Karchev, Konstantin; Montel, Noemi Anau; Coogan, Adam; Weniger, Christoph (2022). [Strong-Lensing Source Reconstruction with Denoising Diffusion Restoration Models](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_169.pdf). NeurIPS Workshop on Machine Learning for the Physical Sciences, . [workshop paper; abstract and bibliographic metadata]

54. **shajib_review2024**. Shajib, A. J.; Vernardos, G.; Collett, T. E.; Motta, V.; Sluse, D.; Williams, L. L. R.; et al. (2024). [Strong Lensing by Galaxies](https://doi.org/10.1007/s11214-024-01105-x). Space Science Reviews, 220, 87. [journal article; abstract and bibliographic metadata]

55. **auger2010**. Auger, M. W.; Treu, T.; Bolton, A. S.; Gavazzi, R.; Koopmans, L. V. E.; Marshall, P. J.; et al. (2010). [THE SLOAN LENS ACS SURVEY. X. STELLAR, DYNAMICAL, AND TOTAL MASS CORRELATIONS OF MASSIVE EARLY-TYPE GALAXIES](https://doi.org/10.1088/0004-637X/724/1/511). The Astrophysical Journal, 724, 511-525. [journal article; abstract and bibliographic metadata]

56. **vegetti2012**. Vegetti, S.; Lagattuta, D. J.; McKean, J. P.; Auger, M. W.; Fassnacht, C. D.; Koopmans, L. V. E. (2012). [Gravitational detection of a low-mass dark satellite galaxy at cosmological distance](https://doi.org/10.1038/nature10669). Nature, 481, 341-343. [journal article; abstract and bibliographic metadata]

57. **filipp2025**. Filipp, Andreas; Hezaveh, Yashar; Perreault-Levasseur, Laurence (2025). [Robustness of Neural Ratio and Posterior Estimators to Distributional Shifts for Population-level Dark Matter Analysis in Strong Gravitational Lensing](https://doi.org/10.3847/1538-4357/adee20). The Astrophysical Journal, 989, 226. [journal article; selected full-text sections]

58. **modrak2025**. Modrák, Martin; Moon, Angie H.; Kim, Shinyoung; Bürkner, Paul; Huurre, Niko; Faltejsková, Kateřina; et al. (2025). [Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity](https://doi.org/10.1214/23-BA1404). Bayesian Analysis, 20, 461-488. [journal article; abstract and bibliographic metadata]

59. **montel2024**. Anau Montel, Noemi; Alvey, James; Weniger, Christoph (2024). [Scalable inference with autoregressive neural ratio estimation](https://doi.org/10.1093/mnras/stae1130). Monthly Notices of the Royal Astronomical Society, 530, 4107-4124. [journal article; abstract and bibliographic metadata]

60. **filipp2026**. Filipp, Andreas; Hezaveh, Yashar; Perreault-Levasseur, Laurence; Gilman, Daniel; LSST Dark Energy Science Collaboration (2026). [LSST Strong Lensing Systems Dark Matter Sensitivity Analysis with Neural Ratio Estimators](https://doi.org/10.3847/1538-4357/ae8fd0). The Astrophysical Journal, 1007, 190. [journal article; abstract and bibliographic metadata]

---

> Historical provenance for the original 4 November review and checkout. The branch, commit, and fingerprints below are not current checkout provenance.

## Local provenance and verification

Prepared in `C:/Users/richa/Desktop/charlie_lensing`, branch `richard_changes`, base commit `6362f2cc228ec2343dd36f347177adcab80bfa1a`. Review cutoff: 4 November 2026.

The manuscript, existing bibliography, experiment runner and training/simulation sources below are unchanged against HEAD. Their contents informed the local-project appraisal, not the external literature findings. Numerical study results were read from the manuscript and existing report; no new training, scoring or raw-output reproduction was performed.

### Source fingerprints (SHA-256)

- `submission/aaai27/reviewer_bundle/paper.tex`: `ace12f26ff9c968312a4932a271231fa9cdab6a7a951913264f7ddef4c250cd0`
- `submission/aaai27/reviewer_bundle/paper.bib`: `afae0350cd77ebd8e26e57030145434a1718592ec76bc22d13e0f10030178fef`
- `docs/equal_budget_final_v1.md`: `de3ec19cdc37eacf951162f74c21ef4e71faa6090e3298b446a740744dd74bc3`
- `scripts/run_equal_budget_final_v1.py`: `29e3647c7dbb20bfc9ad39e8cb9ad6ee7dc729b92ef89b226deb9d3ad556215c`
- `src/ml/robustness_training_v1.py`: `96c1112c6b83155405714a1cbcd8baf836a17510678799f2226e88eda977287a`
- `src/simulation/autolens_backend_v2.py`: `b2ec2a1822c523c8b28a2c280eca1e4e3777cdf8cf19208c56f3904eb4474689`

### Review construction

- Public arXiv records and publisher/Crossref metadata verified bibliographic identity and linked journal/preprint reports. The CSV contains reviewer paraphrases rather than copied abstracts.
- Direct Crossref requests initially encountered one rate-limit response; that citation was retrieved successfully on the later check. HOLISMOKES correction identity is verified, but its full text remains unavailable after publisher HTTP 403 responses.
- Ten works received selected full-text-section assessment. The 60-work count includes reviews, software and general methodology, not solely independent lens-inference experiments.
- The exact retained web-discovery queries and their date are in `search_log.json`. Early broad orientation searches are not retained as exact queries; completeness is not claimed.
- The PDF was created using ReportLab in this repository's `.venv`, checked with pypdf, rendered with Poppler and visually inspected. The source is the review plus readable bibliography and citation audit.
- No manuscript or scientific source changes, repository refactors, external messages, paid research calls or experiment launches were made for this review.

### Computational checks

The validation checks 60 unique topical records; DOI and ID uniqueness; agreement between the evidence CSV and metadata; claim-source IDs and local source paths; BibTeX entry count; JSON readability; missing metadata; encoding/replacement characters; accidental tool references or unresolved drafting markers; and unchanged local source files. These checks establish packet consistency, not independent replication of the scientific literature.
