# Candidate directions after the novelty check

Date: 12 November 2026. Status: proposed questions and experiments, not executed findings or verified priority claims. Decision owner: user; no direction selected. Constraints: approximately 700 original training parents (697 qualified in the maintained manuscript), compact image-only Einstein-radius regression, 256x256 canonical images at 0.05 arcsec/pixel, simulator radius range 0.5-2 arcsec, limited development parents and no new observational validation. Missing production arrays/checkpoints remain an execution dependency.

## Recommendation and evidence boundary

The strongest next question is **whether robustness to individual observing changes survives their combinations, and which extra synthetic examples restore it**. A complementary question is whether a small development sample can reliably identify the repair at all. A third is how to spend a fixed image budget between independent physical scenes and repeated observing conditions.

None is a verified first-of-kind method. These are narrower candidate contributions than another shortcut injection/removal study. Generic compound-shift robustness, experimental interactions and simulation-budget allocation are established concepts. The proposed value is an interpretable lensing result with explicit physical controls and a small-data decision problem.

The [master matrix](../../../literature_review/strong_lens_inference/evidence_matrix.csv) now has **92 works**, after five additions. The [search/extraction record](../../../literature_review/strong_lens_inference/directions_search_2026-11-12.json) records assessed sources, search strings and access limits. This is a bounded AI-assisted, single-reviewer scoping exercise, not exhaustive ADS/Scopus screening. No independent second reviewer was used. Query strings were transcribed during the session rather than exported from a bibliographic service; the log identifies missing exact variants. Earlier bibliography/PDF exports remain dated snapshots.

## What changed after reading the limitations?

| Located evidence | Limitation or implication | Candidate response |
|---|---|---|
| [Filipp et al. 2025](https://arxiv.org/html/2411.05905v2), Sections IV-V | Tests individual deviations while acknowledging simultaneous effects in observations | Quantify interactions and dependence shifts; do not extrapolate its subhalo-posterior results to radius error |
| [Jacobs et al., arXiv 2021](https://arxiv.org/abs/2112.02479) | Sensitivity probes already expose lens-classifier weaknesses | A response sweep alone is insufficient; test whether diagnostic decisions reproduce and transfer |
| [Olischlager et al. 2026](https://arxiv.org/html/2609.39525v1), Sections 5.4/A.8 | Already diagnose representation gaps and use auxiliary supervision in lensing | Avoid claiming novelty for latent probes or predicting simulator parameters from features |
| [Hughes et al. 2024](https://arxiv.org/abs/2403.04349) | Source-complexity experiments already exist for subhalo detection | Study allocation or transfer, not merely more Sersic clumps |
| [Galan et al. 2024](https://arxiv.org/html/2406.08484v2) | Classical source-model and PSF assumptions interact in reconstruction | Physical interactions are known; the open candidate is learned-regressor robustness and repair under matched budgets |
| Existing local policy and secondary conditions | Seven separate training cells; secondary suite already includes Moffat plus half exposure | Do not claim the first combined-condition test even within this project; require a systematic factorial comparison |

## A. Robustness to the pieces versus robustness to their combination

**Plain-language question:** Does practising blurry images and noisy images separately prepare the CNN for an image that is both blurry and noisy?

**Design.** Start with two factors at a fixed pixel scale: Gaussian PSF width and exposure time. For every parent, render a 2x2 family: canonical, wider PSF only, shorter exposure only, both. Add one intermediate level or an independent PSF family later. Hold mass/source/foreground identities fixed; repeat detector draws to distinguish systematic effects from random realization. Exposure and PSF both alter information, so failure is not automatically a shortcut.

Compare canonical training, single-factor variation, and joint-factor variation with the same 697 parent identities, initialization schedule, updates and exposure count. A particularly clean contrast compares two sampling policies with the same frequency of each PSF level and each exposure level but different pairings. Each parent should receive those assignments through a balanced recorded schedule. The target must remain independent of the assigned observation policy. Test new combinations and strengths on fresh parents.

**Feature analysis.** Measure signed prediction response and error in all four cells. A signed second difference, prediction(both)-prediction(PSF)-prediction(exposure)+prediction(canonical), detects non-additive model response to the declared changes. Apply a corresponding contrast to absolute error. Neither is a unique decomposition of learned features: image formation and the absolute-value loss also introduce nonlinearities. Supplement with interpretable arc visibility/contrast descriptors and component renders, without feeding truth-side descriptors to the predictor.

**Potential contribution.** Establish where single-factor training fails despite matched marginal coverage, and whether allocating examples to missing combinations is more efficient than oversampling individually difficult conditions. This changes the question from which condition is hardest to which combinations are missing.

**Feasibility.** High relative to source-family changes: existing PSF/noise controls and CNN can be reused, but generation/scheduling needs new support. Four evaluation cells preserve 150 independent test units; they do not create 600 independent lenses. Begin with a few paired seeds for development, expanding only after the pilot is informative.

**Adversarial check.** If joint training wins only because it sees more severe images, marginal matching failed. If combined error is fully explained by diminished visibility and every policy behaves alike, report that boundary rather than a learned shortcut. Null interaction is useful evidence against an unnecessarily complicated policy.

## B. How many independent scenes are needed to trust a feature diagnosis?

**Plain-language question:** Would a different small set of lenses, or a fresh noise draw, make us choose a different repair?

**Design.** Freeze a small candidate set of repairs and a source-development population. Compare diagnostics based on error alone, rendered prediction responses, and (optionally) one validated attribution method. Allocate the same diagnostic render budget differently: for example, 25 parents x 12 noise draws, 75 x 4, or 150 x 2, giving 300 views per intervention. These numbers are proposed budget cells, not power calculations. Repeat parent-subset selection within development and preserve that repeated-use structure in analysis.

Each diagnostic selects a repair from the same declared candidates. Candidate models can be trained once per paired seed and kept fixed while evaluating many diagnostic allocations; there is no need to retrain for every subset. Final selection policies must be frozen before evaluating fresh parents and fresh noise. Compare selection stability and downstream error, not just visual similarity of explanations. An evaluation-set best-candidate reference is descriptive and uncertain, not a known population oracle.

**Potential contribution.** A practical reliability curve for mechanism-guided data decisions under a finite diagnostic budget. Attribution instability is known from Krell, and seed-dependent shortcuts from D'Amour; the distinct candidate estimand is how those uncertainties affect an actual synthesis decision and its independent outcome.

**Feasibility.** High once a few candidate models/renders exist: much work is repeated inference. Use parent-level inference, not thousands of correlated noise images as independent samples. Ten training seeds are ten model realizations, insufficient for flexible model-level failure prediction.

**Adversarial check.** Stable diagnosis may be stably wrong. Require independent repair benefit. Conversely, different rankings may choose equally effective repairs; ranking disagreement alone is not scientific harm. This is an empirical benchmark contribution, not a new general explanation method.

## C. More physical systems or more observing-condition variants?

**Plain-language question:** With a fixed number of synthetic training images, is it better to show more distinct lenses or show fewer lenses under many conditions?

**Design.** Cross independent-parent count (e.g., 100, 300, 697) with narrow versus broader observing variation while fixing total generated-image count and optimizer updates. Use stratified, repeated parent draws and a common target distribution; do not let a small set accidentally exclude part of the radius range. Separate the rendering budget from optimizer exposures and wall-clock cost. A noiseless scene reused for fresh detector noise is cheaper than a new physical render; report both image-count and actual-cost comparisons.

Evaluate ordinary conditions, held-out combinations from A, and a source-family shift if implemented. Track whether feature-response profiles stabilize with additional independent parents or merely with more noisy views. This is stronger than a standard learning curve because it compares opportunity costs, but it is still related to longstanding augmentation and sample-efficiency research.

**External extension.** Add PSF/noise-aware empirical source templates from COSMOS through the common renderer, or initially use a validated procedural multi-component family. Keep mass and target distributions matched. Empirical-template rendering is not already supported merely because the library exists. Split source-template identities before generating lenses; repeated transforms of one template are not new galaxies. Cluster or use a crossed design if source templates and lens masses are reused.

**Feasibility.** Moderate. The in-family allocation study fits the present scale; empirical morphology adds meaningful implementation and validation work. Do not make a high novelty claim for 100/300/697 alone.

**Adversarial check.** If differences disappear once training convergence or target coverage is controlled, the original interpretation was wrong. If all arms use the same Sersic family, conclusions concern parameter diversity within that family, not realism.

## D. Calibration and tensor-shape cues: prerequisite audit, possible smaller methods study

The current renderer changes array dimensions to preserve field of view when pixel scale changes: nominally 320 pixels at 0.04, 256 at 0.05 and 213 at 0.06 arcsec/pixel. Input geometry is therefore associated with the observing condition even when no metadata vector is supplied. This is a code-level possibility, not evidence the CNN actually exploits it.

Compare explicit pixel-scale conditioning, physically calibrated common-grid preprocessing, and native-shape inputs with carefully controlled cropping/padding. Do not blindly resize and retain an inconsistent angular label. Separate array-shape cues, PSF-in-pixel cues and prior-based size cues. Exact scale-ambiguous synthetic pairs can provide an identifiability control if all angular components and flux conventions are transformed consistently.

The need for angular calibration is established physics, not a novel theorem. This audit could overturn an interpretation of robustness or reveal a hidden information channel. Native shape itself legitimately carries calibration under a fixed-field-of-view contract; it is not automatically leakage. Audit that contract before making cross-instrument claims.

## Dataset compatibility: direct answer

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

## Ranking and decision

Criteria were relevance to synthetic-data decisions, distinct contribution beyond the checked matrix, feasibility at current independent-parent counts, and interpretability of a null result. Ratings are qualitative judgments, not measurements or acceptance forecasts.

| Rank | Direction | Novelty potential | Feasibility | Evidence that would justify proceeding |
|---|---|---|---|---|
| 1 | Compound shifts plus interaction-aware synthesis | Moderate, with direct limitation-based motivation | High for two factors | Reproducible non-additive error or a training-policy difference under matched marginals |
| 2 | Reliability of feature-based repair decisions | Moderate; small-data decision endpoint is the distinction | High after candidate models exist | Parent/noise allocation changes independent repair performance, beyond ranking instability |
| 3 | Independent-parent diversity versus condition coverage | Moderate-low alone; stronger with held-out morphology | Moderate | A reproducible budget frontier after target and optimization controls |
| Supporting | Calibration/shape-cue audit | Uncertain, potentially useful negative methods result | High for initial audit | Demonstrated dependence on a representation convention rather than presumed leakage |

Recommended coherent project: **Do synthetic robustness interventions compose, and can a small diagnostic set identify which missing combinations to simulate?** A tests the failure and repair; B establishes whether that repair choice can be trusted at our sample size. C is a later budget extension. Run D before interpreting scale-related results. No recommendation requires a new architecture, a learned image generator or a large external dataset.

Workflow attribution: literature-review and scientific-brainstorming skills guided source separation, alternative generation and adversarial checks. Their library description is cited outside topical evidence: Kassis et al. (2026), [Scientific Agent Skills](https://arxiv.org/abs/2609.00065), current identity checked. No experiment, manuscript or memory file was changed.
