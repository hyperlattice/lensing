# Reliability study decision log

Append decisions in order. Never rewrite a past decision; supersede it with a dated entry and protocol version if needed.

## 2026-09-25 — D-001: Narrow the paper to reliability of scalar inference

Keep `theta_E_arcsec` as the only primary target, under the named simulator mass-model convention. The research question is when image-based estimates remain reliable under controlled simulation/observation mismatch. Do not expand to arbitrary mass maps, total mass, dark matter, or cosmography in this study.

## 2026-09-25 — D-002: Quarantine historical test evidence

The existing `sim-eng1000-v1` test split and saved CNN test evaluation have already been opened. Preserve the report as historical evidence only. No new model, preprocessing, confidence threshold, or mitigation decision may depend on the old test samples or metrics. Use train/validation for development and reserve a new independent holdout for the final evaluation.

## 2026-09-25 — D-003: Freeze the comparison design

Use matched counterfactual descendants, preserve parent ancestry across splits, change one major nuisance at a time, and estimate paired effects over independent parents. Report seed variation separately from parent bootstrap intervals. Keep the current CNN as the fixed initial model and use the existing classical SIE fitter only on cases for which its assumptions are valid.

## 2026-09-25 — D-004: Freeze hypotheses and practical-effect rules

Freeze H1-H6 and the 0.01 arcsec practical-effect margin in `protocol_v1.md` before new reliability predictions. The margin is informed by the existing baseline seed variation and conservative development gate; it is not a significance threshold. Any different threshold requires a new protocol version before affected predictions are inspected.

## 2026-09-25 — D-005: Limit use of real data

Use `real-castles-v2` for preprocessing/QC and measurable domain-gap analysis. The release's calibration limitations and three preselected astrometric comparisons do not support a general real-domain accuracy claim.

## 2026-09-25 — D-006: Respect the source timeline

Complete the Week 1 audit and freeze before implementing Week 2 counterfactual infrastructure. Follow Weeks 2-12 in the order in `protocol_v1.md`; scientific gates take precedence over calendar dates. Cross-simulator, uncertainty, extra architectures, and hybrid work do not replace the core priority sequence.

## 2026-09-25 — D-007: Branching constraint

The source plan calls for a dedicated reliability-study branch. The environment exposes `.git` as read-only, so branch creation requires the explicit escalation path. The branch request is made only after Week 1 artifacts are reviewable; the restriction is not bypassed.

## 2026-09-25 — D-008: Start Week 2 after the Week 1 gate

The Week 1 audit and protocol freeze are complete on branch `reliability-study-v1`. Proceed to inspect and implement the matched-counterfactual infrastructure. Do not begin the Week 3 pixel-scale evaluation until Week 2 rerender determinism, label invariance, and ancestry/split isolation are validated.

## 2026-09-25 — D-009: Week 2 gate passed

The counterfactual smoke benchmark reproduced all four canonical parents' expected scenes, observed rates, and normalized images exactly. Eight descendants across four development parents retained `theta_E_arcsec` and ancestry-safe split assignments, with zero rejected renders and successful strict-loader validation. The Week 2 report is `week2_counterfactual.md`. Freeze the exact pixel-scale grid and field-of-view convention before Week 3 predictions.

## 2026-09-25 — D-010: Week 3 CNN pixel-scale results; classical comparison still required

The frozen CNN shows a practically important pixel-scale penalty at both 0.04 and 0.06 arcsec/pixel. Angular resampling reduces the observed shift, but both intervals remain inconclusive under the predeclared practical-small rule. Pixel-space-target and metadata-conditioned training improve canonical and 0.04 validation MAE but worsen 0.06 MAE versus the frozen baseline; neither passes the H1 utility rule. All new model selection used the 150-parent validation set; the historical test remains quarantined. The reference also requires a classical-method comparison. The existing fitter initializes from true parameters and conditions on saved point positions, so it is not a valid blind baseline. Keep Week 3 open until a truth-free classical setup is frozen and compared, or the limitation is resolved by an explicit protocol amendment. Details: `week3_pixel_scale_status.md`.

## 2026-09-26 — D-011: Week 3 closed with a limited truth-free classical control

Freeze the observable-only point-image geometry method in `protocol_v1.3.md`. It uses detections from observed images and fits SIS/SIE image geometry, with a training-mean fallback; it does not load validation truth during fitting. On 150 validation parents per scale, the fallback-inclusive pipeline was less accurate than the CNN, with roughly one-third of cases falling back. Conditional fit-only errors are reported separately and are not treated as all-parent performance. The classical method's paired scale effects are inconclusive. Week 3 is complete with this stated limitation; a full pixel-likelihood SIE comparison belongs with Week 6 baseline work. Proceed to Week 4 observation-systematics studies. Full results: `week3_pixel_scale_status.md` and `results/reports/reliability_v1/pixel_scale/classical_geometric/report.md`.

## 2026-09-26 — D-012: Freeze Week 4 observation-systematics conditions

Before Week 4 predictions, freeze the matched conditions and estimands in `protocol_v1.4.md`. Use the same 150 H1 validation parents and the frozen three-seed CNN; test Gaussian PSF width at 0.75x/1.25x, two parent-seeded noise repeats, half exposure, double read noise, zero/double lens-light amplitude, and affine/1st–99th-percentile normalization. Keep all other state fixed one factor at a time. Evaluate the v1.3 truth-free point-image control on physical observation changes. Do not claim an alternate PSF-family comparison: the validated renderer currently supplies Gaussian only. Repeats remain parent-clustered; the historical test remains unopened. Source-position detectability is reserved for its ordered Week 5 study.
## 2026-09-26 — D-013: Close Week 4 observation-systematics milestone

The v1.4 matched release passed with 1,661 descendants (1 train parent for loader validation plus 150 validation parents in each of 11 conditions), zero rejected renders, exact canonical identity, invariant `theta_E_arcsec`, no ancestry crossings, and strict loading. The same 150 H1 validation parents were used with all three frozen CNN seeds; the historical test was not read. Half exposure raised CNN MAE by +0.0463 arcsec (95% parent interval [+0.0297,+0.0649]) and doubled read noise by +0.0350 ([+0.0254,+0.0451]); both satisfy the frozen practical-effect rule. Gaussian PSF width ×1.25 and affine normalization were practically small for the tested conditions; width ×0.75, noise repeats, lens light, and percentile normalization were inconclusive. The repeat-realization prediction shifts averaged 0.0468 and 0.0473 arcsec. No alternate PSF family was tested because the validated renderer provides Gaussian only.

The truth-free point-image control had 44/150 canonical fallbacks and was weaker than the CNN on all nine observation-changing conditions. Its half-exposure penalty was +0.0346 arcsec ([+0.0053,+0.0639]); this remains a limited SIS/SIE point-image comparison. Detailed results and limitations are in `week4_observation_systematics_status.md`; the phase registry is updated. Proceed in order to Week 5 source morphology/position, multiplicity, and shear studies. Source-position detectability is carried here from H4; a mass-family shift remains conditional on renderer/convention validation. Keep the old test quarantined and the Week 11 holdout ungenerated until final freeze.
## 2026-09-26 — D-014: Freeze Week 5 source/position/shear conditions

Freeze the Week 5 design in `protocol_v1.5.md` before CNN predictions: Sersic index 0.5/5.0; effective radius x0.5/x2; source-center coordinate shifts ±0.15 arcsec on x and y; and separate `gamma1=0.05` / `gamma2=0.05` shear components. Use the same 150 validation parents, fixed SIE `theta_E`/ellipticity, and unchanged observation/noise state. Report physical roots, renderer amplitude-threshold images, and observation-only S/N≥5 peak counts separately. The v1.3 classical point-image control remains frozen.

The smoke render passed for 4 independent development parents and 44 descendants: zero rejected renders, exact canonical identities, invariant `theta_E`, ancestry/split isolation, and strict loader validation. Source-position/shear roots were recomputed. The small smoke included changed physical multiplicity for one source offset and one shear component; this is renderer validation only, not model evidence or model selection. The supported rendering path has one Sersic source family and SIE (optionally with shear); no alternate source or mass family is validated. Do not claim those mismatches or improvise a new profile during Week 5.
## 2026-09-26 — D-015: Close Week 5 source/physics milestone

The v1.5 matched release passed with 1,661 descendants, zero rejections, exact canonical identity, invariant `theta_E_arcsec`, no ancestry crossings, and strict loader checks. The same 150 H1 validation parents were scored by all three frozen CNN seeds; the historical test was not accessed. Within-source-family results support increased error for source effective radius x0.5 (+0.0129 arcsec, 95% interval [+0.0041,+0.0222]) and the x−, x+, and y− 0.15-arcsec source shifts (+0.0185 [+0.0008,+0.0374], +0.0248 [+0.0093,+0.0398], and +0.0155 [+0.0013,+0.0306]). The y+ shift and both gamma=0.05 shear components are inconclusive. Larger source radius and Sersic index 5 reduced error for these tested conditions; effects are not monotonic.

Physical point-root count and renderer amplitude-threshold multiplicity agree in at least 99.3% of parents, while the observation-only S/N>=5 peak count agrees with physical roots in only 46.7–58.0%, depending on condition. The peak detector is one measurement procedure, not a ground-truth completeness label. The classical point-image control remains weaker than CNN on all conditions and has 22.7–42.7% fallbacks. The failure mechanism map now combines 22 paired comparisons from Weeks 3–5; its ranking is descriptive and unadjusted for the number of conditions. Report: `week5_source_physics_status.md`.

The validated renderer has no supported alternative source family or EPL/other mass-profile path, so those optional mismatch arms were not attempted. Proceed in order to Week 6 simple baselines, learning curve, broader classical comparison, and shortcut interventions. Keep the historical test quarantined and the fresh blind holdout deferred until Week 11.

## 2026-09-26 — D-016: Freeze Week 6 baselines, learning curve, and image interventions

Freeze the Week 6 analysis in `protocol_v1.6.md` before any Week 6 validation scoring. Use the original 700 train and 150 validation parents, the same three baseline CNN seeds, and a nested deterministic learning curve at 100, 300, and 700 independent train parents (selection seed 6031). Add a train-only standardized radial/angular Ridge baseline with five-fold train-only alpha selection. Compare on canonical data and the six preselected Week 3–5 stress conditions. Run five paired image interventions on canonical validation images: 90-degree rotation, half/double brightness, a 0.50-arcsec central disk mask, and a 0.70–2.20-arcsec annulus mask. Use 10,000 parent bootstrap resamples with seed 6031 and the established 0.01-arcsec practical margin. No historical test or real-data truth is accessed.

The existing `fit_sie_subset` optimizer reads truth-sidecar parameters for initialization and is not a blind comparison. The Week 6 classical comparator therefore remains the validated v1.3 observation-only SIS/SIE point-image geometry estimate with its training-mean fallback. It is explicitly limited and does not satisfy the timeline's full pixel-likelihood SIE comparison. Do not start Week 7 until the Week 6 results and shortcut audit are recorded.

## 2026-09-26 — D-017: Close Week 6 with data-volume and arc-information evidence

The v1.6 learning curve completed nine nested CNN fits at 100, 300, and 700 independent train parents, each evaluated on the same 150 validation parents. Seed-averaged MAE fell from 0.2039 to 0.1163 to 0.0946 arcsec. Paired parent-bootstrap penalties relative to 700 parents were +0.1093 [+0.0911,+0.1277] at 100 and +0.0217 [+0.0135,+0.0299] at 300. The model continued to improve through the available 700-parent range; prioritize additional independent-parent scale/diversity before a major architecture change.

The train-only 45-feature Ridge baseline (five-fold selected alpha 0.01) improved over the training mean but underperformed the frozen CNN on all seven preselected conditions. The observation-only point-image classical control also remained weaker than the CNN, with 29–43% fallbacks across conditions. A truth-free full pixel-likelihood SIE comparison remains unavailable because the current optimizer reads truth-sidecar initialization values.

On the five image interventions, annulus masking at 0.70–2.20 arcsec increased MAE by +0.2632 arcsec [+0.2246,+0.3022]; brightness scaling and 90-degree rotation were practically small in paired MAE, although rotation shifted individual predictions. The central-disk result was inconclusive. The annulus test establishes dependence on information in that radial image region, not an invalid causal shortcut. The Week 6 results, artifacts, and limitations are in `week6_baselines_shortcuts_status.md`; the machine-readable outcomes are under `results/reports/reliability_v1/baselines_week6/`.

Close Week 6 and proceed in order to Week 7 cross-simulator validation. Select an alternate renderer only after a local compatibility audit, freeze conventions and matched cases before its evaluation predictions, and do not make a simulator-independent claim without supporting results. Keep the old test quarantined and leave mitigation for Week 8.

## 2026-09-26 — D-018: Freeze Week 7 independent SIE lens-equation comparison

The environment contains Lenstronomy 1.14.2 and no second installed full renderer. Use the repository's new independently coded closed-form SIS/SIE deflection and numerical point-image root path; keep Lenstronomy's Sersic surface-brightness evaluator, Gaussian PSF, detector noise, and preprocessing shared and fixed. This is a lens-equation implementation comparison, not an independent full rendering stack.

Freeze all 150 `sim-eng1000-v1` validation parents, two conditions (canonical and half exposure), the existing three CNN checkpoints, the root/deflection quality gates, and the paired bootstrap analysis in `protocol_v1.7.md` and `configs/reliability_cross_simulator_v1.yaml`. Run and pass deflection, image-root, and four-parent renderer controls before CNN scoring. Do not retrain or tune on Week 7 outputs; the historical test remains prohibited. Limit conclusions to the tested independent mass-equation/rasterization path.

## 2026-09-26 — D-019: Close Week 7 with matched implementation agreement

The independent deflection control differed from Lenstronomy by at most 7.63×10⁻⁹ arcsec; independently solved root counts matched 150/150 validation systems, with matched positions below the 0.02-arcsec gate. The four-parent smoke reproduced canonical processed images exactly after applying the native pixel-area factor to the independent surface-brightness raster. The schema-checked matched release contains 150 parents in canonical and half-exposure conditions and passed strict loading.

The frozen CNNs showed mean absolute shifts of 3.02×10⁻⁷ arcsec (canonical) and 2.86×10⁻⁷ arcsec (half exposure) relative to the Lenstronomy predictions. Paired MAE changes were effectively zero. Treat this as agreement for the tested equation/root/rasterization convention only; Lenstronomy Sersic light and the project observation/preprocessing components remain shared. Week 7 artifacts are under `results/reports/reliability_v1/cross_simulator_week7/`, and the phase report is `week7_cross_simulator_status.md`.

## 2026-09-26 — D-020: Freeze Week 8 targeted randomization

The Week 3–5 failure ranking supports targeting pixel scale (0.04 and 0.06 arcsec/pixel), half exposure, and doubled read noise. Freeze a single randomized view for each of the 700 training parents with PCG64 seed 6031. Pair each scale with native shapes 320/256/214 so the angular field of view matches the measured Week 3 intervention; retain each canonical training view. Keep the original 150 validation parents, three CNN seeds, architecture, optimizer, and training limits fixed. Score canonical and the four existing stress conditions once after training, saving predictions before target joins. The primary rule requires at least 0.01 arcsec mean stress MAE improvement with a 95% interval below zero and canonical degradation no larger than 0.01 arcsec at its upper interval bound. No historical test or real-data labels may be accessed.

## 2026-09-26 — D-021: Close Week 8 with targeted mitigation pass

The v1.8 release passed ancestry, schema, and strict-loader checks with 1,400 rows from 700 independent training parents. The randomized model improved equal-weight stress MAE by 0.0290 arcsec (parent-bootstrap 95% interval [-0.0351, -0.0231]); canonical MAE improved by 0.0109 arcsec (upper 95% bound -0.0031). Both preregistered rules passed, so the conditional pixel-scale metadata diagnostic was not run. Predictions were saved before target joins; historical test and real labels were not accessed. Results are in `results/reports/reliability_v1/mitigation/` and `week8_mitigation_status.md`.

## 2026-09-26 — D-022: Freeze Week 9 hybrid initialization comparison

Use the successful Week 8 randomized checkpoints as ML starts and compare them with the original train-split target mean and an image-derived median peak radius. Freeze all 150 validation parents in canonical and the four Week 8 stress conditions, the existing observation-only peak detector, and the same bounded SIS/SIE point-image geometry estimator for every start. Save fit records before joining target values. The truth-sidecar-initialized pixel fitter remains excluded. Report convergence, final theta_E error/residual, evaluations, runtime, boundaries, Jacobian conditioning when available, and fallback rates. Limit conclusions to the tested point-image fit and synthetic validation conditions.

## 2026-09-26 — D-023: Close Week 9 with no supported ML speedup

The v1.9 comparison generated 6,750 fit records for 150 validation parents across five conditions, three verified Week 8 checkpoint seeds, and three starts. Predictions and fit records were saved before target joins. Only 198 of 750 parent-condition pairs produced three/four observation-detected peaks. The randomized CNN start preserved final point-fit MAE within the 0.01-arcsec practical margin, but its paired function-evaluation intervals included zero in all five conditions and runtime showed no consistent gain. The heuristic peak-radius start was comparable. Limit the conclusion to the observation-only point-image SIS/SIE control; it is not a full pixel-likelihood comparison. No historical test or real labels were accessed.

## 2026-09-26 — D-024: Freeze Week 10 CASTLES domain-gap analysis

Compare the final Week 8 training release (1,400 views from 700 parents) with all 137 eligible processed CASTLES images representing 69 lens systems. Freeze observable metadata and image proxies, descriptive lens-clustered reporting, and the 90% robust-range rule for coverage ratings in protocol v1.10. Do not read Einstein-radius/redshift labels, score real-image predictions, infer detector noise from simulator variance, or treat unavailable masks, PSFs, or multiplicity labels as zero. Use the existing three-system astrometric comparison only as exploratory context.

## 2026-09-26 — D-025: Clarify Week 10 processed field of view and proxy ratings

The CASTLES quality audit showed that raw image dimensions can exceed the retained 256x256 product and can therefore overstate the field presented by the image pipeline. Amend the v1.10 feature summary to retain raw/native FOV for crop history and use processed-image FOV for the coverage map. Also apply the frozen rule consistently: numeric image features with normalization/proxy semantic mismatch are rated weakly represented, regardless of the numeric overlap. This is a descriptive reporting correction; no labels, predictions, or historical test data are involved. Recompute the Week 10 tables and figures before closing the phase.

## 2026-09-26 — D-026: Close Week 10 with explicit synthetic coverage gaps

The completed v1.10 analysis loaded all 137 processed CASTLES images from 69 lens systems and all 1,400 Week 8 training views from 700 parents. No rows were excluded; no real parameter labels or predictions were read. Pixel-scale coverage is weak (52.6% of real observations within the synthetic robust range), while processed field of view is unsupported by the frozen 90% coverage rule (6.6% in range). Intensity/concentration comparisons remain weak image proxies because processing semantics differ. Instrument-specific rendering, real PSF/masks/multiplicity, and calibrated detector/noise behavior remain unsupported. See `results/reports/domain_gap_v1/` and `week10_domain_gap_status.md`.

## 2026-09-26 — D-027: Freeze Week 11 exactly-once blind synthetic holdout

Select the Week 8 randomized three-seed ensemble as the final model because it passed the preregistered mitigation gate; keep the original CNN ensemble as a fixed comparator. Generate 150 fresh independent parent systems under seed 11031 with canonical and four fixed stress observations per parent. Freeze checkpoint and normalization hashes, preprocessing, scoring, parent bootstrap, and release checks before generation. Write and hash target-free predictions, join theta_E once, and refuse a second scored run. The historical test remains prohibited; no model changes, retuning, or holdout-based selection are allowed.

## 2026-09-26 — D-028: Finalize Week 11 pre-generation integrity checks

Before creating any holdout products, repair the Week 11 runner's condition-column read and prediction-file ordering, add the frozen batch/thread settings, require the five expected native shapes and pixel scales, and record per-checkpoint, normalization, model, dataset-loader, and preprocessing hashes with predictions. Hash every generated file referenced by the observation manifests and retain one final `splits/holdout.csv`. Add an inference-start marker so an interrupted inference cannot be repeated. The frozen descriptive error counts use the already common 0.05- and 0.10-arcsec reporting thresholds. These are pre-generation implementation clarifications; no new holdout release or predictions have been generated or inspected. The final hashes will be recorded before starting the generator.

## 2026-09-26 — D-029: Close Week 11 and begin the final package

The frozen release passed all generation and preflight gates with 150 new parents and 750 observations. The single inference pass produced 4,500 per-checkpoint and 1,500 ensemble predictions; target-free files were hashed before the one target join, and the completion marker/output hashes were verified. The randomized ensemble's equal-weight four-stress paired MAE change versus the fixed canonical baseline was -0.0361 arcsec (10,000 parent-bootstrap 95% interval [-0.0427, -0.0294]); each condition-specific paired interval, including canonical, was below zero. No test or real labels were accessed and no post-holdout tuning occurred. Limit this conclusion to the same synthetic prior/renderer and fixed conditions. Week 12 is synthesis only, using frozen result artifacts.

## 2026-09-26 — D-030: Close Week 12 with the auditable paper package

The final package is built from recorded artifacts without rerunning model inference. It contains a 51-row harmonized paired-effect ledger, final holdout tables and figures, a 750-observation failure catalog, model/research card, manuscript draft, ten-row claim-evidence map, reproducibility statement, and 53-input/21-output SHA-256 manifest. The manuscript places Limitations before Discussion and distinguishes synthetic in-domain, synthetic stress, the limited equation-path comparison, and descriptive CASTLES coverage. It makes no real-image accuracy claim. The workflow timeline is complete.
