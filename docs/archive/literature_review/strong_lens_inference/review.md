# Reliable automated inference of galaxy-scale strong gravitational lenses

Comprehensive scoping and critical narrative review for `charlie_lensing`  
Search cutoff: 4 November 2026. Prepared as a research decision document, not a manuscript revision.

## Assessment

The scientific application is the recovery of the projected mass distribution of a foreground galaxy from the distorted images of a background galaxy or quasar. A fast Einstein-radius estimator addresses one useful component of that inverse problem. Its scientific value depends on whether it recovers the angular mass scale reliably across the observing conditions, galaxy morphologies and selected lens populations encountered in observations.

The literature supports the concern that the current framing is too narrow. Fast CNN regression, observational augmentation, uncertainty prediction, domain adaptation and neural initialization of physical fitting all have substantial precedent. A new paper needs to explain a remaining scientific or methodological uncertainty and resolve it with discriminating experiments. Adding a more elaborate network is not, by itself, the missing contribution.

My recommended direction is a controlled study of **which sources of simulation mismatch cause Einstein-radius bias, which inexpensive interventions remove that bias, and when the remaining errors make an estimate unsuitable for use**. Angular resampling, fresh-noise augmentation, observation-condition randomization and calibrated uncertainty are candidate interventions. Their relative value must be measured rather than assumed. This is a proposed direction, not a claim of established novelty.

The evidence base contains 60 unique scholarly works, plus one linked corrigendum, two official detector/noise documents and one workflow attribution. These are not 60 independent experiments or 60 complete full-text reviews. The evidence table records access depth and task relevance. Preprint/journal versions are linked in one record. No pooled performance estimate is justified.

## 1. Scope and review method

The question is: what is established about fast, reliable galaxy-scale strong-lens parameter inference, particularly Einstein radius, and what evidence would make an observation-robustness study scientifically useful?

Included literature covers classical image inversion; CNN regression; Bayesian neural networks; neural posterior, likelihood and ratio estimation; simulation design; detector and PSF effects; prior mismatch; source morphology; uncertainty validation; real-lens applications; failure analysis; selection effects; and relevant downstream mass/population science. Foundational studies before 2017 are included. Recent work is searched through the stated cutoff. General machine-learning papers are explicitly marked as methodological context rather than evidence of lensing performance.

Discovery used recorded web searches targeting arXiv, journal/publisher records, proceedings and citation chains. Crossref records and arXiv citation metadata were retrieved directly to verify identities and link journal versions. Parallel was unavailable because its CLI and authentication were not configured. Searches were refined as negative evidence and recent observational applications emerged. The exact executed discovery strings are in `search_log.json`; metadata provenance and access limitations are in `references.json` and `coverage.json`.

This is a single-reviewer scoping/narrative review using ranked discovery, not a registered systematic review. There were no complete ADS, Scopus or Web of Science exports, no independent duplicate screening, and no defensible denominator for all search hits screened. Consequently, there is no invented PRISMA flow. Some records were assessed only at abstract level; selected full-text methods, results and limitations were inspected for the closest comparisons. Citation verification establishes identity and retrieved support, not that every result was independently reproduced. Correction checks are bounded to retrieved publisher/arXiv records; absence of a flag is not a comprehensive retraction audit.

## 2. The scientific problem and what Einstein radius measures

Light from a background source is deflected by foreground mass. In the usual thin-lens description, the lens equation maps image-plane angle to source-plane angle through a mass-dependent deflection. Observed pixels then combine the lensed source, foreground light, optical blurring, detector integration and noise. Recovering mass therefore involves disentangling astrophysics from measurement effects.

For a circular lens, the Einstein radius defines an aperture whose mean projected surface density equals the critical lensing density. The corresponding projected mass is

`M_E = pi * (D_l * theta_E)^2 * Sigma_crit`,

where the angle is in radians and the critical density depends on lens/source distances. An SIE model's radius convention must be stated explicitly; equivalent-radius and profile-normalization conventions should not be silently interchanged. An angular radius is not a total galaxy mass, a dark-matter fraction, or a direct measurement of the Hubble constant. Mass conversion additionally needs geometry and assumptions. The SLACS analysis of 73 real lenses illustrates the richer science obtained by combining lensing with stellar masses and velocity dispersions. [Auger et al. 2010](https://doi.org/10.1088/0004-637X/724/1/511)

The application has three connected motivations. First, mass modelling helps study galaxy structure and evolution. Second, detailed image perturbations can reveal dark substructure; a scalar radius alone does not make that measurement. Third, time-delay cosmography needs lens models alongside measured delays, environmental information and other constraints. Observational dark-satellite inference and synthetic neural cosmography demonstrate these distinct uses. [Vegetti et al. 2012](https://doi.org/10.1038/nature10669), [Park et al. 2021](https://doi.org/10.3847/1538-4357/abdfc4)

There is also a fundamental identifiability limit. Mass-sheet and source-position transformations allow different mass/source descriptions to preserve important imaging observables while changing other inferred quantities. Even a visually excellent reconstruction cannot establish that every physical parameter is correct. Accurate radius recovery is valuable but cannot certify a complete mass profile or a cosmological analysis. [Schneider and Sluse 2013](https://doi.org/10.1051/0004-6361/201321882), [2014](https://doi.org/10.1051/0004-6361/201322106)

For orientation, the broader application and cosmography are synthesized in [Shajib et al. 2024](https://doi.org/10.1007/s11214-024-01105-x) and [Birrer et al. 2024](https://doi.org/10.1007/s11214-024-01079-w). These reviews supply context; specific empirical claims here are tied to original studies.

## 3. Why automation matters, and what the comparator should be

Survey forecasts motivated automated modelling well before the present project. Collett's population-based forecasts also show that discoverable lens counts depend on source populations, subtraction assumptions and detection thresholds. Such numbers are forecasts, not observed counts or guarantees about the eventual analysis sample. [Collett 2015](https://doi.org/10.1088/0004-637X/811/1/20)

Classical modelling is a serious, evolving comparator. Semilinear inversion solves pixelized source brightness linearly for a fixed lens model; Bayesian extensions address regularization and model comparison. AutoLens subsequently automated joint lens light, mass and source modelling. These methods do not require optimizing every source pixel as an unrelated nonlinear parameter. [Warren and Dye 2003](https://doi.org/10.1086/375132), [Suyu et al. 2006](https://doi.org/10.1111/j.1365-2966.2006.10733.x), [Nightingale et al. 2018](https://doi.org/10.1093/mnras/sty1264)

More recent work accelerates physical inference with automatic differentiation and GPUs. GIGA-Lens combines gradient-based optimization, variational approximation and Hamiltonian Monte Carlo. TinyLensGPU tests simulated and real SLACS lenses and reports catastrophic failures as well as speed. Wall-clock comparisons need matched hardware, fit complexity, initialization, posterior quality and accounting for simulation/training costs. Comparing a forward pass only against historical expert-intensive fitting would overstate the present computational gap. [Gu et al. 2022](https://doi.org/10.3847/1538-4357/ac6de4), [Cao et al. 2025](https://doi.org/10.1093/mnras/staf891)

Software papers establish useful infrastructure: [lenstronomy](https://doi.org/10.1016/j.dark.2018.11.002), [lenstronomy II](https://doi.org/10.21105/joss.03283), [PyAutoLens](https://doi.org/10.21105/joss.02825), [deeplenstronomy](https://doi.org/10.21105/joss.02854), and [Caustics](https://doi.org/10.21105/joss.07081). A simulator's availability, test suite or throughput does not establish that a particular configuration reproduces a survey. Independent software is useful only if the actual rendered experiment exercises independent implementations and compatible conventions.

## 4. From point regression to calibrated inference

### 4.1 Fast prediction is established

Hezaveh and colleagues established rapid CNN estimation of SIE parameters in 2017, using observational augmentation and demonstrating analysis of observed systems. This is the central historical precedent: neither fast CNN lens modelling nor adding noise/PSF variation is a new research premise today. [Hezaveh et al. 2017](https://doi.org/10.1038/nature23463)

Pearson and colleagues then examined practical dependencies on foreground lens light, colour, SNR and mass-light scatter. HOLISMOKES studied training distributions and lens configurations before extending to parameter uncertainties. These works make training-data assumptions part of the scientific method, not a minor implementation detail. The original HOLISMOKES IV has a publisher-listed corrigendum; its original numerical results are deliberately not used here because the correction text could not be retrieved. [Pearson et al. 2019](https://doi.org/10.1093/mnras/stz1750), [Schuldt et al. 2021](https://doi.org/10.1051/0004-6361/202039574), [corrigendum](https://doi.org/10.1051/0004-6361/202039574e), [Schuldt et al. 2023](https://doi.org/10.1051/0004-6361/202244325)

### 4.2 Confidence intervals on a score are different from uncertainty for a lens

Perreault Levasseur and colleagues evaluated marginal coverage for neural parameter uncertainties in 2017. LEMON likewise predicts uncertainties, rather than only radius point estimates. Thus uncertainty is not an optional decorative addition to a mature inference study. [Perreault Levasseur et al. 2017](https://doi.org/10.3847/2041-8213/aa9704), [Gentile et al. 2023](https://doi.org/10.1093/mnras/stad1325)

A bootstrap interval around mean test MAE describes uncertainty in an aggregate performance estimate under the resampling scheme. It does not provide an interval for an individual lens. Seed-to-seed variation measures training variability; it is also not automatically a calibrated per-lens posterior. A model can have low average error while making confident mistakes on a scientifically important subset.

For this project, an initial uncertainty baseline could be a mean-variance head or a small ensemble, followed by calibration assessment on held-out parents. Neither deserves the label 'reliable' solely because it returns a variance. A Gaussian head can miss asymmetric or multimodal uncertainty; an ensemble can agree on a shared simulator-induced bias. Coverage must be assessed with interval width and conditional error.

### 4.3 Posterior and likelihood learning change the output, not the need for validation

Simulation-based inference can learn a posterior, likelihood or likelihood ratio. A scalar network trained with a regression loss is a point estimator; it is not already a Bayesian inference procedure. Legin and colleagues show how regression outputs can instead be treated as compressed statistics whose likelihood is estimated, permitting explicit inference priors. This offers one way to reuse a clean research core without pretending a point predictor contains complete uncertainty information. [Legin et al. 2023](https://doi.org/10.3847/1538-4357/aca7c2), [Cranmer et al. 2020](https://doi.org/10.1073/pnas.1912789117)

Poh and colleagues compare NPE and BNNs on DES-like simulations with one, five and twelve inferred parameters. Their evaluation includes coverage, rank statistics and posterior predictive checks. Their appendix also documents a sky-magnitude conversion bug that reduced simulated background RMS. This makes the paper a useful diagnostic precedent while limiting any attempt to transfer its precision directly to another simulator or real data. [Poh et al. 2025](https://doi.org/10.1088/1475-7516/2025/05/053)

NPE is not a universal fix. General SBI benchmarking finds that performance depends on task, diagnostic and simulation budget; posterior approximation methods can be overconfident. Autoregressive ratio methods offer additional flexibility but introduce training, sampling and validation choices. Complexity should follow a demonstrated limitation of simpler inference. [Lueckmann et al. 2021](https://proceedings.mlr.press/v130/lueckmann21a.html), [Hermans et al. 2022](https://arxiv.org/abs/2110.06581), [Anau Montel et al. 2024](https://doi.org/10.1093/mnras/stae1130)

## 5. Simulation realism has several independent dimensions

### 5.1 Detector statistics and preprocessing

A basic detector model combines source/background Poisson counts with read noise. Exposure time changes the relation between signal and noise; arbitrary additive noise in a normalized image need not correspond to a physically plausible observation. One must state units, gain, background treatment, exposure combination and the point at which normalization is applied. Official HST exposure-time documentation is a suitable authority for instrument-level noise accounting. [STScI ETC SNR documentation](https://etc.stsci.edu/etcstatic/users_guide/1_2_1_snr.html)

Processed images require additional care. Drizzling/resampling can correlate neighbouring pixels, so a per-pixel variance map alone need not characterize aperture or residual statistics. An independent Gaussian perturbation is not an empirical model of all HST products. A simulator should be matched to a named image product, rather than described generically as 'HST-like'. [STScI DrizzlePac correlated-noise documentation](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise)

Erickson and colleagues provide a close methodological example: empirical HST PSF maps and dither/drizzle treatment enter their simulations. Shajib and colleagues' independent analysis of a real quasar lens shows that PSF treatment can explain important differences between modelling programs. Changing Gaussian to Moffat is a useful controlled stress, but cannot substitute for validating empirical instrument PSFs and image processing. [Erickson et al. 2025](https://doi.org/10.3847/1538-3881/add99f), [Shajib et al. 2022](https://doi.org/10.1051/0004-6361/202243401)

### 5.2 Astrophysical scene realism

Detector realism does not resolve missing source clumps, mass-profile flexibility, foreground-light residuals, contaminants or correlations between mass and light. A clean Sérsic source behind a smooth isothermal lens may be useful for unit-controlled recovery while being an inadequate stress test of a model intended for observations.

Gawade and colleagues compare increasing simulation realism and benchmark against models of 182 real HSC lenses. Their results favour a particular background-injection/subtraction strategy for that setting, with remaining outliers and ellipticity systematics. This is evidence that scene construction and nuisance handling matter; it is not proof that one universal realism ranking exists. [Gawade et al. 2025](https://doi.org/10.1093/mnras/staf935)

Flexible physical models address different assumptions. Chianese and colleagues couple learned source structure to differentiable lensing; Galan and colleagues use wavelets for mass perturbations; Biggio and colleagues represent lens potentials with neural fields, with known source brightness in their demonstration. Diffusion-based source work supplies learned priors, while a later SLACS preprint models source, lens light and PSF jointly. These methods expand the model family; they do not automatically establish real-data calibration. [Chianese et al. 2020](https://doi.org/10.1093/mnras/staa1477), [Galan et al. 2022](https://doi.org/10.1051/0004-6361/202244464), [Biggio et al. 2023](https://doi.org/10.1051/0004-6361/202245126), [Karchev et al. 2022](https://arxiv.org/abs/2211.04365), [Legin et al. 2025, preprint](https://arxiv.org/abs/2511.19595)

### 5.3 Population realism and selection

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

## 6. Randomization and domain adaptation: relevant, but different evidence

Domain randomization varies simulated training conditions without necessarily observing a target domain. Domain adaptation can exploit unlabeled target images during training. A method with target-domain access has an information advantage over a source-only method and should not be compared as if the settings were identical.

Swierc and colleagues test DANN and MMD for Einstein-radius regression from a cleaner synthetic source to a DES-like noisy synthetic target. Agarwal and colleagues combine mean-variance estimation with adaptation and assess uncertainty. Both are particularly close precedents, but neither establishes transfer to arbitrary real observations. Their reported metrics also differ from this project's MAE. [Swierc et al. 2023, workshop](https://arxiv.org/abs/2311.17238), [Agarwal et al. 2024, workshop](https://arxiv.org/abs/2411.03334)

Generic work on domain randomization, active sampling and reduced augmentation search provides methodological context. Robotics and image-classification gains cannot establish effectiveness or novelty in astronomical inference. Physically meaningful image transformations and label conventions must be preserved; a geometric resize can change the pixel-to-angle relation even when it leaves an image visually recognizable. [Tobin et al. 2017](https://arxiv.org/abs/1703.06907), [Mehta et al. 2020](https://proceedings.mlr.press/v100/mehta20a.html), [Cubuk et al. 2020](https://arxiv.org/abs/1909.13719)

For the present project, the relevant question is not simply whether variation helps. It is whether the gain survives controls for fresh noise, image scale, source exposure count and policy-selection effort, and whether it extends beyond the nuisance support optimized during development. Failure-weighted sampling also needs a sufficiently precise comparison with uniform sampling before its additional machinery is justified.

## 7. Failure evidence and uncertainty under mismatch

Filipp and colleagues directly challenge a reassuring matched-distribution evaluation. Controlled changes in source morphology, lens parameters, halo descriptions and noise can bias neural ratio and posterior population inference. This concerns subhalo population inference, so its numerical bias is not a forecast of this project's radius errors. It nevertheless establishes that calibration under an assumed simulator and robustness to modest simulator mismatch are separate questions. [Filipp et al. 2025](https://doi.org/10.3847/1538-4357/adee20)

A blind comparison of six physical modelling methods finds systematic differences even with independently simulated, relatively controlled HST data. No method is uniformly best across properties. Source and PSF choices contribute to discrepancies. The experiment's idealizations restrict generalization of its favourable model-combination result. [Galan et al. 2024](https://doi.org/10.1051/0004-6361/202451095)

Failure analysis also appears in observational automation. Etherington and colleagues examine 59 HST lenses and identify foreground-light subtraction and initial model construction as difficult steps; complete recovery includes pipeline adjustments. A paper should distinguish first-pass success from the final result after rescue. [Etherington et al. 2022](https://doi.org/10.1093/mnras/stac2639)

General uncertainty research finds that calibration can deteriorate under distribution shift. Simulation-based calibration checks computational consistency under an assumed generative model, rather than proving that this model describes the telescope. Modrak and colleagues further show why the choice of SBC test quantities affects sensitivity to failure. Coverage/rank diagnostics should therefore be combined with informativeness, conditional performance and predictive checks. [Ovadia et al. 2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [Talts et al. 2018](https://arxiv.org/abs/1804.06788), [Modrak et al. 2025](https://doi.org/10.1214/23-BA1404)

For scalar radius estimation, an informative failure analysis would ask whether large errors concentrate at low radius relative to PSF width, low arc SNR, strong foreground contamination, crop boundaries, poor normalization, or source/mass families excluded from training. Inspect both worst failures and matched successful examples. A saliency map alone does not identify a causal mechanism; remove or perturb the suspected cue while controlling the rest of the scene.

## 8. What real observations now show

The observational evidence is stronger than a narrative in which all lens inference remains synthetic. LEMON has a pilot SLACS application; Gawade and colleagues analyse real HSC systems; Erickson and colleagues apply posterior inference to 14 real STRIDES lenses. These are different populations, instruments and reference standards. Their performance is not a shared benchmark leaderboard.

The published Euclid Q1 LEMON study is especially relevant. Its 354 selected Q1 comparisons give encouraging Einstein-radius agreement but poor mass-ellipticity agreement. Its appendix discusses uncertainty underestimation on real data after calibration on simulations. Agreement with successful conventional fits is useful cross-validation, while selection and shared model assumptions limit what it proves. [Euclid Collaboration / Busillo et al. 2026](https://doi.org/10.1051/0004-6361/202554538)

Adjacent lens-detection evidence reinforces the distinction between simulated and observed validation. Euclid mock-image studies address faint-arc difficulty, and a later Q1 paper measures a substantial detection-domain gap and benefits from real training examples. These are classification studies: their purity/completeness numbers are not regression accuracy or posterior coverage. [Euclid Collaboration 2024](https://doi.org/10.1051/0004-6361/202347244), [2026](https://doi.org/10.1051/0004-6361/202558456)

A credible observational radius check should document usable bands, pixel scale, PSF, weight maps, masking, foreground subtraction, lens confirmation and reference model conventions. Reference-fit disagreement must be reported, not converted into a fictitious ground-truth label. Candidate samples and already well-modelled lenses answer different applicability questions.

## 9. Population science makes small biases more consequential

Neural inference is being developed for dark-matter populations, not just individual scalar parameters. Wagner-Carena and colleagues combine simulated lens posteriors hierarchically and investigate sequential simulation efficiency. These works are adjacent scientific motivation, not direct evidence that radius augmentation detects substructure. [Wagner-Carena et al. 2023](https://doi.org/10.3847/1538-4357/aca525), [2024](https://doi.org/10.3847/1538-4357/ad6e70)

A 2026 LSST sensitivity analysis explicitly conditions its dark-matter forecast on knowledge of the data-generating process. That is an appropriate boundary for useful synthetic science. Two recent preprints develop contamination-aware cosmology and population-calibrated neural ratio inference with spectroscopic information. They connect automated modelling to scientific endpoints while adding assumptions and observables absent from image-only radius prediction. [Filipp et al. 2026](https://doi.org/10.3847/1538-4357/ae8fd0), [Holloway et al. 2026, preprint](https://arxiv.org/abs/2609.21699), [Jarugula et al. 2026, preprint](https://arxiv.org/abs/2608.23534)

The methodological implication is that small shared biases do not disappear when more lenses are analysed. Increasing the number of examples can reduce statistical scatter while leaving a systematic error dominant. A study aimed at reusable survey inference should quantify bias and failure frequency, alongside average precision.

## 10. What this means for the present manuscript

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

## 11. A defensible next experimental programme

### First: define the intended use and tolerance

Choose a named image product and a concrete use: fast angular-radius screening, initialization of detailed fitting, or calibrated radius inference. These uses have different acceptable error and failure costs. Choose a scientifically justified tolerance before ranking models. For a mass proxy, state the geometry and radius convention; for initialization, measure downstream fitting success and time.

### Second: resolve the present mechanism with a small controlled design

Use the same independent parents, paired seeds and optimization budgets. Compare fixed-condition/fixed-noise, fixed-condition/fresh-noise, uniformly varied/fresh-noise, and failure-weighted/fresh-noise training. Cross the most important comparisons with angular-standardized preprocessing. If the complete cross is expensive, run a documented pilot and retain the comparisons that discriminate the competing explanations. Include policy-development computation in the budget report.

The estimand is the paired difference in held-out-parent loss, conditional on a declared test population. Repeated views of a parent are repeated measurements, not independent lenses. Report sampling uncertainty over parents separately from variation across training seeds; a two-way or hierarchical uncertainty analysis is an option if both populations are part of the intended generalization claim. Do not multiply the independent sample size by views or seeds.

### Third: build physically interpretable stress axes

Select a small set before final evaluation: resolution relative to radius, arc SNR, foreground contrast, PSF family/error, source morphology, mass-family/shear mismatch and correlated/background noise. Start with one-factor tests to localize failure, then add selected physically plausible interactions. Varying all factors arbitrarily at once may expose failure but cannot explain it. Keep development diagnostics and the final holdout separate.

For detector validation, compare simulated blank sky, variance-versus-flux, spatial noise correlation and PSF encircled-energy profiles with the named observational product. Verify units and flux conservation through pixel integration and resampling. These checks test the simulator; predictive accuracy alone does not.

### Fourth: evaluate accuracy, uncertainty and operational failure

Keep one prespecified primary endpoint and use secondary diagnostics to explain it. A useful set is angular MAE, signed bias, median and upper-tail absolute error, fractional error with explicit handling of small radii, and catastrophic-error fraction at a justified tolerance. Report these by the prespecified stress axes. A larger metric list without an interpretation is not greater scientific depth.

For uncertainty models, evaluate 68%, 90% and 95% coverage with interval widths, appropriate proper scoring rules, and conditional calibration. For flexible posteriors, add rank/SBC diagnostics and posterior predictive residual checks. Fit any calibration mapping on a separate development set. Report shift performance after freezing that mapping.

Choose enough independent test parents to resolve the intended differences. For illustration, with 150 independent Bernoulli coverage outcomes at true coverage 0.95, the binomial standard error is about 1.8 percentage points before accounting for subgrouping. Dividing those parents into small strata makes conditional coverage much less precise. More rendered views do not repair a shortage of independent systems.

### Fifth: add an external validation level

Use a truly independent rendering path or a clearly different source/mass family first, with convention and image-difference checks. Then evaluate a curated real sample with documented preprocessing and conventional-model comparisons. Restrict claims if the required raw products are unavailable. Do not optimize on the same real lenses later presented as final validation.

A hybrid comparator is particularly attractive: use neural estimates to initialize physical fitting, then measure convergence, residual structure, catastrophic fits, manual interventions and runtime. Pearson's combined modelling and Euclid's initialization experiments provide direct precedent; the new contribution would need to lie in a better reliability characterization or a demonstrated improvement under a defined mismatch. [Pearson et al. 2021](https://doi.org/10.1093/mnras/stab1547)

## 12. Candidate research questions and contribution boundary

The following are proposed questions, not confirmed literature gaps:

- How much of the observed improvement comes from fresh noise, angular standardization and variation of observation conditions, respectively?
- Which instrument and scene mismatches produce systematic radius bias or catastrophic errors, and can simple interventions reduce them without impairing nominal-condition accuracy?
- Can uncertainty estimates or physical-fit diagnostics identify unreliable cases at a useful retained-sample size?
- Do the conclusions survive independent rendering and observational comparison under a specified survey pipeline?

The first two questions offer a focused controlled study. The latter two extend it towards practical inference reliability. A full cosmology or dark-matter-population paper would require a different scientific target and substantially more modelling; it should not be implied by an Einstein-radius-only experiment.

The most supportable current positioning is a **controlled reliability study of a compact Einstein-radius estimator**, with explicit mechanism tests and bounded validation levels. If source realism, observational calibration and external validation are unavailable, a narrower synthetic result remains legitimate, but its contribution must be argued against the existing controlled-simulation literature. The present evidence does not justify a claim that randomization itself is novel, that failure weighting is superior, or that the system is ready for survey science.

## 13. Reading order and evidence limits

For the immediate redesign, start with Pearson 2019, Pearson 2021, Gawade 2025, Poh 2025, Euclid Q1 LEMON 2026, Filipp 2025 and Etherington 2022. Together they cover nuisance assumptions, hybrid baselines, real-image comparisons, posterior diagnostics, negative shift evidence and operational failures. Then read the foundational Hezaveh/Perreault Levasseur papers and Wagner-Carena's prior-correction framework. The full bibliography and 60-row evidence matrix accompany this document.

The literature does not establish that a particular refactor, architecture or adaptive policy will improve this repository. Nor does the search prove that the proposed mechanism study is unprecedented. A novelty check should be repeated once the exact scientific question, stress family and comparator are fixed. The strongest immediate conclusion is that the application remains worthwhile, but average synthetic MAE after augmentation is too limited an endpoint to carry broad claims about reliable astronomical inference.

## Workflow attribution

The local literature-review and research-lookup skills guided search logging, bibliographic checks and evidence separation. Their library description was verified at the current arXiv record and is cited separately from scientific evidence: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). This attribution is not included in the 60 topical scholarly works.
