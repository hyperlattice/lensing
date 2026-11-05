# Gravitational Lensing Inference: Field Analysis, Current Bottlenecks, and a Contribution Strategy

**Prepared for:** the current strong-lensing inference project  
**Date context:** September 2026  
**Primary focus:** galaxy-scale strong gravitational lensing, image-based parameter inference, simulation-to-observation reliability, and reproducible benchmarking

---

## Executive conclusion

The most scientifically useful direction for this project is **not** to compete with large collaborations by building a larger neural network, attempting full dark-matter mapping immediately, or predicting many lens parameters at once.

A stronger niche is:

> **Build a rigorous, reproducible framework for determining when image-based strong-lens inference is trustworthy, what causes it to fail, and which interventions make it generalize across observational conditions and simulation assumptions.**

This is a real field-level need. The strong-lensing community is rapidly moving from tens or hundreds of carefully modeled systems toward survey-scale samples. Euclid has already demonstrated automated discovery on roughly one million Q1 objects, while Rubin work in 2026 explicitly includes a static strong-lensing data challenge aimed at automated discovery and modelling at LSST scale. Roman is expected to provide another very large high-resolution lens sample. The bottleneck is therefore shifting from “can we find or fit a lens at all?” toward “can we make automated inferences that remain scientifically valid under realistic nuisance variation, model mismatch, selection effects, and domain shift?”

Your current repository is unusually well positioned for this because it already contains:

- a versioned real/simulated data contract;
- ancestry-aware splits;
- deterministic simulation checks;
- a controlled simulator ladder;
- a classical SIE recovery baseline;
- a frozen image-only CNN baseline for Einstein radius;
- seed studies and controlled loss ablations;
- real CASTLES preprocessing and limited real-image checks;
- explicit documentation of what the current results do **not** establish.

The highest-value next step is to turn these engineering safeguards into the **scientific subject of the paper**.

---

# 1. What gravitational lensing contributes to astronomy

Gravitational lensing is unusual because it is simultaneously:

1. a probe of **mass**, including mass that emits little or no light;
2. a natural telescope that magnifies distant sources;
3. a geometrical probe of cosmological distances;
4. a way to test small-scale structure and dark-matter models;
5. a source of time-delay information for cosmography;
6. an inverse problem in which imaging data constrain a latent mass distribution and a latent background source.

That last point is especially important for your work. A strong-lensing image is not a direct picture of mass. It is the output of a forward process containing at least:

\[
\text{mass model}
+ \text{source light}
+ \text{lens light}
+ \text{line-of-sight structure}
+ \text{PSF}
+ \text{pixelization}
+ \text{noise/calibration}
\longrightarrow \text{observed image}.
\]

Inference tries to run this relationship backward. That makes gravitational lensing scientifically powerful, but also makes it easy to obtain a numerically precise answer that is wrong because an assumption in the forward model was wrong.

This is why the important question is increasingly not only:

> “How accurately can a method recover a lens parameter?”

but:

> “Under what assumptions, data conditions, nuisance distributions, and model families does that recovery remain valid?”

---

# 2. The field can be divided into four related problems

## 2.1 Discovery

Given millions or billions of astronomical sources, identify the tiny fraction that are strong gravitational lenses.

This is fundamentally a rare-event classification/ranking problem. Current challenges include false positives, unusual morphologies, small Einstein radii, survey-specific artifacts, and a mismatch between simulated training lenses and real galaxies.

This is already a mature and very active ML area. Euclid Q1 work has deployed multiple networks on roughly one million objects, and the Perseus analysis showed a striking gap between excellent simulated validation performance and low real-data purity. Transformer work on KiDS has similarly required real-data fine-tuning to suppress false positives.

### Implication for your project

**Do not make generic strong-lens classification the main contribution of the current paper.**

It is crowded, requires strong negative datasets, and would pull the project away from the more distinctive inference/reliability infrastructure you already built.

---

## 2.2 Characterization / parameter inference

Once a lens is identified, infer quantities such as:

- Einstein radius;
- lens center;
- ellipticity and orientation;
- external shear;
- density-profile slope;
- source position and morphology;
- time-delay/Fermat-potential quantities;
- substructure properties;
- posterior uncertainty.

This is where your current project sits.

The important distinction is between:

- **point estimation**: output one best estimate;
- **classical Bayesian/likelihood modelling**: fit a forward model and sample/approximate a posterior;
- **simulation-based inference**: learn likelihood/posterior relationships from simulations;
- **hybrid inference**: use ML to initialize, accelerate, constrain, or check a physical model.

Einstein radius is a sensible first target because recent work has found it to be among the most robust lens parameters recoverable from imaging. However, that also means that merely showing a CNN can predict it is not enough novelty by itself. The scientific contribution has to come from **how the estimate behaves under realistic shifts**.

---

## 2.3 Population-level inference

Individual lens parameters are often only intermediate quantities. Scientists ultimately want population statements such as:

- how galaxy density profiles evolve;
- how dark matter is distributed on sub-galactic scales;
- what the subhalo mass function looks like;
- whether dark matter is cold, warm, fuzzy, or self-interacting;
- how the Hubble constant or dark-energy parameters are constrained;
- how distant source populations are distributed.

Population inference introduces a second layer of difficulty: **selection effects**.

A lens sample is not a random sample of galaxies. The probability of being strongly lensed, detected, visually confirmed, successfully modeled, and selected for follow-up can depend on Einstein radius, source brightness, source morphology, ellipticity, redshift, seeing, and the search algorithm itself.

This means a model can be accurate on its selected sample and still yield a biased astrophysical conclusion about the underlying population.

---

## 2.4 Scientific interpretation

The final stage converts lens-model outputs into astrophysical conclusions:

- enclosed mass;
- density slopes;
- dark-matter structure;
- Hubble constant;
- cosmological distances;
- source morphology at high redshift.

This stage is where systematic error is most dangerous. A small bias in a low-level lens parameter can propagate into a highly visible astrophysical claim.

Your current project should **not jump directly to this stage**. It should establish which low-level image-to-parameter inferences can be trusted first.

---

# 3. Major scientific problems in strong-lensing inference right now

## 3.1 Simulation-to-real domain shift

This is one of the clearest current bottlenecks.

ML needs large labeled datasets. Real strong lenses are rare and do not come with exact “ground truth” mass parameters. Simulations solve the label problem, but create a new problem: the model can learn properties of the simulator rather than properties of gravitational lensing.

Simulation mismatch can enter through:

- analytic versus irregular source morphology;
- simple Sérsic lens light versus dust lanes, disks, boxy/disky isophotes, companions, or structure;
- Gaussian versus empirical/spatially varying PSF;
- idealized versus correlated detector noise;
- fixed versus variable pixel scale;
- simplified sky/background structure;
- restricted mass families;
- incorrect parameter priors;
- missing line-of-sight perturbers;
- unrealistic coupling between lens mass and lens light;
- different image-processing pipelines.

Euclid and KiDS results show that models that look excellent on simulations can produce many false positives on real survey images. Domain-adaptive neural posterior estimation research likewise exists specifically because a network trained in one simulated domain does not automatically remain calibrated in another.

### Research opportunity

Instead of treating domain shift as a final “limitation” paragraph, make it the variable being measured.

---

## 3.2 Source-model mismatch

The observed arcs encode both the lensing transformation **and the unknown source structure**.

If the source model is too simple, a fitting method can explain source complexity by changing the lens model. In dark-matter analyses, that is especially dangerous because source complexity can masquerade as small-scale potential perturbations.

Recent work on Abell S1063 showed that inadequate source modelling could produce an apparently strong subhalo detection that disappeared under a more flexible source reconstruction. This is an extreme example of a general issue: model misspecification can look like physics.

### Research opportunity

For a controlled Einstein-radius problem, measure how much the inferred \(\theta_E\) changes when the **same lens mass configuration** is rendered with:

- one Sérsic source;
- multiple Sérsic clumps;
- real-galaxy source cutouts;
- procedurally irregular sources;
- different source sizes;
- different source offsets;
- compact versus extended emission.

This is much more informative than simply adding more random training images.

---

## 3.3 Lens-light mismatch

Foreground lens light overlaps the lensed source. A neural network can use that light as a shortcut if the simulator unintentionally correlates lens-light properties with mass parameters.

Recent HSC work found that lens-light assumptions can strongly affect inferred ellipticity and that the Einstein radius is considerably more robust. This is directly relevant to your project.

### Research opportunity

Build matched tests with:

- no lens light;
- perfect lens-light subtraction;
- Sérsic lens light;
- misaligned mass and light;
- multi-component lens light;
- residual subtraction artifacts;
- real foreground-galaxy cutouts with injected synthetic arcs.

Then ask whether the CNN is using lensing geometry or foreground-light correlations.

---

## 3.4 PSF mismatch

The PSF controls how information is blurred across pixels. It affects arc width, image peak separation, source reconstruction, and small-scale perturbation detection.

Blind multi-code lens-model comparisons have already identified PSF accuracy as one contributor to method-dependent systematic bias. Time-delay cosmography reviews likewise identify accurate PSF reconstruction as important for precision mass models.

Your present synthetic set uses analytic Gaussian PSFs. That is an excellent control model, but it is not enough for a real-data claim.

### Research opportunity

Construct a **PSF robustness matrix**:

- train Gaussian / test Gaussian matched;
- train Gaussian / test Gaussian width shift;
- train Gaussian / test Moffat;
- train Gaussian / test empirical HST PSF;
- train variable / test held-out variable;
- supplied PSF channel versus image-only inference.

This can become one of the central scientific figures in the paper.

---

## 3.5 Pixel scale and angular calibration

This is a particularly important issue for your current repository.

Your engineering simulation release uses a fixed angular pixel scale, while the current CNN receives only the image. Therefore the model can learn a stable mapping between apparent feature size in pixels and Einstein radius in arcseconds.

That is legitimate **within the current synthetic proposal**, but becomes dangerous when transferring to an instrument with a different pixel scale.

### This should become an explicit experiment

Compare at least three strategies:

1. **Angular resampling**: resample every image onto a standard arcsec/pixel grid.
2. **Pixel-space target**: predict Einstein radius in pixels, then convert using metadata.
3. **Metadata-conditioned model**: provide pixel scale as an explicitly allowed non-truth auxiliary input.

Then deliberately test on pixel scales absent from training.

This is exactly the type of small-looking implementation issue that can determine whether a method is scientifically meaningful.

---

## 3.6 Mass-model mismatch and lensing degeneracies

Strong-lensing observables do not uniquely determine a mass map. The mass-sheet transformation and related source-position/shape degeneracies allow different mass distributions to reproduce similar observables.

Therefore a neural network should not be expected to “discover the true mass distribution” from one image unless the training assumptions strongly restrict the allowed solution family.

This is why your earlier framing of “inferring mass distribution” was too broad. The current target—Einstein radius under a specified model convention—is scientifically much cleaner.

### Research opportunity

After Einstein-radius robustness is established, test controlled model-family shift:

- train SIE, test elliptical power law with slope variation;
- train no shear, test shear;
- train single-plane, test weak line-of-sight perturbations;
- train centered mass/light, test offsets;
- train smooth macro-model, test controlled multipoles.

Do not ask only whether MAE rises. Ask **which mismatch produces bias, in which direction, and for which subsets of lenses**.

---

## 3.7 Selection functions

Strong-lens samples are selected twice:

1. by the physics of lensing itself;
2. by the discovery/modeling pipeline.

Neural lens finders have documented preferences for systems with larger Einstein radii and particular source properties. This matters because population studies can inherit the ML selection bias.

### Research opportunity

Even though your current model is an inference model rather than a detector, it can still have an **inference selection function**:

- Which images fail preprocessing?
- Which systems yield large errors?
- Which systems trigger out-of-prior predictions?
- Which systems would pass a confidence threshold?
- Are doubles harder than quads?
- Are small-radius systems systematically overestimated?

Turn these into a failure/completeness map across the physical parameter space.

This is more scientifically useful than one headline MAE.

---

## 3.8 Uncertainty calibration

A point estimate without a trustworthy uncertainty is limited for astronomy.

However, simply adding a predicted standard deviation is not enough. Uncertainty must be tested for **coverage**:

> If the method says “68% interval,” does the true parameter actually fall inside that interval about 68% of the time under the stated data-generating distribution?

And then:

> Does coverage survive a controlled domain shift?

### Research opportunity

Only after deterministic robustness is understood, add one uncertainty method:

- deep ensemble;
- heteroscedastic Gaussian output;
- quantile regression;
- neural posterior estimation.

Then evaluate:

- empirical coverage;
- interval width;
- conditional coverage by Einstein radius/SNR/multiplicity;
- coverage under PSF/noise/source shifts;
- correlation between uncertainty and absolute error.

A model that becomes uncertain when it is wrong is much more useful than a model that is merely accurate on average.

---

## 3.9 Computational and human scalability

Traditional lens modelling can require many CPU hours per system and substantial human setup/review. Automated pipelines such as PyAutoLens have dramatically improved scalability, and neural methods can reduce initial inference to fractions of a second. This means the open problem is no longer simply “make inference fast.”

The harder question is:

> **How do we preserve accuracy and know when automation should defer to expensive physical modelling or human review?**

### Research opportunity

Build a triage system:

- easy systems → CNN estimate accepted;
- moderate systems → CNN initializes physical optimizer;
- high-risk/OOD systems → full modelling or human review.

The scientific variable becomes **resource allocation versus reliability**, not just raw network speed.

---

## 3.10 Cross-method and cross-code systematics

A major blind comparison using six lens-modelling methods found systematic biases in all of them, with no single method uniformly superior. Combining methods reduced the largest biases.

The community has consequently developed standards such as COOLEST to make lens models comparable across software packages.

### Research opportunity

Your repository already has unusually strict manifests and provenance. Extend it toward **code-independent comparison**:

- generate the same physical scene through two simulators where possible;
- translate shared parameters into a common convention;
- quantify image-level differences;
- quantify downstream inference differences;
- store cases using a COOLEST-compatible representation if practical.

A small, carefully validated cross-simulator benchmark could be more scientifically interesting than adding another CNN architecture.

---

# 4. What large collaborations need but often cannot prioritize deeply

It would be inaccurate to say professional lensing researchers “do not have time” for these problems. They are actively working on them. A better framing is:

> **Which high-value tasks are labor-intensive, systematic, reusable, and easy for individual papers to under-prioritize because they are less novel than a new method or astrophysical result?**

Those are excellent contribution opportunities for an independent project.

## 4.1 Failure-case curation

Papers tend to highlight aggregate performance. The community still benefits from carefully organized collections of:

- false positives;
- catastrophic inference failures;
- small-radius lenses;
- doubles versus quads;
- low-SNR arcs;
- poor lens-light subtraction;
- PSF mismatch;
- clipped/cropped systems;
- model-family mismatch;
- OOD examples.

A failure atlas with machine-readable metadata has long-term value.

---

## 4.2 Reproducible nuisance sweeps

It is easy to say “PSF mismatch matters.” It is harder to execute and publish a controlled matrix of:

\[
\text{same lens truth} \times
\text{PSF} \times
\text{noise} \times
\text{source morphology} \times
\text{pixel scale}
\]

while keeping every random seed, ancestry relationship, preprocessing step, checkpoint, and result traceable.

Your repository is already designed for exactly this kind of experiment.

---

## 4.3 Benchmark maintenance rather than one-off model results

Useful scientific infrastructure includes:

- frozen benchmark datasets;
- documented simulator versions;
- baseline implementations;
- standardized metrics;
- provenance hashes;
- test-set access rules;
- model cards;
- reproducibility scripts;
- failure labels.

These outputs are less flashy than a new architecture but can be reused by many later studies.

---

## 4.4 Cross-code convention checking

Strong-lensing software can differ in:

- ellipticity conventions;
- orientation conventions;
- Einstein-radius definitions;
- coordinate orientation;
- pixel-center conventions;
- shear definitions;
- image normalization;
- PSF handling.

These are exactly the kinds of details that generate silent scientific mistakes.

A carefully documented conversion/validation suite between two popular codes would be a meaningful open-science contribution.

---

## 4.5 Negative controls

Astronomy ML needs more experiments designed to prove what the model is **not** learning.

Examples:

- randomize lens light while holding mass fixed;
- randomize source morphology while holding \(\theta_E\) fixed;
- remove arcs but retain foreground light;
- scramble metadata;
- vary pixel scale without changing angular lens geometry;
- invert correlations inserted by the simulator;
- train on one nuisance correlation and test after breaking it.

These tests reveal shortcuts that headline metrics can hide.

---

## 4.6 Small, well-described real-data benchmark sets

A huge real dataset is not necessary for every useful study. A smaller sample with careful metadata and external reference values can be valuable for:

- sanity checking;
- ranking models;
- debugging preprocessing;
- testing calibration drift;
- testing angular-scale conventions;
- checking whether synthetic trends persist qualitatively.

The key is not to pretend the reference values are exact ground truth.

---

# 5. Your strongest potential contribution: an “inference reliability atlas”

## Proposed research question

> **How robust is image-based Einstein-radius inference to controlled mismatches in source morphology, lens light, PSF, noise, pixel scale, and lens-model assumptions, and which training or hybrid-inference strategies best preserve accuracy and calibration?**

This question is narrower than “infer dark matter,” but it is much more scientifically defensible.

## Why this is a contribution

It connects four active problems:

1. scalable strong-lens inference;
2. simulation-to-real transfer;
3. model/systematic uncertainty;
4. reproducibility and benchmark design.

It also generates a reusable artifact: a matrix of inference behavior across controlled shifts.

---

# 6. The central experimental design

## 6.1 Build matched lens families

For each underlying physical lens parent, keep the lens mass truth fixed and generate descendants that differ by exactly one nuisance factor.

Example parent:

```text
parent_0042
  theta_E = 1.12 arcsec
  q = 0.73
  phi = ...
  source position = ...
```

Descendants:

```text
parent_0042_psf_base
parent_0042_psf_wide
parent_0042_psf_moffat
parent_0042_noise_low
parent_0042_noise_high
parent_0042_source_sersic
parent_0042_source_clumpy
parent_0042_pixel_004
parent_0042_pixel_008
...
```

This allows paired effects:

\[
\Delta e_i = |\hat{\theta}_{E,i}^{\text{shift}}-\theta_{E,i}|
            -|\hat{\theta}_{E,i}^{\text{base}}-\theta_{E,i}|.
\]

That is much stronger than comparing two unrelated random datasets.

---

## 6.2 Build a robustness matrix

Rows could be training regimes:

- baseline simulation;
- widened/randomized nuisance training;
- realistic-source training;
- empirical-background training;
- PSF-conditioned model;
- pixel-scale-conditioned model;
- hybrid ML + physical refinement.

Columns could be evaluation domains:

- matched synthetic;
- PSF shift;
- noise shift;
- source shift;
- lens-light shift;
- pixel-scale shift;
- mass-model shift;
- P100 stress;
- cross-simulator;
- small real sanity set.

Report:

- MAE;
- RMSE;
- bias;
- P95 absolute error;
- outlier fraction;
- radius-bin bias;
- multiplicity subgroup;
- error versus SNR;
- out-of-prior prediction rate;
- uncertainty coverage if applicable.

This matrix could be the central result of the paper.

---

# 7. A second high-value contribution: cross-simulator validation

## Question

> If two lens simulators are given physically equivalent parameters, do they produce sufficiently equivalent data for an ML model trained on one to generalize to the other?

## Why it matters

A model that performs only on the simulator that generated its training set may have learned implementation-specific structure.

A cross-simulator test does not prove real-world generalization, but it removes one layer of circularity.

## Possible implementation

Start with a deliberately narrow common subset:

- SIE lens;
- one Sérsic source;
- no shear initially;
- common pixel grid;
- same PSF kernel;
- no noise, then matched noise;
- exact coordinate/convention tests.

Compare:

1. deflection fields;
2. image positions;
3. rendered images;
4. total flux;
5. Einstein-ring geometry;
6. CNN performance across simulator boundary;
7. classical fit recovery across simulator boundary.

Candidates include lenstronomy plus one of Herculens/Caustics where equivalent functionality can be made explicit.

### Scientific standard

Do **not** call two simulators “equivalent” because their images look similar. Establish numerical tolerances on shared analytic controls first.

---

# 8. A third contribution: model shortcut detection

Your baseline can be interrogated for what cues it actually uses.

## Experiments

### 8.1 Lens-light counterfactual

Keep the lensed source fixed. Replace foreground lens light while preserving mass.

If the estimate changes substantially, the model is using lens-light shortcuts.

### 8.2 Source counterfactual

Keep lens mass fixed. Replace the source.

Measure prediction variance across sources:

\[
V_{\mathrm{source}}(\theta_E)
= \operatorname{Var}_s\left[\hat{\theta}_E(I(\theta_E,s))\right].
\]

A physically robust Einstein-radius estimator should not change excessively just because a different reasonable source is behind the same lens.

### 8.3 Rotation test

Rotate a lens scene while preserving its scalar Einstein radius.

A scalar-radius estimator should be approximately invariant:

\[
\hat{\theta}_E(I)
\approx
\hat{\theta}_E(R_\alpha I).
\]

### 8.4 Brightness test

Change source/lens flux while preserving geometry.

### 8.5 Background-only and arc-removal tests

Remove the actual lensed information while retaining nuisance structure. Predictions should degrade in a predictable way rather than remain mysteriously accurate.

These are interpretable tests of learned behavior without relying exclusively on saliency maps.

---

# 9. A fourth contribution: “when should ML defer?”

Instead of trying to replace classical modeling, ask whether ML can act as a reliable triage/initializer.

## Proposed workflow

```text
image
  |
  v
fast ML estimate + OOD/risk indicators
  |
  +--> low-risk --> report fast estimate
  |
  +--> medium-risk --> initialize classical SIE fit
  |
  +--> high-risk --> flag for full modelling/review
```

## Evaluate

- final parameter error;
- optimizer convergence rate;
- number of function evaluations;
- wall time;
- failure rate;
- dependence on initialization;
- whether high ML uncertainty predicts classical difficulty.

ML-assisted initialization is not itself new, so the novelty should be **failure-aware routing under controlled domain shift**.

---

# 10. What not to pursue yet

## 10.1 Full dark-matter mass-map reconstruction

Not yet. A single image does not uniquely determine arbitrary mass structure, and the present simulator/real calibration is not adequate for such a claim.

## 10.2 Dark-matter particle constraints

Not yet. This requires robust macro-model, source, substructure, population, selection, and line-of-sight treatment.

## 10.3 Hubble-constant inference

Not yet. Time-delay cosmography needs measured delays, high-resolution lens modeling, line-of-sight treatment, redshifts, often kinematics, and careful handling of mass-sheet degeneracies.

## 10.4 Predicting many parameters because the network can

Do not expand outputs just to make the model look more sophisticated. Add a parameter only when:

- it has a clear scientific interpretation;
- identifiability is understood;
- its symmetry/convention is correct;
- baselines exist;
- the simulator represents the relevant variation;
- failure analysis is possible.

## 10.5 Large transformers as the next obvious model

The project is currently limited more by **scientific validity and domain coverage** than by parameter count.

---

# 11. How to improve the current simulator scientifically

The simulator should evolve from a data generator into a controlled experimental system.

## Level 1 — current engineering baseline

- SIE;
- analytic Sérsic source;
- analytic foreground light;
- Gaussian PSF;
- scalar background;
- Poisson + read noise.

Keep this permanently as the simple control domain.

## Level 2 — realistic nuisance distributions

Add:

- variable pixel scale;
- spatial/empirical PSFs;
- correlated noise/resampling;
- more realistic backgrounds;
- mass/light misalignment;
- broader lens-light morphologies.

## Level 3 — realistic sources

Use real galaxy cutouts or validated generative source models.

Be careful to keep source assets split by identity so the same real galaxy morphology cannot leak across train/test descendants.

## Level 4 — mass-model mismatch

Add:

- power-law slope variation;
- external shear;
- multipoles;
- offsets;
- weak perturbers;
- line-of-sight approximations.

## Level 5 — survey-conditioned rendering

Build explicitly named observation profiles such as:

```text
HST_like_v1
Euclid_VIS_like_v1
Rubin_i_like_v1
Roman_WFI_like_v1
```

Do not label a profile as a real instrument simulation unless its PSF, pixel scale, noise, exposure, throughput assumptions, and preprocessing are actually documented.

---

# 12. How real data should enter the project

Real data should not initially be treated as a conventional supervised test set because exact latent truth is unavailable.

Use real data in stages.

## Stage A — realism diagnostics

Compare simulated and real distributions of observable quantities:

- pixel histograms;
- background structure;
- power spectra;
- radial profiles;
- segmentation statistics;
- PSF descriptors;
- apparent arc scales;
- brightness ratios;
- morphology embeddings.

Goal: identify where the simulator is visibly/statistically unrealistic.

## Stage B — reference-parameter sanity checks

Use systems with published/modelled Einstein-radius estimates.

Treat those values as **references**, not exact truth.

## Stage C — controlled injection into real backgrounds

A particularly practical bridge is:

1. use real non-lens/foreground cutouts;
2. inject simulated lensed sources;
3. preserve empirical detector/background/lens-light structure;
4. retain exact synthetic lens truth.

This strategy has already shown value in survey lensing ML and is much more realistic than fully analytic scenes.

## Stage D — blind real benchmark

Only once the model/protocol is frozen should a curated real sample be opened for a final external comparison.

---

# 13. Statistical standards the paper should meet

## 13.1 Pre-specified hypotheses

Examples:

**H1:** PSF mismatch increases Einstein-radius error relative to matched PSF controls.

**H2:** training-time PSF randomization reduces the mismatch penalty.

**H3:** source-morphology variation produces larger errors for doubles than quads.

**H4:** fixed-pixel-scale training produces systematic error when evaluated at unseen scales unless angular scale is normalized or explicitly supplied.

**H5:** a model selected on matched-domain validation does not necessarily remain best under stress-domain evaluation.

**H6:** ML initialization reduces classical optimizer cost without degrading final recovery under the matched model family.

---

## 13.2 Paired designs

Whenever possible, change one nuisance variable for the same underlying lens.

This controls parent-to-parent variation and gives a much cleaner estimate of the effect of the intervention.

---

## 13.3 Multiple seeds

Keep multiple training seeds, but be precise:

- seed variance is training stochasticity;
- it is not population uncertainty;
- it is not a confidence interval unless a valid statistical construction is provided.

---

## 13.4 New independent holdout

The existing synthetic test has already been evaluated.

Use current train/validation for development and create a **new independent holdout** after the protocol is frozen.

The holdout should ideally differ in seed namespace and asset ancestry and should never be used for architecture/loss selection.

---

## 13.5 Report conditional metrics

At minimum:

- target-radius bins;
- doubles/quads;
- SNR bins;
- source-size bins;
- PSF-width bins;
- lens-light/source contrast;
- stress-condition category.

A single MAE can hide scientifically important structure.

---

## 13.6 Confidence intervals / resampling

For paired stress effects, report bootstrap intervals across independent parent lenses when appropriate.

For example:

\[
\Delta \mathrm{MAE}_{\mathrm{PSF}}
= \mathrm{MAE}_{\mathrm{shifted}}-\mathrm{MAE}_{\mathrm{matched}}.
\]

Bootstrap over parent systems, not arbitrary descendant images that share the same underlying lens.

---

## 13.7 Effect sizes before significance chasing

The project should emphasize:

- arcsecond error change;
- relative-error change;
- coverage loss;
- outlier-rate increase;
- computational savings.

Do not build the paper around whether a p-value crosses 0.05.

---

# 14. Reproducibility as a scientific contribution

Your existing repository already does much of this well. Strengthen it further.

Every experiment should record:

- Git commit;
- environment/package versions;
- simulator version;
- config hash;
- dataset/release hash;
- split hash;
- parent IDs;
- nuisance intervention ID;
- training seed;
- selected checkpoint;
- exact evaluation sample IDs;
- predictions;
- summary metrics;
- subgroup metrics;
- runtime and hardware.

For lens-model interoperability, investigate exporting or mapping relevant parts of your manifest/results to the **COOLEST** standard rather than inventing a competing full lens-model standard.

Your own schema can remain useful for dataset/release provenance while COOLEST handles model-level conventions.

---

# 15. A concrete paper that would be stronger than the current broad idea

## Working title

**When Does Image-Based Strong-Lens Inference Generalize? A Controlled Robustness Study of Einstein-Radius Regression**

Alternative:

**From Synthetic Accuracy to Reliable Inference: Stress-Testing Einstein-Radius Estimation in Strong Gravitational Lenses**

## Core question

How sensitive is image-based Einstein-radius inference to observational and simulation mismatch, and which mitigation strategies preserve performance without leaking simulator truth?

## Contributions

1. A provenance-controlled benchmark of independent synthetic lens parents.
2. Matched counterfactual descendants isolating nuisance factors.
3. Classical and learned inference evaluated under the same stress conditions.
4. A failure atlas across radius, multiplicity, PSF, source morphology, noise, and pixel scale.
5. Controlled mitigation experiments such as nuisance randomization and allowed auxiliary inputs.
6. A fresh independent holdout protocol.
7. Optional cross-simulator external validation.
8. Reusable release artifacts and model cards.

## What the paper would *not* claim

- accurate physical mass inference on HST;
- calibrated real-domain posterior inference;
- dark-matter particle constraints;
- arbitrary galaxy population generalization;
- replacement of physical lens modelling.

That restraint makes the positive claims stronger.

---

# 16. Suggested primary experiments

## Experiment A — learning curve

Train the current architecture on increasing numbers of independent parents.

Purpose:

- determine whether 700 training parents are limiting performance;
- separate data-scale limitation from architecture limitation.

Use fractions such as 100, 200, 400, 700 and larger newly generated sets if compute allows.

---

## Experiment B — pixel-scale generalization

This should be high priority.

Compare:

- fixed-scale baseline;
- angularly resampled input;
- pixel-space prediction + metadata conversion;
- explicit pixel-scale conditioned model.

Evaluate on unseen pixel scales.

---

## Experiment C — PSF robustness

Paired PSF shifts with matched physical lens parents.

Compare image-only model versus PSF-aware input.

---

## Experiment D — source morphology robustness

Train analytic; test increasingly complex source families.

Then train mixed morphology and measure whether robustness improves without harming matched performance.

---

## Experiment E — lens-light shortcut audit

Break the simulator's correlations between mass and light.

This can reveal whether the network learned geometry or foreground appearance.

---

## Experiment F — noise/background robustness

Include correlated and empirical background conditions where possible.

---

## Experiment G — mass-family shift

SIE training to nearby non-SIE test distributions.

This is especially important before describing the output as a general Einstein-radius estimator.

---

## Experiment H — classical versus CNN failure map

Evaluate both methods on the same parent/stress grid.

Ask:

- where does classical fitting win?
- where does CNN win?
- where do both fail?
- are their errors correlated?

This turns “ML versus traditional” into a scientifically useful complementarity study.

---

## Experiment I — ML initialization

Use CNN predictions to initialize the bounded classical fitter.

Measure accuracy **and** computation.

---

## Experiment J — cross-simulator transfer

Train on one simulator and evaluate on a matched independent renderer after verifying conventions.

Treat this as an external robustness test, not proof of real-domain validity.

---

# 17. Potential second paper: a cross-simulator benchmark

If Experiment J becomes sufficiently substantial, it can stand alone.

## Research question

> How much of strong-lens ML performance is invariant to the simulation implementation when physical assumptions are matched?

## Possible contribution

A small open benchmark specifying:

- common analytic scenes;
- shared parameter conventions;
- render tolerances;
- standardized output images;
- reference image positions/critical curves;
- downstream regression task;
- cross-simulator train/test matrix.

This type of work is methodologically valuable because simulator dependence is often implicit.

---

# 18. Potential third paper: a lightweight real-domain bridge

Only after the synthetic robustness paper.

## Possible question

> Does training on simulated arcs injected into empirical astronomical backgrounds improve Einstein-radius transfer to archival real lenses compared with fully analytic simulation?

This would align directly with findings from survey ML that realistic foreground/background context improves transfer.

Possible data:

- archival HST/CASTLES where calibration is adequate;
- HSC lens samples with published model parameters;
- public Euclid products if suitable and permitted;
- real non-lens galaxies for injection backgrounds.

This paper would require much more careful data provenance and instrument-specific modeling than the current synthetic study.

---

# 19. Areas where you can contribute to open-source lensing directly

A research contribution does not need to be limited to your repository.

## 19.1 COOLEST

Useful contributions may include:

- testing conversions;
- validating conventions;
- example datasets;
- documentation;
- interfaces from another workflow;
- reproducibility utilities.

## 19.2 lenstronomy / Herculens / Caustics / paltas

Potential contribution types:

- edge-case tests;
- consistency tests;
- documentation examples;
- observation-profile validation;
- bug reproduction;
- benchmark scripts;
- interoperability tooling.

Do not begin by proposing a large new feature. First reproduce a scientifically meaningful edge case and open a tightly scoped contribution.

---

# 20. A useful niche: scientific quality assurance

There is a broader identity for this work:

> **Scientific QA for simulation-based astronomical inference.**

This includes:

- defining what “ground truth” actually means;
- preventing leakage;
- testing nuisance dependence;
- verifying units/conventions;
- measuring distribution shift;
- separating structural validation from scientific validation;
- tracking ancestry;
- testing coverage;
- building independent holdouts;
- reporting failure modes.

These skills transfer far beyond gravitational lensing.

They are also precisely the kind of work that becomes more important as astronomy moves toward large automated inference pipelines.

---

# 21. How this connects back to your logbook

Your logbook shows a natural scientific narrowing:

1. initial broad interest in baryons, dark matter, weak versus strong lensing;
2. transition to strong-lens mass inference;
3. realization that classification was already crowded;
4. exploration of SIE forward simulation;
5. recognition that ML may be most useful alongside traditional optimizers;
6. development of metadata/preprocessing infrastructure;
7. creation of a controlled synthetic dataset and first inference target.

The current repository is therefore **more mature than the original abstract framing**. The paper should follow the implementation, not the old brainstorm.

The old question:

> “Can machine learning infer mass from strong gravitational lenses?”

is too broad.

The better question:

> “Which aspects of the simulated observation process control the reliability of image-based Einstein-radius inference?”

is answerable, testable, and publishable at the scale of the existing project.

---

# 22. How this work could eventually lead back to dark matter

The long-term path is:

```text
robust scalar lens-scale inference
        |
        v
robust macro-model parameter inference
        |
        v
source/lens-light/systematic control
        |
        v
calibrated posterior inference
        |
        v
substructure sensitivity tests
        |
        v
population/selection modelling
        |
        v
dark-matter constraints
```

Each layer depends on the one below it.

Your current project is near the first two layers. That is not a weakness. It is the correct place to establish reliability.

---

# 23. Priority ranking for your project

## Highest priority

### 1. Pixel-scale audit

Because it directly threatens the interpretation of an image-only angular-radius predictor.

### 2. Matched nuisance stress framework

Because it becomes the backbone of the scientific paper.

### 3. Source-morphology and PSF shifts

Because both are repeatedly identified in the literature as major inference systematics.

### 4. Classical-versus-CNN failure comparison

Because you already have both implementations.

### 5. New blind synthetic holdout

Because the original test set has already been opened.

---

## Medium priority

### 6. Learning curves

Useful for determining whether to generate more parents.

### 7. Lens-light shortcut tests

Very relevant for transfer.

### 8. Cross-simulator test

High scientific value, but requires careful convention matching.

### 9. ML-assisted physical optimization

Useful if framed around reliability and resource allocation.

---

## Later

### 10. Uncertainty/posterior model

After deterministic behavior is mapped.

### 11. Real-domain benchmark expansion

After synthetic robustness is frozen.

### 12. Multi-parameter inference

After target-by-target identifiability and symmetry issues are understood.

### 13. Dark-matter substructure

Longer-term.

---

# 24. What would make the paper scientifically convincing

A convincing paper would show not simply that the model is accurate, but that you can answer all of these questions:

- What exactly is the estimand?
- Under which lens-model convention is it defined?
- Which variables are available at deployment time?
- Which simulator variables are forbidden as model inputs?
- How independent are train/validation/test lens parents?
- Has the test set been used for model choice?
- Which nuisance factors were intentionally shifted?
- Are comparisons paired on the same physical lens?
- What are the main failure subgroups?
- Are the results robust across training seeds?
- Is a simple geometric/classical baseline competitive?
- Does a better matched-domain model remain better out of domain?
- What happens under source-model mismatch?
- What happens under PSF mismatch?
- What happens when pixel scale changes?
- Does the model know when it is wrong?
- Can another researcher reproduce the dataset and metric table?

If the paper answers these clearly, it is doing real scientific methodology rather than just ML experimentation.

---

# 25. What success would look like

A successful result does **not** require every robustness experiment to work.

Scientifically valuable outcomes include:

- proving pixel scale is a major hidden shortcut;
- finding source morphology barely affects \(\theta_E\) but PSF mismatch does;
- finding doubles fail under domain shift while quads remain stable;
- showing domain randomization helps only some shifts;
- showing a classical fit is slower but substantially more robust to one model change;
- showing ML initialization gives most of the speed advantage without sacrificing physical refinement;
- showing uncertainty is badly miscalibrated out of domain;
- showing cross-simulator transfer fails despite strong in-domain scores.

A negative or failure result can be more useful than another small MAE improvement if the experiment isolates *why* it happens.

---

# 26. Suggested final research identity

Rather than positioning yourself as trying to “solve gravitational lensing with AI,” position the work as:

> **reliable machine learning and simulation-based methodology for strong gravitational-lens inference.**

More specifically:

> **I study how simulation assumptions and observational systematics affect automated inference of strong-lensing parameters, and develop reproducible tests that distinguish genuine lensing information from simulator-specific shortcuts.**

That is narrow enough to be credible and broad enough to support multiple follow-up projects.

---

# 27. Short-term execution plan

## Phase 1 — lock the scientific question

Rewrite the project abstract around Einstein-radius inference robustness.

Freeze:

- current baseline architecture;
- current matched-domain validation results;
- allowed inputs;
- existing test-set status.

## Phase 2 — build intervention infrastructure

Implement descendant generation for:

- pixel scale;
- PSF;
- noise;
- source morphology;
- lens light.

Ensure descendants stay grouped by parent.

## Phase 3 — run baseline stress atlas

No architecture changes yet.

Determine what actually breaks the current model.

## Phase 4 — select the two or three dominant failure modes

Do not attempt to fix everything.

## Phase 5 — mitigation experiments

One factor at a time:

- resampling;
- nuisance randomization;
- allowed auxiliary channel;
- realistic-source mixture;
- PSF input.

## Phase 6 — classical comparison

Run the same stress cases through the classical fitter where scientifically valid.

## Phase 7 — freeze candidate

Choose using validation + robustness criteria, not old test performance.

## Phase 8 — fresh blind holdout

Generate independently and evaluate once.

## Phase 9 — optional cross-simulator validation

Add only if the core paper is already complete.

## Phase 10 — real-data exploratory comparison

Keep claims explicitly bounded.

---

# 28. Literature map for this direction

The following papers/reviews are especially relevant to the research direction above.

1. **Saha et al. (2024), “Essentials of Strong Gravitational Lensing,” Space Science Reviews 220, 12.**  
   Core physical concepts and lensing degeneracies. DOI: 10.1007/s11214-024-01041-w.

2. **Lemon et al. (2024), “Searching for Strong Gravitational Lenses,” Space Science Reviews 220, 23.**  
   Discovery methods, contaminants, populations, and selection biases. DOI: 10.1007/s11214-024-01042-9.

3. **Vegetti et al. (2024), “Strong Gravitational Lensing as a Probe of Dark Matter,” Space Science Reviews 220, 58.**  
   Macro-model, source, lens-light, substructure, and systematic-error issues. DOI: 10.1007/s11214-024-01087-w.

4. **Shajib et al. (2024), “Strong Lensing by Galaxies,” Space Science Reviews.**  
   Review of galaxy-scale lens modelling and open problems including selection functions and degeneracies. DOI: 10.1007/s11214-024-01105-x.

5. **Galan et al. (2024), “Exploiting the diversity of modeling methods to probe systematic biases in strong lensing analyses,” A&A 692, A87.**  
   Blind comparison of six modeling methods; no method wins universally; systematic biases depend on the quantity/system; COOLEST-based comparison. DOI: 10.1051/0004-6361/202451095.

6. **Galan et al. (2023), “COOLEST: COde-independent Organized LEns STandard,” JOSS 8, 5567.**  
   Community standard for sharing/comparing lens-model assumptions and outputs. DOI: 10.21105/joss.05567.

7. **Gawade et al. (2025), “Neural network prediction of model parameters for strong lensing samples from Hyper Suprime-Cam Survey,” MNRAS 540, 3384.**  
   Einstein radius is comparatively robust; training realism and lens-light treatment substantially affect transfer. arXiv:2404.18897.

8. **Swierc et al. (2024), “Domain-Adaptive Neural Posterior Estimation for Strong Gravitational Lens Analysis.”**  
   Directly targets source/target domain mismatch in simulation-based inference. arXiv:2410.16347.

9. **Grespan et al. (2024), “TEGLIE: Transformer encoders as strong gravitational lens finders in KiDS,” A&A 688, A34.**  
   Real-data fine-tuning reduces false positives and illustrates the simulation-to-survey gap. DOI: 10.1051/0004-6361/202449929.

10. **Euclid Collaboration (2025), “Strong Lensing Discovery Engine C — Finding lenses with machine learning.”**  
    Demonstrates real survey-scale ML ranking/discovery on Euclid Q1 data. arXiv:2503.15326.

11. **Euclid Perseus strong-lens CNN study (2025), A&A 697, A14.**  
    Strong simulated performance but low real-data purity; emphasizes realism and human validation. DOI: 10.1051/0004-6361/202451868.

12. **Selection functions of strong lens finding neural networks (2024), MNRAS 534, 1093.**  
    Demonstrates ML selection biases toward larger Einstein radii and particular source properties. DOI: 10.1093/mnras/stae2106.

13. **Ephremidze et al. (2025), “Dark matter substructure or source model systematics? A case study of cluster lens Abell S1063,” MNRAS 542, 2610.**  
    Demonstrates that inadequate source modelling can generate spurious substructure evidence. DOI: 10.1093/mnras/staf1366.

14. **Nightingale et al. (2022), “Automated galaxy–galaxy strong lens modelling: No lens left behind,” MNRAS 517, 3275.**  
    Large-scale automated physical modelling and its computational implications.

15. **Gentile et al. (2023), “LEMON: automated modelling of strong gravitational lenses with Bayesian Neural Networks,” MNRAS 522, 5442.**  
    Demonstrates very fast neural parameter inference and uncertainty estimates.

16. **Treu, Suyu & Marshall (2022), “Strong lensing time-delay cosmography in the 2020s,” Astronomy & Astrophysics Review 30, 8.**  
    Strong discussion of modeling flexibility, line-of-sight effects, PSF, and precision-systematics requirements.

17. **Wedig et al. (2025), “The Roman View of Strong Gravitational Lenses.”**  
    Survey-yield simulations and realistic Roman detector/PSF considerations. arXiv:2506.03390.

18. **Rubin Community Workshop 2026 — Strong Lensing Science session.**  
    Current community work includes a Rubin Static Strong Lensing Data Challenge explicitly targeting automated discovery and modelling at LSST scale.

---

# 29. Final recommendation

The biggest mistake would be to interpret sophistication as “more parameters, a larger model, and a more ambitious astrophysical claim.”

For this project, scientific sophistication should instead mean:

- harder controls;
- cleaner estimands;
- realistic nuisance variation;
- matched counterfactuals;
- stronger baselines;
- independent holdouts;
- explicit domain-shift tests;
- calibrated uncertainty;
- transparent failure modes;
- reusable benchmark artifacts.

The project can make a useful contribution to gravitational lensing by doing something that becomes increasingly necessary as the field scales:

> **turn automated lens inference from a black-box accuracy problem into a measured, auditable reliability problem.**

That is both a realistic next step from the repository you already have and a direction with clear relevance to the Euclid/Rubin/Roman era.

---

# 30. Agent execution workflow and research timeline

This section is written as an **execution specification for a coding/research agent**. The agent should treat the preceding document as the scientific rationale and this section as the operational plan.

The objective is **not** to maximize model complexity. The objective is to build a defensible research study of **when automated strong-lensing inference is reliable, which simulation/observation mismatches cause it to fail, what information the model is actually using, and which mitigations improve robustness**.

The current repository already contains a working simulation pipeline, a real CASTLES preprocessing path, classical SIE fitting, an image-only CNN baseline, ancestry-aware splits, provenance machinery, validation reports, and previously opened synthetic test results. Preserve those assets and extend them rather than rewriting the project from scratch.

## 30.1 Non-negotiable scientific rules

The agent must follow these rules throughout the project:

1. **Do not tune against the existing frozen test split.** Its baseline result is historical evidence only. Use train/validation for all new choices and create a new independent holdout for final evaluation.
2. **Do not silently redefine the scientific target.** The primary target remains `theta_E_arcsec` until the one-parameter study is complete. Do not describe it as total mass or a mass distribution.
3. **Do not leak simulator truth into model inputs.** Parameter seeds, exact nuisance parameters, ancestry IDs, generating labels, truth sidecars, or derived variables that nearly encode `theta_E_arcsec` must remain outside the model input unless an experiment explicitly studies a deployment-available metadata variable.
4. **Use matched counterfactuals whenever possible.** To study a nuisance, rerender the same underlying lens while changing that nuisance alone instead of comparing unrelated samples.
5. **Change one major factor at a time before testing interactions.** First estimate main effects, then selected two-factor interactions.
6. **Preserve ancestry-safe splitting.** Any descendants, rerenders, rotations, PSF variants, source variants, or noise variants of one parent must remain in the same split.
7. **Keep real and simulated evidence conceptually separate.** Real CASTLES data may support preprocessing/QC/domain-gap analysis and bounded exploratory checks, but not calibrated real-domain accuracy unless adequate labels and calibration products are established.
8. **Report more than mean MAE.** At minimum report MAE, RMSE, median absolute/relative error, bias, P90/P95, radius-bin error/bias, subgroup results, seed variation, sample count, runtime, and parameter count where relevant.
9. **Record failures, not only successful runs.** Rejected simulations, optimizer failures, pathological samples, out-of-range predictions, and missing products are scientific evidence.
10. **Prefer interpretable scientific comparisons over architecture sweeps.** Do not spend the project searching dozens of architectures or hyperparameters.
11. **Freeze hypotheses and acceptance criteria before opening final holdout results.** Record them in a versioned protocol file.
12. **Do not upgrade the manuscript claim merely because metrics improve.** Claims must follow from the design of the experiment and the calibration of the data.

## 30.2 Required final research products

By the end of the workflow, the agent should create the following research products.

### Scientific artifacts

- A versioned **research protocol** containing hypotheses, estimands, primary metrics, subgroup definitions, nuisance variables, and stopping/acceptance rules.
- A **simulation reliability benchmark** built from matched parent lenses and controlled nuisance interventions.
- A **pixel-scale audit**.
- A **PSF mismatch audit**.
- A **noise/SNR audit**.
- A **source morphology and source-position audit**.
- A **lens-light audit**.
- A **multiplicity/counterimage-visibility audit**.
- A **mass-model mismatch audit** where feasible.
- A **learning-curve experiment** to determine whether the current problem is data-limited.
- A **cross-simulator or alternate-renderer generalization experiment**, if technically feasible after the primary audits.
- A **classical-versus-neural robustness comparison**.
- A **domain-randomization mitigation experiment** based only on nuisance factors shown to matter.
- A **fresh blind synthetic holdout** used once after model/protocol freeze.
- A **simulation-versus-CASTLES domain-gap report**.
- A **failure-case catalog** containing representative images, metadata, predictions, residuals, and failure categories.

### Reproducibility artifacts

- Immutable experiment YAML/config files.
- Versioned manifests and ancestry assignments.
- Configuration, source, manifest, and prediction hashes.
- Per-run environment/provenance records.
- Deterministic simulator/rerender checks.
- Machine-readable aggregate metric tables.
- Per-sample prediction tables.
- Plots generated from machine-readable results rather than manually entered numbers.
- A final model card / research card stating target, intended use, unsupported uses, performance, calibration limits, and known failure modes.

### Paper artifacts

- A final research question and hypothesis table.
- A literature-positioning matrix explaining how this work differs from prior CNN, NPE/SBI, domain-adaptation, automated modeling, and survey-discovery work.
- Publication-quality figures and tables.
- A complete Methods outline.
- A complete Results outline populated from frozen result files.
- A Limitations section written before the final Discussion is finalized.
- A reproducibility/data-availability statement.
- A final manuscript outline with claim-evidence mapping.

## 30.3 Phase 0 — Repository audit and experiment freeze

### Goal

Establish the exact starting state and prevent accidental contamination of historical results.

### Agent tasks

1. Inspect the repository tree, current branch, Git status, configs, experiment registry, result directories, and existing documentation.
2. Identify all artifacts associated with:
   - `sim-eng1000-v1`;
   - the historical CNN baseline;
   - the three-seed benchmark;
   - Huber ablations;
   - the existing locked test evaluation;
   - classical SIE grid results;
   - `real-castles-v2`.
3. Confirm which existing split has already been opened as test evidence.
4. Mark that split in documentation as **historical/opened — prohibited for further model selection**.
5. Run the repository test suite and record the command, environment, pass/fail count, and any pre-existing failures.
6. Re-run only deterministic validation checks that do not alter the scientific experiment state.
7. Create a new research branch dedicated to the reliability study.
8. Create a `docs/reliability_study/` or equivalent directory containing the protocol, experiment registry, and decision log.
9. Create a machine-readable experiment registry with fields such as:
   - experiment ID;
   - hypothesis;
   - independent variable;
   - controlled variables;
   - dataset/release identity;
   - train/validation/holdout policy;
   - models compared;
   - seeds;
   - primary metric;
   - secondary metrics;
   - acceptance rule;
   - result path;
   - status.
10. Write a baseline-state report so every later change can be compared to a fixed reference.

### Gate to proceed

Proceed only if the baseline artifacts are identifiable, the historical test split is clearly quarantined from development, and the core simulator/loader/model tests are in a known state.

### Deliverables

- `docs/reliability_study/protocol_v1.md`
- `docs/reliability_study/experiment_registry.csv` or YAML
- `docs/reliability_study/baseline_state.md`
- `docs/reliability_study/decision_log.md`

## 30.4 Phase 1 — Reformulate the paper around falsifiable hypotheses

### Goal

Replace the vague objective of “improving lens inference with ML” with a small number of testable claims.

### Agent tasks

Define and freeze hypotheses similar to the following:

- **H1 — Pixel-scale dependence:** an image-only model trained at one angular pixel scale will exhibit systematic error when the same physical lenses are rerendered at unseen pixel scales unless angular scale is normalized or supplied explicitly.
- **H2 — PSF sensitivity:** inference error increases with PSF mismatch, and the sensitivity differs between neural and classical inference.
- **H3 — Source-model sensitivity:** changing source morphology while holding the lens mass parameters fixed causes measurable prediction shift if the model relies on source-family correlations.
- **H4 — Observation-quality sensitivity:** inference error increases as SNR decreases or counterimages become difficult to detect, with different behavior across doubles/quads.
- **H5 — Domain randomization:** training across nuisance ranges identified as important reduces out-of-distribution error without materially degrading in-domain performance.
- **H6 — Hybrid use:** ML initialization can reduce classical optimization cost while maintaining final physical-fit quality better than direct ML predictions alone under mismatch.

For each hypothesis, specify:

- exact intervention;
- what remains fixed;
- target population of simulated parents;
- primary estimand;
- primary metric;
- paired statistical comparison;
- failure/acceptance threshold;
- plots required;
- whether the hypothesis is confirmatory or exploratory.

### Gate to proceed

Do not implement large new model families until the hypotheses and metrics are written and versioned.

## 30.5 Phase 2 — Build the matched-counterfactual benchmark generator

### Goal

Turn the simulator into a controlled scientific instrument.

### Agent tasks

1. Select a development set of independent synthetic parent lenses spanning the established `theta_E` range, shape range, and multiplicity classes.
2. For each parent, save a **canonical latent physical state** that remains fixed across rerenders.
3. Add a counterfactual renderer capable of varying declared nuisance families independently.
4. Initially support:
   - pixel scale;
   - PSF width/shape where possible;
   - exposure/SNR/noise;
   - source brightness;
   - source position;
   - source effective radius/Sersic index;
   - lens-galaxy brightness;
   - external shear;
   - observation normalization;
   - image detectability/counterimage suppression where scientifically definable.
5. Give every counterfactual asset explicit ancestry links back to the same parent.
6. Validate that unchanged settings reproduce the canonical rendering to the declared numerical tolerance.
7. Verify that changing a nuisance does **not** change the ground-truth `theta_E`.
8. Create automated visual contact sheets for a sample of every intervention family.
9. Create a benchmark manifest containing parent ID, intervention family, intervention magnitude, target, and output products.
10. Add unit/integration tests for ancestry, deterministic rerendering, label invariance, and split isolation.

### Gate to proceed

The benchmark is valid only when rerender identity and label invariance tests pass and related counterfactuals cannot leak across splits.

## 30.6 Phase 3 — Pixel-scale audit first

### Why this comes first

The current synthetic engineering release uses a fixed angular pixel scale while the baseline CNN receives only image pixels. This creates a direct risk that the model learns a simulator-specific pixel-radius-to-angle mapping.

### Agent tasks

1. Create matched rerenders over a physically plausible grid of pixel scales.
2. Keep the underlying lens/source state and `theta_E_arcsec` fixed.
3. Evaluate the existing CNN without retraining.
4. Measure prediction shift relative to the canonical rendering for every parent.
5. Plot error versus pixel scale and error versus normalized scale ratio.
6. Stratify by true Einstein radius and multiplicity.
7. Compare at least three mitigation strategies:
   - resampling every input to a fixed angular scale;
   - predicting a scale-normalized or pixel-space radius and converting through metadata;
   - supplying deployment-available pixel-scale metadata as an auxiliary input.
8. Ensure no exact target or simulator truth leaks through the auxiliary metadata path.
9. Repeat across fixed seeds if retraining is required.
10. Select the simplest mitigation that materially reduces scale-induced error without harming validation performance.

### Primary output

A definitive answer to whether the current image-only model is robust to angular sampling.

### Decision gate

Do not move to real-image claims if performance is strongly tied to the training pixel scale and no mitigation has been validated.

## 30.7 Phase 4 — Observation-systematics audits

Run the following studies one at a time using matched parents.

### A. PSF audit

Test:

- PSF FWHM variation;
- train/test PSF mismatch;
- if feasible, Gaussian versus a more realistic or alternate PSF family.

Compare CNN and classical fitting. Record the error slope with mismatch magnitude.

### B. Noise/SNR audit

Test:

- exposure changes;
- Poisson/read-noise changes;
- low-, medium-, and high-SNR regimes;
- repeated noise realizations for identical latent lenses.

Distinguish **noise sensitivity** from **training-seed sensitivity**.

### C. Lens-light audit

Hold mass/source truth fixed and vary foreground lens-light brightness/profile. Test whether the network confuses light morphology with mass scale.

### D. Normalization/preprocessing audit

Compare declared transformations using train-only fitted statistics where applicable. Include the current normalization and selected robust/asinh alternatives. Never fit preprocessing parameters on validation or holdout data.

### Required analysis for every audit

For each nuisance family calculate:

- paired prediction shift;
- paired absolute-error change;
- mean/median shift with uncertainty;
- P90/P95 degradation;
- target-bin bias;
- multiplicity subgroup effects;
- worst cases;
- neural-versus-classical difference;
- representative image panels.

## 30.8 Phase 5 — Physical/source mismatch audits

### Goal

Determine whether inference is robust to changes in astrophysical structure rather than only detector effects.

### Agent tasks

1. Vary source morphology parameters while holding lens mass truth fixed.
2. Introduce source-family alternatives if supported by the simulator.
3. Vary source position around caustics to change image configuration/multiplicity.
4. Separate physical image multiplicity from detectable multiplicity when noise or masking hides images.
5. Test external shear ranges independently of `theta_E`.
6. Where feasible, create deliberate mass-model mismatch:
   - train/fit under SIE-like assumptions;
   - render selected evaluation samples under a related but non-identical mass family.
7. Compare prediction stability with classical-fit residuals and convergence behavior.
8. Identify nuisance variables that correlate with large error even after conditioning on true `theta_E`.

### Scientific output

A ranked **failure-mechanism map** based on measured effects, not subjective importance.

## 30.9 Phase 6 — Baselines and learning curves

### Goal

Determine whether performance comes from genuinely learned image structure and whether more synthetic data is likely to help.

### Baselines to implement or validate

At minimum compare:

1. train-mean predictor;
2. simple geometric/radial estimator where scientifically justified;
3. ridge/elastic-net or boosted tree on fixed, auditable image features;
4. classical SIE fitting;
5. current CNN;
6. one compact residual CNN only if the preceding results justify an architectural challenger.

### Learning-curve experiment

Train the same baseline architecture using increasing numbers of independent parent systems, for example a logarithmically or practically spaced sequence that fits available compute.

For every training size:

- hold the validation set definition fixed;
- use the same seeds where possible;
- record train and validation metrics;
- plot error versus number of independent parents;
- fit only a descriptive trend unless a stronger model is justified.

### Decision gate

If performance is still clearly improving with more parents, prioritize simulation diversity/scale before major architecture changes. If performance has saturated while robustness remains poor, focus on distribution design and representation rather than dataset size alone.

## 30.10 Phase 7 — Model interpretation and shortcut tests

### Goal

Test what information the model actually uses without overinterpreting saliency maps.

### Agent tasks

Use **interventions** as the primary interpretability method.

Perform controlled tests such as:

- erase/mask central lens light while preserving lensed features;
- retain lens light while suppressing selected lensed-source regions;
- radial-only representations;
- rotation tests;
- brightness scaling;
- background replacement;
- PSF swaps;
- source swaps at fixed mass truth;
- matched counterimage removal/suppression where scientifically meaningful.

Optional attribution maps may be used as supporting visualization but must not be treated as proof of causal feature use.

For each test, quantify the change in prediction and error relative to an unchanged paired control.

### Deliverable

Create `shortcut_audit.md` containing evidence for and against specific shortcut hypotheses.

## 30.11 Phase 8 — Cross-simulator / alternate-renderer generalization

### Goal

Test whether the inference pipeline depends on the implementation details of one simulator.

### Agent tasks

1. Identify one feasible alternate renderer/modeling package or independently implemented lens equation path.
2. Construct a small physically matched benchmark with the same declared mass/source parameters and comparable observation settings.
3. Document convention differences explicitly: coordinates, ellipticity, shear, PSF, normalization, source definition, and pixel centering.
4. Validate simple SIS/SIE controls across implementations before ML evaluation.
5. Evaluate the existing model on alternate-renderer images without retraining.
6. Compare:
   - within-simulator validation;
   - same-physics cross-simulator evaluation;
   - nuisance-matched and nuisance-mismatched cases.
7. If cross-simulator performance drops, identify whether the gap is explained by observable image-statistic differences or convention differences.

### Decision gate

Do not claim simulator-independent inference unless cross-simulator evidence supports it.

## 30.12 Phase 9 — Data-driven mitigation rather than arbitrary complexity

### Goal

Mitigate only the failure modes that the audits show are important.

### Agent tasks

1. Rank nuisance families by effect size and scientific relevance.
2. Select the smallest set of high-impact factors for training-time domain randomization.
3. Create a new training proposal with broader distributions for those variables.
4. Keep the validation protocol and model architecture fixed for the first mitigation test.
5. Compare the original and randomized-training models using paired validation cases.
6. Require improvement on mismatch/stress conditions without unacceptable degradation in canonical conditions.
7. If needed, test one additional representation change such as:
   - explicit mask/variance channels;
   - angular-scale normalization;
   - PSF kernel input;
   - compact residual architecture.
8. Change one intervention at a time and record all failed candidates.

### Deliverable

A table of **failure mode → mitigation → measured improvement → remaining limitation**.

## 30.13 Phase 10 — Classical/ML hybrid experiment

### Goal

Test whether ML is more scientifically useful as an initializer than as a final estimator.

### Agent tasks

1. Define classical optimizer initialization strategies:
   - generic/default initialization;
   - heuristic initialization;
   - ML-predicted initialization.
2. Run the same physical fitting procedure from each initialization on matched samples.
3. Record:
   - convergence success;
   - final `theta_E` error;
   - final objective/residual;
   - number of objective evaluations/iterations;
   - wall-clock runtime;
   - boundary failures;
   - Jacobian/local-identifiability diagnostics where available.
4. Repeat under canonical and selected mismatch conditions.
5. Determine whether ML improves **computational efficiency without sacrificing final fit quality**.

### Interpretation rule

Even if direct ML predictions are less robust under domain shift, a reliable speedup for classical modeling may still be a valuable practical contribution.

## 30.14 Phase 11 — Simulation-to-CASTLES domain-gap study

### Goal

Use real observations to measure how the simulation differs from the deployment domain without pretending that real truth labels are available for every sample.

### Agent tasks

1. Restrict to reviewed real observations with known/usable metadata for the comparison being made.
2. Compare simulated and real distributions for measurable quantities such as:
   - pixel scale;
   - image dynamic range;
   - radial brightness profiles;
   - background statistics;
   - masked fraction;
   - apparent PSF proxies where defensible;
   - image size/cropping;
   - multiplicity labels when available;
   - filter/instrument distribution;
   - lens/source morphology proxies.
3. Do not interpret toy variance as calibrated detector variance.
4. Identify regions of real-data feature space not represented by the simulator.
5. Create a **simulation coverage map**: well represented / weakly represented / unsupported.
6. Use the existing three-system astrometric comparison only as exploratory context, not as a general accuracy benchmark.
7. Define what additional calibration data would be required for a genuine blinded real-data parameter benchmark.

### Deliverable

`results/reports/domain_gap_v1/` containing machine-readable comparisons, figures, and a limitations report.

## 30.15 Phase 12 — Optional uncertainty modeling

Only begin this phase after deterministic robustness behavior is reasonably understood.

### Candidate approaches

Choose one simple uncertainty method first:

- deep ensemble;
- heteroscedastic regression;
- quantile regression.

### Required evaluation

Do not report only interval width. Evaluate:

- empirical coverage at declared levels;
- calibration curves;
- interval width versus error;
- coverage under nuisance mismatch;
- coverage by `theta_E` bin and multiplicity;
- failure cases with high error but low predicted uncertainty.

Simulation-based posterior methods such as NPE/NLE/NRE should remain a later extension unless the available simulation scale and validation protocol can support posterior calibration tests such as coverage/SBC.

## 30.16 Phase 13 — Build a fresh blind holdout

### Goal

Create the only final synthetic evaluation set used for the new paper.

### Agent tasks

1. Finalize the simulator proposal after all development decisions are complete.
2. Define the holdout distribution and sample size in the protocol **before generating predictions**.
3. Generate entirely new parent systems with new seeds and no ancestry overlap with development data.
4. Include both canonical proposal cases and predeclared robustness strata if the paper makes robustness claims.
5. Validate release integrity and ancestry isolation.
6. Freeze the final selected model, preprocessing, and evaluation script.
7. Hash all relevant identities.
8. Evaluate the holdout once.
9. Store predictions and metrics immutably.
10. Do not return to model selection based on these results. If the final result is disappointing, report it and explain it rather than retuning on the holdout.

## 30.17 Phase 14 — Statistical analysis and final figures

### Required statistical approach

Prefer paired analysis because many experiments use the same latent parent before and after a nuisance intervention.

For major comparisons compute:

- paired error difference;
- bootstrap confidence intervals over independent parent systems where appropriate;
- effect sizes;
- subgroup estimates with sample counts;
- seed variation separately from parent/sample variation;
- multiple-comparison caution for large exploratory sweeps.

Do not describe three-seed standard deviation as a confidence interval.

### Minimum final figure set

1. End-to-end research pipeline diagram.
2. Baseline predicted-versus-true `theta_E` plot.
3. Learning curve.
4. Pixel-scale robustness curve.
5. PSF mismatch robustness curve.
6. Noise/SNR robustness curve.
7. Source/morphology robustness figure.
8. Neural versus classical robustness comparison.
9. Failure-mode heat map or ranked effect-size plot.
10. Domain-randomization mitigation comparison.
11. Simulation-versus-real domain-gap figure.
12. Final blind-holdout performance figure.
13. Representative failure-case image panel.

### Minimum final table set

1. Dataset/release definitions.
2. Simulation parameter ranges.
3. Model and baseline specifications.
4. In-domain validation metrics.
5. Robustness/stress metrics.
6. Classical versus neural comparison.
7. Mitigation results.
8. Blind-holdout results.
9. Real-data metadata/calibration limitations.

## 30.18 Phase 15 — Manuscript construction

### Recommended paper structure

#### 1. Introduction

Establish:

- why scalable strong-lens inference matters;
- why simulation-trained inference is attractive;
- why simulation-to-observation mismatch threatens scientific validity;
- why Einstein radius is a controlled first target;
- the paper's reliability-focused contribution.

End with explicit research questions/hypotheses.

#### 2. Data and simulations

Describe:

- CASTLES role and limitations;
- simulator physics;
- engineering dataset;
- counterfactual benchmark;
- split/ancestry rules;
- new blind holdout.

#### 3. Inference methods

Describe:

- simple baselines;
- CNN;
- classical SIE fitting;
- any selected hybrid/mitigation model.

#### 4. Reliability experiments

Organize by nuisance family rather than by code implementation.

#### 5. Results

Report:

- baseline performance;
- learning curve;
- nuisance sensitivities;
- shortcut tests;
- classical-neural comparison;
- mitigation;
- blind holdout;
- bounded real-domain comparison.

#### 6. Discussion

Address:

- what the network appears to use;
- which mismatches matter most;
- where classical and learned methods differ;
- implications for simulation-trained lens inference;
- what is required before real scientific deployment.

#### 7. Limitations

Explicitly discuss:

- analytic source families;
- PSF simplifications;
- population/selection priors;
- CASTLES calibration limitations;
- one-parameter target;
- finite simulator diversity;
- lack of population-level real labels;
- any untested lens-mass complexity.

#### 8. Conclusion

Conclude with measured reliability findings, not a general claim that the model “solves” lens modeling.

## 30.19 Claim-evidence map the agent must maintain

Before manuscript completion, create a table with columns:

- manuscript claim;
- exact experiment supporting it;
- dataset/release;
- metric/figure/table;
- limitations;
- allowed wording;
- prohibited stronger wording.

Examples:

- Allowed: “The baseline recovered `theta_E` accurately for held-out samples from the same synthetic proposal.”
- Not allowed: “The model accurately measures Einstein radii from HST lenses.”
- Allowed: “Pixel-scale mismatch increased prediction error in matched synthetic counterfactuals.”
- Not allowed: “Pixel scale is the dominant error source in all real strong-lens analyses.”
- Allowed: “Domain-randomized training improved robustness over the tested nuisance range.”
- Not allowed: “Domain randomization solves the simulation-to-real gap.”

## 30.20 Suggested execution timeline

The timeline below is an **ordering and milestone schedule**, not a requirement to force experiments to fit a calendar. Scientific gates take precedence over dates.

### Week 1 — Freeze the research design

- Audit repository and existing results.
- Quarantine historical test set from development.
- Run/record tests and validations.
- Create research branch and reliability-study documentation.
- Freeze hypotheses H1-H6 and primary metrics.
- Create experiment registry and decision log.

**Milestone:** reproducible starting point and signed-off protocol.

### Week 2 — Counterfactual benchmark infrastructure

- Implement canonical-parent + rerender mechanism.
- Implement ancestry tracking for interventions.
- Add deterministic/label-invariance tests.
- Create intervention manifests and preview tooling.
- Run a small smoke benchmark.

**Milestone:** same physical lens can be safely rerendered under controlled nuisance changes.

### Week 3 — Pixel-scale study

- Generate scale-matched counterfactuals.
- Evaluate baseline CNN and classical method.
- Test angular-resampling / normalized-target / auxiliary-metadata alternatives.
- Produce pixel-scale report and decision.

**Milestone:** resolve whether fixed angular sampling is a shortcut or major domain risk.

### Week 4 — PSF and observation-quality studies

- Run PSF-width/mismatch audit.
- Run noise/SNR audit with repeated noise realizations.
- Run lens-light and normalization audits.
- Produce paired-effect tables and plots.

**Milestone:** quantify major observation-systematic sensitivities.

### Week 5 — Source/physics mismatch studies

- Run source morphology/position interventions.
- Analyze physical versus detectable multiplicity.
- Run shear study.
- Add a limited mass-model mismatch experiment if validated.
- Build first failure-mechanism ranking.

**Milestone:** separate astrophysical mismatch from detector/processing mismatch.

### Week 6 — Baselines, learning curves, and shortcut tests

- Implement/validate simple non-neural baselines.
- Run learning curve over independent-parent count.
- Run key image interventions/shortcut tests.
- Compare neural versus classical behavior.

**Milestone:** know whether the problem is data-limited, representation-limited, or distribution-limited.

### Week 7 — Cross-simulator study

- Validate conventions with an alternate renderer.
- Build a small matched benchmark.
- Run cross-simulator inference.
- Document convention and implementation differences.

**Milestone:** estimate simulator-specific dependence independently of real-data complications.

### Week 8 — Mitigation study

- Select high-impact nuisance variables from Weeks 3-7.
- Build a domain-randomized training proposal.
- Retrain with the established seeds.
- Test the smallest justified input/architecture modification if needed.
- Compare robustness and canonical performance.

**Milestone:** demonstrate whether identified failure modes can be reduced in a controlled way.

### Week 9 — Hybrid classical/ML study

- Implement ML-based optimizer initialization.
- Compare default/heuristic/ML starts.
- Measure accuracy, convergence, objective evaluations, failures, and runtime.
- Repeat on canonical and selected mismatch conditions.

**Milestone:** determine whether ML is more useful as a final predictor, an initializer, or both.

### Week 10 — Real-domain gap analysis

- Review CASTLES eligibility and metadata quality.
- Compare real/synthetic measurable image statistics.
- Create simulator coverage map.
- Identify unsupported real regimes.
- Specify requirements for a future blinded real benchmark.

**Milestone:** clearly state what the simulator does and does not represent about HST/CASTLES observations.

### Week 11 — Freeze final method and create blind holdout

- Review all development evidence.
- Select final preprocessing/model/mitigation protocol using validation only.
- Freeze code/config/hypotheses.
- Generate new independent holdout.
- Validate release and hashes.
- Evaluate exactly once.

**Milestone:** final unbiased synthetic evidence for the paper.

### Week 12 — Analysis and manuscript package

- Complete paired statistical analysis.
- Generate final figures/tables from frozen result files.
- Build failure-case catalog.
- Write model/research card.
- Write Methods and Results directly from recorded artifacts.
- Complete claim-evidence map.
- Write Limitations before final Discussion.
- Assemble manuscript draft and reproducibility statement.

**Milestone:** complete auditable paper package.

## 30.21 If the full timeline is too ambitious

Prioritize in this order:

1. research protocol and test-set quarantine;
2. counterfactual benchmark infrastructure;
3. pixel-scale audit;
4. PSF audit;
5. source morphology audit;
6. classical-versus-CNN robustness comparison;
7. learning curve;
8. domain-randomization mitigation;
9. new blind holdout;
10. CASTLES domain-gap analysis;
11. manuscript/reproducibility package.

Cross-simulator testing, uncertainty modeling, additional architectures, and hybrid optimization are valuable extensions but should not displace the core robustness study.

## 30.22 Agent completion checklist

The agent should not consider the project complete until all applicable items below are satisfied.

- [ ] Historical synthetic test split is not used for new model selection.
- [ ] Research hypotheses and primary metrics are versioned before final experiments.
- [ ] Counterfactual rerenders preserve physical truth and ancestry.
- [ ] Pixel-scale sensitivity has been explicitly tested.
- [ ] PSF and noise sensitivity have been explicitly tested.
- [ ] At least one source/morphology mismatch has been tested.
- [ ] Neural performance is compared with meaningful non-neural/classical baselines.
- [ ] Learning-curve evidence exists.
- [ ] Failure cases are preserved and categorized.
- [ ] Mitigation is driven by measured failure modes rather than arbitrary architecture search.
- [ ] A new independent blind holdout has been evaluated only after freeze.
- [ ] Real-data claims remain within CASTLES calibration/label limits.
- [ ] All figures/tables can be regenerated from machine-readable results.
- [ ] Every major result records configuration, code/data identity, seeds, and sample counts.
- [ ] Limitations and unsupported claims are explicitly documented.
- [ ] Claim-evidence map is complete.
- [ ] Final manuscript language distinguishes synthetic in-domain, synthetic stress, cross-simulator, and real observational evidence.

## 30.23 Final instruction to the agent

Implement this plan as a **scientific research program**, not as a feature checklist. At each phase, inspect the existing repository first, reuse validated infrastructure, make the smallest change necessary to answer the current hypothesis, and stop to analyze the result before expanding scope.

The desired final contribution is not “a more complicated neural network.” It is a reproducible body of evidence showing:

1. what information supports Einstein-radius inference;
2. which observational and simulation assumptions cause inference error;
3. whether classical and learned methods fail differently;
4. whether targeted mitigation improves robustness;
5. which parts of the real observational domain remain unsupported;
6. what evidence is required before automated strong-lens inference can be trusted more broadly.

If an experiment falsifies the expected hypothesis, preserve and report that result. Negative robustness results are part of the contribution, not implementation failures.
