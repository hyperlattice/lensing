# Understanding and improving strong-lens predictions with synthetic data

**Research idea and proposed study design**  
**Date:** 5 November 2026  
**Status:** Planning report. The experiments described below are proposed; no new results are reported here. The exact novelty of the proposal remains to be established through a focused literature comparison.

## 1. The idea in plain language

We want to understand how a computer estimates the strength of a gravitational lens from an image. We then want to use that understanding to create better practice images for its training.

Think of the model as a student. Synthetic images are its practice questions. If the student keeps making a particular mistake, we investigate the reason and prepare exercises that address it. A separate exam checks whether the student has learned something useful.

In this project, the answer being predicted is the **Einstein radius**: an angular measure of the scale of the lensing effect. Our simulator supplies a known value for this parameter, so we can measure prediction errors directly within the simulated setting.

The proposed process is:

**Create controlled images → investigate model behaviour → identify a candidate explanation → generate targeted training examples → test on fresh systems.**

The main research focus is explainability. Synthetic data provides both the experimental control needed to investigate the model and the training material needed to test an improvement.

Some useful terms:

| Term | Plain-language meaning |
|---|---|
| Foreground lens | The nearer galaxy whose gravity bends the light |
| Background source | The more distant galaxy being distorted |
| Parent system | An underlying lens system whose rendered variants are grouped together |
| PSF, or point-spread function | The instrument's blurring response |
| MAE, or mean absolute error | The average distance between a prediction and the known answer |
| Counterfactual image | A controlled alternative image in which specified scene properties change |

## 2. The central research question

> Can controlled synthetic experiments explain a model's failure patterns, and can those explanations guide training-data generation more effectively than uniform diversity or error-based targeting?

This question has three parts:

1. **Explanation:** Which changes to a scene systematically affect predictions and errors?
2. **Intervention:** Can a specific training-data change reduce the diagnosed dependence?
3. **Generalization:** Does the improvement persist on independent systems and conditions that were withheld during development?

The intended explanation is a testable description of model behaviour. The study would not establish a complete account of the network's internal reasoning.

## 3. How this differs from the current manuscript

The [current manuscript](../submission/aaai27/reviewer_bundle/paper.tex) compares canonical training, uniform observation variation, and failure-weighted observation sampling under matched final training budgets. Its [equal-budget report](archive/equal_budget_final_v1.md) documents an improvement from uniform variation and an inconclusive additional benefit from failure weighting.

That comparison asks whether a training package helps. The new proposal asks **what the model depends on, why a particular failure occurs, and whether a diagnosis predicts a useful training intervention**.

The current comparison does not isolate fresh-noise exposure from observing-condition variation. Angular representation is another possible explanation for some scale-related gains. These need a common, documented treatment in the new experiment so they do not obscure the effect of scene generation.

There is useful existing groundwork:

- [Counterfactual rendering](../src/simulation/counterfactual.py) supports controlled changes to foreground light, source properties, and observing conditions while protecting the target and identity fields.
- [Source-physics configurations](../configs/reliability/source_physics.yaml) specify source-profile, size, position, and shear interventions.
- The [earlier shortcut audit](../src/ml/reliability_week6.py) uses masks, brightness changes, and rotations.
- The [existing claim-evidence map](archive/reliability_study/final/claim_evidence_map.md) correctly distinguishes sensitivity to an image region from proof of an invalid shortcut.

These components are starting points. Their presence does not establish that all required datasets and checkpoints are available or that the proposed study has already been implemented.

## 4. What synthetic data means here

Synthetic data means images assembled through a controlled image-formation process. It does not require a learned image generator.

| Approach | What it supplies | Main advantage | Main limitation |
|---|---|---|---|
| Procedural simulation | Explicit profiles, disks, clumps, foreground light, and observation effects | Clear control over each change | Can omit important structures and correlations |
| Real galaxy templates | Prepared galaxy images used as source or foreground components | Empirical morphological diversity | Finite library; existing PSF and noise require treatment |
| Real-background injection | Simulated lensed light inserted into reviewed observational cutouts | Measured background structure and contaminants | Calibration, units, additional shot noise, and asset reuse need control |
| Learned generation | New galaxy components sampled from a trained model | Potentially broader and controllable morphology | Can memorize, omit rare structures, or introduce artefacts |

For a learned source generator, the proposed pipeline is:

**Generate an unlensed galaxy → ray-trace it through a known lens → apply the instrument model → save the image and simulator-defined radius.**

This keeps the lens label tied to an explicit physical renderer. It still requires validation of the generated source population. A visually plausible galaxy is not evidence that its distribution is appropriate for the intended observations.

## 5. The first concrete example: foreground-light dependence

The proposed first case is foreground-light dependence because it permits interpretable interventions and connects to existing rendering controls. This is a candidate case, not a finding that the current model uses an invalid shortcut.

### 5.1 Construct matched scene families

Hold the lens mass, background source, geometry, and radius convention fixed. Render versions with different foreground brightness, size, and orientation.

Each family contains repeated views of one underlying lens system. These views remain grouped during splitting and analysis.

An illustrative question is:

> When the foreground galaxy changes but the lens mass stays fixed, does the predicted radius shift systematically, and under which visibility conditions?

Begin with one component at a time. Selected interactions, such as bright foreground light combined with poor resolution, can follow once individual effects are understood.

### 5.2 Distinguish rival explanations

| Candidate explanation | Expected observation | Discriminating check |
|---|---|---|
| Reliance on a training correlation | Prediction shifts when the foreground cue changes relative to lens strength | Alter the correlation while preserving declared marginal distributions and evaluate independent systems |
| Loss of useful arc information | Errors increase as foreground light obscures arcs | Separate clearly visible and heavily obscured regimes; compare high-quality/noiseless controls |
| Preprocessing sensitivity | A rendering change changes normalization or cropping in a consequential way | Inspect intermediate products and repeat with a frozen alternative preprocessing rule |
| Inadequate scene coverage | Failures concentrate in plausible combinations underrepresented during training | Add those combinations and evaluate on independent related cases |
| Training or architecture limitation | The dependence persists despite verified additional coverage | Examine development learning curves and, later, a bounded model comparison |

Noise should be repeated across scenes rather than allowing one random realization to determine the conclusion. Pair random streams where meaningful and record their construction; using the same seed does not make different signal-dependent noise processes identical.

### 5.3 Avoid an unrealistic invariance requirement

A fixed true radius does not imply that every changed image contains the same information. A brighter foreground can hide arcs and legitimately make estimation less precise. The study should test systematic bias, conditional errors, and responsiveness to real signal changes, rather than demand exactly equal predictions for every nuisance intervention.

Some controlled combinations may also be uncommon in nature. Mark diagnostic probes separately from simulations intended to represent an observational population.

## 6. What counts as an explanation?

A useful explanation makes a prediction before the next independent test.

For example, a candidate explanation could be:

> The estimator relies on a relationship between foreground appearance and lens strength in its training population. It will therefore be biased when that relationship changes. Training on additional, physically plausible combinations should reduce that bias.

That statement can fail. If the suspected dependence is absent, if a different mechanism explains the errors, or if the proposed retraining intervention does not help, the explanation is unsupported or incomplete.

Heatmaps may illustrate where a network is sensitive. Their role is supplementary: visual appeal alone does not establish explanation quality. Adebayo and colleagues show that some saliency methods can be insensitive to learned parameters or the data-generating process. Any retained attribution method should receive appropriate model/data randomization checks. [Adebayo et al., 2018](https://arxiv.org/abs/1810.03292)

Within a correctly implemented simulator, controlled interventions support statements about model response to specified scene changes. Extending that explanation to actual telescopes requires separate observational evidence.

## 7. Test the value of explanation-guided synthesis

After development diagnostics, freeze a candidate explanation and a training-data intervention. Start with a single retraining cycle so the intervention is interpretable.

| Arm | Training-data strategy | Purpose |
|---|---|---|
| A: reference | Current-style scene distribution under the common new observation/preprocessing setup | Establish the starting point |
| B: uniform diversity | Sample a declared scene-variation library uniformly | Measure the benefit of broad diversity |
| C: error-guided | Weight the same library using development errors | Test whether emphasizing difficulty is sufficient |
| D: explanation-guided | Weight or pair examples from the same library using the frozen diagnosis | Test whether the explanation provides additional value |

The key comparisons are **D versus B** and **D versus C**. D versus A alone cannot separate useful diagnosis from a generic diversity benefit.

Use the same available scene library for B/C/D. Record their realized coverage and retain a declared exploration allocation so targeting does not silently discard most of the scene space. In the first study, use a paired parent schedule and let the selection strategy choose scene variants for those parents. This isolates variant selection from a change in parent exposure.

All arms should share:

- Architecture, optimizer settings, training exposures, and final update budget.
- Paired initialization seeds and training-parent schedules.
- A common fresh-noise policy and frozen angular-preprocessing rule selected on development data.
- The same inference-time inputs and target convention.
- Documented selection, generation, and diagnostic budgets.

Match the final training budget and report total development, rendering, and training costs separately. Image dimensions can change processing cost even when exposure counts match.

The strategy for D must specify exactly how a diagnosis changes sampling or pairing. It must not be an informal choice of attractive examples after inspecting final results. If the diagnosis does not distinguish any examples from C's selection, D is not a distinct experimental condition.

## 8. Where learned generators and adaptive synthesis fit

### Learned source generation

Once the first intervention study is interpretable, compare procedural, real-template, and learned sources under a common allocation strategy. This separates the value of source construction from the value of targeting.

A learned generator could be conditioned on source size, brightness, elongation, or clumpiness. Each control must be checked against measurements on the generated images; requesting a property does not establish that the generator delivered it.

Required checks include held-out morphology coverage, flux and size distributions, nearest-neighbour/memorization inspection, and independent downstream evaluation. Existing PSF and noise in template or generator training images must be accounted for before applying the simulated instrument response.

Direct generation or style transfer of complete lensed images would need an additional label-preservation study. It is a separate extension from generating source components before physical rendering.

### Data feasibility and unresolved route choice

The [data feasibility and options note](literature_review/synthetic_generation_benchmark/data_options.md) records the subsequent discussion of the current 700-parent training split, external labelled lens data, and ordinary-galaxy libraries. A small training set is a limitation, not a validated impossibility threshold for learned generation. External datasets are not established as drop-in additions: target-distribution and observation differences could create dataset-origin shortcuts, while source-galaxy libraries require physical lens rendering to supply radius labels. The note compares keeping the current data, rendering more parents, incorporating external labelled lenses, and generating source components. No data route or generator experiment has been selected; this does not replace the intervention study above.

### Adaptive synthesis

A later version can repeat: train, diagnose on development data, generate examples, and retrain. Compare it with uniform allocation and a simple error-based policy under declared total budgets.

High error or high uncertainty alone does not establish that a scene is useful for learning. An adaptive strategy should demonstrate benefit on separate development systems and preserve coverage. Its final policy and stopping rule must be frozen before confirmation.

The current manuscript's fixed failure-weighted observation policy provides a relevant comparator, but it does not establish the value of adaptively generating source or foreground structure.

## 9. Evaluation and success criteria

Separate the questions of accuracy, explanation, and practical reliability.

| Outcome | Proposed measurement | Interpretation |
|---|---|---|
| Primary predictive outcome | Paired difference in parent-averaged angular MAE on a prespecified shifted population | Does explanation-guided synthesis improve the declared target task? |
| Systematic error | Signed radius bias by prespecified scene regime | Does the diagnosed bias decrease? |
| Severe failures | Fraction exceeding a justified error tolerance, plus upper-tail error | Are large failures reduced? |
| Explanation test | Prediction and error response to matched interventions | Does behaviour change in the way the explanation predicted? |
| Preservation of useful signal | Response when the true lens strength changes, with targets recomputed | Has reduced sensitivity merely made the predictor unresponsive? |
| Nominal performance | MAE under ordinary benchmark conditions | Does improvement incur an unacceptable nominal penalty? |
| Efficiency | Development, rendering, training, and inference costs | Is the additional machinery worthwhile? |

The target-changing control requires a separate validated rendering path that recomputes truth and dependent geometry. The current label-preserving counterfactual interface deliberately protects the radius field and should not be bypassed to create it.

A convincing positive result combines the predicted behavioural change, better independent predictive performance, and an acceptable nominal tradeoff. A predictor that becomes nearly constant may appear insensitive to nuisances while losing useful information.

Do not choose tolerances or metric weights from final scores. Define the intended use, a practically meaningful MAE improvement, an allowable nominal penalty, and any catastrophic-error threshold during planning. Values are not fixed in this report.

The primary superiority question can be D versus B. If D also claims an advantage from explanation beyond error targeting, D versus C must receive a declared confirmatory comparison and multiplicity treatment. Otherwise label that second comparison exploratory. An interval containing zero does not establish equivalence.

## 10. Data separation and statistical validity

Use distinct pools for training, diagnosis/selection, and final confirmation. The previously opened manuscript holdout may inform development context; it should not be presented as an untouched confirmation set for the new idea.

Split lens parents and reused source, foreground, and background assets before rendering variants. Keep all related views together. If shared assets create dependence between several systems, the design and uncertainty analysis must reflect that dependence.

Evaluate all arms on common test parents and report paired differences. Keep views grouped when resampling parents. Report training-seed variation separately; if the claim includes future training runs as well as future parents, use an analysis that accounts for both sources of variation.

Choose the number of independent parents and training seeds using pilot variability and the intended effect or precision. Additional views do not replace independent systems. No numerical sample-size or power claim is made here.

Preserve resolved configurations, asset lineage, generator/checkpoint identities, random seeds, prediction seals, and analysis inputs. Validate image units, flux treatment, PSF conventions, geometry, and preprocessing independently of predictive accuracy.

## 11. Relationship to prior work and the contribution boundary

| Prior work | Established precedent | Consequence for this proposal |
|---|---|---|
| [Hezaveh et al., 2017](https://arxiv.org/html/1708.08842) | CNN parameter prediction using real and clumpy sources with extensive on-the-fly observation variation | Synthetic diversity and fresh augmentation are established |
| [Pearson et al., 2019](https://arxiv.org/abs/1904.06199) | Performance depends on foreground light, colour, signal quality, and mass-light assumptions | Foreground dependence itself is not a new discovery |
| [Gawade et al., 2025](https://arxiv.org/abs/2404.18897) | Compares simulation realism strategies and real HSC model references; supports real-cutout injection in that setting | Real templates and backgrounds are serious baselines; reference fits remain imperfect |
| [Karchev et al., 2022, workshop](https://arxiv.org/abs/2211.04365) | Uses a diffusion-based source prior for reconstruction | Learned priors have precedent; this does not establish training-data utility for radius regression |
| [Barco et al., 2025, workshop](https://arxiv.org/abs/2511.04792) | Joint source and lens inference with a score-based prior on simulated cases | Generative inference is an existing direction, distinct from the proposed retraining comparison |
| [Wagner-Carena et al., 2024](https://arxiv.org/abs/2404.14487) | Sequentially refines simulation distributions for lensing population inference | Adaptive simulation allocation has direct lensing precedent |

The proposed contribution is the experimentally demonstrated link between **a behavioural diagnosis, a specific synthetic-data intervention, and an independently tested change in failure behaviour**. A targeted novelty review is still needed once the intervention and comparator are fixed. The existing [literature packet](literature_review/strong_lens_inference/README.md) is a scoping/narrative review, not proof that this exact study is unprecedented.

## 12. A bounded first study

The suggested first slice is one foreground-light dependence study using the existing physical simulator and a fixed CNN.

1. Verify data/checkpoint availability and reproduce a small development prediction set.
2. Qualify matched foreground-light interventions and useful-signal controls.
3. Use development-only results to record a candidate explanation and its rivals.
4. Freeze the scene library, generation rules, arm definitions, metrics, and analysis plan.
5. Run a paired pilot to determine feasibility and uncertainty requirements.
6. Fit the prespecified arms and evaluate a fresh confirmatory population once.
7. Report positive, negative, and inconclusive outcomes, including cases contradicting the explanation.

A learned generator is an optional next stage if morphology-related limitations warrant it. An adaptive loop follows only after the one-cycle intervention has a clear interpretation. Full-image fitting and real-observation validation can later assess practical usefulness.

## 13. What different outcomes would mean

| Result | Supported interpretation |
|---|---|
| D improves over B and C, with the predicted behavioural change | Evidence that the tested explanation provides useful synthesis guidance in the declared setting |
| D improves over A but not B | Additional diversity may explain the benefit; explanation-guided targeting has not established extra value |
| C and D perform similarly with precise enough uncertainty | Simple error targeting may be sufficient within the tested scope; equivalence still requires a specified margin and analysis |
| Sensitivity decreases but independent errors do not | Behaviour changed, but its practical benefit or the original diagnosis remains unresolved |
| Shifted performance improves while nominal failures increase | A tradeoff requiring explicit interpretation and the prespecified acceptance rule |
| No candidate dependence is supported | The initial explanation is unsupported; retain the result and investigate another cause on development data |

Synthetic tests directly assess model behaviour under the chosen generator. They do not by themselves establish that the simulator is unrealistic, that the model has learned the physical mechanism, or that predictions are accurate on real observations.

## 14. Decisions required before implementation

- Confirm foreground light as the first component and define plausible intervention ranges.
- Inspect whether the existing training distribution contains the suspected dependence; do not assume it does.
- Choose the intended use, preprocessing/noise baseline, and independent evaluation population.
- Specify how D turns a diagnosis into a rule distinct from error weighting.
- Set meaningful effect/tolerance criteria and a pilot-based sample-size rationale.
- Resolve asset availability, source lineage, total compute accounting, and target-changing control implementation.

These are scientific design choices for the next stage. Implementation and new experiment execution are future work.

## Workflow attribution and evidence note

This report was drafted with AI assistance using the scientific-writing, hypothesis-generation, and experimental-design skills. Their procedural library is described by Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). This attribution is separate from the scientific evidence for the proposal.

The report draws on local manuscript/code inspection and primary sources retrieved during the accompanying discussion. Source access ranges from abstracts to selected full-text sections; no complete new systematic review or independent reproduction of existing results was performed. Human scientific review remains necessary before this planning document is converted into a final protocol or publication.
