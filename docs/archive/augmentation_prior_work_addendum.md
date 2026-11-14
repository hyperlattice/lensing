# Broader augmentation precedents and the contribution we can defend

Research date: 2026-10-27. This supplements the immutable earlier lensing
[prior-work ledger](prior_work_and_novelty_analysis.md). These primary sources
were checked before any new model-performance comparison was run.

## Established ideas

[Mehta et al., Active Domain Randomization](https://arxiv.org/abs/1904.04762)
(2019 preprint; see also [CoRL proceedings](https://proceedings.mlr.press/v100/mehta20a.html))
learns a simulation-parameter sampling strategy emphasizing informative
variations rather than uniform domain randomization. Its experiments concern
reinforcement-learning/robot tasks. Our inference: preferentially training on
hard simulator conditions is an established general idea. A fixed positive-
degradation weighting rule for lens images must not be presented as inventing
active domain randomization or as an exact reproduction of that algorithm.

[Akkaya et al., Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113)
(2019) uses automatic domain randomization to expand environment difficulty
and reports simulation-to-real robotic performance. Our fixed seven-cell,
source-scored policy does not adapt ranges or establish real-world transfer.
Our inference: automatically choosing difficult simulated conditions and
source-only transfer cannot serve as generic priority claims for this project.

[Cubuk et al., RandAugment](https://arxiv.org/abs/1909.13719) (2019 preprint)
reduces automated image-augmentation search complexity and studies image
classification/detection. Our inference: uniform random augmentation and
limiting search overhead are strong precedents. Our physical-condition G
policy is not an exact RandAugment implementation. A primary R-versus-G win
tests allocation over our identical seven cells, not superiority over every
learned/compositional augmentation policy. T supplies a separate historical
compound-condition comparator, but does not exhaust the wider literature.

[Hendrycks et al., AugMix](https://arxiv.org/abs/1912.02781) (2019/2020)
studies robustness to unforeseen image corruptions and classifier uncertainty.
Our inference: testing corruptions absent from training is established, and
point-regression MAE cannot establish superior uncertainty calibration to
AugMix. A regression/physical-observation port would be a separate method,
not an exact reproduction of its classifier training objective.

## Consequences for our claims

The potential contribution is an empirical lens-inference comparison under
controlled physical conventions, parent separation, equal candidate source
exposure and explicit target access. The question is whether a fixed
source-failure allocation adds a measurable benefit beyond uniform sampling,
fresh noise, a compound-condition recipe, and competent MMD ports on this
benchmark. The weighting formula alone is not a demonstrated new algorithm.

Correctly matched A/B renderings are expected to agree. Our numerical controls
therefore establish cross-code consistency and remove convention artifacts;
they do not independently create a difficult astrophysical distribution shift.
The held-out Moffat observing family supplies that challenge. Equivalent A/B
accuracy is not evidence of robustness to different mass/source populations,
empirical telescope PSFs, real galaxy morphology or real observations.

Keep the source-only and target-aware MMD comparisons separate. Reproducing an
author kernel on a common single-band backbone is a method port. Neither a
synthetic win nor the pinned-code audit proves that a published result was
invalid or that our method has better real-data performance.

Report the reference-training/scoring cost used to build R's policy in
addition to each candidate's matched training budget. Bootstrap uncertainty
over candidate seeds and final parents is conditional on the fixed source
sample, source-reference ensemble, augmentation policy and tuning procedure;
it does not capture every source-dataset or policy-construction uncertainty.

No first-of-its-kind claim, global state-of-the-art augmentation claim or
universal superiority claim is supported by this review. A submission-ready
result still needs the frozen comparisons, uncertainty/guardrails and an
honest claim-evidence map. Publication acceptance is not implied by passing
software or numerical tests.
