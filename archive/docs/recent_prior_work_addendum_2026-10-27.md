# Additional prior-work check during benchmark execution

Checked primary arXiv records on 2026-10-27. This supplements the immutable
original literature ledger. Search coverage is not proof of priority or an
exhaustive systematic review. Abstract-level findings below must not become
claims of detailed implementation equivalence.

| Primary source | What it establishes | Consequence for our contribution |
|---|---|---|
| [Swierc et al., Domain-Adaptive Neural Posterior Estimation, 2024](https://arxiv.org/abs/2410.16347) | Combines posterior estimation and unsupervised adaptation for a noiseless-to-survey-noise synthetic shift; reports improved inference and posterior coverage. | Adaptation for lens inference is established beyond point regression. Our scalar MAE cannot establish improved posterior coverage. Keep the common-backbone active MMD point port, and describe the limited task. |
| [Zhong et al., GPU-accelerated multi-survey strong-lens simulation, September 2026](https://arxiv.org/abs/2609.18180) | A differentiable PyTorch simulator includes SED-dependent multiband photometry and survey PSFs/limits/zero points. Identical physical lenses are simulated for KiDS, LSST and Euclid; a deblending application is evaluated. | Simulating one lens under multiple telescope conditions is already prior work. Our shared physical scenes and alternate renderer are controls, not a new simulation concept. No multiband realism or simulator-speed claim follows from our experiment. |
| [Holloway et al., Lens Modeling and Cosmological Inference from an Impure Sample, September 2026](https://arxiv.org/abs/2609.21699) | Studies neural lens inference and cosmological inference with false positives; the abstract reports 3.7% mean Einstein-radius precision with calibrated uncertainties and a contamination-aware cosmological framework. | Their precision percentage is not our MAE. Our preselected synthetic multiple-image population excludes contaminants and cannot establish better survey-wide inference or cosmological performance. Explicitly disclose conditional selection. |
| [Sayed et al., FlowLensing, revised November 2025](https://arxiv.org/abs/2510.07878) | Uses flow matching for conditional strong-lens image synthesis, including dark-matter classes/continuous parameters, and reports simulator speed improvements. | Image synthesis speed/fidelity is a separate task. Our numerical qualification checks a deterministic rendering pipeline; it does not outperform generative simulators or establish a novel image generator. |

Our potentially useful result remains narrow and measurable: **does a source-
failure-weighted allocation of a fixed observation-randomization budget beat
a tuned uniform allocation with exactly the same transformations, optimizer
and source exposures, on new paired lens parents under a declared PSF-family
challenge?** Does that allocation also improve a functioning MMD point baseline
at each explicitly named target-information level?

The representative-target MMD arms receive unlabeled Moffat images. For these
arms Moffat is a target domain, not an unseen family. It remains unseen in
optimization for source-only and Gaussian-target-only arms. Source-only
selection never uses target predictive accuracy. Do not merge these access
levels into a claim of unseen-family performance for every method.

Broader failure-guided/random augmentation precedents are documented in
[augmentation_prior_work_addendum.md](augmentation_prior_work_addendum.md).
The weighting recipe is an experimental intervention, with no generic
algorithmic first claim. A workshop contribution would be an audited common-
benchmark measurement and transfer boundary, supported by independent final
parents, paired multi-seed inference, substantive comparators and released
provenance. Whether its accuracy advantage exists is still unmeasured.

## Comparisons that remain prohibited without further experiments

- No ranking our historical 27.4% own-control reduction against another paper's
  percentage, R-squared, precision, coverage or unrelated dataset.
- No claim to beat NPE, MVE uncertainty, DANN, real-HSC modeling, multiband
  simulation, lens detection or cosmological population inference from the
  present point-only matrix.
- No claim that near-identical A/B images establish astrophysical robustness.
- No attribution of an allocation gain to a new CNN architecture: it is shared.
- No universal superiority claim if one or several required comparisons fail.

Keep the predeclared primary and guardrails even if a different contrast is
more favorable. Report a negative or conditional result with the same care.
