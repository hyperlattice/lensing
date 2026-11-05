# One-parameter model landscape for strong-lens inference

This report maps the model families that are realistically useful for this
project. The immediate task remains deliberately narrow: infer one continuous
parameter, `theta_E_arcsec`, from one simulated lens image. Every design below
is judged both as a one-target model and as a possible foundation for later
inference of lens ellipticity, orientation, shear, source position, or a joint
posterior.

This is a comprehensive landscape of relevant approaches, not a claim that
every machine-learning algorithm should be tried. Models that do not match the
image geometry, data volume, or scientific question are included only when
they provide a useful control or a clear future path.

## 1. Current experimental boundary

The starting benchmark is `theta-e-cnn-seeds-v1`:

| Constraint | Current state |
|---|---|
| Target | One scalar: `theta_E_arcsec` |
| Input | One normalized 256 x 256 simulated image |
| Training data | 700 parent scenes |
| Validation data | 150 parent scenes |
| Reference model | Four-block CNN, 248,409 trainable parameters |
| Reference loss | Huber with delta `1.0` on normalized targets |
| Reference validation MAE | `0.09210 +/- 0.00872` arcsec over three seeds |
| Best completed loss ablation | Huber delta `0.25`: `0.08535` mean validation MAE |
| Model-selection data | Validation only; the existing test split is not used for tuning |

The small training set changes the model ranking. Compact convolutional,
equivariant, feature-based, and physics-assisted methods are more defensible
next experiments than a large transformer trained from scratch.

All proposed models must initially obey the same contract:

```text
image and approved observational channels
                 |
              encoder
                 |
      one-target prediction head
                 |
          theta_E_arcsec only
```

The encoder and prediction head should be separate modules. Expansion then
replaces the head without silently changing the image representation during a
one-target comparison.

## 2. Executive recommendation

The most useful sequence is:

1. Add non-neural and radial-feature controls so the CNN has meaningful
   scientific baselines.
2. Compare a small residual CNN with approximately the same parameter budget
   as the current CNN.
3. Compare a rotation/reflection-equivariant residual CNN because Einstein
   radius is unchanged by image rotation.
4. Test a neural initializer followed by a one-parameter physical lens-model
   refinement. This is the most practical first physics/ML hybrid.
5. Build a differentiable forward-model hybrid in which predicted parameters
   must also reconstruct the observation through the lens equation.
6. Add calibrated uncertainty: first a heteroscedastic head or deep ensemble,
   then neural posterior estimation when the simulation budget is much larger.

A conventional physics-informed neural network (PINN) is not the leading
choice. Classic PINNs learn a field by minimizing differential-equation
residuals at collocation points. This project is instead an inverse problem:
an image contains a lensed, unknown source convolved with a PSF and corrupted
by noise. The useful analogue is a **physics-constrained inverse model** with a
differentiable lens renderer, not a generic PDE PINN. The distinction follows
the original [PINN formulation](https://arxiv.org/abs/1711.10561) and the
[differentiable strong-lensing formulation](https://arxiv.org/abs/1910.06157).

## 3. Model-family comparison

Priority is relative to the present data and target, not a universal judgment
about the model family.

| Family | One-parameter implementation | Later expansion | Main strength | Main risk | Priority |
|---|---|---|---|---|---|
| Constant mean/median | Predict the training-set center | None | Required sanity floor | No image information | Control |
| Ridge/elastic net | Regress from moments or radial features | One regressor per target | Transparent and data-efficient | Feature ceiling | Control |
| Gradient-boosted trees | Regress from engineered features | Multi-output or per-target | Strong nonlinear tabular baseline | Depends on feature design | High control |
| Radial-profile MLP/1D CNN | Encode azimuthal profiles and radial moments | Add channels/heads | Directly represents angular scale | Can discard asymmetry | High |
| Wavelet scattering | Fixed multiscale features plus ridge/MLP | Replace output head | Stable small-data representation | Fixed features may miss task detail | Medium-high |
| Current compact CNN | Existing scalar regression head | Multi-head or density head | Established baseline | Regression to the mean at range edges | Reference |
| Small residual CNN | Same-size residual blocks plus scalar head | Natural multi-head encoder | Better optimization at controlled capacity | Still learns symmetries from data | Highest next |
| Rotation-equivariant CNN | Invariant scalar readout for `theta_E_arcsec` | Equivariant heads for angles/vectors | Encodes known rotational symmetry | Requires careful head design | Highest next |
| Multi-scale CNN | Pyramid/dilated features plus scalar head | Shared pyramid, multiple heads | Captures compact and extended arcs | More design variables | High |
| Coordinate CNN | Add normalized `x` and `y` channels | Same encoder, wider head | Makes image geometry explicit | Sensitive to centering/domain shift | Medium |
| Compact modern ConvNet | Small DenseNet/EfficientNet/ConvNeXt-like encoder | Straightforward | Tests improved convolutional blocks | Easy to over-size comparison | Medium |
| U-Net encoder | Pool bottleneck or add scalar branch | Add masks and parameter heads | Couples localization with regression | Needs auxiliary target/loss to justify decoder | Medium-high |
| Keypoint model | Detect image/arc positions, then regress radius | Add geometry parameters | Interpretable intermediate representation | Weak/diffuse arcs and blends | Medium-high |
| Set transformer/GNN | Regress from a variable set of detected components | Natural multi-object/vector output | Handles doubles/quads explicitly | Detection errors become bottleneck | Medium |
| Polar-image network | CNN/1D model in polar coordinates | Multi-channel polar features | Makes radial scale explicit | Center choice and interpolation loss | Medium-high |
| CNN-attention hybrid | Convolutions plus small attention blocks | Wider/vector heads | Combines local arcs and global layout | More data and tuning | Medium |
| Vision Transformer/Swin | Patch tokens plus scalar token/head | Natural vector/density head | Flexible global relationships | Too data-hungry at current scale | Defer |
| Autoencoder/VAE regressor | Latent code plus scalar head | Structured latent/decoder | Can learn compact scene representation | Reconstruction may ignore target detail | Medium-low |
| Self-supervised encoder | Pretrain on unlabeled images, then regress | Reuse for many parameters | Useful if abundant unlabeled images exist | Pretext task/domain may not align | Future |
| Gaussian process | GP on compact learned/engineered features | Multi-output GP | Data-efficient uncertainty | Does not scale to raw pixels or large data | Medium control |
| Heteroscedastic neural regressor | Predict mean and variance of one scalar | Diagonal or structured covariance | Simple input-dependent uncertainty | Gaussian/unimodal assumption | High after point model |
| Quantile regressor | Predict selected `theta_E_arcsec` quantiles | Quantiles per target | Distribution-free intervals | Quantile crossing and no full density | Medium-high |
| Mixture-density network | Predict a 1D Gaussian mixture | Multivariate mixtures | Represents multimodality | Unstable components in higher dimensions | Medium |
| Bayesian NN / MC dropout | Distribution over predictions/weights | Same network, wider output | Epistemic-uncertainty attempt | Approximation can be miscalibrated | Medium |
| Deep ensemble | Train independent point/probabilistic models | Same for vector targets | Strong practical uncertainty baseline | Multiplies training and storage cost | High |
| Neural posterior estimation | Conditional normalizing flow posterior | Direct joint posterior over parameters | Full amortized posterior | Needs many realistic simulations | Highest long-term |
| Neural likelihood estimation | Learn likelihood, sample with chosen prior | Natural vector inference | Prior can change after training | Posterior sampling cost | Long-term |
| Neural ratio estimation/MNRE | Learn likelihood-to-evidence ratio | Target selected marginals | Well suited to a chosen parameter subset | Specialized training/calibration | Long-term |
| Neural initializer + physical fit | CNN estimate initializes one-parameter optimizer | Add physical fit parameters | Practical, interpretable hybrid | Forward-model mismatch | Highest physics step |
| Differentiable renderer hybrid | Supervised estimate plus image/physics residual | Joint lens/source latent variables | Enforces observational consistency | Degeneracies and simulator mismatch | Highest research step |
| Learned source + exact lens model | Neural source decoder inside physical renderer | Full lens/source inference | Separates source morphology from lens physics | Complex identifiability problem | Long-term |
| Simulator emulator/neural operator | Surrogate forward simulation, then inference | Fast multi-parameter simulation | Accelerates likelihood/SBI pipelines | Not itself an inverse estimator | Supporting research |
| Conventional PDE PINN | PDE residual plus scalar output | Field and parameter inference | Useful when a governing PDE is directly observed | Poor fit to current image inverse task | Do not prioritize |

## 4. Deterministic point-estimation models

### 4.1 Non-neural scientific controls

The project should retain a training-set mean or median predictor and add two
stronger controls:

- A ridge or elastic-net regressor on flux moments, radial intensity bins,
  detected-component separations, and simple shape statistics.
- A gradient-boosted tree model on exactly the same features.

These models answer whether the neural network is learning more than an
obvious image-scale statistic. Their failures are also easier to diagnose.
They can expand by fitting one regressor per new target, but they are not the
preferred route to a correlated joint posterior.

### 4.2 Radial, polar, and scattering representations

Einstein radius is fundamentally a scale-like quantity. A radial-profile
model can compute several profiles rather than a single azimuthal average:
total intensity, maximum intensity, angular variance, and mask coverage per
radius. A small MLP or 1D CNN then predicts the scalar target.

A polar-image CNN preserves azimuthal structure while making radius an
explicit image axis. It is more expressive, but the assumed lens center and
the Cartesian-to-polar interpolation must be varied in robustness tests.

Wavelet scattering is a fixed multiscale representation with stability to
small deformations. It is a credible low-data control between hand-crafted
features and learned CNN features; see the original
[scattering-transform paper](https://arxiv.org/abs/1203.1513).

### 4.3 Residual and multi-scale CNNs

The immediate learned candidate should be a small residual CNN whose parameter
count is close to 248,409. Residual connections improve optimization without
requiring a large pretrained network; the architectural principle comes from
[Deep Residual Learning](https://arxiv.org/abs/1512.03385).

A subsequent multi-scale candidate can merge features at two or three spatial
resolutions. This is scientifically plausible because compact image systems
and extended arcs express the same radius differently. It should be a separate
ablation from residual blocks so any gain remains attributable.

### 4.4 Rotation-equivariant CNNs

Rotating the whole observation should not change `theta_E_arcsec`. A group-
equivariant CNN can encode rotations, and optionally reflections, by weight
sharing rather than making the model relearn them from 700 scenes. The core
method is described in [Group Equivariant Convolutional
Networks](https://arxiv.org/abs/1602.07576).

For the current target, global pooling should produce a rotation-invariant
scalar. Later orientation inference must not use that invariant head. Lens
orientation should use a covariant/equivariant head, ideally represented by
`cos(2 * phi_rad)` and `sin(2 * phi_rad)` to respect the 180-degree symmetry.

### 4.5 U-Net, keypoints, sets, and graphs

A U-Net-like encoder becomes useful if the simulator can provide arc masks,
critical-curve masks, or image-position heatmaps. The scalar prediction is made
from the bottleneck while the decoder receives an auxiliary localization loss.
The original [U-Net](https://arxiv.org/abs/1505.04597) demonstrated the value of
localization-aware training in a small-data image setting, although this
project would need its own lens-specific validation.

A more explicit pipeline is:

```text
image -> heatmaps/keypoints -> detected component set -> scalar geometry model
```

Simple analytic features or a set transformer/GNN can consume the detections.
This naturally handles a variable number of images and later predicts source
position, ellipticity, or shear. It is interpretable, but a missed arc cannot
be recovered by the downstream model. The raw-image model must remain a
comparison.

### 4.6 Transformers and attention

A small attention block after a convolutional stem may help relate separated
lensed images. A full Vision Transformer is a lower-priority experiment at
the current scale. The original [Vision Transformer](https://arxiv.org/abs/2010.11929)
reported its strongest results with large-scale pretraining, while this project
currently has only 700 training parents. Reconsider transformers after the
simulation set grows, after scientifically compatible pretraining, or after a
compact CNN-attention hybrid establishes a need for longer-range interactions.

## 5. Physics-informed and physics-constrained approaches

### 5.1 Why a standard PINN is not the default

A standard PINN approximates an unknown function and evaluates a differential-
equation residual at selected coordinates. The present input is not a sparse
sample of an unknown PDE solution. It is a detector image generated by several
stages:

```text
source light -> lens mapping -> detector sampling -> PSF -> noise and masks
```

The unknown source morphology creates genuine degeneracies with lens
parameters. Adding the lens equation as an extra scalar penalty is therefore
not sufficient: the training graph must model or marginalize the source and
the observation process.

`LensPINN` provides an interesting precedent for combining a learned vision
model with the lens equation, but its reported task is lens-class/morphology
classification rather than direct `theta_E_arcsec` regression. It should be
treated as a design example, not as evidence that a PINN will improve this
project's accuracy. See the [LensPINN workshop
paper](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf).

### 5.2 Neural initializer plus physical refinement

This is the lowest-risk physics hybrid:

1. The image encoder predicts an initial `theta_E_arcsec`.
2. A conventional forward lens model holds the allowed nuisance assumptions
   fixed or fits a small declared nuisance set.
3. A bounded optimizer refines `theta_E_arcsec` by minimizing the image
   residual.
4. The report compares the neural initialization, refined result, convergence
   failures, runtime, and residual structure.

Begin by refining only the one target. Later releases can add lens center,
ellipticity, shear, and source parameters one at a time. The neural estimate
reduces optimizer search cost; the physical stage provides an interpretable
consistency check. It can still be biased when the source, PSF, or noise model
is wrong, so forward-model mismatch must be a named robustness test.

### 5.3 Differentiable forward-model hybrid

The more ambitious design makes rendering part of training:

```text
encoder(image) -> theta_E, lens nuisance latent, source latent
                              |
                  differentiable lens renderer
                              |
                     reconstructed image
```

A first loss can be written as

```text
L = L_theta
  + lambda_image * L_variance_weighted_image
  + lambda_lens * L_lens_equation
  + lambda_prior * L_latent_prior
```

`L_theta` remains the supervised one-parameter loss. The renderer must use
predicted or inferred nuisance values; feeding hidden ground-truth simulator
parameters into it would leak information unavailable at inference time.
Image residuals should use approved masks and calibrated variance when those
products are valid.

Before a full experiment:

1. Verify gradients of the renderer against finite differences.
2. Demonstrate recovery on a noiseless, known-source control.
3. Add source-latent uncertainty and PSF/noise stages.
4. Compare supervised-only and physics-constrained training with identical
   encoders and seeds.
5. Stress test deliberate source, PSF, and noise mismatch.

Differentiable strong-lens modeling has been demonstrated with exact lens
physics and learned source representations in
[Differentiable Strong Lensing](https://arxiv.org/abs/1910.06157). This makes
the hybrid scientifically credible, but not automatically superior under the
current simulator or data volume.

### 5.4 Learned source models and simulator surrogates

A VAE or other learned source decoder can generate the unlensed source while
an exact differentiable lens operator performs the mapping. This is an
inverse-graphics model, useful when expanding to joint lens/source inference.
Its main challenge is identifiability: a flexible source decoder can absorb
lens-model errors.

A neural operator or emulator addresses a different bottleneck. It learns a
fast approximation to the forward simulator and can make likelihood-free
inference or large simulation campaigns cheaper. It should not be compared as
though it directly predicts Einstein radius; its quality is measured by
forward fidelity and downstream posterior bias.

## 6. Probabilistic inference and uncertainty

Point accuracy and uncertainty quality are separate questions. Do not promote
a probabilistic model only because its mean prediction has a lower MAE.

### 6.1 Lightweight uncertainty heads

- **Heteroscedastic Gaussian:** predict `mu_theta` and `log_variance_theta`.
  This is the easiest input-dependent uncertainty baseline, but it assumes a
  unimodal conditional distribution.
- **Quantile regression:** predict, for example, the 5th, 16th, 50th, 84th,
  and 95th percentiles. It gives useful intervals without a Gaussian
  assumption; use a monotonic parameterization or a crossing penalty.
- **Mixture-density network:** predict mixture weights, means, and scales for
  a one-dimensional density. It can represent ambiguity, but component
  collapse and expansion to many dimensions require care.
- **Gaussian process on learned features:** freeze or jointly learn a compact
  encoder, then place a GP over its embedding. This is credible for 700
  examples but not for full-resolution pixels directly.

### 6.2 Deep ensembles and Bayesian neural networks

Deep ensembles are the preferred practical epistemic-uncertainty baseline:
train independently initialized models and combine their predictive
distributions. The existing three seeds do not automatically constitute a
validated ensemble; ensemble inference, calibration, and the frozen member set
must be declared explicitly. The method is described in
[Simple and Scalable Predictive Uncertainty Estimation using Deep
Ensembles](https://arxiv.org/abs/1612.01474).

Bayesian neural networks and MC dropout may also be tested, particularly to
separate uncertainty caused by limited training coverage. Their posterior
approximations must be checked empirically; the word "Bayesian" is not a
calibration guarantee. Strong-lens work has combined BNN outputs with
hierarchical inference; see
[Hierarchical inference with Bayesian neural networks](https://doi.org/10.3847/1538-4357/abdf59).

### 6.3 Simulation-based inference

Simulation-based inference (SBI) is the best long-term route when the desired
output is a posterior rather than a point estimate. A neural posterior
estimator learns

```text
p(theta_E | image)
```

now, and later

```text
p(theta_E, q, phi, shear, source_position, ... | image).
```

Normalizing flows are common density models for this purpose; see the
[normalizing-flow review](https://arxiv.org/abs/1912.02762). Strong-lensing
parameter inference with neural density estimation is demonstrated in
[Simulation-Based Inference of Strong Gravitational Lensing
Parameters](https://arxiv.org/abs/2112.05278).

Three variants deserve separate experiments:

- **Neural posterior estimation (NPE):** direct, fast amortized posterior
  sampling, but tied to the training prior unless corrected or retrained.
- **Neural likelihood estimation (NLE):** learns the observation likelihood
  and permits prior changes, but needs posterior sampling at inference.
- **Neural ratio estimation (NRE/MNRE):** learns likelihood-to-evidence ratios;
  marginal NRE is attractive when only `theta_E_arcsec` is currently needed
  from a larger nuisance space. Strong-lens marginal ratio estimation is
  discussed in [this MNRAS study](https://academic.oup.com/mnras/article/518/2/2746/6821314).

SBI should not be trained seriously on only the current 700 scenes. Its next
step is a simulation-budget study with coverage diagnostics, prior-predictive
checks, and deliberately held-out simulator settings. A recent comparison of
NPE and BNN inference for one, five, and twelve simulated strong-lens
parameters is available in
[Deep inference of simulated strong lenses](https://arxiv.org/abs/2501.08524).

## 7. Prediction-head design for safe expansion

Keep a common encoder interface and make the head explicit:

| Stage | Head output | Loss | Interpretation |
|---|---|---|---|
| Current | One unrestricted or bounded scalar | Huber/MAE/MSE/log-cosh | Point estimate |
| First uncertainty | Mean plus positive variance | Gaussian NLL | Unimodal predictive density |
| Robust intervals | Ordered quantiles | Pinball loss | Conditional intervals |
| Multimodal scalar | Mixture or 1D flow parameters | Negative log likelihood | Full 1D density |
| First vector model | Normalized parameter vector | Per-target masked losses | Joint point estimate |
| Final probabilistic model | Conditional multivariate flow | Density objective | Joint posterior |

Bounded outputs should reflect a versioned scientific prior, not a convenient
hard-coded training range. Always report the out-of-prior prediction fraction
even if a bounded transform makes it zero, because clipping can hide failure.

For later parameters:

- Predict `q` with a transform consistent with its allowed interval.
- Predict orientation as `cos(2 * phi_rad)` and `sin(2 * phi_rad)`.
- Predict shear as Cartesian components rather than magnitude plus a raw
  wrapped angle.
- Normalize each scalar using training-only statistics.
- Use missing-target masks if observational labels are incomplete.
- Report each target in physical units; do not rank a vector model only by an
  averaged normalized loss.

## 8. Metrics by output type

All candidates retain the current point-estimate metrics: MAE, median absolute
error, RMSE, P90/P95 absolute error, signed bias, radius-bin error, threshold
accuracy, calibration slope/intercept, subgroup results, prediction range,
runtime, and parameter count.

Add metrics according to the model:

| Model output | Additional required checks |
|---|---|
| Probabilistic scalar | Negative log likelihood, CRPS, 50/68/90/95% coverage, interval width, PIT or rank histogram |
| SBI posterior | Simulation-based calibration ranks, posterior coverage, posterior predictive checks, prior-sensitivity study |
| Keypoint/set model | Detection completeness/purity, localization error, error by true and detected multiplicity |
| Physical hybrid | Optimizer convergence, normalized residual map, image likelihood or chi-square only when noise is calibrated, failure rate |
| Reconstruction model | Held-out image residuals plus target metrics; reconstruction quality alone is insufficient |
| Equivariant model | Prediction consistency under controlled rotations/reflections |
| Simulator emulator | Forward-image fidelity and bias induced in downstream parameter inference |

Continue using paired three-seed validation comparisons and the existing
acceptance gate. Probabilistic calibration adds requirements; it does not
replace the MAE and tail-error gate.

## 9. Controlled implementation queue

Each experiment initially predicts only `theta_E_arcsec`, uses the frozen
training/validation parents and three established seeds, and changes one
declared factor relative to its parent.

| ID | Candidate | Controlled question | Advance when |
|---|---|---|---|
| M00 | Current CNN | Recorded reference | Already available |
| M01 | Ridge and boosted trees on fixed features | How much signal is explained by simple geometry? | Reproducible control report |
| M02 | Radial-profile MLP/1D CNN | Is explicit radial structure competitive? | Beats non-neural controls or diagnoses CNN behavior |
| M03 | Parameter-matched residual CNN | Do residual blocks improve scalar accuracy? | Passes current MAE/tail/bias gate |
| M04 | Rotation-equivariant residual CNN | Does encoded symmetry improve data efficiency? | Improves gate and rotation consistency |
| M05 | Multi-scale CNN | Are scale-separated features useful? | Paired gain over selected convnet |
| M06 | U-Net/keypoint auxiliary model | Does explicit localization help? | Target gain without unacceptable detection failures |
| M07 | Neural initializer plus physical refinement | Does a lens-model fit improve the estimate? | Better error/tails with acceptable convergence/runtime |
| M08 | Differentiable renderer hybrid | Does image consistency improve inference? | Better validation and mismatch robustness |
| M09 | Heteroscedastic head and deep ensemble | Are intervals calibrated and useful? | Coverage and sharpness beat simple residual baselines |
| M10 | One-dimensional NPE/NRE pilot | Can SBI recover a calibrated scalar posterior? | Passes SBC/coverage on a larger simulation release |
| M11 | First multi-parameter head/posterior | Does the selected encoder generalize? | Only after every added target has a scalar benchmark |

M01 and M02 may be implemented alongside M03 because they are independent
controls, but none should use the test split for model choice. M07 should
precede M08: it isolates the benefit of physical refinement before adding the
complexity of end-to-end differentiable training.

## 10. Models to defer

Do not prioritize the following in the next implementation block:

- A large pretrained or from-scratch Vision Transformer.
- A generic stock ResNet whose parameter count dwarfs the baseline.
- A conventional PDE PINN with no complete source/PSF/noise forward model.
- A high-dimensional normalizing-flow posterior trained on only 700 scenes.
- A neural operator presented as though it were a direct inverse model.
- A multi-parameter network before each new target has a scalar baseline and
  symmetry-appropriate metric.

These are deferrals, not permanent exclusions. Data scale, simulation realism,
or a new scientific objective can change their priority.

## 11. Definition of a useful winner

The next winner is not simply the architecture with the smallest aggregate
MAE. It must:

1. Improve paired three-seed validation MAE by the predeclared margin.
2. Avoid worse P95 error and edge-radius bias.
3. Remain stable under rotation, PSF, noise, source-morphology, multiplicity,
   and P100 stress checks appropriate to the model.
4. Record parameter count, runtime, convergence failures, and provenance.
5. Preserve a clean one-target interface that can accept a new prediction head.
6. Avoid truth-only simulator inputs that will be absent from observations.
7. Make synthetic-domain claims only until real-domain validation is defined.

Under those rules, the recommended immediate implementation is M01-M04, with
M03 as the main learned challenger. The recommended physics path is M07 then
M08. The recommended posterior path is M09 then M10 after the simulation
budget and observation model are expanded.
