# Step 3 — claim and prior-work comparison matrix

Date: 2026-11-01  
Purpose: decide what the workshop paper can claim from the completed artifacts.

## Recommended paper question

**When CNN architecture, optimizer, update count, initialization, source-parent
schedule, and source-image exposure are matched, does source-only variation of
observation conditions improve Einstein-radius prediction on parent-disjoint
synthetic lens systems?**

This is an attribution and reliability question. It is not a claim that the
training policy is a new architecture, that it beats every augmentation method,
or that it transfers to real telescope data.

## Evidence available now

| Comparison | Result | Interpretation |
|---|---:|---|
| Canonical C, four-stress composite | 0.141251 arcsec | Fixed-budget baseline |
| Uniform variation G, four-stress composite | 0.111507 arcsec | Primary intervention |
| Failure-weighted variation R, four-stress composite | 0.107385 arcsec | Secondary policy |
| G − C | -0.029744 arcsec; 95% parent interval [-0.045725, -0.012141] | Reliable improvement under this synthetic benchmark |
| R − G | -0.004122 arcsec; 95% interval [-0.026208, 0.009341] | No reliable evidence that weighting beats uniform variation |
| Renderer A/B image identity | 1,350/1,350 pairs exactly equal | No simulator-induced image shift in rapid-v2 |

The older 27.4% result is retained as historical context, but its canonical and
randomized procedures received different numbers of optimizer updates and image
exposures. It must not be presented as the causal estimate of randomization.

## Prior-work comparison

| Study | Target and setting | Shift / method | Information access and metric | Direct numerical comparison? | What we may claim |
|---|---|---|---|---|---|
| Hezaveh et al. (2017) | CNN lens-parameter inference from simulated strong lenses | Noise, PSF and image-processing variation | Supervised synthetic training; different targets/data/metrics | No | CNN inference and observational variation are established foundations |
| Pearson et al. (2019) | Multiple lens parameters in LSST/Euclid-like simulations | Lens light, colour, S/N and mass/light scatter | Different population, target set and evaluation summaries | No | Training diversity and sensitivity analysis are established; our result is a controlled one-target extension |
| Swierc et al. (2023) | Einstein-radius regression with deeplenstronomy domains | DANN/MMD domain adaptation from source to DES-like target | Unlabeled target images used during adaptation; reports (R^2) | No | Closest methodological context; our source-only result is a different information-access setting, not a win over DANN/MMD |
| Agarwal et al. (2024/25) | Einstein-radius inference with mean/variance prediction | MMD adaptation and uncertainty estimation | Unlabeled target images; signed residuals, likelihood and calibration metrics | No | Their uncertainty/adaptation capability is not tested by our point-MAE CNN |
| Gawade et al. (2025) | Einstein radius and ellipticity for HSC-oriented simulations/real lenses | Increasing simulation realism and real-data comparison | Real HSC evidence and different radius conventions | No | They provide stronger simulation-to-real evidence than we do; no real-data superiority claim is permitted |
| Galan et al. (2024) | Classical lens modeling across independently simulated/code-generated data | Cross-code comparison with explicit conventions | Different task and classical methods | No | Code switching alone is not novel; our A/B result is only a consistency control |
| Mehta et al. (2019/20), RandAugment, AugMix | General domain randomization/augmentation | Non-lensing active or corruption-focused policies | Different tasks and objectives | No | Failure-guided allocation is a benchmark-specific policy, not a first active-randomization method |
| Filipp et al. (2025), Poh et al. (2025) | Population posterior inference or multi-parameter posterior estimation | Distribution shift, NPE/BNN and calibration | Different targets and uncertainty objectives | No | Cite as robustness/posterior context; scalar MAE cannot establish superiority |

## Claim-evidence map

| Candidate sentence | Status | Decision |
|---|---|---|
| “Observation-condition variation improves robustness under a matched CNN/update/exposure budget.” | Supported by equal-budget final result | Use as the main result, bounded to the tested synthetic generator and four stresses |
| “Failure-weighted sampling is better than uniform variation.” | Not supported; R−G interval includes zero | Do not claim; report as an inconclusive secondary comparison |
| “The method transfers across simulators.” | Not supported; A/B image products are identical | Remove from title, abstract and conclusion |
| “Our method beats Swierc or Agarwal.” | Not tested on a common benchmark or exact reproduction | Do not claim |
| “Our method works on real telescope images.” | No trustworthy per-image reference labels or scored real benchmark | Do not claim |
| “The study contributes a reproducible robustness benchmark and controlled attribution.” | Supported, with narrow scope | Use as the methodological contribution |

## Submission positioning

Recommended title:

> **Equal-budget observation-condition randomization improves synthetic
> Einstein-radius regression robustness**

Recommended contribution bullets:

1. A parent-disjoint, target-free evaluation that matches optimizer updates,
   initialization, source-parent schedule and source-image exposure across
   canonical and varied-condition CNN training.
2. A statistically paired result showing a -0.029744 arcsec four-stress MAE
   change for uniform variation versus canonical training, with a parent-level
   95% interval excluding zero.
3. A negative result for the proposed failure-weighted allocation relative to
   uniform variation, preventing the paper from mistaking a policy heuristic
   for a demonstrated algorithmic advantage.
4. An explicit renderer audit showing that the current AutoLens/Lenstronomy
   release is a numerical equivalence control, not independent simulator
   transfer evidence.

Recommended abstract boundary:

> We study whether source-only observation-condition variation improves a CNN
> estimate of the Einstein radius when optimization and source exposure are
> matched. On 150 parent-disjoint synthetic systems, uniform variation reduced
> four-stress MAE by 0.029744 arcsec relative to canonical training (parent
> bootstrap 95% interval [-0.045725, -0.012141]). A failure-weighted allocation
> did not show a reliable gain over uniform sampling. The benchmark uses one
> synthetic lens/observation generator; a paired AutoLens/Lenstronomy audit
> produced identical saved images, so we make no simulator-transfer or
> real-observation accuracy claim.

## Required manuscript edits

- Make the equal-budget result primary and label the 27.4% result as the
  earlier budget-confounded comparison.
- Add the R-versus-G null result; omitting it would overstate the intervention.
- Replace “cross-simulator transfer” with “renderer-equivalence control.”
- Include the prior-work table with target, generator, shift, target-data access,
  metric and comparability fields; never rank incompatible MAE, (R^2), signed
  residual or coverage numbers.
- State that the contribution is a controlled synthetic robustness measurement,
  not a new CNN architecture, universal augmentation method, real-data result,
  or published-method leaderboard win.
