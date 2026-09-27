# Reliability study baseline state

**Audit date:** 2026-09-25  
**Purpose:** record the repository and evidence state before new reliability experiments.

## Repository snapshot

- Current branch at audit: `pre-ml-completion`.
- HEAD at audit: `30874ee` (`add project structure`), seven commits ahead of `origin/pre-ml-completion`.
- Tracked files were clean at audit. The user-provided reference `gravitational_lensing_field_analysis_and_contribution_plan (1).md` was untracked and is preserved as supplied.
- Python: 3.13.6.
- The source plan's dedicated branch is now active as `reliability-study-v1`; it was created on 2026-09-25 after the required explicit escalation because `.git` is read-only in this environment.

## Existing artifacts identified

| Artifact | Location | Baseline state / use |
|---|---|---|
| Engineering synthetic release | `data/manifests/sim-eng1000-v1/` and local `data/simulations/sim-eng1000-v1/` | 1,000 independent SIE parents; 700 train / 150 validation / 150 test; ancestry leakage check passed; primary target `theta_E_arcsec`; P100 separate from training. Manifest SHA-256: `189c1351e768c6a6563c2af3243419df92d39f14a9670ff6385b693405e06a44`. |
| Original CNN | `results/ml/theta-e-cnn-baseline-v1/` | Image-only four-block CNN; 248,409 parameters; one 256 x 256 image; one-time historical test evaluation exists. |
| Three-seed baseline | `results/ml/theta-e-cnn-seeds-v1/` | Validation only; seeds 12001-12003; mean MAE 0.092097 arcsec, sample SD 0.008719; mean P95 0.237815 arcsec. Radius-bin mean bias is +0.0466 arcsec for 0.5-1.0 arcsec and -0.0507 arcsec for 1.5-2.0 arcsec. |
| Huber ablations | `results/ml/comparison-huber-025-v1/`, `results/ml/comparison-huber-050-v1/` | Validation-only comparisons; neither passes its conservative gate. Delta 0.25 mean MAE 0.085348; delta 0.50 mean MAE 0.089680. Not adopted as a new baseline. |
| Historical opened test | `results/ml/theta-e-cnn-baseline-v1/evaluation-test/` | 150 samples; MAE 0.0876, RMSE 0.1292, bias -0.0090, R2 0.9127. Historical evidence only; prohibited for all new selection. |
| Classical SIE recovery | `results/reports/fit-sie-grid-v1/` | Six scenarios, 13/13 optimizer successes and local identifiability checks; maximum `theta_E` relative error 0.1171. Noiseless control maximum relative error 0.0003. Not a general real-data benchmark. |
| Real CASTLES release | `data/releases/real-castles-v2/` | 137 observations from 69 systems; empirical PSF/ERR/DQ products are absent and toy variance is used. |
| Real astrometric comparison | `results/reports/real-comparison-v1/` | Three preselected systems; exploratory scale check only, not calibrated pixel-level inference or unbiased accuracy. |

The engineering release metadata records 21,009 artifacts totaling 2,573,791,870 bytes. The tracked audit mirror and local release manifests, split assignments, configuration, checksum ledger, rejection log, and validation report were identified. `configs/simulation_engineering_v1.yaml` specifies a 0.05 arcsec/pixel grid, SIE mass model, analytic Sersic source, Gaussian PSF, scalar background, and independent Poisson/read noise.

## Week 1 checks

| Check | Command / evidence | Result |
|---|---|---|
| Automated tests | `python -B -m pytest -q -p no:cacheprovider --basetemp=.pytest-work` | **98 passed in 33.61s**. |
| Engineering release deterministic validation | `python -B -m src.testing.validate_simulations data/simulations/sim-eng1000-v1/manifest.csv` | **Passed:** 1,000 parents, 0 errors, `passed=true`, residual tolerance `1e-10` arcsec. Deterministic rerenders and product checks completed. |
| Split integrity | `data/manifests/sim-eng1000-v1/engineering_release.json` | 1,000 unique parents; 700/150/150 split; no ancestry leakage; validation passed; all release gates true. |

## Gate and next step

The baseline artifacts are identifiable, the opened synthetic test is explicitly quarantined in `protocol_v1.md`, the dedicated `reliability-study-v1` branch is active, and both the test suite and deterministic engineering-release validation pass. The Week 1 gate is complete. The next scientific phase is Week 2 counterfactual infrastructure; no pixel-scale or other new model evaluation is authorized before that infrastructure passes ancestry and label-invariance checks.
