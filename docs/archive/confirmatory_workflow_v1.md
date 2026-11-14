# Executable workflow for the bounded robustness study

Status: implemented workflow; scientific tuning/freeze/pilot/final stages are
not complete. Read the current [execution handoff](research_execution_2026-10-27.md)
for running sessions. All commands run from `D:\lensing`. Source ML uses global
Python; B rendering uses `.venv-autolens/Scripts/python.exe`. There is no CUDA.

## Development and method decisions

The active bank build is `scripts.build_all_robustness_banks_v4`, at most two
rendering workers globally and sequential roles. V3 stopped before completing
the source-train audit and its partial files are excluded. V4 checkpoints every
ten finished proposals and validates product hashes before resuming. It retains
all independently agreed multiple-image parents from the original 700/150
source proposal pools. Their final accepted counts must come from completion
reports. Target banks select the first 500 eligible parents from 570 fixed
proposals. Numerical failures block eligibility; physical single-image
rejections are audited.

The lightweight controller waits for every bank's completion, then executes:

```powershell
python -B -m scripts.run_robustness_development_v1 reference
python -B -m scripts.run_robustness_development_v1 weights
python -B -m scripts.run_robustness_development_v1 tune
python -B -m scripts.review_robustness_development_v1 summarize
```

Do not launch these again while the controller is active. The initial search
is 12 settings × two paired seeds × ten experiments. References are three
separate canonical fits. Refresh development-only progress with
`python -B -m scripts.report_robustness_progress_v1`.

Inspect source tuning boundaries. Expand ranges when needed, recording all
additional trials before choosing a final optimizer. A zero-weight-decay
optimum is a physical boundary, unlike a lower learning-rate limit. Give
competing families comparable opportunities. No target predictive error is
available for these decisions. Then run matched finalist curves:

```powershell
python -B -m scripts.review_robustness_development_v1 curves
```

The default curves cover 880/1760/3520 updates and require source-only review;
they do not automatically certify convergence. If an expanded search is
needed, version its runner/config/review and use its reviewed summary instead
of freezing the initial one. G-selected optimizer settings are shared with R,
likewise within Gaussian-target and representative-target G/R+MMD pairs. The
independently selected R settings remain development diagnostics.

Write a source review after inspecting actual evidence. Required fields:

```json
{
  "status": "source_only_boundary_and_convergence_review_complete",
  "target_predictive_scores_used": false,
  "final_images_used": false,
  "source_tuning_summary": "results/ml/robustness-development-v1/source_tuning_review.json",
  "optimizer_updates": "REPLACE WITH REVIEWED INTEGER",
  "boundary_review": {"resolved": true, "evidence_paths": ["ACTUAL SEARCH EVIDENCE"], "reason": "ACTUAL DECISION"},
  "convergence_review": {"resolved": true, "evidence_paths": ["ACTUAL CURVE EVIDENCE"], "reason": "ACTUAL DECISION"},
  "training_seconds_per_seed_all_methods": "MEASURED POSITIVE NUMBER",
  "rendering_seconds_per_paired_parent": "MEASURED/PROJECTED POSITIVE NUMBER WITH RATIONALE"
}
```

This is a schema example, not a completed review. Timing projections are used
only to rank powered designs by estimated cost, not to claim efficiency.
The method-freeze builder validates qualified banks, policy provenance,
comparator coverage, paired optimizer settings and actual review evidence.

## Freeze 1 and independent variance pilot

After source review, create the new study identity once:

```powershell
python -B -m scripts.prepare_robustness_method_freeze_v1 configs/ml/robustness_development_v1.yaml results/ml/robustness-development-v1/source_review.json results/ml/robustness-confirmatory-v1
```

The builder seals sources, scripts, qualified banks, allocation, reviewed
settings and a 120-parent pilot-generation design. Pilot training seeds are
720811–720815; confirmation candidates 720821–720840. These are separate from
reference and tuning seeds. No method retuning follows target-family pilot
predictions. See [analysis specification](confirmatory_analysis_protocol_v1.md)
and [statistical limits](statistical_prior_work_addendum.md).

Fit the five pilot seeds for every fixed method, without pilot images:

```powershell
python -B -m scripts.run_robustness_fixed_fits_v1 pilot results/ml/robustness-confirmatory-v1/method_freeze.json results/ml/robustness-confirmatory-v1/fixed_training_settings.json results/ml/robustness-confirmatory-v1/pilot-fits
```

Prepare and render the pilot in sequential phases:

```powershell
python -B -m scripts.generate_paired_evaluation_v1 prepare results/ml/robustness-confirmatory-v1/pilot_generation_design.json
python -B -m scripts.generate_paired_evaluation_v1 lenstronomy data/simulations/robustness-power-pilot-v1 --workers 2
.venv-autolens/Scripts/python.exe -B -m scripts.generate_paired_evaluation_v1 autolens data/simulations/robustness-power-pilot-v1 --workers 2
python -B -m scripts.generate_paired_evaluation_v1 assemble data/simulations/robustness-power-pilot-v1
```

Every proposal is retained in a vault/identity ledger, including unused pool
members. Public image manifests contain required variance/PSF/masks and units,
without targets or physical parameters. Both engines must agree on eligible
root counts and independently checked positions. All numerical failures block
assembly. Shared observation code and the finite qualification scope remain
explicit limitations.

Pilot predictions are written and sealed before its truth is joined. Only
paired variance components and planning results are exported, not rankings:

```powershell
python -B -m scripts.plan_robustness_power_v1 pilot results/ml/robustness-confirmatory-v1/method_freeze.json data/simulations/robustness-power-pilot-v1/manifest.json results/ml/robustness-confirmatory-v1/pilot-fits/fixed_fits_complete.json results/ml/robustness-confirmatory-v1/power-pilot
python -B -m scripts.plan_robustness_power_v1 plan results/ml/robustness-confirmatory-v1/method_freeze.json results/ml/robustness-confirmatory-v1/power-pilot/variance_components.json results/ml/robustness-confirmatory-v1/power_design.json
```

Planning uses stipulated delta -0.010 arcsec against a -0.005 margin, not the
pilot treatment mean. It simulates the actual 10,000-draw crossed bootstrap
with 200 Monte Carlo trials for candidate designs, reports Wilson uncertainty,
variance doubling and Gaussian upper-component sensitivity. The capped grid
is 10/15/20 seeds and 600/1000/1500/2000 parents. An underpowered cap is reported
as such, without changing the hypothesis. Guardrails are not jointly powered.

## Confirmatory fits and Freeze 2

```powershell
python -B -m scripts.run_robustness_fixed_fits_v1 confirmation results/ml/robustness-confirmatory-v1/method_freeze.json results/ml/robustness-confirmatory-v1/fixed_training_settings.json results/ml/robustness-confirmatory-v1/confirmatory-fits --power-design results/ml/robustness-confirmatory-v1/power_design.json
python -B -m scripts.prepare_robustness_evaluation_freeze_v1 results/ml/robustness-confirmatory-v1/method_freeze.json results/ml/robustness-confirmatory-v1/fixed_training_settings.json results/ml/robustness-confirmatory-v1/power_design.json results/ml/robustness-confirmatory-v1/confirmatory-fits/fixed_fits_complete.json data/simulations/robustness-power-pilot-v1 results/ml/robustness-confirmatory-v1
```

Freeze 2 checks actual checkpoint identities, fixed settings, target access,
initialization/parent-exposure pairing and training provenance. It binds N/S,
new generation/prediction identities, all earlier role exclusions including
the full pilot pool, and the exact contrasts/analysis code before final images
exist. Reusing a verified completed fit is explicit; partially started fits
and inference/analysis markers do not silently restart.

Generate final data using the same prepare/A/B/assemble sequence above, with
`final_generation_design.json` and `data/simulations/robustness-final-evaluation-v1`.
Both methods and rendering conventions are already fixed.

## Sealed predictions, once-only scoring and publication artifacts

```powershell
python -B -m scripts.predict_paired_evaluation_v1 results/ml/robustness-confirmatory-v1/evaluation_freeze.json data/simulations/robustness-final-evaluation-v1/manifest.json
python -B -m scripts.score_paired_evaluation_v1 results/ml/robustness-confirmatory-v1/evaluation_freeze.json
python -B -m scripts.plot_robustness_results_v1 results/ml/robustness-confirmatory-v1/evaluation_freeze.json results/figures/robustness-confirmatory-v1
```

Every checkpoint predicts every parent/condition/renderer without truth. The
workflow checks axes and hashes before the one-way truth join. Primary R−G
meaningful superiority requires its bound below -0.005 arcsec and all canonical
and tail gates. Five secondary claims use the prespecified adjusted alpha.
Percentage gains name their comparator and use jointly resampled denominators.

`final_analysis.json`, `result_overview.md`, per-contrast results and standalone
PDF/PNG figures supply the empirical publication record. Write the manuscript
claim-evidence table from these exact gates, retaining failed comparisons and
all physical/numerical/fit failures. An engineering checkpoint, a reused
development score, or an unrelated paper's percentage is not the final result.
