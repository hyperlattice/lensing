# Active execution of the robustness comparison

## Current checkpoint (updated 2026-11-01; supersedes historical handoff below)

Item 1 is complete: the equal-budget final comparison is in
[equal_budget_final_v1](equal_budget_final_v1.md). Ten paired seeds were fit for
canonical (C), uniform variation (G), and frozen failure-weighted variation (R),
all with 880 updates, batch 16, identical initial weights and parent schedules.
On the existing 150-parent blind synthetic holdout, four-stress MAE was C
0.141251, G 0.111507, and R 0.107385 arcsec. G-C was -0.029744 arcsec with
parent-bootstrap 95% interval [-0.045725, -0.012141]; R-G was -0.004122 with
interval [-0.026208, 0.009341]. This establishes an equal-budget source-only
robustness effect for varied-condition training and shows no reliable extra
benefit from failure weighting over uniform variation. Predictions were sealed
before the target join; source-train and holdout parent IDs have zero overlap.
The result is synthetic and does not establish cross-simulator, real-data, or
published-method superiority.

Item 2 rapid A/B release is now complete as an audit, not as a positive
cross-simulator result.  The new 150-parent `cross-simulator-rapid-v2` cohort
rendered 1,350 paired Lenstronomy/AutoLens image products and sealed 40,500
paired predictions.  All image pairs and predictions are exactly identical at
saved precision.  The adapter's convention mapping and shared image-formation
stack therefore produce a renderer-equivalence control, while the Moffat
conditions provide an unseen PSF-family shift shared by both labels.  See
[`cross_simulator_rapid_v2_audit.md`](cross_simulator_rapid_v2_audit.md) and
the machine-readable
[`renderer_identity_audit.json`](../results/ml/cross-simulator-rapid-v2/renderer_identity_audit.json).
The project must not claim simulator-induced transfer from v2; a genuinely
different backend-native image-forming path requires a new immutable release
identity.

Step 3 (claim and prior-work comparison) is now frozen in
[`step3_claim_comparison_matrix_2026-11-01.md`](step3_claim_comparison_matrix_2026-11-01.md).
The defensible workshop contribution is equal-budget, source-only observation
randomization robustness within the tested synthetic generator. The failure-
weighted policy is inconclusive versus uniform variation; no numerical
comparison to Swierc, Agarwal, Hezaveh, Pearson or Gawade is valid without a
common benchmark or faithful reproduction. The manuscript should therefore
avoid published-method superiority, simulator-transfer and real-data claims.

Submission packaging is now prepared: the current manuscript draft is
[`manuscript_submission_v2.md`](../reliability_study/final/manuscript_submission_v2.md),
the two equal-budget figures are under
`results/ml/equal-budget-final-v1/figures/`, and the artifact/environment
checklist is [`reproducibility_checklist_submission_v1.md`](reproducibility_checklist_submission_v1.md).

2026-11-01 steering: the user requested that tuning be ignored and asked for
the immediate result relative to other studies. The 240-fit background process
(PID 1160) was stopped after six completed tuning fits; no completed fit files
were removed. Its source-development outputs are exploratory and have no bearing
on the immediate paper claim. See [immediate result](immediate_paper_result_2026-11-01.md).
The strongest finished evidence is the Week 11 blind synthetic holdout:
150 new lens systems, four-stress MAE 0.131980 canonical versus 0.095875
randomized, delta -0.036105 arcsec (27.356% relative), parent-bootstrap
95% interval [-0.042736, -0.029419]. This is a comparison between our own
CNN training recipes, not published-model superiority. The matched-budget
source-development follow-up delta is -0.016971 arcsec. No B/Moffat final
accuracy comparison or real-telescope accuracy result exists.

Renderer qualification v3 PASSED: 315/315 cross-engine views and 56/56
independent fine integration checks. Worst adaptive-versus-fine relative L1
was 0.000621125 (0.0621%), below the 0.5% gate; worst A/B expected-image L1
was 1.29436e-8. Of 106 proposals, one independently agreed physical single
image was rejected; 105 multiple-image parents were tested at three geometries.
Shared integration/Jacobian stencil/PSF/detector limit engine independence.

The dense-root, eight-total-worker bank build exhausted RAM and produced no
eligible bank. Preserve `robustness-study-*-v1` failed attempts and the ledger
`results/reports/robustness-bank-generation-attempts-v1.json`. This machine has
8.28 GB RAM, 12 CPU threads and no CUDA. All failed workers were stopped and
verified exited. Bank arrays are now read-only memory maps.

The lower-memory renderer then PASSED qualification v5 on all 315 views,
worst expected-image L1 1.96031e-9 and root-position error 4.36742e-10 arcsec.
See [protocol v6](renderer_qualification_protocol_v6.md). Preserve its bound
sources. The earlier failed analytical-only qualification v4 remains intact.

The v3 source-train render did not finish: it stopped at proposal 672/700 with
4,700 product arrays, but no proposal audit, completion report, or bank manifest.
The files are preserved as a failed/interrupted attempt and are not data for
training. Its execution session was gone on 2026-10-29. No source-development or
target bank was created. The attempt ledger records this explicitly.

Historical v4 execution commands were `python -B -m scripts.build_all_robustness_banks_v4` and the
lightweight waiting development controller
`python -B -m scripts.execute_robustness_development_v3 --config
configs/ml/robustness_development_v4.yaml --wait-for-banks`. Current execution
sessions: bank build 73838; controller 27281. V4 uses new versioned output
identities, the same frozen eligibility protocol, two workers, and atomic
checkpoints every ten completed proposals. On restart it verifies recorded
product hashes and rerenders only proposals without valid checkpoint records.
It processes roles sequentially, with AutoLens isolated for the target role.
The source-train bank has completed and validated: 697 eligible parents from
700 proposals; the three exclusions were independently agreed physical
single-image systems; zero unresolved failures; runtime 2,126 seconds and
985,448,396 bytes. Source-development completed cleanly: 150/150 eligible
parents, zero exclusions/failures, runtime 471 seconds and 212,507,229 bytes.
Target-adaptation completed cleanly: 500 selected from 567 eligible proposals
out of the fixed 570 pool; three approved single-image exclusions and 67
unused eligible proposals; zero unresolved failures; runtime 945 seconds and
1,207,461,007 bytes. All three source-only reference fits are now complete
(seeds 720601-720603; canonical source-development MAEs 0.117699, 0.103356,
0.102936 arcsec). Source-only allocation scoring finished with probabilities
0.301/0.235/0.185/0.146/0.0357/0.0612/0.0357 for pixel-scale-004,
pixel-scale-006, exposure-half, read-noise-double, Gaussian FWHM 0.75,
Gaussian FWHM 1.25 and fresh-noise respectively. It used no target or final
images. These are development choices, not final performance evidence.

The 240-fit source tuning grid was stopped after six exploratory fits at the
user's direction; no tuning output was used to select the equal-budget final
settings. Its partial directories remain preserved. The equal-budget result
above is the current final item-1 evidence. Do not resume that grid unless the
user explicitly changes the no-tuning priority.
Refresh with `python -B -m scripts.report_robustness_progress_v4`. Do not infer
success from partial files.

Fourteen focused statistical, integrated training and truth/prediction-gate
tests pass: `results/reports/robustness-development-v1/controls_tests_v3.xml`.
An earlier test invocation failed because its temporary parent directory was
missing; the earlier XML is retained. Tests are not accuracy evidence.

Additional final-workflow controls now bring the focused suite to 23 passing
tests (`confirmatory_controls_v1.xml`), including weighted P95 versus explicit
whole-parent resampling, variance-only power controls, image-only native
products and a canonical guardrail that blocks an otherwise favorable MAE.
The implemented final pipeline has not yet run on scientific final data.

Development controller: `python -B -m
scripts.execute_robustness_development_v3 --wait-for-banks`.
It waits for ALL banks before running one fit at a time. Physical memory
measured only 618 MB available during simultaneous rendering/tests, so avoid
overlapping fits with rendering. The lightweight waiter does not import Torch.
Earlier waiting controllers 16654 and 80349 were stopped before starting fits.
Do not launch duplicates.
Refresh development-only status with
`python -B -m scripts.report_robustness_progress_v1`.

Implemented but not scientifically executed: method/evaluation freeze builders,
fixed-setting pilot/confirmation fitter, paired native generator, truth-free
predictor, pilot variance/power runner and once-only final scorer. Inspect their
CLI help; no method/evaluation freeze exists. Use source-only tuning/boundary
and convergence evidence to create the required review before Freeze 1.

Next: finish v4 banks; reference/weights/tuning actions in
`scripts/run_robustness_development_v1.py`; boundary/convergence review;
Freeze 1; variance-only pilot; powered N/S and confirmatory fits; Freeze 2;
fresh final generation, sealed target-free predictions and one scoring pass;
claim-evidence report/manuscript. G/R share G-selected optimizer settings for
the allocation claim, as do corresponding G/R+MMD pairs. Independent R tuning
is diagnostic. All point arms use Huber, so they are common-benchmark ports.
See [development protocol](robustness_development_protocol_v1.md) and
[analysis protocol](confirmatory_analysis_protocol_v1.md), declared before fits.
The latter allocates alpha .025 primary plus .025 secondary family and fixes
guardrails and variance-planning rules. No final freeze exists yet.

## Historical handoff (outdated running-job status retained below)

This is a working handoff, not a final result or method freeze. The user has
authorized continuing the complete strengthening plan and wants a quantified,
finished or publishable result. Do not stop after another infrastructure-only
milestone or call the study publishable because tests pass.

## Completed evidence

- Existing matched-budget/source-validation evidence remains unchanged. All
  73 hashes in its evidence manifest and eight hashes in the foundations
  checkpoint were rechecked and match. Old final images/labels remain unused.
- Pinned public adaptation repository commit:
  `17e42acb13f0c3111c8a4e665222e215b10d2e7b`. Eight files fetched with hashes.
  Isolated code audit verified that the pinned MVE training utility ignores
  the DA weight in its optimized loss. Weight 0 versus 1.4 produced exactly
  identical updates on a CPU fixture. The notebook includes target-label
  checkpoint selection. This does not establish what generated paper models.
  See [audit](prior_baseline_code_audit_v1.md).
- Active paper-kernel MMD port agrees with the inspected kernel to 2.98e-7
  on a fixture, with nonzero feature gradients. The comparison must use a
  functioning, competently tuned baseline, not exploit the disabled term.
- A source generative/image-only target bank interface and integrated eight-arm
  development trainer are implemented. Unit fixtures verify identical initial
  weights, parent schedules, source exposures and full optimizer batch sizes
  across C/N/G/T/R and C/G/R+MMD. This is not lens-accuracy evidence.
- Crossed seed/parent bootstrap and variance-component planning helpers are
  implemented and tested. More parents cannot erase seed-only uncertainty.

## Numerical qualification: failures retained, fixes being checked

V1 used 100 fresh prior draws plus six edge cases at three native geometries.
One prior draw had a single point image, giving three rejected view records.
Of 315 other views, 36 failed the proposed factor-4-to-8 clean-image tolerance
0.005; worst relative L1 was 0.0856617. Two also failed the expected-image
tolerance. B's central-floor profile mapping passed all 36 controls at worst
relative error 1.89e-13. Twelve expected-image comparisons failed the stricter
cross-code 1e-5 gate while clean images agreed near 1e-9.

Adaptive pixel integration now uses successive-grid estimates at tolerance
1e-4 with cap 512, following an explicitly recorded prototype/amendment.
All 315 A views passed the INTERNAL estimate. This does not yet certify them.
Both packages also use their own deflections with a shared five-point
Jacobian stencil: installed Lenstronomy's default SIE/NIE Hessian used forward
differences at step 1e-10, while the initial B amplitude used a two-point
stencil. The new point convention leaves all original experiment files intact.

Protocols and exact scope:
[v1](renderer_qualification_protocol_v1.md),
[v2](renderer_qualification_protocol_v2.md),
[point preflight](renderer_qualification_protocol_v3.md),
[current corrected-point qualification](renderer_qualification_protocol_v4.md).

Current A references: `data/simulations/renderer-qualification-v3/` (315 views,
zero export failures). They reuse immutable v2 CLEAN arrays, regenerate
expected images with corrected A amplitudes, and preserve old versions.
Two independent phases are running:

```powershell
python -B -m scripts.qualify_adaptive_renderer_v3 refine
.venv-autolens/Scripts/python.exe -B -m scripts.qualify_adaptive_renderer_v3 validate
```

Do not edit the sources/protocols bound in those references while phases run.
The fine phase tests all previous failing views, edge parents and 12 seeded
additional views against uniform 16/32/64 integration. The B phase tests all
315 views, point checks, and independently classifies the single-image draw.
Only then run `python -B -m scripts.qualify_adaptive_renderer_v3 assemble`.
A `passed: false` report blocks bank generation; preserve and diagnose it.

## Ordered continuation

1. Finish numerical qualification and summarize all unconditional proposals,
   physical rejections, numerical failures and conditional agreement. Do not
   loosen the declared gates to pass existing images.
2. `scripts/generate_robustness_banks_v1.py` is implemented, but not yet run.
   It requires the passing v3 qualification, then generates 700 original
   source-training parents, 150 reused source-development parents, and 500
   fresh target-adaptation parents using the new versioned renderer. It writes
   raw rates/background-retained images, not old processed images. Source
   uses fresh draws from five cached expected scenes to implement seven cells.
   T ports the original canonical-plus-compound-condition recipe to the new
   fidelity. No old release is overwritten. Target labels are not serialized.
   Only independently confirmed physical one-image target proposals may be
   replaced; any numerical failure blocks completion. Review its cohort and
   geometry validation before large generation.
3. Finish source-only score artifact, tuning runner, budget curves and R's
   empirical weights. No new scientific models have been fitted yet and no
   augmentation weights are frozen. Keep source, initialization, view/noise
   and target sampling streams separate. Record all trials/compute/failures.
4. Primary comparison remains R versus G on new B/Moffat parents. Add a
   separately labeled representative-target MMD comparison before claiming
   superiority over UDA that receives unlabeled images from the evaluation
   PSF family. The existing core Gaussian-only adaptation setting tests
   extrapolation beyond available target conditions; it cannot answer the
   stronger target-aware question. The target bank generator provides separate
   image-only Gaussian and representative-condition indices for the same
   adaptation parents. The training protocol, exact contrast family and
   information-access table must be frozen before their results are scored.
5. Follow the plan's two freezes: settings before a separate power pilot;
   final N/S/checkpoints and analysis before generating final images. At least
   ten paired seeds, crossed uncertainty, meaningful margins and canonical/
   tail guardrails are planned. Power/sample size still need actual pilot data.
6. Run final predictions without truth, hash/verify them, join labels once,
   score the frozen contrasts, then write the claim-evidence report/manuscript.
   A failed primary comparison is a result; do not replace the endpoint.

Available resources measured at start: 12 CPU threads, no CUDA GPU, approximately
90 GB free. Measure actual bank/training costs before launching the full run
matrix. The one-month schedule is a planning target, not a verified deadline.

The optional DANN/MVE/real-image tracks are needed only for claims about those
methods or capabilities. No real-data superiority, posterior calibration or
new-method priority claim is supported by the current project.
