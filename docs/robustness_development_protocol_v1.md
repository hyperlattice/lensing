# Source-only development and comparison protocol

Declared before new scientific fits. Configuration:
`configs/ml/robustness_development_v1.yaml`. This supplements the proposed
superiority plan; it is not Freeze 1 or a final preregistration.

Use qualified, versioned raw-rate banks with background retained: 700 existing
source-training parents, 150 reused source-development parents and 500 fresh
unlabeled adaptation parents. No old final images/labels enter this workflow.
Original source parameters are rerendered with qualified adaptive integration
and precise point amplitudes. These images do not replace old reported data.
Eligibility amendment before scientific fits: the original 700/150 source
proposal pools contain physical single-image systems (first detected at source
index 78, parent 0109, with independent one-root agreement). Apply the declared
two/four-image benchmark population to source roles as well as target roles.
Keep every agreed eligible source parent, with no replenishment; report the
original pool size, physical rejections and resulting count. All methods use
the identical accepted source pool. Numerical/root disagreements remain fatal.
The interrupted v2 attempt is retained; v3 records every worker result before
eligibility checks. This is a pre-training population definition, not selection
using any prediction error. New percentages must name this changed cohort.
Images alone enter the predictor; source-train-only global image/target
statistics are identical across arms. Source noise is regenerated from cached
expected scenes where the policy calls for fresh draws.

## Models and fixed first-stage controls

Backbone: original four-block image-only CNN, channels 8/16/32/64, pooled size 4,
128 hidden units, dropout .1. Objective: Huber loss on standardized radius,
delta 1. Optimizer: AdamW. Fixed source batch size 16. Each seed/arm gets the
same realized parent schedule and initial tensor values. T uses grouped native-
geometry forwards with one optimizer update for the full prescribed batch.
No resizing or padding is inserted to mix native geometries.

C uses fixed canonical noise; N uses fresh canonical noise; G uniformly samples
the seven cells; R samples those identical cells with frozen source-failure
probabilities; T uses the original canonical-plus-compound-condition recipe,
rerendered at the common fidelity. C/G/R+MMD use the active, audited adaptive
five-kernel MMD port at pooled features and unlabeled B batches. This is a
common-backbone method port, not an exact multiband MVE paper reproduction.
Source labels, views, actual noise graphs, source/target exposures, pixels,
histories, input hashes and costs are recorded.

The eight-arm core gives MMD only Gaussian-condition B images. To avoid
claiming superiority over target-aware adaptation from that extrapolation-only
setting, add G+MMD and R+MMD with a separately labeled representative unlabeled
target bank containing Gaussian/Moffat views of the SAME 500 adaptation parents.
Source/target exposure budgets match within adaptation comparisons. No target
regression labels enter optimization or model selection. The total first-stage
matrix is ten labeled experiments; source-only R/G still never receive Moffat
images. Final parents are fresh and disjoint from every adaptation parent.

## Reference fits and policy scores

Train three C reference fits at 880 updates using seeds 720601/720602/720603,
learning rate .001 and weight decay .0001. They are development/reference cost,
not confirmatory seeds. For each source-development parent evaluate canonical
plus seven conditions using TWO deterministic independent noise realizations
per condition, shared across reference models. Noise namespace 720610 is
derived with parent parameter seed, condition index and replicate index.

Average absolute errors over noise replicates. For each cell, average its
paired error increase over reference seeds and source-development parents.
Use positive degradation weights with epsilon .25; fall back to uniform if
all increases are nonpositive. Save error cube, ordered parent/condition IDs,
reference checkpoint and bank hashes, and probabilities. No target error or
final observation can alter these weights.

## Shared tuning score and initial search

Use the SAME canonical-plus-seven-source-cell MAE, averaged equally over
parents, conditions and the two fixed noise replicates, to select settings
for every method. Canonical validation histories are additional diagnostics.
The source-development set has already informed prior study choices and is
explicitly reused; these scores are not independent final evidence.

Initial tuning: 880 source updates, two paired seeds 720611/720612 and twelve
configurations per experiment. Source-only arms use learning rates
.0003/.0006/.001/.003 crossed with weight decays 0/.0001/.001. MMD arms use
learning rates .0003/.001/.003 crossed with MMD weights .003/.01/.1/1.4;
weight decay remains .0001. Delta and architecture remain fixed. The inspected
paper's MMD weight 1.4 is included. MMD sums five adaptive-bandwidth kernels,
so its scale is not interchangeable with the earlier fixed-kernel average.

Before any scientific fit: retain the common Huber objective throughout this
point-regression track. This deliberately isolates augmentation/alignment; it
does not reproduce the paper's mean-variance likelihood. For the confirmatory
allocation contrast, select G's optimizer settings by the shared source score
and use those identical settings for R. Likewise use each G+MMD information-
access setting's selected optimizer/MMD weight for its paired R+MMD arm. R's
independently tuned configurations remain development diagnostics and cannot
replace these primary comparisons. C, N and T retain their own source-selected
settings; comparisons with them concern complete training recipes. All arms
use the common update budget chosen from the convergence review.

Record all runs and failures. A boundary optimum triggers a source-only
search-range review and documented expansion with comparable opportunity for
competing families. It does not justify choosing a weak reference. Examine
880/1760/3520 learning curves, extend if necessary, and choose a common
confirmatory source-update budget before Freeze 1. No target-family predictive
accuracy is available during these choices. Do not claim completed tuning,
convergence, power or superiority from the initial 880-update stage alone.

## Remaining freeze requirements

Declare exact confirmatory contrasts, multiplicity/alpha allocation, canonical
and tail guardrails, variance-only pilot procedure and N/S selection before
Freeze 1. The proposed main contrast is R-minus-G B/Moffat MAE with a 0.005
arcsec meaningful-improvement margin. Target-aware comparisons must retain
their additional image access in tables and claims. At least ten paired
confirmatory seeds and a separate power pilot are planned; actual N/S cannot
be justified until the pilot variance analysis is complete.

The statistical and prediction-seal helpers are implemented but no scientific
method/evaluation seal exists yet. A seal must be written before the relevant
data stage, with actual qualified inputs, chosen settings, source-only policy,
all required method checkpoints and complete analysis/recovery rules.
