# Two-experiment plan for an eight-page manuscript

Date: 13 November 2026.

The manuscript is limited to eight pages, excluding references. Its core should
contain two scientific experiments. HST preparation and compatibility checks are
supporting methodology rather than additional headline experiments.

This document narrows the paper scope from the broader
[HST experiment plan](hst_experiment_plan.md). The detailed plan remains useful
as an implementation workflow; its stages do not all need separate experiments
in the manuscript.

## Central research question

> Does varying synthetic observing conditions improve Einstein-radius
> estimation, and do those improvements persist on real telescope images?

The contribution is a controlled comparison of synthetic training distributions
with observational validation, conditional on obtaining a defensible real-data
cohort. Feature analysis is outside the core scope of this manuscript.

## Experiment 1: Synthetic robustness

**Question:** Does synthetic training variation improve robustness?

Retain the existing C/G/R comparison:

- **C:** canonical synthetic observing conditions.
- **G:** general variation in observing conditions.
- **R:** variation weighted toward conditions found difficult during development.

Use the same CNN, matched training budgets, and ten paired training seeds.
Evaluate all models on the same synthetic parent systems and observing
conditions. The primary comparison is G versus C; the secondary comparison is
R versus G.

This experiment establishes whether general variation helps and whether
targeting difficult conditions adds value under the declared synthetic tests.
The existing results support G over C. R over G remains inconclusive.

Implementation note: the checked-out composite-bootstrap code has now been
corrected. The historical confidence intervals need verification against the
original saved predictions before these inferential statements are finalized.
See the [implementation and reanalysis guide](hst_study.md).

## Experiment 2: Transfer to real HST images

**Question:** Do the synthetic-training benefits transfer to real observations?

After resolving compatibility, evaluate the frozen C/G/R models on an untouched
eligible HST cohort:

- Give all three arms identical prepared observations.
- Compare predictions with declared classical Einstein-radius references.
- Report paired differences, uncertainty, and individual-lens behaviour.
- Keep the five development lenses outside the confirmation cohort, including
  other bands or releases of those same systems.

This experiment tests whether the synthetic result extends to selected real
observations. A failure to transfer is also a useful result. The references are
model-dependent estimates, not exact physical truth, so the observational claim
must describe agreement with those references and the eligible cohort.

This experiment is proposed. The compatibility work completed so far does not
establish observational prediction accuracy or a C/G/R ranking on telescope data.

## Supporting work and scope boundaries

| Work | Role in the paper |
|---|---|
| Calibration, centering, PSF, coverage, and reference checks | Data and preprocessing methods |
| Synthetic checks of numerical preparation | Brief validation statement; details in an appendix if permitted, or in the repository |
| Five-system HST pilot | Development procedure, not a headline result |
| Feature-reliance interventions and saliency analysis | Deferred to a separate study |

Supporting checks still need to be carried out. Describing them concisely does
not remove their importance to scientific validity. The main paper must retain
the evidence necessary to assess its headline claims.

## Suggested page allocation

References are excluded from the following eight-page allocation.

| Section | Pages |
|---|---:|
| Introduction and contributions | 0.75 |
| Related work | 0.75 |
| Synthetic data, HST data, and preprocessing | 1.25 |
| CNN and C/G/R training design | 1.00 |
| Experiment 1: synthetic results | 1.50 |
| Experiment 2: HST results | 1.50 |
| Discussion, limitations, and conclusion | 1.25 |
| **Total** | **8.00** |

Prioritize one synthetic-results figure, one HST-results figure, and a compact
comparison table. Feature analysis should not consume the main paper's space.

## If observational qualification is unsuccessful

If a defensible HST cohort cannot be established, retain the synthetic-only claim
rather than force an observational result into the paper. Report the limits of
the available observational evidence honestly and revise the framing and page
allocation accordingly.
