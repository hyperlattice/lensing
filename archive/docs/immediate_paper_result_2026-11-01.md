# Result available for a paper today

The completed, blinded synthetic holdout has 150 new lens systems and 750
observations across canonical imaging and four observing-condition stresses.
A three-seed randomized-training CNN ensemble reduced the equal-weight four-
stress mean absolute Einstein-radius error from about 0.1320 to 0.0959 arcsec,
a change of -0.03610 arcsec or 27.4% relative to the canonical-trained CNN
ensemble. The parent-bootstrap 95% interval for the error change is
[-0.04274, -0.02942] arcsec. Canonical-image MAE was 0.06548 versus 0.08277
arcsec. Source: `results/reports/reliability_v1/blind_holdout_week11/report.md`
and `results/reports/final_v1/analysis_summary.json`.

This is the result to present immediately. State that it supports improved
robustness **within this synthetic lens/observation generator**, for the
specified pixel-scale, exposure and read-noise shifts. The interval resamples
lens parents for fixed trained ensembles; it does not include uncertainty over
new training runs. The original randomized training also added a view per
source parent, so this blind result alone does not isolate randomization from
training-exposure budget.

The matched-budget follow-up used final 880-update checkpoints, common parent
exposures and shared normalization. On 150 source-development systems it found
four-stress MAE change -0.01697 arcsec (descriptive parent-bootstrap interval
[-0.02391, -0.01005]). This is supporting development evidence and cannot
replace the blind holdout. Source:
`results/reports/matched-budget-validation-v1/report.md`.

No completed result compares our method with an exact reproduction of a
published model on the same split, image generator, label definition and
metric. Swierc et al. (2023) study DANN/MMD adaptation on a different
deeplenstronomy-to-DES-like setup; Agarwal et al. (2024) study MVE plus UDA and
uncertainty. Their reported gains cannot be ranked against our MAE percentage.
Our independent second-renderer/Moffat evaluation and real-telescope accuracy
are unfinished. Do not write that this result beats either paper.

Suggested results sentence: "On a blinded synthetic holdout of 150 lens
systems, randomized observing-condition training lowered mean absolute
Einstein-radius error across four predefined shifts by 0.0361 arcsec (27.4%)
relative to our canonical-trained CNN, with a parent-bootstrap 95% interval
of 0.0294-0.0427 arcsec improvement. A separate matched-training-budget
development comparison retained the same direction of effect."

Suggested contribution: a measured, paired robustness effect with an openly
stated budget confound in the blinded study and a matched-budget development
check. The paper can be an honest synthetic robustness workshop submission.
It is not a validated cross-simulator or published-method superiority paper.
