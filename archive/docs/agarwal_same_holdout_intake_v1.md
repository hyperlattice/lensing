# Agarwal et al. same-image benchmark intake (2026-11-01)

**Status:** The exact public target and source archives have been located, but
neither archive has yet passed a local checksum. The 3.8 GB target download is
in progress in `data/external_reference_research/agarwal_zenodo_v1/`. Two
author checkpoints have been downloaded at pinned Git blob hashes and
loaded successfully with CPU PyTorch. No target-test labels have been opened
and no model has been scored.

Primary sources: [paper](https://arxiv.org/html/2411.03334v3),
[author repository](https://github.com/deepskies/DomainAdaptiveMVEforLensModeling),
[Zenodo dataset](https://zenodo.org/records/13647416).

| Resource | Bytes | Published MD5 |
|---|---:|---|
| `mb_paper_source_final.zip` | 3,768,480,940 | `e79d4cef608ed4203c5e2e48dd81cf29` |
| `mb_paper_target_final.zip` | 3,804,478,112 | `9839a0ae65593405b1d4ebe6d1076117` |

At pinned author commit `17e42acb13f0c3111c8a4e665222e215b10d2e7b`,
`src/training/MVEUDA/train_utils.py:create_dataloader` loads the NPY image
array and CSV metadata, takes the label column
`PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g`, and uses contiguous rows:
0-69,999 for training, 70,000-89,999 for test, 90,000-99,999 for validation
when the released dataset has the paper's 100,000 objects. The test rows must
be verified against the actual archives before use. Images are three-band
40x40 arrays. The author's `paper_models` include saved MVE-only and MVE-UDA
checkpoints; the published repository lists a visualization notebook for them.

The headline Table 2 target values are selected MVE-only 0.0818 and selected
MVE-UDA 0.0425 arcsec; its five-run medians are 0.0585 and 0.0461 arcsec,
respectively. **The paper prose defines mean residual as a signed
mean, but the pinned `ModelVizPaper.ipynb` cell 13 computes
`mean(abs(prediction - truth))` and prints exactly 0.0818 and 0.0425.** Thus
the reported table numbers are consistent with MAE in the released evaluation
code; retain the prose/code discrepancy in the audit. A shared benchmark must
report MAE, signed bias, RMSE and paired uncertainty for both author and
project methods on exactly the same test rows. The author provides five
initialization runs; the notebook's cell 2 names `RunA_snll` and
`RunA_noDA_snll` as the first checkpoints in its five-run lists. Both files
contain `state_dict` and were loaded with `torch.load(..., weights_only=True)`.
The notebook's cell 8 output associates its first run with seed 56, whose
target MAEs are 0.0436 (UDA) and 0.0693 (MVE-only). Cell 13 has a saved output
of 0.0425 and 0.0818 for its `i=0` computation, inconsistent with cell 8's
saved output. Those notebook outputs were evidently generated in different
states or with changed checkpoint paths. **Do not identify RunA as the Table 2
selected checkpoint until inference reproduces the value and its metadata is
checked.** The exact test images and metric remain identifiable from the
dataset and evaluation code regardless of that notebook-output inconsistency.

Our saved 256x256 single-band CNN is incompatible with the paper's 40x40
three-band input. A fair direct comparison requires training an adapted version
of our source-only randomization method on the paper's **source training rows**,
keeping target test rows untouched. It also requires checking which unlabeled
target training rows the author MVE-UDA model accessed, and distinguishing
source-only from target-aware training. The author's published checkpoints can
then be scored on the same target test images. This work has not been completed.
For the chosen target test, an MAE below 0.0818 would beat the selected MVE-only
number, but an MAE below 0.0425 would be needed to beat the selected MVE-UDA
number. Those thresholds alone are insufficient for a superiority claim:
report all seeds, matched information access, and paired uncertainty.

The pinned code audit in `docs/prior_baseline_code_audit_v1.md` found that one
released utility did not backpropagate its MMD term. This is a code provenance
question, not evidence that the paper's saved checkpoints are invalid or that
our method is better. Score released models first; describe any exact-code
reproduction discrepancy separately.

If the archive or adaptation cannot be completed for the deadline, pivot to an
honest own-benchmark paper: report the blinded 150-parent synthetic holdout and
the matched-budget development check, cite this paper as related work, and
avoid published-method superiority. An author-inspired port on our simulator
could later support a *common-benchmark method* comparison, but would still
not be an exact reproduction of the published 40x40 multiband result.
