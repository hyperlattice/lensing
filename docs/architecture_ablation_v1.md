# CNN architecture ablation v1

This is an exploratory five-seed capacity check for the maintained synthetic
robustness study. It compares the reference CNN with one compact alternative
under the uniform-variation policy. It is separate from the ten-seed C/G/R
experiment.

## Data and protocol

- Training: 697 qualified source-training parents from
  `robustness-study-source-train-v4`.
- Holdout: 150 independent parents, each rendered under one canonical and four
  stress conditions (750 images total) from `reliability-blind-holdout-v1`.
- Training arm: G, uniform sampling of the seven source observation cells.
- Budget: 880 optimizer updates, batch size 16, and 14,080 source exposures per
  fit.
- Seeds: 730001 through 730005, paired across architectures.
- Optimizer and loss: AdamW with learning rate 0.001 and weight decay 0.0001;
  Huber loss with delta 1.0.
- Evaluation: condition-specific MAE and the parent-averaged equal-weight MAE
  over the four stress conditions. The 95% interval resamples parents and holds
  the five training seeds fixed.

The code and release metadata are maintained in the public repository:
<https://github.com/nix-life/lensing>. Large image arrays and checkpoints are
local release products; the result files below retain the compact numerical
record and configuration identities.

## Architectures

Both variants use four blocks. Each block contains a 5x5 stride-2 convolution,
GroupNorm, SiLU, a 3x3 convolution, GroupNorm, and SiLU, followed by adaptive
4x4 average pooling and a scalar regression head.

| Variant | Channels | Hidden units | Trainable parameters |
|---|---|---:|---:|
| Compact | [4, 8, 16, 32] | 64 | 62,397 |
| Reference | [8, 16, 32, 64] | 128 | 248,409 |

The reference checkpoints are the existing G-arm fits from the primary study.
The compact fits were trained with the same source-parent schedule and
observation policy.

When the local qualified banks and primary reference checkpoints are available,
the five-seed run is reproducible with:

```powershell
python -m scripts.run_architecture_ablation_v1 all
```

## Result

| Architecture | Canonical MAE (arcsec) | Four-stress MAE (arcsec) |
|---|---:|---:|
| Compact | 0.092197 | 0.117802 |
| Reference | 0.082345 | 0.107728 |

Compact minus reference on the four-stress endpoint was **+0.010075 arcsec**;
the parent-bootstrap 95% interval was **[+0.007529, +0.012557]**. The estimate
is averaged across the five paired seeds; the interval resamples the 150 parents
with seeds held fixed.

| Seed | Compact MAE | Reference MAE |
|---:|---:|---:|
| 730001 | 0.116951 | 0.109919 |
| 730002 | 0.123526 | 0.112743 |
| 730003 | 0.116266 | 0.103415 |
| 730004 | 0.119868 | 0.106952 |
| 730005 | 0.112401 | 0.105610 |

The reference model had lower MAE in both summaries across these five seeds.
This exploratory comparison tests one compact alternative on the same synthetic
holdout; it does not establish a general ranking of CNN architectures or change
the primary claim, because C/G/R keep the architecture fixed.

Machine-readable outputs are in
`results/reports/architecture_ablation_v1/`:

- `report.json` — protocol, model settings, metrics, and limitations;
- `metrics.csv` — condition-level metrics;
- `predictions.csv` — architecture predictions and joined targets used for this
  exploratory score;
- `report.md` — compact result table and per-seed stress errors.
