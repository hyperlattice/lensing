"""Build the Week 12 paper package from frozen study artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/reports/final_v1"
DOCS = ROOT / "docs/reliability_study/final"
W11 = ROOT / "results/reports/reliability_v1/blind_holdout_week11"
RELEASE = ROOT / "data/simulations/reliability-blind-holdout-v1"
FREEZE = ROOT / "docs/reliability_study/week11_freeze_manifest.json"
CONDITIONS = [
    "canonical",
    "pixel-scale-004",
    "pixel-scale-006",
    "exposure-half",
    "read-noise-double",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def verify_week11_inputs() -> tuple[dict, dict, dict, dict]:
    summary = json.loads((W11 / "summary.json").read_text(encoding="utf-8"))
    completion = json.loads((W11 / "holdout_evaluation_complete.json").read_text(encoding="utf-8"))
    preflight = json.loads((W11 / "holdout_preflight.json").read_text(encoding="utf-8"))
    generation = json.loads((RELEASE / "holdout_generation_manifest.json").read_text(encoding="utf-8"))

    if not completion.get("scored_once") or completion.get("target_join_count") != 1:
        raise ValueError("Week 11 completion marker does not certify one scored pass and one label join")
    if not summary.get("target_joined_once") or not summary.get("predictions_written_before_target_join"):
        raise ValueError("Week 11 summary does not certify target-free predictions before scoring")
    if any(summary.get(key) for key in ("historical_test_accessed", "real_data_labels_accessed", "post_holdout_tuning_or_checkpoint_selection")):
        raise ValueError("Week 11 summary violates the frozen claim boundary")
    if not preflight.get("preflight_passed"):
        raise ValueError("Week 11 preflight did not pass")
    if preflight.get("release_manifest_sha256") != generation.get("release_manifest_sha256"):
        raise ValueError("Week 11 preflight and generated release hashes do not match")
    if summary.get("release_manifest_sha256") != generation.get("release_manifest_sha256"):
        raise ValueError("Week 11 score and generated release hashes do not match")
    if summary.get("freeze_manifest_sha256") != sha256(FREEZE):
        raise ValueError("Week 11 score does not match the recorded freeze manifest")
    if generation.get("release_input_hashes_sha256") != sha256(RELEASE / "release_input_hashes.csv"):
        raise ValueError("Week 11 release input-hash table changed")
    if generation.get("release_manifest_sha256") != sha256(RELEASE / "manifest.csv"):
        raise ValueError("Week 11 root release manifest changed")

    for path_text, expected in summary["prediction_hashes_before_target_join"].items():
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != expected:
            raise ValueError(f"Week 11 target-free prediction hash mismatch: {path_text}")
    if completion.get("prediction_hashes_before_target_join") != summary["prediction_hashes_before_target_join"]:
        raise ValueError("Completion marker and Week 11 prediction hashes differ")
    for key, path in (
        ("scored_output_sha256", W11 / "holdout_scored_predictions.csv"),
        ("scored_per_checkpoint_sha256", W11 / "holdout_scored_per_checkpoint.csv"),
        ("summary_sha256", W11 / "summary.json"),
        ("report_sha256", W11 / "report.md"),
    ):
        if completion.get(key) != sha256(path):
            raise ValueError(f"Week 11 completion marker hash mismatch: {key}")
    if len(pd.read_csv(W11 / "holdout_predictions_before_scoring.csv")) != 1500:
        raise ValueError("Week 11 ensemble prediction row count is not 1,500")
    if len(pd.read_csv(W11 / "holdout_predictions_per_checkpoint_before_scoring.csv")) != 4500:
        raise ValueError("Week 11 per-checkpoint prediction row count is not 4,500")
    return summary, completion, preflight, generation


def build_evidence_ledger() -> pd.DataFrame:
    rows = []

    def add_effects(
        experiment_id,
        path,
        *,
        condition_col,
        effect_col,
        low_col,
        high_col,
        count_col,
        metric,
        unit,
        interpretation,
        limit,
        filter_fn=None,
    ):
        frame = pd.read_csv(ROOT / path)
        if filter_fn is not None:
            frame = frame.loc[filter_fn(frame)]
        for record in frame.to_dict(orient="records"):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "condition_or_contrast": str(record[condition_col]),
                    "metric": metric,
                    "estimate": float(record[effect_col]),
                    "ci95_low": float(record[low_col]),
                    "ci95_high": float(record[high_col]),
                    "unit": unit,
                    "parent_count": int(record[count_col]),
                    "interpretation": interpretation,
                    "limitation": limit,
                    "source_artifact": Path(path).as_posix(),
                }
            )

    add_effects(
        "H1-PIXEL-SCALE",
        "results/reports/reliability_v1/pixel_scale/frozen_baseline/paired_effects.csv",
        condition_col="intervention_id",
        effect_col="paired_mean_delta_mae_arcsec",
        low_col="bootstrap_95_low_arcsec",
        high_col="bootstrap_95_high_arcsec",
        count_col="parent_count",
        metric="paired_mae_change_vs_canonical",
        unit="arcsec",
        interpretation="Positive values indicate higher MAE than canonical.",
        limit="Matched synthetic validation conditions; not a real-instrument estimate.",
    )
    add_effects(
        "H2/H4-OBSERVATION-SYSTEMATICS",
        "results/reports/reliability_v1/observation_systematics/cnn_paired_effects.csv",
        condition_col="intervention_id",
        effect_col="paired_delta_mae_arcsec",
        low_col="bootstrap_95_low_arcsec",
        high_col="bootstrap_95_high_arcsec",
        count_col="parent_count",
        metric="paired_mae_change_vs_canonical",
        unit="arcsec",
        interpretation="Positive values indicate higher MAE than canonical.",
        limit="Gaussian PSF width and the project's synthetic observation/noise model only.",
    )
    add_effects(
        "H3/H5-SOURCE-PHYSICS",
        "results/reports/reliability_v1/source_physics/cnn_paired_effects.csv",
        condition_col="intervention_id",
        effect_col="paired_delta_mae_arcsec",
        low_col="bootstrap_95_low_arcsec",
        high_col="bootstrap_95_high_arcsec",
        count_col="parent_count",
        metric="paired_mae_change_vs_canonical",
        unit="arcsec",
        interpretation="Positive values indicate higher MAE than canonical.",
        limit="Validated Sersic source, source-position, and fixed shear interventions; no alternate mass family.",
    )
    add_effects(
        "W6-SHORTCUTS",
        "results/reports/reliability_v1/baselines_week6/shortcuts/shortcut_paired_effects.csv",
        condition_col="intervention_id",
        effect_col="paired_delta_mae_arcsec",
        low_col="parent_bootstrap_95_low_arcsec",
        high_col="parent_bootstrap_95_high_arcsec",
        count_col="parent_count",
        metric="paired_mae_change_vs_unmodified_image",
        unit="arcsec",
        interpretation="Positive values indicate higher MAE after image intervention.",
        limit="Artificial image ablations; the annulus result shows dependence on information, not an invalid causal shortcut.",
    )
    add_effects(
        "W7-CROSS-SIMULATOR",
        "results/reports/reliability_v1/cross_simulator_week7/paired_effects.csv",
        condition_col="condition",
        effect_col="paired_delta_mae_arcsec",
        low_col="paired_delta_mae_bootstrap_95_low_arcsec",
        high_col="paired_delta_mae_bootstrap_95_high_arcsec",
        count_col="parent_count",
        metric="paired_mae_change_independent_equation_path_minus_lenstronomy",
        unit="arcsec",
        interpretation="Near-zero change indicates agreement for this matched implementation path.",
        limit="Not a fully independent rendering stack; source-light, PSF, detector, and preprocessing components were shared.",
    )
    add_effects(
        "W8-TARGETED-RANDOMIZATION",
        "results/reports/reliability_v1/mitigation/paired_effects.csv",
        condition_col="condition",
        effect_col="paired_delta_mae_arcsec",
        low_col="bootstrap_95_low_arcsec",
        high_col="bootstrap_95_high_arcsec",
        count_col="parent_count",
        metric="paired_mae_change_randomized_minus_canonical_trained",
        unit="arcsec",
        interpretation="Negative values favor the randomized-training model.",
        limit="Fixed development validation parents and the four selected synthetic stress conditions.",
    )
    mitigation_summary = json.loads(
        (ROOT / "results/reports/reliability_v1/mitigation/summary.json").read_text(encoding="utf-8")
    )
    mitigation_decision = mitigation_summary["decision"]
    rows.append(
        {
            "experiment_id": "W8-TARGETED-RANDOMIZATION",
            "condition_or_contrast": "four_stress_composite",
            "metric": "paired_mae_change_randomized_minus_canonical_trained",
            "estimate": float(mitigation_decision["mean_stress_delta_mae_arcsec"]),
            "ci95_low": float(mitigation_decision["stress_bootstrap_95_low_arcsec"]),
            "ci95_high": float(mitigation_decision["stress_bootstrap_95_high_arcsec"]),
            "unit": "arcsec",
            "parent_count": int(mitigation_decision["stress_parent_count"]),
            "interpretation": "Negative values favor the randomized-training model.",
            "limitation": "Fixed development validation parents and the four selected synthetic stress conditions.",
            "source_artifact": "results/reports/reliability_v1/mitigation/summary.json",
        }
    )
    add_effects(
        "W9-HYBRID-INITIALIZATION",
        "results/reports/reliability_v1/hybrid_week9/paired_initializer_effects.csv",
        condition_col="condition",
        effect_col="mean",
        low_col="low",
        high_col="high",
        count_col="parent_count",
        metric="training_mean_optimizer_evaluations_minus_ml_start",
        unit="objective evaluations",
        interpretation="Positive values favor the ML initializer; intervals including zero do not support a speed advantage.",
        limit="Only optimizer-eligible point-geometry fits with three or four detected peaks.",
        filter_fn=lambda frame: frame["metric"] == "function_evaluations",
    )
    add_effects(
        "W11-INDEPENDENT-BLIND-HOLDOUT",
        "results/reports/reliability_v1/blind_holdout_week11/paired_effects.csv",
        condition_col="condition",
        effect_col="mean",
        low_col="bootstrap_95_low",
        high_col="bootstrap_95_high",
        count_col="parent_count",
        metric="paired_mae_change_randomized_minus_canonical_trained",
        unit="arcsec",
        interpretation="Negative values favor the final randomized-training ensemble.",
        limit="Independent parents from the same synthetic prior and renderer; no real images or alternate full renderer.",
        filter_fn=lambda frame: frame["metric"] == "delta_abs_error_randomized_minus_baseline_arcsec",
    )
    return pd.DataFrame(rows)


def build_failure_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = pd.read_csv(W11 / "holdout_scored_predictions.csv")
    randomized = scored.loc[scored["model_group"] == "domain_randomized"].copy()
    baseline = scored.loc[
        scored["model_group"] == "canonical_baseline",
        [
            "parent_sample_id",
            "condition",
            "theta_E_arcsec_predicted",
            "absolute_error_arcsec",
        ],
    ].rename(
        columns={
            "theta_E_arcsec_predicted": "canonical_baseline_prediction_arcsec",
            "absolute_error_arcsec": "canonical_baseline_absolute_error_arcsec",
        }
    )
    catalog = randomized.merge(baseline, on=["parent_sample_id", "condition"], validate="one_to_one")
    release_manifest = RELEASE / "manifest.csv"
    available = set(pd.read_csv(release_manifest, nrows=0).columns)
    desired = [
        "sample_id",
        "condition",
        "pixel_scale",
        "exposure_time",
        "read_noise_electrons",
        "psf_fwhm_arcsec",
        "original_shape",
        "processed_shape",
        "image_count_physical",
        "image_count_detectable",
        "masked_fraction",
    ]
    metadata_columns = [name for name in desired if name in available]
    metadata = pd.read_csv(release_manifest, usecols=metadata_columns)
    catalog = catalog.merge(metadata, on=["sample_id", "condition"], validate="one_to_one")
    error = catalog["absolute_error_arcsec"].astype(float)
    catalog["error_band"] = np.select(
        [error <= 0.05, error <= 0.10],
        ["within_0.05_arcsec", "over_0.05_to_0.10_arcsec"],
        default="over_0.10_arcsec",
    )
    catalog["error_reduction_vs_baseline_arcsec"] = (
        catalog["canonical_baseline_absolute_error_arcsec"] - catalog["absolute_error_arcsec"]
    )
    catalog["rank_by_final_model_absolute_error"] = error.rank(method="first", ascending=False).astype(int)
    catalog["exceeds_0.05_arcsec"] = error > 0.05
    catalog["exceeds_0.10_arcsec"] = error > 0.10
    catalog = catalog.sort_values("absolute_error_arcsec", ascending=False).reset_index(drop=True)
    catalog_path = RESULTS / "tables/failure_case_catalog.csv"
    write_table(catalog_path, catalog)
    counts = (
        catalog.groupby(["condition", "error_band"], observed=False)
        .size()
        .rename("observation_count")
        .reset_index()
    )
    condition_order = {name: index for index, name in enumerate(CONDITIONS)}
    band_order = {"within_0.05_arcsec": 0, "over_0.05_to_0.10_arcsec": 1, "over_0.10_arcsec": 2}
    counts["_condition_order"] = counts["condition"].map(condition_order)
    counts["_band_order"] = counts["error_band"].map(band_order)
    counts = counts.sort_values(["_condition_order", "_band_order"]).drop(columns=["_condition_order", "_band_order"])
    write_table(RESULTS / "tables/failure_counts_by_condition.csv", counts)
    return catalog, counts


def build_figures(ensemble_metrics: pd.DataFrame, paired_effects: pd.DataFrame, catalog: pd.DataFrame) -> None:
    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    effect_metric = "delta_abs_error_randomized_minus_baseline_arcsec"
    effect_order = CONDITIONS + ["four_stress_composite"]
    effect_rows = paired_effects.loc[paired_effects["metric"] == effect_metric].set_index("condition").reindex(effect_order)
    x = np.arange(len(effect_order))
    mean = effect_rows["mean"].to_numpy(float)
    low = effect_rows["bootstrap_95_low"].to_numpy(float)
    high = effect_rows["bootstrap_95_high"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.vlines(x, low, high, color="#315f8c", linewidth=1.8)
    axis.scatter(x, mean, color="#315f8c", s=38, zorder=3)
    axis.axhline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_xticks(x, effect_order, rotation=20, ha="right")
    axis.set_ylabel("Paired MAE change (randomized minus baseline, arcsec)")
    axis.set_title("Independent synthetic holdout: parent-bootstrap 95% intervals")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "week11_paired_effects.png", dpi=180)
    plt.close(fig)

    metrics = ensemble_metrics.set_index(["model_group", "condition"])
    positions = np.arange(len(CONDITIONS))
    width = 0.36
    baseline_mae = [float(metrics.loc[("canonical_baseline", name), "mae_arcsec"]) for name in CONDITIONS]
    randomized_mae = [float(metrics.loc[("domain_randomized", name), "mae_arcsec"]) for name in CONDITIONS]
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.bar(positions - width / 2, baseline_mae, width, label="Canonical-trained ensemble", color="#d99258")
    axis.bar(positions + width / 2, randomized_mae, width, label="Domain-randomized ensemble", color="#527da8")
    axis.set_xticks(positions, CONDITIONS, rotation=20, ha="right")
    axis.set_ylabel("Ensemble MAE (arcsec)")
    axis.set_title("Frozen ensembles on the independent synthetic holdout")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "week11_ensemble_mae.png", dpi=180)
    plt.close(fig)

    data = [catalog.loc[catalog["condition"] == condition, "absolute_error_arcsec"].to_numpy(float) for condition in CONDITIONS]
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.boxplot(data, tick_labels=CONDITIONS, showfliers=False, whis=(5, 95))
    axis.axhline(0.05, color="#666666", linewidth=1, linestyle=":", label="0.05 arcsec reporting threshold")
    axis.axhline(0.10, color="#aa5d39", linewidth=1, linestyle="--", label="0.10 arcsec reporting threshold")
    axis.set_ylabel("Absolute error (arcsec)")
    axis.set_title("Final ensemble error distribution by condition")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "week11_error_distribution.png", dpi=180)
    plt.close(fig)


def build_documents(summary: dict, ledger: pd.DataFrame, counts: pd.DataFrame, curve: pd.DataFrame) -> None:
    random_mae = summary["condition_ensemble_mae_arcsec"]["domain_randomized"]
    baseline_mae = summary["condition_ensemble_mae_arcsec"]["canonical_baseline"]
    composite = summary["equal_weight_four_stress_paired_mae_delta_arcsec"]
    ci_low, ci_high = summary["equal_weight_four_stress_paired_mae_delta_95_ci_arcsec"]
    condition_rows = ledger.loc[
        (ledger["experiment_id"] == "W11-INDEPENDENT-BLIND-HOLDOUT")
        & (ledger["metric"] == "paired_mae_change_randomized_minus_canonical_trained")
    ]
    effect_lookup = condition_rows.set_index("condition_or_contrast")

    result_table = [
        "| Condition | Baseline MAE | Randomized MAE | Paired MAE change | Parent-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    label = {
        "canonical": "Canonical",
        "pixel-scale-004": "Pixel scale 0.04",
        "pixel-scale-006": "Pixel scale 0.06",
        "exposure-half": "Half exposure",
        "read-noise-double": "Double read noise",
    }
    for condition in CONDITIONS:
        row = effect_lookup.loc[condition]
        result_table.append(
            f"| {label[condition]} | {baseline_mae[condition]:.4f} | {random_mae[condition]:.4f} | "
            f"{row['estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    result_table.append(
        f"| Equal-weight four-stress composite | — | — | {composite:+.4f} | [{ci_low:+.4f}, {ci_high:+.4f}] |"
    )

    manuscript = f"""# Reliability of synthetic Einstein-radius regression under controlled observation mismatch

## Abstract

We evaluated a convolutional regressor for the Einstein radius, `theta_E`, using matched synthetic lens observations and a preregistered reliability study. Earlier development experiments found increased validation error at pixel scales 0.04 and 0.06 arcsec/pixel (+0.0707 and +0.0489 arcsec versus canonical), half exposure (+0.0463 arcsec), and doubled read noise (+0.0350 arcsec). Targeted domain-randomized training over those nuisance factors improved the fixed validation stress composite by -0.0290 arcsec. The final model was then frozen and compared with a canonical-trained ensemble on 150 new independent parent systems, each rendered under five conditions (750 observations total). The randomized ensemble's mean absolute error (MAE) was 0.0655 arcsec in the canonical condition and 0.0862–0.1158 arcsec across the four stress conditions. Its equal-weight parent-paired stress MAE change versus the canonical-trained ensemble was **{composite:+.4f} arcsec** (10,000-resample parent-bootstrap 95% interval [{ci_low:+.4f}, {ci_high:+.4f}]). The study supports improved performance over the fixed comparator for these tested synthetic conditions. The holdout uses the same simulator and parent prior; a separate CASTLES comparison was descriptive and did not score real images. These results do not establish real-observation `theta_E` accuracy or close the simulation-to-real gap.

## Introduction

Learned parameter estimates can change when image scale, exposure, noise, source structure, and rendering assumptions shift. We studied one target, `theta_E`, with controlled matched-parent counterfactuals. The research question was whether measured synthetic failure modes could guide a bounded nuisance-randomization intervention and whether the resulting model retained its advantage on a new synthetic holdout. The study was designed to keep validation model selection, final holdout evaluation, cross-implementation checks, and real-domain descriptive evidence distinct.

## Methods

### Synthetic data and model

The core development release contains 1,000 simulated SIE systems partitioned by independent parent. The Week 8 training release contains 1,400 views from 700 training parents: each parent retains its canonical view and adds one view randomized over pixel scale (0.04/0.05/0.06 arcsec per pixel with matched native shapes 320/256/214), exposure multiplier (0.5/1.0), and read-noise multiplier (1.0/2.0). The final predictor is the arithmetic mean of three frozen CNN checkpoints (seeds 12001, 12002, 12003), each using its own training-only normalization. The original canonical-trained three-seed ensemble is a fixed comparator.

### Development perturbations and comparisons

Matched counterfactuals used 150 validation parents. The pixel-scale study compared 0.04 and 0.06 arcsec/pixel with canonical 0.05; the observation study included Gaussian PSF-width changes and observation-quality interventions; source-physics experiments varied Sersic profile/size, source position, and fixed external shear. Week 6 compared the CNN with train-mean, train-only Ridge, and an observation-only point-image geometry control, and evaluated image ablations and learning-curve subsets. The Week 7 alternate path independently implemented the SIE deflection/root/rasterization equations while sharing source-light rendering, PSF, detector noise, and preprocessing. Week 9 compared point-geometry fit initializations. These development results did not use the historical synthetic test split.

### Domain randomization and independent holdout

The Week 8 targeted-randomization decision used validation only. For final evaluation, protocol v1.11 and 117 input hashes were frozen before generation. A fresh seed 11031 produced 150 new parent systems and five matched conditions per parent: canonical, pixel scale 0.04, pixel scale 0.06, half exposure, and double read noise. Root and observation IDs and parameter/noise seeds were checked against development train/validation. The final split contains only `holdout` observations. Release products and manifests passed ancestry, identity, schema, strict-loader, and hash checks before inference.

Both frozen ensembles were evaluated once. Per-checkpoint and ensemble predictions were written to target-free files and hashed before `theta_E` was joined once for scoring. Parent-level paired effects average the four stress-condition error differences within parent for the composite. Confidence intervals use 10,000 bootstrap resamples of 150 independent parents, seed 6031. Descriptive error counts use fixed thresholds of 0.05 and 0.10 arcsec. No model selection or tuning followed holdout scoring.

### CASTLES observable-domain comparison

The Week 10 domain comparison included 137 processed CASTLES images from 69 lens systems and the 1,400-view synthetic training distribution. It measured available metadata and image proxies only. It did not read real `theta_E`/redshift labels or score model predictions.

## Results

### Development sensitivity and baselines

On matched synthetic validation parents, the frozen CNN's MAE increased by +0.0707 arcsec at pixel scale 0.04 and +0.0489 arcsec at 0.06. Half exposure increased MAE by +0.0463 arcsec and doubled read noise by +0.0350 arcsec. Tested Gaussian PSF-width changes were small or inconclusive. Source-position shifts produced increases up to +0.0248 arcsec for the tested x+0.15 arcsec condition; the fixed shear interventions were inconclusive. The observation-only S/N>=5 peak detector matched renderer multiplicity in 46.7–58.0% of Week 5 cases, so detected and physical image counts were not interchangeable.

The 700-parent learning-curve validation MAE was 0.0946 arcsec, compared with 0.1163 at 300 parents and 0.2039 at 100 parents. The train-only Ridge model improved on the train mean but underperformed the CNN on all seven reported conditions. The point-image geometry control had 44/150 fallbacks at canonical conditions and an all-parent MAE of 0.2007 arcsec. Masking the fixed 0.70–2.20 arcsec annulus increased MAE by +0.2632 arcsec (95% parent interval [+0.2246, +0.3022]); this indicates dependence on information in that image region, not an invalid causal shortcut.

### Independent equation-path comparison and mitigation

The Week 7 closed-form SIE deflection differed from Lenstronomy by at most 7.63e-9 arcsec on the tested grid; root counts agreed for 150/150 validation parents. The comparison shares important renderer components and is not evidence for general simulator independence. Week 8 domain-randomized training reduced the four-stress validation MAE by -0.0290 arcsec (95% parent interval [-0.0351, -0.0231]); canonical MAE changed by -0.0109 arcsec ([-0.0189, -0.0031]). Week 9 found no consistent optimizer-evaluation or runtime advantage for the CNN point-fit initializer; the result applies only to optimizer-eligible point-geometry cases.

### Independent synthetic holdout

The holdout contains 150 independent parents and 750 matched observations. Each condition has 150 parents. The final ensemble MAEs and paired changes versus the canonical-trained ensemble are:

{chr(10).join(result_table)}

![Paired errors on the independent holdout](../../../results/reports/final_v1/figures/week11_paired_effects.png)

![Ensemble MAE by condition](../../../results/reports/final_v1/figures/week11_ensemble_mae.png)

All five condition-specific paired error intervals and the four-stress composite interval are below zero. The randomized model had 73/150 canonical errors above 0.05 arcsec and 26/150 above 0.10 arcsec. Under pixel scale 0.04 those counts were 102/150 and 67/150, respectively. The full 750-case final-model catalog, comparator errors, seed spread, and available observation diagnostics are included with the package tables.

### CASTLES coverage

Of the processed CASTLES observations, 52.6% fell within the synthetic pixel-scale p01–p99 range. Only 6.6% fell within the synthetic processed-field-of-view range under the frozen coverage rule. These descriptive measurements have semantic limitations: intensity proxies use different calibration/normalization, radial features use cutout centers, and instrument-specific rendering, calibrated PSF/noise, masks, and reviewed multiplicity were not represented. No real-image prediction accuracy was estimated.

## Limitations

The independent holdout samples the same synthetic parent prior and renderer as development. Its nuisance conditions cover only the selected pixel-scale, exposure, and read-noise changes. Synthetic detector variance is not calibrated HST noise. The Week 7 comparison changes the mass-equation/root path while sharing Sersic light, PSF, noise, and preprocessing. The model predicts one scalar target and does not cover alternate mass families or a population-calibrated prior. The point-image baseline is not a full pixel-likelihood fit and has substantial fallbacks. The 137 CASTLES images were used only for observable metadata/proxy coverage; labels and model scores were not used. The three-seed ensemble spread is not a calibrated predictive interval. Accordingly, no real-observation accuracy, calibrated uncertainty, or simulator-to-real transfer claim is supported.

## Discussion

The study links measured synthetic sensitivity to a targeted training-distribution change and evaluates that change on new parents drawn from the same synthetic proposal. The randomized ensemble retained lower MAE than the fixed canonical-trained comparator on all five holdout conditions, including canonical. That is useful evidence for robustness within the tested simulator and nuisance range. Earlier development results also show why that result needs a narrow boundary: native pixel-scale and observation-quality changes raised error, source-position changes mattered for selected interventions, and a regional mask changed predictions substantially.

The Week 7 implementation agreement checks a limited equation/root/rasterization path. The Week 10 CASTLES audit, by contrast, identifies domain coverage gaps without scoring real lens parameters. Together they motivate a future blinded benchmark using calibrated image products, reviewed lens-center/multiplicity information, independently governed labels, and a specified evaluation split. Such a benchmark is needed before making claims about measured `theta_E` in astronomical observations.

## Conclusion

Targeted nuisance randomization improved paired `theta_E` MAE over a fixed canonical-trained ensemble on an independent synthetic holdout from the same prior and renderer. The experiments identify tested sensitivity mechanisms and show bounded mitigation, while the observable-domain audit leaves material simulation-to-real gaps unresolved. The evidence supports a reproducible synthetic reliability result, not real-image accuracy.

## Source artifacts

- Protocols and decisions: [study protocols](../), [Week 11 freeze](../week11_freeze_manifest.json)
- Week 3–10 phase reports: [pixel scale](../week3_pixel_scale_status.md), [observation systematics](../week4_observation_systematics_status.md), [source physics](../week5_source_physics_status.md), [baselines and shortcuts](../week6_baselines_shortcuts_status.md), [equation path](../week7_cross_simulator_status.md), [mitigation](../week8_mitigation_status.md), [hybrid initialization](../week9_hybrid_status.md), [CASTLES domain gap](../week10_domain_gap_status.md)
- Week 11 report and tables: [frozen holdout results](../../../results/reports/reliability_v1/blind_holdout_week11/report.md)
"""
    (DOCS / "manuscript_draft.md").write_text(manuscript, encoding="utf-8")

    model_card = f"""# Research / model card — `theta_E` synthetic regression

## Model and task

- **Target:** scalar Einstein radius, `theta_E` (arcsec), for simulated SIE lens systems.
- **Model:** arithmetic mean of three fixed convolutional-network checkpoints, seeds 12001, 12002, and 12003.
- **Final training regime:** domain-randomized training on 1,400 views from 700 independent parents, retaining one canonical and adding one randomized view per parent. Normalization is training-only and checkpoint-specific.
- **Fixed comparator:** three-seed canonical-trained CNN ensemble.

## Intended use

Use for controlled synthetic benchmarking of `theta_E` regression under the recorded simulator, prior, preprocessing, and tested nuisance conditions. Use the machine-readable paired outputs and parent counts when comparing conditions.

## Unsupported use

Do not use this evidence to report Einstein radii for real CASTLES/HST observations, estimate real-data uncertainty, claim instrument-calibrated noise robustness, or assert transfer across full renderers. Do not extrapolate beyond the tested SIE/Sersic prior and fixed perturbations.

## Performance on the independent synthetic holdout

The holdout has 150 new parents and 750 observations. The randomized ensemble MAE was {random_mae['canonical']:.4f} arcsec canonical; its equal-weight four-stress paired change versus the fixed comparator was {composite:+.4f} arcsec (95% parent-bootstrap interval [{ci_low:+.4f}, {ci_high:+.4f}]).

| Condition | Randomized ensemble MAE | Fixed baseline MAE |
|---|---:|---:|
{chr(10).join(f"| {label[c]} | {random_mae[c]:.4f} | {baseline_mae[c]:.4f} |" for c in CONDITIONS)}

## Calibration and uncertainty

The standard deviation across three seed predictions is retained as an ensemble-spread descriptor. It was not calibrated against target coverage and must not be interpreted as a predictive interval.

## Known failure modes and limits

- Pixel-scale 0.04 produced the largest randomized-ensemble holdout MAE (0.1158 arcsec); 67/150 errors exceeded 0.10 arcsec.
- Error distributions have substantial tails; see [`failure_case_catalog.csv`](../../../results/reports/final_v1/tables/failure_case_catalog.csv).
- The holdout uses the same synthetic prior and renderer; the CASTLES study was descriptive and did not score labels.
- The tested cross-implementation path shared source-light, PSF, detector, and preprocessing components.
- The point-image classical comparator is a limited geometry fit with fallback behavior, not a pixel-likelihood lens model.

## Data and provenance

The frozen inputs, release hashes, exact-once marker, bootstrap seed, environment, and build procedure are recorded in [`reproducibility_statement.md`](reproducibility_statement.md) and [`package_manifest.json`](../../../results/reports/final_v1/package_manifest.json).
"""
    (DOCS / "model_research_card.md").write_text(model_card, encoding="utf-8")

    claims = [
        ["Native pixel-scale shifts increased error in matched synthetic parents.", "H1-PIXEL-SCALE", "sim-eng1000-v1; 150 validation parents", "Week 3 paired effects; final evidence ledger", "Two scale values in one simulator and prior.", "The tested 0.04 and 0.06 arcsec/pixel conditions increased synthetic validation MAE.", "Pixel scale is the dominant error source in real lens analyses."],
        ["Half exposure and doubled read noise increased synthetic validation error.", "H2-PSF / H4-OBSERVATION-QUALITY", "reliability-observation-systematics-v1; 150 validation parents", "Week 4 paired effects", "Synthetic observation/noise model; Gaussian PSF controls only.", "The tested exposure and read-noise changes increased error under this simulator.", "The result calibrates HST detector noise or establishes PSF-family robustness."],
        ["Some tested source-position and source-size shifts changed error; fixed shear effects were inconclusive.", "H3-SOURCE / W5-SHEAR", "reliability-source-physics-v1; 150 validation parents", "Week 5 paired effects and multiplicity table", "One validated Sersic family and fixed interventions; no alternate mass family.", "Selected source-position/source-size shifts affected synthetic estimates.", "These shifts characterize all source populations or real-lens caustic selection."],
        ["The fixed annulus ablation substantially increased error.", "W6-SHORTCUTS", "canonical synthetic validation; 150 parents", "Week 6 shortcut table", "Artificial fixed-radius mask removes potentially useful lensed-source signal.", "The CNN depends on information in the tested image annulus.", "The model uses an invalid causal shortcut."],
        ["The tested point-image baseline had weaker all-parent performance and fallbacks than the CNN.", "W6-BASELINES", "sim-eng1000-v1 and selected stress conditions", "Week 6 comparison table", "Observation-only SIS/SIE geometry estimate; training-mean fallback; not pixel likelihood.", "The evaluated point-image pipeline underperformed on these synthetic conditions.", "Classical Bayesian/pixel-likelihood modeling is generally inferior."],
        ["Two tested SIE equation/root paths produced closely matching predictions.", "W7-CROSS-SIMULATOR", "reliability-independent-sie-v1; 150 parents, two conditions", "Week 7 controls and paired effects", "Shares source-light, PSF, noise, and preprocessing components.", "The implementations agreed for the tested equation/root/rasterization convention.", "The model is independent of simulator or real-domain shift."],
        ["Targeted nuisance randomization improved fixed-validation performance.", "W8-TARGETED-RANDOMIZATION", "700 training parents; 150 fixed validation parents", "Week 8 paired effects and decision", "Four tested synthetic nuisance factors and one randomized view per parent.", "Randomized training reduced MAE over the tested validation stress composite.", "Domain randomization solves simulation-to-real transfer."],
        ["The randomized ensemble had lower paired MAE than the fixed comparator on the independent holdout.", "W11-INDEPENDENT-BLIND-HOLDOUT", "reliability-blind-holdout-v1; 150 new parents, 750 observations", "Week 11 report; `week11_paired_effects.png`; W11 paired-effects table", "Same synthetic prior/renderer; only five conditions; one exactly-once score.", "On this independent synthetic holdout, the randomized ensemble had lower paired MAE in the tested conditions.", "The model accurately measures Einstein radii in real HST/CASTLES lenses."],
        ["CASTLES feature coverage is incomplete under the frozen observable comparison.", "W10-CASTLES-DOMAIN-GAP", "137 processed observations from 69 lens systems; 1,400 synthetic views", "Coverage map; Week 10 report", "Descriptive image/metadata proxies; no real labels or predictions.", "Pixel-scale overlap was weak and processed angular field coverage was unsupported under the stated rule.", "The CASTLES comparison establishes model accuracy or a calibrated real benchmark."],
        ["The Week 9 CNN initializer showed no consistent evaluation-count speedup.", "W9-HYBRID-INITIALIZATION", "150 validation parents; optimizer-eligible subset", "Week 9 paired initializer effects", "Observation-only point-image fit; optimizer cases only.", "No supported speed advantage was found for the tested point-geometry initializer.", "CNN initialization is generally useless or slower in full lens modeling."],
    ]
    claim_columns = [
        "manuscript_claim",
        "exact_experiment",
        "dataset_release",
        "metric_figure_or_table",
        "limitations",
        "allowed_wording",
        "prohibited_stronger_wording",
    ]
    claim_frame = pd.DataFrame(claims, columns=claim_columns)
    write_table(RESULTS / "tables/claim_evidence_map.csv", claim_frame)
    md_header = "| " + " | ".join(claim_columns) + " |"
    md_separator = "|" + "|".join(["---"] * len(claim_columns)) + "|"
    md_rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in claim_frame.itertuples(index=False, name=None)
    ]
    claim_md = "# Claim-evidence map\n\n" + "\n".join([md_header, md_separator, *md_rows]) + "\n"
    (DOCS / "claim_evidence_map.md").write_text(claim_md, encoding="utf-8")

    reproducibility = f"""# Reproducibility statement

## Frozen final evaluation

- **Protocol/config:** `docs/reliability_study/protocol_v1.11.md` and `configs/reliability_blind_holdout_v1.yaml`.
- **Freeze manifest:** [`week11_freeze_manifest.json`](../week11_freeze_manifest.json), SHA-256 `{summary['freeze_manifest_sha256']}`.
- **Holdout:** 150 independent parents; seed 11031; 750 rows in canonical, pixel-scale 0.04, pixel-scale 0.06, half-exposure, and double-read-noise conditions.
- **Models:** fixed checkpoint seeds 12001/12002/12003 for both ensembles; checkpoint and normalization hashes are recorded in the freeze manifest and prediction files.
- **Bootstrap:** 10,000 parent resamples, seed 6031.
- **Environment:** Python {summary['evaluation_environment']['python']}, PyTorch {summary['evaluation_environment']['torch']}, CPU, {summary['evaluation_environment']['torch_num_threads']} Torch threads.
- **Prediction provenance:** 4,500 per-checkpoint rows and 1,500 ensemble rows were saved target-free and hashed before the single label join. The completion marker certifies one scored pass. Do not rerun holdout inference.
- **Quarantine:** no historical test data or real-parameter labels were accessed; no post-holdout model selection or tuning occurred.

## Regeneration

The synthetic study workflows and phase-specific commands are recorded under `docs/reliability_study/` and their frozen protocols/configs. The Week 11 sequence was:

```text
python -m src.ml.reliability_week11 generate
python -m src.ml.reliability_week11 preflight
python -m src.ml.reliability_week11 evaluate
```

The `evaluate` command is exactly-once for this release. Regenerate tables, figures, and documents from the saved result files with:

```text
python scripts/build_reliability_final_package.py
```

The freeze and package manifests record the relevant code, protocol, config, release, checkpoint, normalization, prediction, and output hashes. The package does not rerun model inference.
"""
    (DOCS / "reproducibility_statement.md").write_text(reproducibility, encoding="utf-8")

    status = f"""# Week 12 status — final analysis and manuscript package

Week 12 is complete. The package is built from the recorded Week 1–11 protocols, phase reports, machine-readable effects, the Week 11 exactly-once predictions/score, and the frozen release manifest. It does not rerun model inference or select a model from holdout results.

## Delivered

- Harmonized paired-effect evidence ledger covering Weeks 3–11.
- Final holdout ensemble/seed metrics, paired effects, target-range coverage, and fixed-threshold error counts.
- Full 750-observation final-model failure catalog with comparator error, seed spread, condition, error tier, and available observation diagnostics; top-error table and error-count summary.
- Three reproducible figures: parent-paired effects, ensemble MAE, and error distributions.
- Research/model card, manuscript draft, claim-evidence map, and reproducibility statement.
- Input/output package manifest with SHA-256 hashes.

The paper states Limitations before Discussion and distinguishes synthetic in-domain, synthetic stress, the limited Week 7 equation path, and descriptive CASTLES evidence. It makes no real-image accuracy claim.

## Primary final result

The randomized ensemble's equal-weight four-stress paired MAE change versus the fixed canonical-trained ensemble was **{composite:+.4f} arcsec** (95% parent-bootstrap interval [{ci_low:+.4f}, {ci_high:+.4f}]) on 150 independent synthetic parents. This supports a result for the tested simulator/prior and conditions only.

## Package files

- [Manuscript draft](final/manuscript_draft.md)
- [Research/model card](final/model_research_card.md)
- [Claim-evidence map](final/claim_evidence_map.md)
- [Reproducibility statement](final/reproducibility_statement.md)
- [Tables and figures](../../results/reports/final_v1/)
- [Package manifest](../../results/reports/final_v1/package_manifest.json)
"""
    (ROOT / "docs/reliability_study/week12_final_package_status.md").write_text(status, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    summary, completion, preflight, generation = verify_week11_inputs()

    metrics = pd.read_csv(W11 / "metrics_by_condition_seed_and_ensemble.csv")
    ensemble_metrics = metrics.loc[metrics["predictor"] == "three_seed_ensemble"].copy()
    seed_metrics = metrics.loc[metrics["predictor"] != "three_seed_ensemble"].copy()
    ensemble_metrics["_condition_order"] = ensemble_metrics["condition"].map({name: index for index, name in enumerate(CONDITIONS)})
    ensemble_metrics["_model_order"] = ensemble_metrics["model_group"].map({"canonical_baseline": 0, "domain_randomized": 1})
    ensemble_metrics = ensemble_metrics.sort_values(["_condition_order", "_model_order"]).drop(columns=["_condition_order", "_model_order"])
    seed_metrics["_condition_order"] = seed_metrics["condition"].map({name: index for index, name in enumerate(CONDITIONS)})
    seed_metrics = seed_metrics.sort_values(["_condition_order", "model_group", "predictor"]).drop(columns=["_condition_order"])
    paired_effects = pd.read_csv(W11 / "paired_effects.csv")
    target_range = pd.read_csv(W11 / "target_range_coverage.csv")
    ledger = build_evidence_ledger()
    catalog, failure_counts = build_failure_catalog()
    learning_curve_json = json.loads((ROOT / "results/reports/reliability_v1/baselines_week6/learning_curve/learning_curve_summary.json").read_text(encoding="utf-8"))
    learning_curve = pd.DataFrame(learning_curve_json["curve"])

    write_table(RESULTS / "tables/week11_ensemble_metrics.csv", ensemble_metrics)
    write_table(RESULTS / "tables/week11_seed_metrics.csv", seed_metrics)
    write_table(RESULTS / "tables/week11_paired_effects.csv", paired_effects)
    write_table(RESULTS / "tables/week11_target_range_coverage.csv", target_range)
    write_table(RESULTS / "tables/study_paired_effect_ledger.csv", ledger)
    write_table(RESULTS / "tables/learning_curve.csv", learning_curve)
    write_table(RESULTS / "tables/week11_largest_30_errors.csv", catalog.head(30))
    domain_coverage = pd.read_csv(ROOT / "results/reports/domain_gap_v1/coverage_map.csv")
    write_table(RESULTS / "tables/domain_gap_coverage.csv", domain_coverage)

    build_figures(ensemble_metrics, paired_effects, catalog)
    build_documents(summary, ledger, failure_counts, learning_curve)

    composite = float(summary["equal_weight_four_stress_paired_mae_delta_arcsec"])
    ci = [float(value) for value in summary["equal_weight_four_stress_paired_mae_delta_95_ci_arcsec"]]
    final_summary = {
        "package_id": "RELIABILITY-STUDY-FINAL-V1",
        "week11_release_id": generation["release_id"],
        "week11_parent_count": summary["parent_count"],
        "week11_observation_count": summary["observation_count"],
        "week11_target_join_count": completion["target_join_count"],
        "week11_checkpoint_count": summary["evaluated_checkpoint_count"],
        "week11_four_stress_paired_mae_delta_arcsec": composite,
        "week11_four_stress_parent_bootstrap_95_ci_arcsec": ci,
        "week11_condition_mae_arcsec": summary["condition_ensemble_mae_arcsec"],
        "week11_historical_test_accessed": False,
        "week11_real_data_labels_accessed": False,
        "week11_post_holdout_tuning": False,
        "failure_catalog_rows": int(len(catalog)),
        "failure_rows_over_005_arcsec": int(catalog["exceeds_0.05_arcsec"].sum()),
        "failure_rows_over_010_arcsec": int(catalog["exceeds_0.10_arcsec"].sum()),
        "week10_real_observation_count": 137,
        "week10_real_lens_system_count": 69,
        "week10_real_labels_or_predictions_used": False,
        "scope": "Synthetic prior/renderer and preregistered stress conditions; CASTLES analysis is descriptive observable coverage only.",
        "manuscript_section_order": ["Abstract", "Introduction", "Methods", "Results", "Limitations", "Discussion", "Conclusion"],
    }
    write_json(RESULTS / "analysis_summary.json", final_summary)

    source_paths = {
        Path("gravitational_lensing_field_analysis_and_contribution_plan (1).md"),
        Path("docs/reliability_study/baseline_state.md"),
        Path("docs/reliability_study/week2_counterfactual.md"),
        *(Path(f"docs/reliability_study/week{week}_{name}.md") for week, name in [
            (3, "pixel_scale_status"),
            (4, "observation_systematics_status"),
            (5, "source_physics_status"),
            (6, "baselines_shortcuts_status"),
            (7, "cross_simulator_status"),
            (8, "mitigation_status"),
            (9, "hybrid_status"),
            (10, "domain_gap_status"),
            (11, "blind_holdout_status"),
        ]),
        *(Path(f"docs/reliability_study/protocol_v1{suffix}.md") for suffix in ("", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9", ".10", ".11")),
        Path("docs/reliability_study/week11_freeze_manifest.json"),
        Path("docs/reliability_study/experiment_registry.csv"),
        Path("results/reports/reliability_v1/pixel_scale/frozen_baseline/paired_effects.csv"),
        Path("results/reports/reliability_v1/observation_systematics/cnn_paired_effects.csv"),
        Path("results/reports/reliability_v1/source_physics/cnn_paired_effects.csv"),
        Path("results/reports/reliability_v1/baselines_week6/shortcuts/shortcut_paired_effects.csv"),
        Path("results/reports/reliability_v1/baselines_week6/learning_curve/learning_curve_summary.json"),
        Path("results/reports/reliability_v1/cross_simulator_week7/paired_effects.csv"),
        Path("results/reports/reliability_v1/cross_simulator_week7/summary.json"),
        Path("results/reports/reliability_v1/mitigation/paired_effects.csv"),
        Path("results/reports/reliability_v1/mitigation/summary.json"),
        Path("results/reports/reliability_v1/hybrid_week9/paired_initializer_effects.csv"),
        Path("results/reports/reliability_v1/hybrid_week9/summary.json"),
        Path("results/reports/domain_gap_v1/coverage_map.csv"),
        Path("results/reports/domain_gap_v1/summary.json"),
        Path("results/reports/reliability_v1/blind_holdout_week11/summary.json"),
        Path("results/reports/reliability_v1/blind_holdout_week11/report.md"),
        Path("results/reports/reliability_v1/blind_holdout_week11/holdout_evaluation_complete.json"),
        Path("results/reports/reliability_v1/blind_holdout_week11/holdout_preflight.json"),
        Path("results/reports/reliability_v1/blind_holdout_week11/holdout_predictions_before_scoring.csv"),
        Path("results/reports/reliability_v1/blind_holdout_week11/holdout_predictions_per_checkpoint_before_scoring.csv"),
        Path("results/reports/reliability_v1/blind_holdout_week11/holdout_scored_predictions.csv"),
        Path("results/reports/reliability_v1/blind_holdout_week11/metrics_by_condition_seed_and_ensemble.csv"),
        Path("results/reports/reliability_v1/blind_holdout_week11/paired_effects.csv"),
        Path("results/reports/reliability_v1/blind_holdout_week11/target_range_coverage.csv"),
        Path("data/simulations/reliability-blind-holdout-v1/holdout_generation_manifest.json"),
        Path("data/simulations/reliability-blind-holdout-v1/release_input_hashes.csv"),
        Path("data/simulations/reliability-blind-holdout-v1/manifest.csv"),
        Path(__file__).resolve().relative_to(ROOT),
    }
    missing = sorted(path for path in source_paths if not (ROOT / path).is_file())
    if missing:
        raise FileNotFoundError("Missing package input(s): " + ", ".join(str(path) for path in missing))
    input_manifest = {
        "manifest_version": 1,
        "package_id": "RELIABILITY-STUDY-FINAL-V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "week11_freeze_manifest_sha256": sha256(FREEZE),
        "week11_completion_marker_sha256": sha256(W11 / "holdout_evaluation_complete.json"),
        "inputs_sha256": {path.as_posix(): sha256(ROOT / path) for path in sorted(source_paths)},
    }
    write_json(RESULTS / "input_manifest.json", input_manifest)

    output_paths = sorted(
        [path for path in RESULTS.rglob("*") if path.is_file() and path.name != "package_manifest.json"]
        + [path for path in DOCS.rglob("*") if path.is_file()]
        + [ROOT / "docs/reliability_study/week12_final_package_status.md"]
    )
    output_manifest = {
        "manifest_version": 1,
        "package_id": "RELIABILITY-STUDY-FINAL-V1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "inputs_manifest_sha256": sha256(RESULTS / "input_manifest.json"),
        "outputs_sha256": {repo_path(path): sha256(path) for path in output_paths},
        "week11_exact_once_marker_sha256": sha256(W11 / "holdout_evaluation_complete.json"),
    }
    write_json(RESULTS / "package_manifest.json", output_manifest)
    print(
        json.dumps(
            {
                "package_id": final_summary["package_id"],
                "week11_four_stress_delta_arcsec": composite,
                "week11_parent_bootstrap_95_ci_arcsec": ci,
                "failure_catalog_rows": len(catalog),
                "input_hash_count": len(input_manifest["inputs_sha256"]),
                "output_hash_count": len(output_manifest["outputs_sha256"]),
                "manifest": "results/reports/final_v1/package_manifest.json",
                "manuscript": "docs/reliability_study/final/manuscript_draft.md",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
