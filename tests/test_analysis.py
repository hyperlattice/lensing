"""Preserve the recorded comparison and reject settings drift before data access."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ml import analysis, artifacts, predict, train

runner = SimpleNamespace(
    load_config=artifacts.load_config, fit=train.fit, predict=predict.predict, score=analysis.score
)


def test_composite_bootstrap_averages_conditions_before_quantiles():
    candidate = np.tile([1.0, 3.0], (2, 3, 1))
    comparator = np.full_like(candidate, 2.0)
    result = analysis._bootstrap_difference(candidate, comparator, draws=100)
    assert result["difference_arcsec"] == 0.0
    assert result["bootstrap_95_low_arcsec"] == 0.0
    assert result["bootstrap_95_high_arcsec"] == 0.0


def test_composite_bootstrap_matches_direct_with_unequal_condition_effects():
    candidate = np.random.default_rng(12).uniform(size=(3, 5, 4))
    comparator = np.random.default_rng(13).uniform(size=candidate.shape)
    result = analysis._bootstrap_difference(candidate, comparator, seed=42, draws=80)
    indices = np.random.default_rng(42).integers(0, 5, size=(80, 5))
    direct = [candidate[:, ids, :].mean() - comparator[:, ids, :].mean() for ids in indices]
    np.testing.assert_allclose(
        [result["bootstrap_95_low_arcsec"], result["bootstrap_95_high_arcsec"]],
        np.quantile(direct, [0.025, 0.975]),
    )


def seal_config(config):
    root = Path(config["output"])
    root.mkdir(exist_ok=True)
    (root / "resolved_config.json").write_text(json.dumps({"config": config}), encoding="utf-8")


def test_default_comparison_retains_recorded_budget_and_pairing():
    config = runner.load_config()
    assert config["seeds"] == list(range(730001, 730011))
    assert config["arms"] == ["C", "G", "R"]
    assert config["conditions"] == [
        "canonical",
        "pixel-scale-004",
        "pixel-scale-006",
        "exposure-half",
        "read-noise-double",
    ]
    assert config["training"] == {
        "experiment_id": "equal-budget-final-v1",
        "fit_purpose": "confirmatory_fit",
        "optimizer_updates": 880,
        "validation_interval_updates": 440,
        "batch_size": 16,
        "torch_threads": 6,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "huber_delta": 1.0,
        "mmd_weight": 0.0,
        "model": {
            "channels": [8, 16, 32, 64],
            "pooled_size": 4,
            "hidden_features": 128,
            "dropout": 0.1,
        },
    }
    assert config["bootstrap"] == {"resamples": 10000, "stress_seed": 8731, "canonical_seed": 8732}
    assert config["source_train"] == "data/simulations/robustness-study-source-train-v4/bank.json"
    assert (
        config["source_development"]
        == "data/simulations/robustness-study-source-development-v4/bank.json"
    )
    assert config["holdout"] == "data/simulations/reliability-blind-holdout-v1/splits/holdout.csv"


@pytest.mark.parametrize("stage", [runner.fit, runner.predict, runner.score])
def test_changed_settings_rejected_before_reading_scientific_inputs(tmp_path, monkeypatch, stage):
    config = runner.load_config()
    config["output"] = str(tmp_path / "run")
    seal_config(config)
    (Path(config["output"]) / "fit_complete.json").write_text("{}", encoding="utf-8")
    config["training"]["optimizer_updates"] += 1

    def forbidden(*args, **kwargs):
        raise AssertionError("Scientific input was accessed before settings validation")

    monkeypatch.setattr(train, "ObservationBank", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="configuration changed"):
        stage(config)


def test_legacy_run_is_not_given_retroactive_provenance(tmp_path):
    config = runner.load_config()
    config["output"] = str(tmp_path)
    (tmp_path / "fit_complete.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        runner.fit(config)
    assert not (tmp_path / "resolved_config.json").exists()


@pytest.mark.parametrize(
    "change", ["duplicate_seed", "condition_order", "arm_order", "unknown_field"]
)
def test_invalid_comparison_definition_rejected(tmp_path, change):
    config = deepcopy(runner.load_config())
    if change == "duplicate_seed":
        config["seeds"] = [1, 1]
    elif change == "condition_order":
        config["conditions"].reverse()
    elif change == "arm_order":
        config["arms"].reverse()
    else:
        config["unexpected"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_scoring_repeats_from_saved_predictions(tmp_path):
    config = runner.load_config()
    config.update(output=str(tmp_path), holdout=str(tmp_path / "holdout.csv"), seeds=[11, 12])
    config["bootstrap"]["resamples"] = 100
    seal_config(config)
    truth = []
    predictions = []
    for parent in range(3):
        target = 1.0 + parent * 0.2
        for condition in config["conditions"]:
            row = {
                "sample_id": f"p{parent}-{condition}",
                "parent_sample_id": f"p{parent}",
                "condition": condition,
            }
            truth.append({**row, "theta_E_arcsec": target})
            for arm, offset in (("C", 0.2), ("G", 0.1), ("R", 0.08)):
                for seed in config["seeds"]:
                    predictions.append(
                        {
                            **row,
                            "arm": arm,
                            "seed": seed,
                            "theta_E_arcsec_predicted": target + offset + (seed - 11) * 0.01,
                        }
                    )
    holdout = Path(config["holdout"])
    pd.DataFrame(truth).to_csv(holdout, index=False)
    prediction_path = tmp_path / "predictions_before_scoring.csv"
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    (tmp_path / "predictions_complete.json").write_text(
        json.dumps({"prediction_path": str(prediction_path)})
    )
    runner.score(config)
    first = (tmp_path / "final_report.json").read_bytes()
    analysis.score(config, report_output=tmp_path / "corrected_analysis")
    assert (tmp_path / "corrected_analysis" / "final_report.json").read_bytes() == first
    assert (tmp_path / "final_report.json").read_bytes() == first
    runner.score(config)
    assert (tmp_path / "final_report.json").read_bytes() == first
    holdout.write_text(holdout.read_text() + "\n")
    runner.score(config)
    assert (tmp_path / "final_report.json").read_bytes() == first
