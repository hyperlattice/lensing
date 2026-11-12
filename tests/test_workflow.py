"""Run a tiny C/G/R study with explicit fixtures, keeping all outputs temporary."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_bank

from src.data.dataset import ObservationBank
from src.ml import analysis, predict, train
from src.ml.artifacts import load_config


def test_train_predict_analyze_and_plot_without_production_access(tmp_path, monkeypatch):
    from src.ml.plots import plot

    source = fixture_bank(tmp_path / "source", "source_train", 1000)
    development = fixture_bank(tmp_path / "development", "source_development", 2000)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"probabilities": [1 / 7] * 7}))
    rows = []
    # The retained primary manifest uses release-relative paths two levels up.
    holdout = tmp_path / "release" / "splits" / "holdout.csv"
    holdout.parent.mkdir(parents=True)
    for parent in range(150):
        for condition in load_config()["conditions"]:
            sample_id = f"p{parent}-{condition}"
            image = holdout.parent.parent / f"{sample_id}.npy"
            np.save(image, np.full((32, 32), 0.4 + parent * 0.001, np.float32))
            rows.append(
                {
                    "sample_id": sample_id,
                    "parent_sample_id": f"p{parent}",
                    "condition": condition,
                    "processed_path": image.name,
                    "theta_E_arcsec": 0.8 + parent * 0.002,
                }
            )
    pd.DataFrame(rows).to_csv(holdout, index=False)
    config = load_config()
    config.update(
        output=str(tmp_path / "results"),
        source_train=str(source),
        source_development=str(development),
        policy=str(policy),
        holdout=str(holdout),
        seeds=[11],
    )
    config["training"].update(
        optimizer_updates=1,
        validation_interval_updates=1,
        batch_size=2,
        torch_threads=2,
        model={"channels": [2, 4, 4, 4], "pooled_size": 1, "hidden_features": 8, "dropout": 0.0},
    )
    config["bootstrap"]["resamples"] = 20
    actual_bank = ObservationBank

    def fixture_loader(path, *, expected_role):
        return actual_bank(path, expected_role=expected_role, allow_fixture=True)

    monkeypatch.setattr(train, "ObservationBank", fixture_loader)
    actual_train_arm = train.train_arm

    def fixture_train(*args, **kwargs):
        return actual_train_arm(*args, **kwargs, allow_fixture=True)

    monkeypatch.setattr(train, "train_arm", fixture_train)
    train.fit(config)
    recorded_reads = []
    actual_read_csv = pd.read_csv

    def record_read(path, **kwargs):
        recorded_reads.append(kwargs.get("usecols"))
        return actual_read_csv(path, **kwargs)

    monkeypatch.setattr(predict.pd, "read_csv", record_read)
    predict.predict(config)
    assert recorded_reads == [["sample_id", "parent_sample_id", "condition", "processed_path"]]
    root = Path(config["output"])
    assert (root / "predictions_complete.json").exists()
    analysis.score(config)
    report = json.loads((root / "final_report.json").read_text())
    assert report["parent_count"] == 150
    assert report["seed_count"] == 1
    assert np.isfinite([item["difference_arcsec"] for item in report["contrasts"].values()]).all()
    plot(config)
    assert (root / "figures/equal_budget_mae_by_condition.png").exists()
    assert (root / "figures/equal_budget_paired_contrasts.pdf").exists()


@pytest.mark.parametrize("action", ["generate", "develop", "train", "predict", "analyze", "plot"])
def test_command_help_does_not_load_data(action):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.study", action, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
