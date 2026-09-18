import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.ml.data import TorchLensDataset, compute_training_statistics
from src.ml.evaluate import evaluate_checkpoint
from src.ml.experiments import aggregate_runs, load_registered_experiment, run_seed_suite, validate_suite
from src.ml.metrics import improvement_fraction, mean_baseline_metrics, regression_metrics, target_bin_metrics
from src.ml.data import file_sha256
from src.ml.model import EinsteinRadiusCNN
from src.ml.train import train_experiment


def _write_release(root):
    root = Path(root)
    (root / "splits").mkdir(parents=True)
    rows = []
    assignments = {
        "train": ((1.0, 0.6), (2.0, 1.0), (3.0, 1.4), (4.0, 1.8)),
        "validation": ((1.5, 0.8), (3.5, 1.6)),
        "test": ((1.25, 0.7), (3.75, 1.7)),
    }
    for split, values in assignments.items():
        split_rows = []
        for index, (pixel_value, target) in enumerate(values):
            sample_id = f"{split}-{index}"
            raw_path = root / f"{sample_id}.fits"
            image_path = root / f"{sample_id}.npy"
            raw_path.write_bytes(b"fixture")
            np.save(image_path, np.full((256, 256), pixel_value, dtype=np.float32))
            row = {
                "sample_id": sample_id,
                "lens_id": sample_id,
                "raw_path": raw_path.name,
                "processed_path": image_path.name,
                "pixel_scale": 0.05,
                "domain": "simulated",
                "eligible_for": "simulation;ml",
                "theta_E_arcsec": target,
                "theta_E_target_available": True,
                "mass_target_available": False,
                "parameter_seed": index,
                "source_x_arcsec": 100.0,
                "image_count_physical": 2 if index % 2 == 0 else 4,
                "psf_fwhm_arcsec": 0.12 + 0.02 * index,
            }
            rows.append(row)
            split_rows.append(row)
        pd.DataFrame(split_rows).to_csv(root / "splits" / f"{split}.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    return root


def test_cnn_forward_shape_and_validation():
    model = EinsteinRadiusCNN(channels=(2, 4, 8, 8), pooled_size=2, hidden_features=8)
    assert model(torch.ones((3, 1, 64, 64))).shape == (3,)
    assert model.trainable_parameter_count > 0
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones((3, 64, 64)))


def test_torch_adapter_uses_training_only_statistics_and_returns_no_provenance(tmp_path):
    release = _write_release(tmp_path / "release")
    raw = TorchLensDataset(release, split="train")
    statistics = compute_training_statistics(raw)
    dataset = TorchLensDataset(release, split="train", statistics=statistics)
    sample = dataset[0]

    assert statistics["image_mean"] == pytest.approx(2.5)
    assert statistics["target_mean"] == pytest.approx(1.2)
    assert set(sample) == {"image", "target", "target_raw", "sample_id"}
    assert sample["image"].shape == (1, 256, 256)
    assert "parameter_seed" not in sample
    assert "source_x_arcsec" not in sample
    with pytest.raises(ValueError, match="train split"):
        compute_training_statistics(TorchLensDataset(release, split="validation"))


def test_regression_metrics_compare_with_mean_baseline():
    targets = np.array([0.5, 1.0, 1.5])
    predictions = np.array([0.6, 0.9, 1.4])
    metrics = regression_metrics(targets, predictions)
    baseline = mean_baseline_metrics(1.0, targets)
    assert metrics["mae_arcsec"] == pytest.approx(0.1)
    assert metrics["bias_arcsec"] == pytest.approx(-1.0 / 30.0)
    assert improvement_fraction(metrics["mae_arcsec"], baseline["mae_arcsec"]) > 0


def test_one_epoch_training_and_evaluation_smoke(tmp_path):
    release = _write_release(tmp_path / "release")
    output = tmp_path / "run"
    config = {
        "experiment_id": "test-baseline",
        "release": release.as_posix(),
        "output": output.as_posix(),
        "target_field": "theta_E_arcsec",
        "seed": 7,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "huber_delta": 1.0,
        "patience": 1,
        "num_workers": 0,
        "model": {"channels": [2, 4, 8, 8], "pooled_size": 2, "hidden_features": 8},
    }
    summary = train_experiment(config, device_request="cpu")
    report = evaluate_checkpoint(
        output / "best.pt",
        split="test",
        output_dir=output / "test",
        device_request="cpu",
        batch_size=2,
    )

    assert summary["test_samples_accessed"] == 0
    assert summary["validation_metrics"]["count"] == 2
    assert report["metrics"]["count"] == 2
    assert (output / "training_curves.png").exists()
    assert (output / "test" / "predictions.csv").exists()
    assert summary["seed"] == 7
    assert summary["checkpoint_sha256"] == file_sha256(output / "best.pt")
    assert summary["environment"]["packages"]["torch"]
    assert summary["provenance"]["source_sha256"]["src/ml/train.py"] == file_sha256("src/ml/train.py")


def test_tail_metrics_and_fixed_bin_boundaries():
    targets = np.array([0.4, 0.5, 1.0, 1.5, 2.0, 2.1])
    predictions = targets + np.arange(6)
    metrics = regression_metrics(targets, predictions)
    assert metrics["p90_absolute_error_arcsec"] == pytest.approx(4.5)
    assert metrics["p95_absolute_error_arcsec"] == pytest.approx(4.75)
    bins = target_bin_metrics(targets, predictions, [0.5, 1.0, 1.5, 2.0])
    assert [row["count"] for row in bins] == [1, 1, 1, 2, 1]
    assert bins[3]["metrics"]["bias_arcsec"] == pytest.approx(3.5)
    assert bins[0]["metrics"]["r2"] is None
    empty = target_bin_metrics([1.1], [1.2], [0.5, 1.0, 1.5, 2.0])
    assert empty[1]["metrics"] is None
    with pytest.raises(ValueError, match="strictly increasing"):
        target_bin_metrics(targets, predictions, [0.5, 1.0, 1.0])


def _seed_suite(root):
    release = _write_release(root / "release")
    base = {
        "experiment_id": "fixture-baseline", "release": release.as_posix(),
        "output": (root / "original-baseline").as_posix(), "target_field": "theta_E_arcsec",
        "seed": 7, "epochs": 1, "batch_size": 2, "learning_rate": 0.001,
        "weight_decay": 0.0, "huber_delta": 1.0, "patience": 1, "num_workers": 0,
        "model": {"channels": [2, 4, 8, 8], "pooled_size": 2, "hidden_features": 8},
    }
    path = root / "baseline.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return {
        "schema_version": 1, "experiment_id": "fixture-seeds", "parent_baseline_id": "fixture-baseline",
        "base_config": path.as_posix(), "output": (root / "suite").as_posix(), "seeds": [17, 18],
        "target_bin_edges_arcsec": [0.5, 1.0, 1.5, 2.0],
    }


def test_seed_suite_trains_validation_only_resumes_and_rejects_changed_artifacts(tmp_path, monkeypatch):
    suite = _seed_suite(tmp_path)
    output = Path(suite["output"])
    opened_splits = []

    def guarded_dataset(*args, **kwargs):
        opened_splits.append(kwargs["split"])
        assert kwargs["split"] in {"train", "validation"}
        return TorchLensDataset(*args, **kwargs)

    monkeypatch.setattr("src.ml.train.TorchLensDataset", guarded_dataset)
    report = run_seed_suite(suite, device_request="cpu")
    assert report["status"] == "completed_validation_benchmark"
    assert report["test_samples_accessed"] == 0
    assert set(opened_splits) == {"train", "validation"}
    assert report["seeds"] == [17, 18]
    maes = [run["validation_metrics"]["mae_arcsec"] for run in report["runs"]]
    assert report["validation_metrics"]["mae_arcsec"]["mean"] == pytest.approx(np.mean(maes))
    assert report["validation_metrics"]["mae_arcsec"]["sample_std"] == pytest.approx(np.std(maes, ddof=1))
    assert (output / "seed_comparison.png").exists()
    assert (output / "report.md").exists()
    assert len(pd.read_csv(output / "per_seed_metrics.csv")) == 2
    from src.ml.compare import compare_suites
    comparison = compare_suites(output, output, tmp_path / "comparison")
    assert comparison["passes_gate"] is False
    assert comparison["metrics"]["mae_arcsec"]["delta"] == 0
    assert not (tmp_path / "original-baseline").exists()

    def no_training(*args, **kwargs):
        pytest.fail("Resume should reuse both verified checkpoints")

    monkeypatch.setattr("src.ml.experiments.train_experiment", no_training)
    resumed = run_seed_suite(suite, device_request="cpu", resume=True)
    assert resumed == report
    with pytest.raises(FileExistsError):
        run_seed_suite(suite, device_request="cpu")
    altered_suite = {**suite, "seeds": [17, 19]}
    with pytest.raises(ValueError, match="Cannot resume"):
        run_seed_suite(altered_suite, device_request="cpu", resume=True)

    incompatible_runs = copy.deepcopy(report["runs"])
    incompatible_runs[1]["test_samples_accessed"] = 1
    with pytest.raises(ValueError, match="validation-only"):
        aggregate_runs(incompatible_runs)
    incompatible_runs[1]["test_samples_accessed"] = 0
    incompatible_runs[1]["split_manifest_sha256"]["validation"] = "changed"
    with pytest.raises(ValueError, match="split_manifest_sha256"):
        aggregate_runs(incompatible_runs)

    checkpoint = output / "seed-17" / "best.pt"
    checkpoint.write_bytes(b"altered checkpoint")
    with pytest.raises(ValueError, match="Artifact hash changed"):
        run_seed_suite(suite, device_request="cpu", resume=True)


def test_smoke_run_label_and_controls(tmp_path):
    suite = _seed_suite(tmp_path)
    report = run_seed_suite(suite, device_request="cpu", smoke=True)
    assert report["status"] == "smoke_only"
    output = Path(suite["output"] + "-smoke")
    assert (output / "resolved_suite.json").exists()
    config = json.loads((output / "seed-17" / "resolved_config.json").read_text())
    assert config["epochs"] == 2
    assert report["runs"][0]["run_controls"] == {"max_train_samples": 32, "max_validation_samples": 16}
    with pytest.raises(ValueError, match="Cannot resume"):
        run_seed_suite(suite, output_override=output, device_request="cpu", resume=True)


@pytest.mark.parametrize("seeds", [[1], [1, 1], [1, -2], [1, True], [1, 2**32], [1, 2.5]])
def test_suite_rejects_invalid_seed_lists(seeds):
    suite = load_registered_experiment("theta-e-cnn-seeds-v1")
    suite["seeds"] = seeds
    with pytest.raises(ValueError):
        validate_suite(suite)


def test_registry_rejects_unknown_experiment():
    with pytest.raises(ValueError, match="Unknown experiment"):
        load_registered_experiment("not-registered")
