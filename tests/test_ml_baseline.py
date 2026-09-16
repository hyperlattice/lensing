from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.ml.data import TorchLensDataset, compute_training_statistics
from src.ml.evaluate import evaluate_checkpoint
from src.ml.metrics import improvement_fraction, mean_baseline_metrics, regression_metrics
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
