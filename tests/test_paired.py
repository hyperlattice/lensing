"""The secondary study keeps sealed prediction and truth access separate."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import paired_release

from src.data.dataset import PairedEvaluationImages
from src.ml import analysis, predict
from src.ml.artifacts import sha256
from src.ml.model import build_model


def test_paired_prediction_analysis_and_identity_audit(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    manifest_path, manifest = paired_release(release)
    root = tmp_path / "secondary"
    root.mkdir()
    method = root / "method_freeze.json"
    method.write_text(json.dumps({"study_id": "fixture"}))
    truth = release / "truth_vault/targets.json"
    truth.parent.mkdir()
    truth.write_text(json.dumps({parent: 1.0 for parent in manifest["parent_ids"]}))
    manifest.update(
        role="power_pilot", method_freeze_sha256=sha256(method), truth_sha256=sha256(truth)
    )
    for row in manifest["observations"]:
        row["role"] = "power_pilot"
    manifest_path.write_text(json.dumps(manifest))
    for renderer in manifest["renderer_order"]:
        (release / f"render-{renderer}.json").write_text(json.dumps({"backend_versions": {}}))
    torch.set_num_threads(2)
    model = build_model(
        {"channels": [2, 4, 4, 4], "pooled_size": 1, "hidden_features": 8, "dropout": 0.0}
    )
    checkpoint = tmp_path / "final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.model_config,
            "statistics": {
                "image_mean": 0.0,
                "image_std": 1.0,
                "target_mean": 1.0,
                "target_std": 0.2,
            },
        },
        checkpoint,
    )
    predictions = root / "predictions_before_scoring.csv"
    for module in (predict, analysis):
        monkeypatch.setattr(module, "PAIRED_ROOT", root)
        monkeypatch.setattr(module, "PAIRED_RELEASE", manifest_path)
        monkeypatch.setattr(module, "PAIRED_PREDICTIONS", predictions)
        monkeypatch.setattr(module, "PAIRED_SEEDS", (11,))
    monkeypatch.setattr(predict, "METHOD", method)
    monkeypatch.setattr(predict, "_checkpoint", lambda arm, seed: checkpoint)
    monkeypatch.setattr(analysis, "PAIRED_AUDIT", root / "renderer_identity_audit.json")
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == truth:
            raise AssertionError("Prediction opened the truth vault")
        return original_read(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_text", guarded_read)
        predict.predict_paired()
    assert len(pd.read_csv(predictions)) == 2 * 9 * 2 * 3
    analysis.score_paired()
    audit = json.loads((root / "renderer_identity_audit.json").read_text())
    assert audit["image_exact_fraction"] == 1.0
    assert audit["prediction_exact_fraction"] == 1.0
    report = json.loads((root / "final_report.json").read_text())
    assert report["parent_count"] == 2
    assert report["seed_count"] == 1
    assert all(np.isfinite(value["difference_arcsec"]) for value in report["contrasts"].values())


def test_paired_release_rejects_different_method_identity(tmp_path):
    path, _ = paired_release(tmp_path)
    with pytest.raises(ValueError, match="frozen method identity"):
        PairedEvaluationImages(path, role="power_pilot", freeze_sha256="different-method")
