"""Scientific controls for versioned, source-only matched training."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from test_ml_baseline import _seed_suite, _write_release
from src.ml.experiments_v2 import run_seed_suite
from src.ml.matched_training import paired_schedule, run


def test_versioned_suite_complete_control_schema_and_resume(tmp_path):
    torch.set_num_threads(2)
    suite = _seed_suite(tmp_path)
    report = run_seed_suite(suite, device_request="cpu")
    output = Path(suite["output"] + "-controls-v2")
    assert not Path(suite["output"]).exists()
    assert set(report["runs"][0]["run_controls"]) == {
        "max_train_samples", "max_validation_samples", "train_sample_count",
        "train_sample_ids_sha256", "sample_selection_id"}
    assert run_seed_suite(suite, device_request="cpu", resume=True)["seed_count"] == 2
    path = output / "seed-17" / "training_summary.json"
    summary = json.loads(path.read_text())
    summary["run_controls"]["sample_selection_id"] = "changed"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="run_controls changed"):
        run_seed_suite(suite, device_request="cpu", resume=True)


def test_schedule_deterministic_full_batches_parent_balance_and_geometry():
    parents = ["a", "b", "a", "b"]
    shapes = [(32, 32), (32, 32), (64, 64), (64, 64)]
    kwargs = {"updates": 100, "batch_size": 7, "seed": 19}
    schedule = paired_schedule(shapes, parents, **kwargs)
    assert schedule == paired_schedule(shapes, parents, **kwargs)
    assert len(schedule) == 100
    assert all(len(batch) == 7 and len({shapes[i] for i in batch}) == 1 for batch in schedule)
    with pytest.raises(ValueError, match="same number of views"):
        paired_schedule(shapes[:3], parents[:3], **kwargs)


def test_matched_run_pairs_actual_updates_parents_normalization_and_initialization(tmp_path):
    canonical = _write_release(tmp_path / "canonical")
    for path in [canonical / "manifest.csv", *(canonical / "splits").glob("*.csv")]:
        frame = pd.read_csv(path)
        frame["parent_system_id"] = frame["sample_id"]
        frame["original_parent_sample_id"] = frame["sample_id"]
        frame.to_csv(path, index=False)
    varied = tmp_path / "varied"
    shutil.copytree(canonical, varied)
    train = pd.read_csv(varied / "splits/train.csv")
    extra = train.copy()
    for index in extra.index:
        sample = str(extra.loc[index, "sample_id"]) + "__varied"
        image_path = sample + ".npy"
        # Deliberately change geometry and brightness to catch arm-specific normalization.
        np.save(varied / image_path, np.full((128, 128), 10 + index, dtype=np.float32))
        extra.loc[index, "sample_id"] = sample
        extra.loc[index, "processed_path"] = image_path
    pd.concat([train, extra]).to_csv(varied / "splits/train.csv", index=False)
    others = pd.read_csv(varied / "manifest.csv")
    pd.concat([others, extra]).to_csv(varied / "manifest.csv", index=False)
    config = yaml.safe_load(Path("configs/ml/matched_budget_v1.yaml").read_text())
    config.update(canonical_release=str(canonical), randomized_release=str(varied),
                  output=str(tmp_path / "out"), seeds=[17], optimizer_updates=2,
                  validation_interval_updates=1, batch_size=2, torch_threads=2,
                  model={"channels": [2, 4, 8, 8], "pooled_size": 2, "hidden_features": 8})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run(path)
    a, b = result["runs"]
    assert a["run_controls"] == b["run_controls"]
    assert a["run_controls"]["optimizer_updates"] == 2
    assert a["run_controls"]["image_exposures"] == 4
    assert result["test_images_accessed"] == result["target_simulator_images_accessed"] == 0
    assert result["independent_training_parents"] == 4
    with pytest.raises(FileExistsError, match="No implicit resume"):
        run(path)
    config["output"] = str(tmp_path / "repeat")
    path.write_text(yaml.safe_dump(config))
    repeated = run(path)
    for arm in ("canonical", "randomized"):
        first = pd.read_csv(tmp_path / "out" / "seed-17" / arm / "final_validation_predictions.csv")
        second = pd.read_csv(tmp_path / "repeat" / "seed-17" / arm / "final_validation_predictions.csv")
        pd.testing.assert_frame_equal(first, second, check_exact=True)
