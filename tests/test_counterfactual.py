import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from src.simulation.counterfactual import (
    apply_intervention,
    create_counterfactual_benchmark,
    validate_counterfactual_ancestry,
)
from src.simulation.generate import generate_observation_pilot
from src.ml.pixel_scale_audit import resample_image_to_angular_scale


def test_intervention_preserves_target_and_applies_declared_multiplier():
    state = {
        "theta_E_arcsec": 1.2,
        "lens_light_amp": 0.08,
        "parameter_seed": 11,
    }
    result, applied = apply_intervention(
        state,
        {
            "intervention_id": "brighter-lens",
            "family": "lens_light",
            "overrides": {"lens_light_amp": {"multiply": 1.5}},
        },
    )
    assert result["theta_E_arcsec"] == state["theta_E_arcsec"]
    assert result["lens_light_amp"] == pytest.approx(0.12)
    assert applied == {"lens_light_amp": pytest.approx(0.12)}


def test_intervention_rejects_target_or_unknown_field_changes():
    state = {"theta_E_arcsec": 1.2, "lens_light_amp": 0.08}
    with pytest.raises(ValueError, match="protected"):
        apply_intervention(state, {"overrides": {"theta_E_arcsec": 1.3}})
    with pytest.raises(ValueError, match="Unsupported"):
        apply_intervention(state, {"overrides": {"not_a_field": 12}})


def test_ancestry_validator_rejects_descendant_in_a_different_split():
    parent = pd.DataFrame(
        [{
            "sample_id": "parent-sample",
            "parent_system_id": "parent-system",
            "split_group_id": "parent-system",
            "source_template_id": "source-parent",
            "background_template_id": "background-parent",
            "theta_E_arcsec": 1.1,
        }]
    )
    child = pd.DataFrame(
        [{
            "sample_id": "parent-sample__psf-wide",
            "parent_sample_id": "parent-sample",
            "parent_system_id": "parent-system",
            "split_group_id": "parent-system",
            "source_template_id": "source-parent",
            "background_template_id": "background-parent",
            "theta_E_arcsec": 1.1,
            "split": "validation",
        }]
    )
    with pytest.raises(ValueError, match="split differs"):
        validate_counterfactual_ancestry(parent, child, {"parent-sample": "train"})


def test_angular_resampling_preserves_centered_constant_image_and_canonical_grid():
    shifted = np.full((320, 320), 2.5, dtype=np.float32)
    resampled = resample_image_to_angular_scale(
        shifted,
        input_pixel_scale_arcsec=0.04,
        output_shape=(256, 256),
    )
    assert resampled.shape == (256, 256)
    np.testing.assert_allclose(resampled, 2.5)

    canonical = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    identical = resample_image_to_angular_scale(
        canonical,
        input_pixel_scale_arcsec=0.05,
        output_shape=(256, 256),
    )
    np.testing.assert_array_equal(identical, canonical)


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_smoke_rerender_is_deterministic_label_invariant_and_ancestry_safe(tmp_path):
    pytest.importorskip("astropy")
    source_root = tmp_path / "source"
    source = generate_observation_pilot(
        source_root,
        count=1,
        seed=911,
        simulation_id="cf-test-source-v1",
        config_path="configs/simulation_engineering_v1.yaml",
        create_preview=False,
    )
    parent_manifest = pd.read_csv(source_root / "manifest.csv")
    splits_dir = source_root / "splits"
    splits_dir.mkdir()
    parent_manifest.to_csv(splits_dir / "train.csv", index=False)
    parent_manifest.iloc[0:0].to_csv(splits_dir / "validation.csv", index=False)
    interventions = [
        {"intervention_id": "canonical", "family": "canonical", "overrides": {}},
        {
            "intervention_id": "lens-light-smoke",
            "family": "lens_light",
            "overrides": {"lens_light_amp": {"multiply": 1.25}},
        },
        {
            "intervention_id": "pixel-scale-004",
            "family": "pixel_scale",
            "overrides": {"pixel_scale_arcsec": 0.04, "shape": [320, 320]},
        },
    ]

    outputs = []
    for name in ("benchmark-a", "benchmark-b"):
        output = tmp_path / name
        summary = create_counterfactual_benchmark(
            source_root / "manifest.csv",
            {"train": splits_dir / "train.csv", "validation": splits_dir / "validation.csv"},
            output,
            interventions,
            max_parents_per_split=1,
            create_preview=False,
        )
        assert summary["passed"]
        assert summary["identity_checks_passed"]
        assert summary["strict_loader_validation"]["train"]["sample_count"] == 3
        assert summary["strict_loader_validation"]["validation"]["sample_count"] == 0
        rows = pd.read_csv(output / "manifest.csv")
        assert len(rows) == 3
        assert rows["theta_E_arcsec"].nunique() == 1
        assert rows["split"].tolist() == ["train", "train", "train"]
        assert set(rows["parent_system_id"]) == {parent_manifest.iloc[0]["parent_system_id"]}
        assert json.loads((output / rows.iloc[0]["parameters_path"]).read_text())["theta_E_arcsec"] == pytest.approx(
            float(parent_manifest.iloc[0]["theta_E_arcsec"])
        )
        outputs.append(output)

    first = pd.read_csv(outputs[0] / "manifest.csv").sort_values("sample_id")
    second = pd.read_csv(outputs[1] / "manifest.csv").sort_values("sample_id")
    for path_a, path_b in zip(first["processed_path"], second["processed_path"]):
        np.testing.assert_array_equal(
            np.load(outputs[0] / path_a),
            np.load(outputs[1] / path_b),
        )
    first_by_id = first.set_index("intervention_id")
    canonical_image = np.load(outputs[0] / first_by_id.loc["canonical", "processed_path"])
    altered_image = np.load(outputs[0] / first_by_id.loc["lens-light-smoke", "processed_path"])
    assert not np.array_equal(canonical_image, altered_image)
    pixel_image = np.load(outputs[0] / first_by_id.loc["pixel-scale-004", "processed_path"])
    assert pixel_image.shape == (320, 320)
    assert (outputs[0] / "splits" / "train__pixel-scale-004.csv").exists()
