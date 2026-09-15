import numpy as np
import pandas as pd

from src.processing.preprocess import center_crop_or_pad
from src.processing.create_splits import ancestry_components, create_ancestry_splits, create_splits


def test_center_crop_or_pad_produces_canonical_shape_and_dtype():
    result = center_crop_or_pad(np.ones((128, 128), dtype=np.float64))
    assert result.shape == (256, 256)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_split_assignments_do_not_leak_lenses(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "lens_id": ["lens1", "lens1", "lens2", "lens3"],
        "processed_path": ["a", "b", "c", "d"],
    }).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "splits"
    result = create_splits(manifest_path, output_dir=output_dir, seed=1)
    assert result.groupby("lens_id")["split"].nunique().max() == 1
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "assignments.csv").exists()
    assert not (tmp_path / "data" / "splits" / "train.csv").exists()


def test_split_assignments_can_be_preserved(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "lens_id": ["lens1", "lens2", "lens3"],
        "processed_path": ["a", "b", "c"],
    }).to_csv(manifest_path, index=False)
    existing_path = tmp_path / "existing.csv"
    pd.DataFrame({
        "lens_id": ["lens1", "lens2"],
        "split": ["test", "train"],
    }).to_csv(existing_path, index=False)

    result = create_splits(
        manifest_path,
        output_dir=tmp_path / "splits",
        seed=1,
        existing_assignments_path=existing_path,
    )
    assignments = result.set_index("lens_id")["split"].to_dict()
    assert assignments["lens1"] == "test"
    assert assignments["lens2"] == "train"


def test_ancestry_components_close_transitive_template_links():
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "parent_system_id": ["p1", "p2", "p3", "p4"],
            "split_group_id": ["p1", "p2", "p3", "p4"],
            "source_template_id": ["s1", "s1", "s3", "s4"],
            "background_template_id": ["b1", "b2", "b2", "b4"],
        }
    )

    result = ancestry_components(manifest)
    groups = result.set_index("sample_id")["effective_group_id"]
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_ancestry_splits_keep_every_reused_asset_in_one_split(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    rows = []
    for index in range(40):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "parent_system_id": f"parent-{index}",
                "split_group_id": f"parent-{index}",
                "source_template_id": f"source-{index // 2}",
                "background_template_id": f"background-{index // 2}",
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    result, report = create_ancestry_splits(
        manifest_path,
        tmp_path / "splits",
        seed=7,
    )

    assert report["leakage_detected"] is False
    assert result.groupby("source_template_id")["split"].nunique().max() == 1
    assert result.groupby("background_template_id")["split"].nunique().max() == 1
    assert set(result["split"]) == {"train", "validation", "test"}
