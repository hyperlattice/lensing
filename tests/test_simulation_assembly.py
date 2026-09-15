import pandas as pd

from src.simulation.assemble import assemble_pilot_release


def _stage(root, name, sample_id):
    stage = root / name
    (stage / "raw").mkdir(parents=True)
    (stage / "raw" / f"{sample_id}.fits").write_bytes(b"fixture")
    pd.DataFrame(
        {
            "sample_id": [sample_id],
            "parent_system_id": [f"{sample_id}-parent"],
            "lens_id": [f"{sample_id}-parent"],
        }
    ).to_csv(stage / "manifest.csv", index=False)
    pd.DataFrame(columns=["sample_id", "parent_system_id", "stage", "error_type", "message"]).to_csv(
        stage / "rejection_log.csv", index=False
    )
    return stage


def test_assembly_combines_unique_parents_and_rewrites_release_version(tmp_path):
    first = _stage(tmp_path, "first", "sample-1")
    second = _stage(tmp_path, "second", "sample-2")
    output = tmp_path / "release"

    summary = assemble_pilot_release([first, second], output, create_preview=False)
    manifest = pd.read_csv(output / "manifest.csv")

    assert summary["count_accepted"] == 2
    assert summary["parent_count"] == 2
    assert set(manifest["dataset_version"]) == {"sim-p100-cumulative-v1"}
    assert manifest["sample_id"].tolist() == ["sample-1", "sample-2"]
    assert (output / "raw" / "sample-1.fits").exists()
