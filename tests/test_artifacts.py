"""Write-once identities and scientific qualification fail closed."""

import json

import pytest

from src.data.validation import qualify_rendering
from src.ml.artifacts import create_once, freeze_methods, verify_method_freeze
from src.simulation.generate import run


def test_method_record_binds_artifact_bytes(tmp_path):
    artifact = tmp_path / "design.json"
    artifact.write_text("original design")
    keys = (
        "primary_endpoint",
        "secondary_contrasts",
        "alpha_allocation",
        "margins",
        "source_selection_rule",
        "training_budget",
        "noise_and_seed_policy",
        "numerical_qualification",
        "failure_and_recovery_policy",
        "power_planning_rule",
    )
    design = dict.fromkeys(keys)
    design.update(
        training_roles=["source_train", "source_development", "target_adaptation"],
        arms=["C", "G", "R"],
        target_information_access=dict.fromkeys(["C", "G", "R"], "none"),
    )
    record = tmp_path / "method.json"
    freeze_methods(record, study_id="fixture", design=design, artifacts=[artifact])
    assert verify_method_freeze(record)["study_id"] == "fixture"
    artifact.write_text("changed design")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_method_freeze(record)


def test_write_once_preserves_existing_record(tmp_path):
    path = tmp_path / "record.json"
    create_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        create_once(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_qualification_rejects_incomplete_controls_before_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    references = tmp_path / "data/simulations/renderer-qualification-v3"
    references.mkdir(parents=True)
    (references / "references.json").write_text(json.dumps({"cases": [{}]}))
    with pytest.raises(ValueError, match="complete 315-case"):
        qualify_rendering(["lenstronomy"])
    assert not (tmp_path / "results").exists()


def test_source_generator_rejects_retired_adaptation_role_before_io():
    with pytest.raises(ValueError, match="Only source"):
        run("target_adaptation", 1)
