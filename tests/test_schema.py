from src.data.schema import (
    CONDITIONAL_FIELDS,
    FIELD_SPECS,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    validate_columns,
    validate_row,
)


def _base_row():
    return {
        "sample_id": "sample-1",
        "lens_id": "lens-1",
        "domain": "real",
        "schema_version": SCHEMA_VERSION,
        "dataset_version": "dev-v2",
        "raw_path": "data/raw/sample.fits",
        "science_path": "data/processed/science/sample.npy",
        "processed_path": "data/processed/normalized/sample.npy",
        "image_unit": "unknown",
        "calibration_status": "unknown",
        "original_shape": "256x256",
        "processed_shape": "256x256",
        "pixel_scale": 0.05,
        "processing_config_id": "baseline-v2",
        "processing_version": SCHEMA_VERSION,
        "background_status": "unknown",
        "source_type": "unknown",
        "mass_target_available": False,
        "theta_E_target_available": False,
        "structural_status": "pass",
        "observation_status": "review",
        "science_status": "review",
        "eligible_for": "preprocessing",
        "exclusion_reason": None,
    }


def test_schema_registry_covers_required_and_conditional_fields():
    assert REQUIRED_FIELDS.issubset(FIELD_SPECS)
    assert CONDITIONAL_FIELDS["simulated"].issubset(FIELD_SPECS)


def test_minimal_real_row_is_valid():
    row = _base_row()
    assert validate_columns(row) == []
    assert validate_row(row) == []


def test_minimal_simulated_row_requires_truth_and_reproducibility_fields():
    row = _base_row() | {"domain": "simulated", "theta_E_target_available": True}
    errors = validate_row(row)
    assert "simulation_id is required for simulated rows" in errors

    row.update(
        {
            "simulation_id": "run-1",
            "parent_system_id": "parent-1",
            "split_group_id": "parent-1",
            "mass_model": "SIS",
            "theta_E_arcsec": 1.0,
            "simulator_name": "lenstronomy",
            "simulator_version": "test",
            "parameter_seed": 1,
            "noise_seed": 2,
            "parameters_path": "data/labels/simulations/run-1/sample.json",
        }
    )
    assert validate_row(row) == []


def test_schema_rejects_invalid_path_and_redshifts():
    row = _base_row() | {
        "processed_path": r"data\processed\sample.npy",
        "lens_redshift": 1.0,
        "source_redshift": 0.5,
    }
    errors = validate_row(row)
    assert any("processed_path" in error for error in errors)
    assert any("redshifts" in error for error in errors)
