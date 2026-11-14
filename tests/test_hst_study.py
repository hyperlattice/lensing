"""HST scientific boundaries and a temporary end-to-end paired inference smoke run."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.hst import (
    REVIEW_FIELDS,
    fit_light_center,
    freeze_confirmation,
    implementation_identity,
    initialize_observations,
    load_hst_config,
    load_prepared,
    prepare_image,
    prepare_release,
    read_observations,
    verify_confirmation,
)
from src.ml.analysis import score_hst
from src.ml.artifacts import inspect_model_assets, load_config, sha256, write_json
from src.ml.model import build_model
from src.ml.predict import predict_hst, predict_images
from src.simulation.hst_bridge import (
    DevelopmentViews,
    development_views,
    numerical_controls,
    validate_preparation,
)


@pytest.fixture
def hst_fixture(tmp_path):
    config = load_hst_config(Path("configs/hst_study.yaml"))
    for key, relative in (
        ("output", "results"),
        ("prepared_root", "prepared"),
        ("observations", "observations.json"),
        ("references", "references.json"),
        ("reserved_manifest", "reserved.json"),
        ("study_config", "study.yaml"),
    ):
        config[key] = str(tmp_path / relative)
    config.update(
        eligibility_rules="All reviewed fixture lenses",
        primary_band_rule="F606W only",
        reference_strategy="Declared fixture classical references",
    )
    config["bootstrap"]["resamples"] = 100
    config["bridge"]["tolerances"] = {
        "max_center_error_pixels": 1e-4,
        "max_unit_roundtrip_abs": 1e-5,
        "max_geometry_error_arcsec": 1e-10,
        "max_psf_sum_error": 1e-10,
    }
    write_json(
        config["reserved_manifest"],
        {"systems": [{"system_id": "DEV", "aliases": ["reserved-alias"]}]},
    )
    study = load_config()
    study["output"] = str(tmp_path / "models")
    study["seeds"] = [11, 12]
    for name in ("source_train", "source_development", "policy", "holdout"):
        study[name] = str(tmp_path / f"{name}.json")
        write_json(study[name], {})
    study["training"]["model"] = {
        "channels": [2, 4, 4, 4],
        "pooled_size": 1,
        "hidden_features": 8,
        "dropout": 0.1,
    }
    import yaml

    Path(config["study_config"]).write_text(yaml.safe_dump(study))
    model = build_model(study["training"]["model"])
    statistics = {"image_mean": 0.0, "image_std": 1.0, "target_mean": 1.0, "target_std": 0.2}
    fits = []
    for arm_index, arm in enumerate(study["arms"]):
        for seed in study["seeds"]:
            torch.manual_seed(seed)
            path = Path(study["output"]) / "fits" / arm / f"seed-{seed}" / "final.pt"
            path.parent.mkdir(parents=True)
            # Constant-output fixtures isolate ordering/aggregation from CNN learning.
            for parameter in model.parameters():
                parameter.data.zero_()
            model.regressor[-1].bias.data.fill_(arm_index + (seed - 11) * 0.5)
            torch.save(
                {
                    "model_config": model.model_config,
                    "model_state": model.state_dict(),
                    "statistics": statistics,
                    "arm": arm,
                    "seed": seed,
                    "optimizer_updates": study["training"]["optimizer_updates"],
                    "target_information_access": None,
                },
                path,
            )
            fits.append({"arm": arm, "seed": seed, "checkpoint": str(path)})
    write_json(
        Path(study["output"]) / "fit_complete.json",
        {
            "settings": study["training"],
            "fits": fits,
            "target_labels_accessed": False,
            "final_images_accessed": False,
        },
    )
    assets = inspect_model_assets(study)
    write_json(Path(config["output"]) / "model_assets.json", assets)
    observations, references = [], []
    for index, system in enumerate(("DEV", "FINAL0", "FINAL1", "FINAL2")):
        path = tmp_path / f"{system}.npy"
        np.save(path, np.full((40, 40), index + 10.0, dtype=np.float32))
        psf = tmp_path / f"{system}-psf.npy"
        np.save(psf, np.ones((3, 3)))
        observations.append(
            {
                "system_id": system,
                "aliases": [],
                "observation_id": f"{system}_F606W",
                "survey": "fixture",
                "band": "F606W",
                "cohort": "development" if system == "DEV" else "confirmation",
                "eligibility": "include",
                "exclusion_reason": "",
                "science": {"path": str(path), "sha256": sha256(path)},
                "psf": {"path": str(psf), "sha256": sha256(psf)},
                "valid_mask": None,
                "pixel_transform_arcsec": [[0.05, 0], [0, 0.05]],
                "psf_pixel_transform_arcsec": [[0.05, 0], [0, 0.05]],
                "input_unit": "fixture_DN/s",
                "unit_multiplier": 7.0,
                "background_raw": 2.0,
                "center": {
                    "method": "reviewed_external",
                    "xy": [19.5, 19.5],
                    "evidence": "known fixture center",
                },
                "crop_shape_yx": [32, 32],
                "center_tolerance_arcsec": 0.0,
                "psf_center_tolerance_pixels": 1e-8,
                "review": {key: "explicit fixture review" for key in REVIEW_FIELDS},
            }
        )
        references.append(
            {
                "system_id": system,
                "median_arcsec": 1.15 + index * 0.1,
                "error_lower": 0.1,
                "error_upper": 0.2,
                "definition": "fixture_radius",
                "mass_model": "fixture SIE",
                "interval_semantics": "16/50/84 quantiles",
                "source": "explicit test reference",
                "alternatives": [],
            }
        )
    write_json(
        config["observations"], {"schema": "hst_observations_v1", "observations": observations}
    )
    write_json(config["references"], {"schema": "hst_references_v1", "references": references})
    return config, study, observations


def write_validation_gate(config):
    output = Path(config["output"])
    write_json(
        output / "preparation_validation.json",
        {
            "passed": True,
            "smoke": False,
            "settings": config["bridge"],
            "implementation": implementation_identity(),
            "source_bank_sha256": sha256(load_config(config["study_config"])["source_development"]),
            "parent_count": 3,
            "available_parent_count": 3,
            "assets_sha256": sha256(output / "model_assets.json"),
            "prepared_development_sha256": sha256(
                Path(config["prepared_root"]) / "development" / "manifest.json"
            ),
        },
    )


def test_paired_development_and_confirmation_pipeline(hst_fixture, monkeypatch):
    config, _, _ = hst_fixture
    torch.set_num_threads(2)
    for cohort in ("development", "confirmation"):
        report = prepare_release(config, cohort)
        assert report["observations"]
    predict_hst(config, "development")
    development = score_hst(config, "development")
    assert development["contrasts"]["G_minus_C"]["bootstrap_95_low_arcsec"] is None
    write_validation_gate(config)
    freeze_confirmation(config)
    predict_hst(config, "confirmation")

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis opened images or checkpoints")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    report = score_hst(config, "confirmation")
    first = (Path(config["output"]) / "confirmation" / "analysis.json").read_bytes()
    assert report["system_count"] == 3
    score_hst(config, "confirmation")
    assert (Path(config["output"]) / "confirmation" / "analysis.json").read_bytes() == first
    frame = pd.read_csv(Path(config["output"]) / "confirmation" / "reference_disagreements.csv")
    expected = frame.groupby(["system_id", "arm"])["absolute_disagreement_arcsec"].mean().unstack()
    assert report["contrasts"]["G_minus_C"]["difference_arcsec"] == pytest.approx(
        (expected["G"] - expected["C"]).mean()
    )
    from src.ml.plots import plot_hst

    plot_hst(config, "confirmation")
    assert (
        Path(config["output"]) / "confirmation" / "figures" / "hst_reference_agreement.pdf"
    ).is_file()


def test_prediction_never_reads_reference_values(hst_fixture, monkeypatch):
    config, _, _ = hst_fixture
    prepare_release(config, "development")
    Path(config["references"]).unlink()
    import src.data.hst as hst

    monkeypatch.setattr(hst, "read_references", lambda *args: pytest.fail("References opened"))
    predict_hst(config, "development")


@pytest.mark.parametrize("change", ["missing_seed", "duplicate_seed"])
def test_analysis_rejects_incomplete_or_duplicated_predictions(hst_fixture, change):
    config, _, _ = hst_fixture
    prepare_release(config, "development")
    predict_hst(config, "development")
    root = Path(config["output"]) / "development"
    prediction_path = root / "predictions_before_scoring.csv"
    frame = pd.read_csv(prediction_path)
    if change == "missing_seed":
        frame = frame.iloc[1:]
    else:
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    frame.to_csv(prediction_path, index=False)
    seal_path = root / "predictions_complete.json"
    seal = json.loads(seal_path.read_text())
    seal.update(prediction_sha256=sha256(prediction_path), rows=len(frame))
    write_json(seal_path, seal)
    with pytest.raises(ValueError, match="one finite value per lens/arm/seed"):
        score_hst(config, "development")


@pytest.mark.parametrize("change", ["reserved_alias", "cross_cohort", "duplicate_lens"])
def test_identity_boundaries(hst_fixture, change):
    config, _, rows = hst_fixture
    if change == "reserved_alias":
        rows[1]["aliases"] = ["reserved alias"]
    elif change == "cross_cohort":
        rows[1]["aliases"] = ["DEV"]
    else:
        rows[2]["system_id"] = rows[1]["system_id"]
    write_json(config["observations"], {"schema": "hst_observations_v1", "observations": rows})
    with pytest.raises(ValueError):
        read_observations(config)


def test_unreviewed_inputs_cannot_enter_prediction(hst_fixture):
    config, _, rows = hst_fixture
    rows[0]["review"]["photometry"] = ""
    write_json(config["observations"], {"schema": "hst_observations_v1", "observations": rows})
    report = prepare_release(config, "development")
    assert not report["observations"]
    with pytest.raises(ValueError, match="qualified"):
        predict_hst(config, "development")


def test_units_crop_masks_and_psf_grid_are_explicit(hst_fixture):
    _, _, rows = hst_fixture
    row = rows[0]
    science = np.arange(1600, dtype=float).reshape(40, 40)
    valid = np.ones(science.shape, dtype=bool)
    product = prepare_image(science, np.ones((3, 3)), valid, row)
    np.testing.assert_array_equal(product.image, ((science[4:36, 4:36] - 2) * 7).astype(np.float32))
    assert product.metadata["origin_xy_pixels"] == [4, 4]
    assert product.psf.sum() == pytest.approx(1)
    valid[20, 20] = False
    with pytest.raises(ValueError, match="unsupported"):
        prepare_image(science, np.ones((3, 3)), valid, row)
    changed = copy.deepcopy(row)
    changed["psf_pixel_transform_arcsec"][0][0] = 0.04
    with pytest.raises(ValueError, match="PSF must already match"):
        prepare_image(science, np.ones((3, 3)), np.ones(science.shape, bool), changed)
    changed = copy.deepcopy(row)
    changed["crop_shape_yx"] = [60, 60]
    with pytest.raises(ValueError, match="padding"):
        prepare_image(science, np.ones((3, 3)), np.ones(science.shape, bool), changed)


def test_fits_preparation_preserves_sky_coordinates_and_source_bytes(hst_fixture, tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS

    from src.data.hst import read_array

    _, _, rows = hst_fixture
    row = copy.deepcopy(rows[0])
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [20.5, 20.5]
    wcs.wcs.cdelt = [-0.05 / 3600, 0.05 / 3600]
    wcs.wcs.crval = [9, -9]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    path = tmp_path / "native.fits"
    fits.PrimaryHDU(np.full((40, 40), 10.0), header=wcs.to_header()).writeto(path)
    before = path.read_bytes()
    science, header = read_array({"path": str(path), "sha256": sha256(path), "hdu": 0})
    row["pixel_transform_arcsec"] = (wcs.pixel_scale_matrix * 3600).tolist()
    row["psf_pixel_transform_arcsec"] = row["pixel_transform_arcsec"]
    product = prepare_image(science, np.ones((3, 3)), np.ones(science.shape, bool), row, header)
    output_wcs = WCS(product.header)
    np.testing.assert_allclose(
        output_wcs.all_pix2world([[15.5, 15.5], [0, 0]], 0),
        wcs.all_pix2world([[19.5, 19.5], [4, 4]], 0),
        atol=1e-10,
    )
    assert before == path.read_bytes()


def test_real_renderer_bridge_separates_native_cropping_from_preparation(hst_fixture, monkeypatch):
    import src.simulation.hst_bridge as bridge

    _, _, rows = hst_fixture
    fixture = json.loads((Path(__file__).parent / "fixtures" / "render_reference.json").read_text())

    def bounded_clean(engine, condition):
        # Bounded renderer smoke; this does not constitute numerical qualification.
        return engine.clean(condition, factor=4), {"fixture": True}

    monkeypatch.setattr(bridge, "adaptive_clean", bounded_clean)
    row = copy.deepcopy(rows[0])
    row["center_tolerance_arcsec"] = 0.04
    result = development_views(fixture["parameters"], row, noise_seed=31)
    views = result.images
    np.testing.assert_array_equal(views["sampling"], views["canonical"])
    center_xy = (
        127.5
        + np.array(
            [
                fixture["parameters"]["lens_center_x_arcsec"],
                fixture["parameters"]["lens_center_y_arcsec"],
            ]
        )
        / 0.05
    )
    x0, y0 = np.floor(center_xy - 15.5 + 0.5).astype(int)
    np.testing.assert_allclose(views["field"], views["canonical"][y0 : y0 + 32, x0 : x0 + 32])
    np.testing.assert_allclose(views["prepared"], views["field"], atol=1e-6)


def test_center_recovery_and_numeric_conversion_controls(hst_fixture):
    _, _, rows = hst_fixture
    values = numerical_controls(rows[0], [[1.25, -0.75], [0, 0]])
    assert values["max_center_error_pixels"] < 1e-5
    assert values["max_unit_roundtrip_abs"] < 1e-5
    with pytest.raises(ValueError, match="no positive light"):
        fit_light_center(np.zeros((20, 20)), np.ones((20, 20), bool), [0, 20, 0, 20])


def test_bridge_uses_the_declared_gaussian_center_method_on_shifted_scenes(
    hst_fixture, monkeypatch
):
    import src.simulation.hst_bridge as bridge

    _, _, rows = hst_fixture
    row = copy.deepcopy(rows[0])
    row["center"] = {
        "method": "gaussian_profile",
        "roi_yx": [5, 35, 5, 35],
        "evidence": "known analytic light-profile fixture",
    }
    row["reviewed_center_xy"] = [19.5, 19.5]
    row["center_tolerance_arcsec"] = 0.04
    parameters = {"lens_center_x_arcsec": -0.1, "lens_center_y_arcsec": 0.15}

    class GaussianScene:
        def __init__(self, parameters, renderer):
            self.parameters = parameters

        def clean(self, condition, *, factor):
            yy, xx = np.indices(condition["shape"], dtype=float)
            scale = condition["pixel_scale_arcsec"]
            height, width = condition["shape"]
            x = (xx - (width - 1) / 2) * scale - self.parameters["lens_center_x_arcsec"]
            y = (yy - (height - 1) / 2) * scale - self.parameters["lens_center_y_arcsec"]
            return 1 + 20 * np.exp(-0.5 * ((x / 0.12) ** 2 + (y / 0.16) ** 2))

        def expected(self, condition, *, factor, clean):
            return {"expected": clean}

        def observe(self, condition, *, factor, noise_seed, expected):
            return {"image": expected["expected"].astype(np.float32)}

    monkeypatch.setattr(bridge, "SceneEngine", GaussianScene)
    monkeypatch.setattr(
        bridge, "adaptive_clean", lambda engine, condition: (engine.clean(condition, factor=4), {})
    )
    result = development_views(parameters, row, noise_seed=31)
    assert result.center_recovery_tested
    assert result.preparation["center"]["method"] == "gaussian_profile"
    assert result.center_error_pixels < 1e-5


@pytest.mark.parametrize("change", ["missing_seed", "wrong_arm", "wrong_statistics"])
def test_checkpoint_assets_do_not_silently_reduce_or_change_matrix(hst_fixture, change):
    config, study, _ = hst_fixture
    path = Path(study["output"]) / "fits" / "C" / "seed-11" / "final.pt"
    if change == "missing_seed":
        path.unlink()
        report = inspect_model_assets(study)
        assert not report["ready"] and str(path) in report["missing"]
        return
    checkpoint = torch.load(path, weights_only=True)
    if change == "wrong_arm":
        checkpoint["arm"] = "G"
    else:
        checkpoint["statistics"]["image_mean"] = 0.5
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        inspect_model_assets(study)


def test_confirmation_requires_full_current_validation_and_unchanged_inputs(hst_fixture):
    config, _, _ = hst_fixture
    for cohort in ("development", "confirmation"):
        prepare_release(config, cohort)
    with pytest.raises(FileNotFoundError):
        predict_hst(config, "confirmation")
    write_validation_gate(config)
    gate = Path(config["output"]) / "preparation_validation.json"
    report = json.loads(gate.read_text())
    report["smoke"] = True
    write_json(gate, report)
    with pytest.raises(ValueError, match="complete, current"):
        freeze_confirmation(config)
    report["smoke"] = False
    write_json(gate, report)
    freeze_confirmation(config)
    row = load_prepared(config, "confirmation")[0][0]
    image = Path(config["prepared_root"]) / "confirmation" / row["image"]["path"]
    np.save(image, np.zeros((32, 32), np.float32))
    with pytest.raises(ValueError, match="Frozen HST input changed"):
        verify_confirmation(config, images=True)


def test_native_shape_inference_preserves_order_and_is_deterministic(hst_fixture):
    _, study, _ = hst_fixture
    path = Path(study["output"]) / "fits" / "C" / "seed-11" / "final.pt"
    arrays = [np.ones(shape, dtype=np.float32) for shape in ((40, 40), (32, 32), (40, 40))]
    np.testing.assert_array_equal(predict_images(path, arrays, batch_size=2), np.ones(3))
    np.testing.assert_array_equal(predict_images(path, arrays, batch_size=2), np.ones(3))


@pytest.mark.parametrize(
    "action", ["assets", "prepare", "validate-preparation", "freeze", "predict", "analyze", "plot"]
)
def test_hst_cli_help_does_not_open_scientific_inputs(action):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.hst_study", action, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_initialization_keeps_pilot_unqualified_and_references_separate(hst_fixture):
    config, _, rows = hst_fixture
    Path(config["observations"]).unlink()
    Path(config["references"]).unlink()
    audit = Path(config["output"]) / "pilot_audit.json"
    config["pilot_audit"] = str(audit)
    write_json(
        audit,
        {
            "systems": [
                {
                    "system_id": "DEV",
                    "survey": "fixture",
                    "band": "F606W",
                    "native_products": rows[0]["science"],
                    "reference": {"median_arcsec": 1.0, "table_source": "fixture"},
                }
            ]
        },
    )
    initialize_observations(config)
    initialized = read_observations(config)
    assert initialized[0]["eligibility"] == "pending"
    assert "reference" not in initialized[0]
    assert prepare_release(config, "development")["observations"] == []


def test_bridge_runs_complete_pipeline_on_temporary_parent_fixture(hst_fixture, monkeypatch):
    """Exercise real bridge orchestration; renderer/bank are bounded explicit fixtures."""
    config, study, _ = hst_fixture
    prepare_release(config, "development")
    parameters = Path(config["output"]) / "parameters.json"
    write_json(parameters, {"theta_E_arcsec": 1.0})
    bank_path = Path(config["output"]) / "bank.json"
    write_json(bank_path, {})
    from types import SimpleNamespace

    import yaml

    study["source_development"] = str(bank_path)
    Path(config["study_config"]).write_text(yaml.safe_dump(study))
    parents = {
        f"p{index}": {
            "source_parameter_path": str(parameters),
            "source_parameter_sha256": sha256(parameters),
        }
        for index in range(3)
    }
    bank = SimpleNamespace(by_id=parents, targets=dict.fromkeys(parents, 1.0))
    import src.simulation.hst_bridge as bridge

    def load_bank(path, *, expected_role):
        assert expected_role == "source_development"
        return bank

    def views(parameters, row, *, noise_seed):
        images = {
            name: np.ones((32, 32), np.float32)
            for name in ("canonical", "sampling", "field", "prepared")
        }

        return DevelopmentViews(images, {}, False, 0.0)

    monkeypatch.setattr(bridge, "ObservationBank", load_bank)
    monkeypatch.setattr(bridge, "development_views", views)
    smoke = validate_preparation(config, smoke=True)
    assert smoke["parent_count"] == 2 and not smoke["passed"]
    full = validate_preparation(config, smoke=False)
    assert full["passed"] and full["parent_count"] == 3
    assert (Path(config["output"]) / "preparation_validation" / "paired_predictions.csv").is_file()
