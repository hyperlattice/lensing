"""Matched source-development checks of the actual native HST preparation path.

Analytic synthetic PSFs remain the source simulator's PSFs. These controls test
geometry and numerical preparation, not realistic HST detector/noise fidelity.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import ObservationBank
from src.data.hst import (
    fit_light_center,
    implementation_identity,
    load_prepared,
    prepare_image,
    read_array,
    read_observations,
)
from src.ml.artifacts import (
    environment_record,
    load_config,
    sha256,
    verified_hst_assets,
    write_json,
)
from src.ml.predict import predict_images
from src.simulation.observation import condition_for
from src.simulation.render import SceneEngine, adaptive_clean, native_square_transform


@dataclass(frozen=True)
class DevelopmentViews:
    """Matched images plus independently known foreground-center diagnostics."""

    images: dict[str, np.ndarray]
    preparation: dict[str, Any]
    center_recovery_tested: bool
    center_error_pixels: float


def numerical_controls(row: dict[str, Any], offsets: list[list[float]]) -> dict[str, float]:
    """Recover injected Gaussian centers and test reviewed unit/crop/PSF arithmetic."""
    yy, xx = np.indices((41, 41), dtype=float)
    center_errors = []
    for offset_x, offset_y in offsets:
        center_x, center_y = 20 + offset_x, 20 + offset_y
        light = 7 + 30 * np.exp(-0.5 * (((xx - center_x) / 3) ** 2 + ((yy - center_y) / 4) ** 2))
        fitted = fit_light_center(light, np.ones(light.shape, dtype=bool), [0, 41, 0, 41])
        center_errors.append(float(np.linalg.norm(np.asarray(fitted["xy"]) - [center_x, center_y])))
    physical = (xx + 2 * yy + 1) / 13
    multiplier = row["unit_multiplier"]
    background = row["background_raw"]
    raw = physical / multiplier + background
    control_row = copy.deepcopy(row)
    control_row.update(
        center={
            "method": "reviewed_external",
            "xy": [20.0, 20.0],
            "evidence": "known synthetic control center",
        },
        crop_shape_yx=[31, 31],
        center_tolerance_arcsec=0.0,
    )
    psf = np.exp(-0.5 * ((xx - 20) ** 2 + (yy - 20) ** 2) / 2**2)
    product = prepare_image(raw, psf, np.ones(raw.shape, dtype=bool), control_row)
    expected = physical[5:36, 5:36]
    transform = np.asarray(row["pixel_transform_arcsec"])
    before = transform @ (np.array([20.0, 20.0]) - [5, 5])
    after = transform @ np.asarray(product.metadata["center_xy_prepared"])
    return {
        "max_center_error_pixels": max(center_errors),
        "max_unit_roundtrip_abs": float(np.max(abs(product.image - expected))),
        "max_geometry_error_arcsec": float(np.max(abs(before - after))),
        "max_psf_sum_error": float(abs(product.psf.sum() - 1)),
    }


def development_views(
    parameters: dict[str, Any], row: dict[str, Any], *, noise_seed: int
) -> DevelopmentViews:
    """Separate sampling, field loss and exact raw-unit/crop preparation controls.

    The mass parameter and scene stay fixed. Canonical and sampled grids use
    seed-matched detector sampling, not identical physical noise realizations.
    Field loss crops that SAME sampled image, isolating loss from a new draw.
    """
    transform = np.asarray(row["pixel_transform_arcsec"], dtype=float)
    scale = float(np.sqrt(abs(np.linalg.det(transform))))
    native, _ = read_array(row["science"])
    requested = row["crop_shape_yx"] or list(native.shape)
    canonical_condition = condition_for("fresh-noise")
    sampled_shape = np.maximum(np.ceil(12.8 / scale).astype(int), native.shape)
    # Keep parity with native shape to make the field crop a pure integer crop.
    sampled_shape += (sampled_shape - np.array(native.shape)) % 2
    sampled_condition = {
        **canonical_condition,
        "shape": sampled_shape.tolist(),
        "pixel_scale_arcsec": scale,
        "pixel_transform_arcsec": transform.tolist(),
    }
    native_square_transform(sampled_condition)
    engine = SceneEngine(parameters, "lenstronomy")
    views = {}
    for name, condition in (("canonical", canonical_condition), ("sampling", sampled_condition)):
        clean, _ = adaptive_clean(engine, condition)
        expected = engine.expected(condition, factor=4, clean=clean)
        views[name] = engine.observe(condition, factor=4, noise_seed=noise_seed, expected=expected)[
            "image"
        ]
    origin = (np.asarray(views["sampling"].shape) - np.asarray(native.shape)) // 2
    y0, x0 = origin
    field = views["sampling"][y0 : y0 + native.shape[0], x0 : x0 + native.shape[1]]
    # Source parents have nonzero lens centers; array midpoint is not truth.
    center_arcsec = np.array(
        [parameters["lens_center_x_arcsec"], parameters["lens_center_y_arcsec"]]
    )
    center_xy = (np.array(field.shape)[::-1] - 1) / 2 + np.linalg.solve(transform, center_arcsec)
    preparation_row = copy.deepcopy(row)
    center_recovery_tested = row["center"]["method"] == "gaussian_profile"
    if center_recovery_tested:
        if "reviewed_center_xy" not in row:
            raise ValueError(
                "Gaussian bridge requires the qualified observation's recovered center"
            )
        delta_x, delta_y = np.rint(center_xy - row["reviewed_center_xy"]).astype(int)
        y0, y1, x0, x1 = row["center"]["roi_yx"]
        preparation_row["center"]["roi_yx"] = [
            int(y0 + delta_y),
            int(y1 + delta_y),
            int(x0 + delta_x),
            int(x1 + delta_x),
        ]
    else:
        preparation_row["center"] = {
            "method": "reviewed_external",
            "xy": center_xy.tolist(),
            "evidence": "known simulator foreground center; external estimator not tested",
        }
    raw = field.astype(float) / row["unit_multiplier"] + row["background_raw"]
    product = prepare_image(raw, np.ones((1, 1)), np.ones(raw.shape, dtype=bool), preparation_row)
    crop_origin = np.array(product.metadata["origin_xy_pixels"])
    crop_x, crop_y = crop_origin
    target = field[crop_y : crop_y + requested[0], crop_x : crop_x + requested[1]]
    views["field"] = target.copy()
    views["prepared"] = product.image
    measured_center = np.asarray(product.metadata["center"]["xy"])
    return DevelopmentViews(
        views,
        product.metadata,
        center_recovery_tested,
        float(np.linalg.norm(measured_center - center_xy)),
    )


def validate_preparation(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    """Save numerical gates plus descriptive paired CNN changes; never train."""
    output = Path(config["output"])
    if (output / "confirmation_freeze.json").exists():
        raise ValueError("Preparation validation is frozen; choose a new study identity")
    assets_path = output / "model_assets.json"
    assets = verified_hst_assets(assets_path)
    prepared_rows, _ = load_prepared(config, "development")
    prepared_by_id = {row["observation_id"]: row for row in prepared_rows}
    templates = []
    for row in read_observations(config):
        if row["observation_id"] in prepared_by_id:
            template = copy.deepcopy(row)
            template["reviewed_center_xy"] = prepared_by_id[row["observation_id"]]["preparation"][
                "center"
            ]["xy"]
            templates.append(template)
    study = load_config(config["study_config"])
    import torch

    torch.set_num_threads(study["training"]["torch_threads"])
    bank = ObservationBank(study["source_development"], expected_role="source_development")
    settings = config["bridge"]
    if (
        type(settings["smoke_parent_count"]) is not int
        or settings["smoke_parent_count"] < 1
        or type(settings["seed"]) is not int
        or settings["seed"] < 0
    ):
        raise ValueError("Bridge needs a positive smoke count and a nonnegative seed")
    offsets = np.asarray(settings["offsets_xy_pixels"], dtype=float)
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 2
        or not len(offsets)
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("Bridge centering offsets must be a finite nonempty [offset, xy] array")
    parents = sorted(bank.by_id)
    if smoke:
        parents = parents[: settings["smoke_parent_count"]]
    thresholds = settings["tolerances"]
    controls = {
        row["observation_id"]: numerical_controls(row, offsets.tolist()) for row in templates
    }
    required = set(next(iter(controls.values())))
    if set(thresholds) != required:
        raise ValueError("Bridge tolerances must match the declared numerical controls")
    thresholds_set = all(value is not None for value in thresholds.values())
    if thresholds_set and (
        not np.isfinite(list(thresholds.values())).all() or min(thresholds.values()) < 0
    ):
        raise ValueError("Numerical validation tolerances must be finite and nonnegative")
    numerical_passed = thresholds_set and all(
        value <= thresholds[key] for control in controls.values() for key, value in control.items()
    )
    destination = output / ("preparation_smoke" if smoke else "preparation_validation")
    if destination.exists():
        raise FileExistsError("Bridge outputs already exist; use a new output identity")
    destination.mkdir(parents=True)
    records = []
    products = []
    roundtrip_errors = []
    scene_center_errors = []
    for parent_index, parent in enumerate(parents):
        metadata = bank.by_id[parent]
        parameter_path = Path(metadata["source_parameter_path"])
        if sha256(parameter_path) != metadata["source_parameter_sha256"]:
            raise ValueError(f"Development parent parameters changed: {parent}")
        parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
        if not np.isclose(parameters["theta_E_arcsec"], bank.targets[parent], rtol=0, atol=1e-12):
            raise ValueError("Bridge physical radius differs from development truth")
        for template in templates:
            result = development_views(
                parameters, template, noise_seed=settings["seed"] + parent_index
            )
            views = result.images
            if result.center_recovery_tested:
                scene_center_errors.append(result.center_error_pixels)
            names = list(views)
            roundtrip_errors.append(float(np.max(abs(views["prepared"] - views["field"]))))
            archive_path = destination / f"{parent}_{template['observation_id']}.npz"
            np.savez_compressed(archive_path, **views)
            products.append(
                {
                    "parent_system_id": parent,
                    "observation_id": template["observation_id"],
                    "path": str(archive_path),
                    "sha256": sha256(archive_path),
                    "preparation": result.preparation,
                    "center_recovery_tested": result.center_recovery_tested,
                    "center_error_pixels": result.center_error_pixels,
                }
            )
            for checkpoint in assets["checkpoints"]:
                values = predict_images(
                    Path(checkpoint["path"]), list(views.values()), batch_size=config["batch_size"]
                )
                for name, value in zip(names, values, strict=True):
                    records.append(
                        {
                            "parent_system_id": parent,
                            "observation_id": template["observation_id"],
                            "condition": name,
                            "arm": checkpoint["arm"],
                            "seed": checkpoint["seed"],
                            "truth_arcsec": bank.targets[parent],
                            "prediction_arcsec": float(value),
                            "absolute_error_arcsec": float(abs(value - bank.targets[parent])),
                        }
                    )
    frame = pd.DataFrame(records)
    keys = ["parent_system_id", "observation_id", "arm", "seed"]
    baseline = frame[frame["condition"] == "canonical"][
        keys + ["prediction_arcsec", "absolute_error_arcsec"]
    ]
    paired = frame.merge(baseline, on=keys, suffixes=("", "_canonical"), validate="many_to_one")
    paired["prediction_change_arcsec"] = (
        paired["prediction_arcsec"] - paired["prediction_arcsec_canonical"]
    )
    paired["absolute_error_change_arcsec"] = (
        paired["absolute_error_arcsec"] - paired["absolute_error_arcsec_canonical"]
    )
    field_losses = frame[frame["condition"] == "field"][
        keys + ["prediction_arcsec", "absolute_error_arcsec"]
    ]
    paired = paired.merge(field_losses, on=keys, suffixes=("", "_field"), validate="many_to_one")
    paired["prediction_change_from_field_arcsec"] = (
        paired["prediction_arcsec"] - paired["prediction_arcsec_field"]
    )
    paired["error_change_from_field_arcsec"] = (
        paired["absolute_error_arcsec"] - paired["absolute_error_arcsec_field"]
    )
    paired.to_csv(destination / "paired_predictions.csv", index=False)
    numerical_passed = bool(
        numerical_passed and max(roundtrip_errors) <= thresholds["max_unit_roundtrip_abs"]
    )
    if scene_center_errors:
        numerical_passed = bool(
            numerical_passed and max(scene_center_errors) <= thresholds["max_center_error_pixels"]
        )
    report = {
        "schema": "hst_preparation_validation_v1",
        "smoke": smoke,
        "passed": bool(numerical_passed and not smoke),
        "numerical_controls_passed": bool(numerical_passed),
        "tolerances_declared": thresholds_set,
        "controls": controls,
        "settings": settings,
        "actual_unit_roundtrip_max_abs": max(roundtrip_errors),
        "source_scene_center_recovery_count": len(scene_center_errors),
        "source_scene_center_max_error_pixels": max(scene_center_errors)
        if scene_center_errors
        else None,
        "implementation": implementation_identity(),
        "environment": environment_record(torch.device("cpu")),
        "parent_ids": parents,
        "parent_count": len(parents),
        "available_parent_count": len(bank.by_id),
        "source_bank_sha256": sha256(study["source_development"]),
        "products": products,
        "assets_sha256": sha256(assets_path),
        "prepared_development_sha256": sha256(
            Path(config["prepared_root"]) / "development" / "manifest.json"
        ),
        "limitations": [
            "Synthetic Gaussian-center recovery does not establish unbiased HST light centers.",
            "External/manual centers use simulator truth in scene controls; their estimator is not validated here.",
            "Analytic source PSF and source detector model; not an HST likelihood validation.",
            "Seed-matched noise across different grids is not the same physical noise draw.",
            "Prediction sensitivity is descriptive and is not a preparation acceptance threshold.",
        ],
    }
    write_json(
        output / ("preparation_smoke.json" if smoke else "preparation_validation.json"), report
    )
    return report
