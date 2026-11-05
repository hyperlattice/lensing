"""Audit whether the rapid A/B release contains a genuine image-domain shift.

This is a target-free audit: it reads the paired manifest products and the
sealed predictions, but never opens the truth vault.  A different package name
is not sufficient evidence of simulator transfer if the saved image products
are identical after the declared conversions and shared rasterization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.benchmark_contract_v1 import sha256


ROOT = Path("results/ml/cross-simulator-rapid-v2")
RELEASE = Path("data/simulations/cross-simulator-rapid-v2/manifest.json")
PREDICTIONS = ROOT / "predictions_before_scoring.csv"
OUTPUT = ROOT / "renderer_identity_audit.json"


def audit_release(*, write: bool = True) -> dict:
    """Return and optionally save exact A/B identity checks for the release."""
    manifest = json.loads(RELEASE.read_text(encoding="utf-8"))
    release_root = RELEASE.parent
    observations = manifest["observations"]
    by_key = {(row["parent_system_id"], row["renderer"], row["condition"]): row
              for row in observations}
    parents = list(manifest["parent_ids"])
    conditions = list(manifest["condition_order"])
    renderers = list(manifest["renderer_order"])
    if renderers != ["lenstronomy", "autolens"]:
        raise ValueError(f"Unexpected renderer order: {renderers}")

    image_pairs = 0
    exact_image_pairs = 0
    max_image_abs_difference = 0.0
    max_image_mean_abs_difference = 0.0
    for parent in parents:
        for condition in conditions:
            left = by_key[(parent, "lenstronomy", condition)]["products"]["image"]["path"]
            right = by_key[(parent, "autolens", condition)]["products"]["image"]["path"]
            image_left = np.load(release_root / left)
            image_right = np.load(release_root / right)
            difference = np.abs(image_left.astype(np.float64) - image_right.astype(np.float64))
            image_pairs += 1
            exact_image_pairs += int(np.array_equal(image_left, image_right))
            max_image_abs_difference = max(max_image_abs_difference, float(difference.max()))
            max_image_mean_abs_difference = max(max_image_mean_abs_difference,
                                                float(difference.mean()))

    predictions = pd.read_csv(PREDICTIONS)
    key_columns = ["parent_system_id", "condition", "arm", "seed"]
    paired = predictions.pivot_table(index=key_columns, columns="renderer",
                                     values="theta_E_arcsec_predicted", aggfunc="first")
    if set(paired.columns) != {"lenstronomy", "autolens"}:
        raise ValueError("Prediction table does not contain both renderer columns")
    prediction_difference = (paired["lenstronomy"] - paired["autolens"]).abs()
    prediction_pairs = int(len(prediction_difference))
    exact_prediction_pairs = int((prediction_difference == 0).sum())

    backend_versions = {}
    for renderer in renderers:
        report = json.loads((release_root / f"render-{renderer}.json").read_text(encoding="utf-8"))
        backend_versions[renderer] = report["backend_versions"]

    result = {
        "schema": "cross_simulator_renderer_identity_audit_v1",
        "release_manifest_sha256": sha256(RELEASE),
        "truth_opened": False,
        "parent_count": len(parents),
        "condition_count": len(conditions),
        "image_pairs": image_pairs,
        "exact_image_pairs": exact_image_pairs,
        "image_exact_fraction": exact_image_pairs / image_pairs,
        "max_image_absolute_difference": max_image_abs_difference,
        "max_image_mean_absolute_difference": max_image_mean_abs_difference,
        "prediction_pairs": prediction_pairs,
        "exact_prediction_pairs": exact_prediction_pairs,
        "prediction_exact_fraction": exact_prediction_pairs / prediction_pairs,
        "max_prediction_absolute_difference": float(prediction_difference.max()),
        "backend_versions": backend_versions,
        "interpretation": (
            "The released A/B products are byte-identical at the saved float32 image "
            "precision. This is a matched-renderer equivalence control, not a "
            "simulator-induced image-domain shift; the Moffat conditions remain "
            "an observation-family shift shared by both renderer labels."
        ) if exact_image_pairs == image_pairs and exact_prediction_pairs == prediction_pairs else (
            "The released products contain a measurable A/B difference; inspect "
            "the magnitude and provenance before making a transfer claim."
        ),
    }
    if write:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit_release(), indent=2))
