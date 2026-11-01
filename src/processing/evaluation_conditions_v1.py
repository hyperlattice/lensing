"""Fixed nine-condition observation matrix for the paired lens benchmark.

This defines physical conditions only, without data loading or label access.
The Moffat family is unavailable to source-only methods; representative-target
adaptation has explicitly declared unlabeled access to three Moffat views.
"""
from src.processing.observation_conditions_v1 import condition_for

CONDITIONS = ("canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half",
              "read-noise-double", "moffat-2.5", "moffat-3.5", "moffat-4.5",
              "moffat-3.5-exposure-half")
RENDERERS = ("lenstronomy", "autolens")
PRIMARY_CONDITIONS = ("moffat-2.5", "moffat-3.5", "moffat-4.5")


def evaluation_condition(name):
    if name not in CONDITIONS:
        raise ValueError("Unknown frozen evaluation condition")
    if name == "canonical":
        return {**condition_for("fresh-noise"), "name": name}
    if name.startswith("moffat-"):
        return {**condition_for("fresh-noise"), "name": name,
                "psf_family": "moffat", "moffat_beta": float(name.split("-")[1]),
                "exposure_multiplier": .5 if name.endswith("exposure-half") else 1.}
    return condition_for(name)
