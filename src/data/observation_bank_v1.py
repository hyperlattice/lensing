"""Generative source bank and image-only target bank for development training.

Source expected-rate arrays are simulator internals used solely to draw fresh
observations. They are not predictor features. Target banks expose only fixed
observed images and never return target labels, expected scenes or parameters.
Qualification and identity audit are mandatory outside explicit unit fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.benchmark_contract_v1 import contained_path, validate_lineage
from src.processing.observation_conditions_v1 import CELLS, condition_for
from src.simulation.autolens_backend_v2 import sample_detector


class ObservationBank:
    def __init__(self, path, *, expected_role, allow_fixture=False):
        if expected_role not in {"source_train", "source_development", "target_adaptation"}:
            raise ValueError("This bank cannot load pilot/final data")
        self.path = Path(path)
        self.root = self.path.parent
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document["schema"] != "observation_bank_v1" or document["role"] != expected_role:
            raise ValueError("Bank schema/role mismatch")
        self.role = expected_role
        if document["stage"] == "unit_fixture":
            if not allow_fixture:
                raise ValueError("Unit fixtures are not scientific data")
        else:
            if document["stage"] != "qualified_development":
                raise ValueError("Observation bank is not qualified")
            qualification = document["qualification"]
            report_path = Path(qualification["path"])
            report = json.loads(report_path.read_text())
            if report["passed"] is not True or report["rendering_factor"] != document["rendering_factor"]:
                raise ValueError("Failed or inconsistent rendering qualification")
        self.document = document
        if document["stage"] != "unit_fixture" and (
            document.get("image_unit") != "electrons_per_second" or
            document.get("target_definition") != "lenstronomy_SIE_circularized_parameter_arcsec"):
            raise ValueError("Missing image units or target convention")
        self.parents = [{**p, "latent_id": p.get("latent_id", p.get("latent_sha256", p["parent_system_id"]))} for p in document["parents"]]
        self.by_id = {p["parent_system_id"]: p for p in self.parents}
        if not self.parents or len(self.by_id) != len(self.parents):
            raise ValueError("Empty/duplicate parent identities")
        validate_lineage(self.parents, document["exclusion_ledger"])
        seeds, noise_seeds, fingerprints = {}, set(), set()
        for p in self.parents:
            if p["role"] != expected_role:
                raise ValueError("Parent crossed bank role")
            fingerprint = p["latent_id"]
            if fingerprint in fingerprints or p["parameter_seed"] in seeds or p["noise_seed"] in noise_seeds:
                raise ValueError("Duplicate latent parent or parameter/noise seed alias")
            fingerprints.add(fingerprint)
            seeds[p["parameter_seed"]] = p["parent_system_id"]
            noise_seeds.add(p["noise_seed"])
        self._arrays = {}
        if expected_role == "target_adaptation":
            if document["targets"] is not None or any("theta_E_arcsec" in p for p in self.parents):
                raise ValueError("Target adaptation bank contains labels")
            for p in self.parents:
                if set(p).difference({"latent_sha256"}) != {"parent_system_id", "role", "parameter_seed", "noise_seed", "ancestry_ids", "latent_id", "images"}:
                    raise ValueError("Target bank contains unexpected physical metadata")
                self._arrays[p["parent_system_id"]] = [self._array(a, nonnegative=False) for a in p["images"]]
                if not self._arrays[p["parent_system_id"]]:
                    raise ValueError("Target parent has no observations")
            self.targets = None
        else:
            target = document["targets"]
            path = contained_path(self.root, target["path"])
            self.targets = json.loads(path.read_text())
            if set(self.targets) != set(self.by_id) or any(not np.isfinite(v) or v <= 0 for v in self.targets.values()):
                raise ValueError("Source labels must match parent identities")
            for p in self.parents:
                if set(p["expected"]) != {"base", "pixel-scale-004", "pixel-scale-006", "gaussian-fwhm-075", "gaussian-fwhm-125"}:
                    raise ValueError("Source bank missing shared transformation support")
                self._arrays[p["parent_system_id"]] = {k: self._array(a, nonnegative=True) for k,a in p["expected"].items()}
                if document["stage"] != "unit_fixture":
                    for key,array in self._arrays[p["parent_system_id"]].items():
                        cell="fresh-noise" if key=="base" else key
                        if array.shape!=tuple(condition_for(cell)["shape"]):
                            raise ValueError("Source cell has the wrong declared native geometry")
                detector = p["detector"]
                if set(detector) != {"exposure_s", "read_noise_electrons", "background_rate"} or not np.isfinite(list(detector.values())).all():
                    raise ValueError("Invalid detector metadata")
                if detector["exposure_s"]<=0 or min(detector["read_noise_electrons"],detector["background_rate"])<0:
                    raise ValueError("Invalid detector exposure/read/background units")

    def _array(self, record, *, nonnegative):
        path = contained_path(self.root, record["path"])
        # Read-only file-backed arrays avoid committing the entire multi-GB
        # source/target bank to private RAM on the 8-GB development machine.
        a = np.load(path, allow_pickle=False, mmap_mode="r")
        if a.ndim != 2 or a.dtype != np.float32 or not np.isfinite(a).all() or (nonnegative and np.any(a < 0)):
            raise ValueError("Invalid native-grid observation array")
        return a

    def source_image(self, parent, *, cell="fresh-noise", noise_seed=None, historical=False):
        if self.targets is None:
            raise ValueError("Target bank cannot generate source observations")
        p = self.by_id[parent]
        if historical:
            condition = p["historical_condition"]
            key = {0.04: "pixel-scale-004", 0.06: "pixel-scale-006"}.get(condition["pixel_scale_arcsec"], "base")
        else:
            condition = condition_for(cell)
            key = cell if cell in {"pixel-scale-004", "pixel-scale-006", "gaussian-fwhm-075", "gaussian-fwhm-125"} else "base"
        d = p["detector"]
        fixed = p.get("historical_noise_seed", p["noise_seed"]) if historical else p["noise_seed"]
        seed = fixed if noise_seed is None else noise_seed
        raw, _ = sample_detector(self._arrays[parent][key], d["exposure_s"]*condition["exposure_multiplier"],
            d["read_noise_electrons"]*condition["read_noise_multiplier"], d["background_rate"], seed)
        return raw.astype(np.float32)

    def target_image(self, parent, view_index):
        if self.targets is not None:
            raise ValueError("Source bank cannot be used as target adaptation")
        return self._arrays[parent][view_index % len(self._arrays[parent])]

    def source_statistics(self):
        if self.role != "source_train":
            raise ValueError("Normalization is source-training-only")
        total, squared, pixels = 0., 0., 0
        for parent in self.by_id:
            a = self.source_image(parent).astype(float)
            total += a.sum()
            squared += np.square(a).sum()
            pixels += a.size
        y = np.asarray(list(self.targets.values()))
        mean = total/pixels
        std = np.sqrt(max(squared/pixels-mean**2, 0.))
        if min(std, y.std()) <= 0:
            raise ValueError("Degenerate source normalization")
        return {"image_mean": float(mean), "image_std": float(std),
                "target_mean": float(y.mean()), "target_std": float(y.std())}
