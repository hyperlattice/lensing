"""Export original development references, then validate the alternate stack.

Run export in the original environment and validate in .venv-autolens. The
reference files contain no trained-model predictions or final-holdout data.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from scipy.special import gamma
import yaml

ROOT = Path("data/simulations/cross-simulator-controls-v2-nojit")
REPORT = Path("results/reports/cross-simulator-controls-v2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(root=ROOT):
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    if version("lenstronomy") != "1.14.2":
        raise ValueError("Original reference requires lenstronomy 1.14.2")
    from src.simulation.models import build_light_model, ellipticity_from_q_phi
    from src.simulation.observation import render_observation_scene, observation_psf
    from src.simulation.priors import draw_observation_parameters
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Reference output already exists")
    root.mkdir(parents=True, exist_ok=True)
    center = np.array([-.13, .07])
    radius = np.geomspace(.0003, 3.0, 128)
    angle = np.arange(16) * 2 * np.pi / 16
    xy = center + np.stack([np.outer(radius, np.cos(angle)).ravel(),
                            np.outer(radius, np.sin(angle)).ravel()], axis=1)
    np.save(root / "xy.npy", xy)
    light = build_light_model(["SERSIC_ELLIPSE"])
    records = []
    for n in (.5, 1., 2., 4.):
        for q in (1., .7, .4):
            for phi in (0., .3, 1.1):
                name = f"light-{len(records):03d}"
                e1, e2 = ellipticity_from_q_phi(q, phi)
                kwargs = {"amp": 1.3, "R_sersic": .2, "n_sersic": n,
                          "e1": e1, "e2": e2, "center_x": center[0], "center_y": center[1]}
                values = light.surface_brightness(xy[:, 0], xy[:, 1], [kwargs])
                np.save(root / f"{name}.npy", values)
                records.append({"name": name, "n": n, "q": q, "phi": phi,
                                "intensity": 1.3, "radius": .2, "center": center.tolist(),
                                "total_flux": float(light.total_flux([kwargs])[0]),
                                "floored_center_brightness": float(light.surface_brightness(
                                    center[:1], center[1:], [kwargs])[0])})
    source_config = Path("configs/simulation_engineering_v1.yaml")
    config = yaml.safe_load(source_config.read_text())
    scenes = []
    for index in range(4):
        p = draw_observation_parameters(61031, index, config)
        p["supersampling_factor"] = 2
        scene = render_observation_scene(p, shape=(256, 256), pixel_scale_arcsec=.05,
                                         psf_fwhm_arcsec=p["psf_fwhm_arcsec"], supersampling_factor=2)
        name = f"scene-{index:03d}"
        np.savez(root / f"{name}.npz", clean=scene.clean_scene, expected=scene.expected_scene,
                 psf=observation_psf(psf_fwhm_arcsec=p["psf_fwhm_arcsec"], pixel_scale_arcsec=.05).values,
                 positions=np.column_stack([scene.point_positions_x_arcsec, scene.point_positions_y_arcsec]),
                 amplitudes=scene.point_amplitudes)
        scenes.append({"name": name, "parameters": p})
    sources = [Path("src/simulation") / name for name in ("observation.py", "models.py", "priors.py")]
    metadata = {"stage": "development_controls", "numba_jit": "disabled", "light_cases": records, "scenes": scenes,
                "environment": {key: version(key) for key in ("lenstronomy", "numpy", "scipy")},
                "source_sha256": {str(p): digest(p) for p in [*sources, Path(__file__), source_config]},
                "array_sha256": {p.name: digest(p) for p in root.iterdir() if p.is_file()},
                "final_holdout_accessed": False, "cnn_inference_run": False}
    write(root / "references.json", metadata)
    print(f"Exported {len(records)} light controls and {len(scenes)} development scenes", flush=True)


def validate(root=ROOT, output=REPORT):
    from src.simulation import autolens_backend_v2 as adapter
    if adapter.backend_version() != "2026.10.26.1":
        raise ValueError("Wrong alternate backend version")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Control evidence already exists")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "references.json").read_text())
    for name, expected_hash in metadata["array_sha256"].items():
        if digest(root / name) != expected_hash:
            raise ValueError("Original reference array hash changed")
    xy = np.load(root / "xy.npy")
    al = adapter.backend()
    light_records, center_records = [], []
    for row in metadata["light_cases"]:
        p = {f"source_{key}": value for key, value in {
            "amp": row["intensity"], "re_arcsec": row["radius"], "n": row["n"],
            "q": row["q"], "phi_rad": row["phi"]}.items()}
        grid = al.Grid2DIrregular(values=xy[:, ::-1].copy())
        mapped = adapter._light(p, "source", row["center"])
        direct = al.lp.Sersic(centre=tuple(row["center"])[::-1],
                              ell_comps=adapter._ellipticity(row["q"], row["phi"]),
                              intensity=row["intensity"], effective_radius=row["radius"], sersic_index=row["n"])
        expected = np.load(root / f"{row['name']}.npy")
        values = np.asarray(mapped.image_2d_from(grid=grid))
        naive = np.asarray(direct.image_2d_from(grid=grid))
        denominator = np.maximum(abs(expected), 1e-200)
        relative_error = float(np.max(abs(values - expected) / denominator))
        conversion = adapter.mapped_light_parameters(row["intensity"], row["radius"], row["n"])
        b = conversion["b_backend"]
        total = (2 * np.pi * row["n"] * conversion["intensity"] * conversion["effective_radius"]**2
                 * np.exp(b) * gamma(2 * row["n"]) / b**(2 * row["n"]))
        flux_error = abs(total / row["total_flux"] - 1)
        light_records.append({"name": row["name"], "max_relative_error": relative_error,
                              "direct_parameter_max_relative_error": float(np.max(abs(naive - expected) / denominator)),
                              "total_flux_relative_error": float(flux_error),
                              "passed": bool(relative_error <= 1e-10 and flux_error <= 1e-10)})
        center_grid = al.Grid2DIrregular(values=[row["center"][::-1]])
        center_records.append({"name": row["name"], "original_floored": row["floored_center_brightness"],
                               "alternate_unfloored": float(np.asarray(mapped.image_2d_from(grid=center_grid))[0])})
    scene_records = []
    for row in metadata["scenes"]:
        p = row["parameters"]
        result = adapter.render_scene(p, {"shape": [256, 256], "pixel_scale_arcsec": .05,
                                            "psf_family": "gaussian", "exposure_multiplier": 1.,
                                            "read_noise_multiplier": 1.})
        reference = np.load(root / f"{row['name']}.npz")
        errors = {key: float(np.sum(abs(result[key] - reference[key])) / np.sum(abs(reference[key])))
                  for key in ("clean", "expected")}
        positions = result["positions_xy"]
        if len(positions) != len(reference["positions"]):
            scene_records.append({"name": row["name"], "passed": False, "reason": "point multiplicity mismatch"})
            continue
        distance = np.linalg.norm(positions[:, None] - reference["positions"][None], axis=2)
        i, j = linear_sum_assignment(distance)
        point_error = float(distance[i, j].max())
        amplitude_error = float(np.max(abs(result["point_amplitudes"][i] / reference["amplitudes"][j] - 1)))
        psf_error = (float(abs(result["psf"] - reference["psf"]).sum())
                     if result["psf"].shape == reference["psf"].shape else None)
        scene_records.append({"name": row["name"], "normalized_l1_errors": errors,
                              "max_point_error_arcsec": point_error, "max_amplitude_relative_error": amplitude_error,
                              "psf_l1_error": psf_error,
                              "passed": max(errors.values()) <= 1e-5 and point_error <= 1e-5
                              and amplitude_error <= 1e-3 and psf_error is not None and psf_error <= 1e-6})
    kernel_records = []
    for family in ("gaussian", "moffat"):
        kernel = adapter.psf_kernel(.15, .05, family)
        impulse = np.zeros((128, 128)); impulse[64, 64] = 1
        flux_error = float(abs(fftconvolve(impulse, kernel, mode="same").sum() - 1))
        kernel_records.append({"family": family, "sum": float(kernel.sum()), "impulse_flux_error": flux_error,
                               "passed": bool(np.all(kernel >= 0) and abs(kernel.sum() - 1) <= 1e-6 and flux_error <= 1e-6)})
    noise_records = []
    for exposure in (50., 100.):
        for read in (0., 2.):
            for signal in (.1, 1.):
                expected = np.full((128, 128), signal)
                samples = np.stack([adapter.sample_detector(expected, exposure, read, .03, 81000 + seed)[0]
                                    for seed in range(32)])
                variance = ((signal + .03) * exposure + read**2) / exposure**2
                z = float(abs(samples.mean() - signal - .03) / np.sqrt(variance / samples.size))
                variance_error = float(abs(samples.var(ddof=1) / variance - 1))
                noise_records.append({"exposure": exposure, "read_noise": read, "signal_rate": signal,
                                      "mean_z": z, "variance_relative_error": variance_error,
                                      "passed": z <= 5 and variance_error <= .02})
    groups = {"light_profiles": light_records, "images_and_points": scene_records,
              "psf": kernel_records, "detector_moments": noise_records}
    result = {"protocol": "docs/cross_simulator_validation_protocol_v2.md", "groups": groups,
              "passed": all(row["passed"] for group in groups.values() for row in group),
              "central_floor_diagnostics_not_equivalence_gate": center_records,
              "reference_metadata_sha256": digest(root / "references.json"),
              "adapter_sha256": digest(adapter.__file__), "control_script_sha256": digest(__file__),
              "original_environment": metadata["environment"],
              "alternate_environment": {k: version(k) for k in ("autolens", "autogalaxy", "numpy", "scipy")},
              "cnn_inference_run": False, "final_holdout_accessed": False,
              "limitations": ["Development grid and four parents only", "Original central radial floor is not imposed on alternate profile",
                              "Project integration, root search, PSF convolution and NumPy detector; not native SimulatorImaging",
                              "No pixel-integration convergence or empirical instrument validation", "No accuracy benchmark qualification"]}
    write(output / "controls.json", result)
    print(json.dumps({"passed": result["passed"], "groups": {k: sum(row["passed"] for row in v)
                      for k, v in groups.items()}}, indent=2))
    if not result["passed"]:
        raise ValueError("Development controls failed; inspect saved evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "validate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    export(args.root) if args.action == "export" else validate(args.root, args.output)
