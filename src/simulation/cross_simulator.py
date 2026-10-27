"""Development controls and releases for the optional alternate simulation stack.

No model checkpoints, historical test labels, or frozen holdout products are
read. A passed deflection gate is necessary but not sufficient for image-level
or real-domain accuracy claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from src.simulation import autolens_backend as adapter
from src.simulation.independent_sie import independent_sie_deflection
from src.simulation.priors import draw_observation_parameters
from src.processing.preprocess import preprocess_image

CONFIG = Path("configs/cross_simulator_development_v1.yaml")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def deflection_controls(config):
    """Compare independent analytic SIE equations to the external mass profile."""
    required = str(config["required_backend_version"])
    actual = adapter.backend_version()
    if actual != required:
        raise ValueError(f"Expected autolens {required}, got {actual}")
    grid = np.linspace(-2.5, 2.5, 21)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    records = []
    for q in config["controls"]["tested_axis_ratios"]:
        for phi in config["controls"]["tested_orientations_rad"]:
            parameters = {"theta_E_arcsec": 1.2, "q":q, "phi_rad":phi,
                          "lens_center_x_arcsec":0.031, "lens_center_y_arcsec":-0.047}
            alpha = adapter.deflections_xy(adapter.mass_profile(parameters),xy)
            ax, ay = independent_sie_deflection(xy[:,0],xy[:,1], theta_E_arcsec=1.2,
                       axis_ratio=q,phi_rad=phi,center_x_arcsec=0.031,center_y_arcsec=-0.047)
            error = float(np.max(np.abs(alpha-np.column_stack([ax,ay]))))
            records.append({"q":float(q),"phi_rad":float(phi),"max_error_arcsec":error})
    maximum = max(row["max_error_arcsec"] for row in records)
    return {"backend":"autolens", "version":actual, "checks":records,
            "max_error_arcsec":maximum,
            "passed":maximum <= float(config["controls"]["deflection_max_error_arcsec"]),
            "comparison":"external mass profile versus project independent analytic SIE",
            "source_light_mapping_validated":False, "image_level_equivalence_validated":False}


def run(config_path=CONFIG, *, generate=False):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report = {"experiment_id":config["experiment_id"],"stage":"development",
              "cnn_inference_run":False, "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "python":platform.python_version(), "deflection_controls":deflection_controls(config)}
    report_path = Path(config["report"])
    if not report["deflection_controls"]["passed"]:
        _json(report_path,report)
        raise ValueError("External-backend deflection controls failed; no release generated")
    if not generate:
        _json(report_path,report)
        return report
    if config["split"] != "development" or not str(config["status"]).startswith("development"):
        raise ValueError("This runner only creates development releases")
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new development release ID; existing outputs are immutable")
    output.mkdir(parents=True,exist_ok=True)
    _json(output/"generation_started.json",{"config_sha256":report["config_sha256"]})
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    rows, failures = [], []
    for index in range(int(config["parent_count"])):
        p = draw_observation_parameters(int(config["seed"]),index,source_config)
        p["supersampling_factor"] = int(source_config["pilot"]["sie_extended"]["supersampling_factor"])
        parent_id = f"{config['experiment_id']}-parent-{index+1:04d}"
        for condition in config["conditions"]:
            sample_id = f"{parent_id}__{condition['name']}"
            folder = output / sample_id
            folder.mkdir()
            try:
                rendered = adapter.render_scene(p,condition,include_point_sources=config["include_point_sources"])
                header = fits.Header()
                header["BUNIT"] = "ELECTRONS/S"
                header["SIMULATR"] = "autolens"
                raw_path = folder / "raw.fits"
                fits.writeto(raw_path,rendered["raw"],header=header)
                processed = preprocess_image(raw_path,target_shape=tuple(condition["shape"]),
                                            normalization_method="none",image_unit="electrons_per_second")
                for name in ("expected","clean","expected_variance","psf","positions_xy","point_amplitudes"):
                    np.save(folder/f"{name}.npy",rendered[name])
                np.save(folder/"image.npy",processed["normalized"])
                np.save(folder/"science.npy",processed["science"])
                _json(folder/"parameters.json",{**p,"condition":condition,
                       "light_mapping":"direct backend Sersic parameters; not yet matched to Lenstronomy light",
                       "target_definition":config["target_definition"]})
                _json(folder/"processing.json",{"background":processed["background"].to_dict(),
                       "normalization":processed["normalization"].to_dict(),
                       "geometry":processed["geometry_transform"].to_dict()})
                relative = folder.relative_to(output)
                rows.append({"sample_id":sample_id,"parent_system_id":parent_id,"split":"development",
                             "domain":"simulated","simulator_name":"autolens",
                             "simulator_version":adapter.backend_version(),"condition":condition["name"],
                             "raw_path":(relative/"raw.fits").as_posix(),
                             "processed_path":(relative/"image.npy").as_posix(),
                             "theta_E_arcsec":p["theta_E_arcsec"],"pixel_scale":condition["pixel_scale_arcsec"],
                             "shape":"x".join(map(str,condition["shape"])),
                             "parameter_seed":p["parameter_seed"],"noise_seed":p["noise_seed"],
                             "point_image_count":len(rendered["positions_xy"]),
                             "accuracy_evaluation_eligible":False})
            except Exception as exc:
                failures.append({"sample_id":sample_id,"error_type":type(exc).__name__,"message":str(exc)})
    pd.DataFrame(rows).to_csv(output/"development_manifest.csv",index=False)
    _json(output/"rejections.json",failures)
    report.update({"requested_parents":int(config["parent_count"]),
                   "requested_observations":int(config["parent_count"])*len(config["conditions"]),
                   "accepted_observations":len(rows),"rejected_observations":len(failures),
                   "source_config_sha256":hashlib.sha256(source_config_path.read_bytes()).hexdigest(),
                   "source_hashes":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       [Path(__file__),Path(adapter.__file__),Path("src/simulation/priors.py")]},
                   "environment":{name:importlib.metadata.version(name) for name in
                       ["autolens","autogalaxy","autoarray","autofit","numpy","scipy","astropy"]},
                   "failures":failures,
                   "limitations":["Development samples, not a blinded holdout",
                                  "Light and full-image cross-backend equivalence not yet validated",
                                  "Project-owned root search, integration, PSF and detector code",
                                  "Shared physical proposal and preprocessing; no real-instrument calibration"]})
    hashes = {path.relative_to(output).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.rglob("*")) if path.is_file()}
    _json(output/"checksums.json",hashes)
    _json(output/"generation_complete.json",report)
    _json(report_path,report)
    if failures:
        raise RuntimeError(f"Development release has {len(failures)} recorded failures; inspect {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["controls","generate"])
    parser.add_argument("--config",default=str(CONFIG))
    args = parser.parse_args()
    print(json.dumps(run(args.config,generate=args.action=="generate"),indent=2))


if __name__ == "__main__":
    main()
