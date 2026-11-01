"""Check integration convergence of the corrected development renderer."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.simulation import autolens_backend_v2 as adapter


def main():
    reference = Path("data/simulations/cross-simulator-controls-v2-nojit/references.json")
    output = Path("results/reports/cross-simulator-convergence-v1")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Convergence evidence already exists")
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(reference.read_text())["scenes"]
    records = []
    condition = {"shape": [256, 256], "pixel_scale_arcsec": .05, "psf_family": "gaussian",
                 "exposure_multiplier": 1., "read_noise_multiplier": 1.}
    for case in cases:
        previous, differences = None, {}
        for factor in (2, 4, 8):
            parameters = {**case["parameters"], "supersampling_factor": factor}
            rendered = adapter.render_scene(parameters, condition)
            current = {key: rendered[key] for key in ("clean", "expected")}
            if previous is not None:
                differences[f"{factor//2}_to_{factor}"] = {
                    key: float(np.sum(abs(current[key] - previous[key])) / np.sum(abs(current[key])))
                    for key in current}
            previous = current
        passed = all(differences["4_to_8"][key] <= min(.005, differences["2_to_4"][key])
                     for key in ("clean", "expected"))
        records.append({"case": case["name"], "normalized_l1_differences": differences,
                        "passed": bool(passed)})
        print(json.dumps(records[-1]), flush=True)
    result = {"protocol": "docs/cross_simulator_convergence_protocol_v1.md", "cases": records,
              "passed": all(row["passed"] for row in records),
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cnn_inference_run": False, "final_holdout_accessed": False,
              "scope": "Four development draws, canonical Gaussian condition only"}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Pixel-integration convergence", "",
             f"Predeclared development gate passed: {result['passed']}.", "",
             "| Case | Clean 2-to-4 L1 | Clean 4-to-8 L1 | Expected 2-to-4 L1 | Expected 4-to-8 L1 | Passed |",
             "|---|---:|---:|---:|---:|---|"]
    for row in records:
        a, b = row["normalized_l1_differences"]["2_to_4"], row["normalized_l1_differences"]["4_to_8"]
        lines.append(f"| {row['case']} | {a['clean']:.6g} | {b['clean']:.6g} | {a['expected']:.6g} | {b['expected']:.6g} | {row['passed']} |")
    lines += ["", "These checks cover four development draws at canonical geometry. They neither establish arbitrary-parameter convergence nor qualify a final dataset. The predeclared tolerance was 0.5% normalized L1 plus monotonic refinement.", ""]
    (output / "report.md").write_text("\n".join(lines))
    if not result["passed"]:
        raise ValueError("Integration convergence gate failed; preserve evidence")


if __name__ == "__main__":
    main()
