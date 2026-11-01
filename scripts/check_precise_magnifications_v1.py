"""Check fourth-order magnification stability on all v1 point-related failures."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.simulation.precise_point_renderer_v1 import PreciseSceneEngine, fourth_order_amplitudes


def main():
    references = Path("data/simulations/renderer-qualification-v1-complete")
    m = json.loads((references/"references.json").read_text())
    v1 = json.loads(Path("results/reports/renderer-qualification-v1/report.json").read_text())
    failed = {r["name"] for r in v1["scene_checks"] if not r["cross_engine_passed"]}
    rows = []
    done = set()
    for r in m["cases"]:
        if r["name"] not in failed or r["index"] in done:
            continue
        done.add(r["index"])
        engine = PreciseSceneEngine(r["parameters"], "autolens")
        positions, amplitudes = engine.points()
        half = fourth_order_amplitudes(engine, positions, step=5e-5)
        with np.load(references/f"{r['name']}.npz") as original:
            distances = np.linalg.norm(positions[:,None]-original["positions_xy"][None],axis=2)
            i,j=linear_sum_assignment(distances)
            error = float(max(abs(amplitudes[i]/original["point_amplitudes"][j]-1)))
        stability = float(max(abs(amplitudes/half-1)))
        rows.append({"index":r["index"],"relative_step_difference":stability,
                     "relative_amplitude_error_vs_A":error,"passed":stability<=1e-6})
    output=Path("results/reports/renderer-qualification-v3/point_preflight.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump({"cases":rows,"passed":all(r["passed"] for r in rows)},stream,indent=2)
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
