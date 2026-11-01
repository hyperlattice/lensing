"""Validate low-memory A roots/images against immutable qualified references."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.qualify_benchmark_renderer_v1 import digest,relative_l1,write
from src.simulation.low_memory_renderer_v1 import LowMemorySceneEngine


def main():
    parent_path=Path("results/reports/renderer-qualification-v3/qualification.json")
    parent=json.loads(parent_path.read_text())
    if not parent["passed"]:
        raise ValueError("Parent numerical qualification failed")
    for path,expected in parent["source_sha256"].items():
        if digest(path)!=expected:
            raise ValueError("Parent qualified source changed")
    output=Path("results/reports/renderer-qualification-v4")
    if output.exists():
        raise FileExistsError("Qualification already started")
    output.mkdir(parents=True)
    sources={**parent["source_sha256"],**{p:digest(p) for p in
        ["src/simulation/low_memory_renderer_v1.py",__file__,"docs/renderer_qualification_protocol_v5.md"]}}
    write(output/"started.json",{"source_sha256":sources,"parent_qualification_sha256":digest(parent_path)})
    references=Path("data/simulations/renderer-qualification-v3")
    metadata=json.loads((references/"references.json").read_text())
    rows,failures,current,engine=[],[],None,None
    started=time.perf_counter()
    for row in metadata["cases"]:
        try:
            if row["index"]!=current:
                current=row["index"]
                engine=LowMemorySceneEngine(row["parameters"],"lenstronomy")
            with np.load(references/f"{row['name']}.npz") as reference:
                rendered=engine.expected(row["condition"],factor=4,clean=reference["clean"])
                a,b=rendered["positions_xy"],reference["positions_xy"]
                if len(a)!=len(b):
                    raise ValueError("Low-memory root multiplicity differs")
                distance=np.linalg.norm(a[:,None]-b[None],axis=2)
                i,j=linear_sum_assignment(distance)
                position=float(distance[i,j].max())
                amplitude=float(np.max(abs(rendered["point_amplitudes"][i]/reference["point_amplitudes"][j]-1)))
                image=relative_l1(rendered["expected"],reference["expected"])
            rows.append({"name":row["name"],"position_error_arcsec":position,"amplitude_relative_error":amplitude,
                "expected_normalized_l1":image,"passed":bool(position<=1e-5 and amplitude<=1e-3 and image<=1e-5)})
        except Exception as exc:
            failures.append({"name":row["name"],"error":type(exc).__name__,"message":str(exc)})
    passed=not failures and len(rows)==315 and all(r["passed"] for r in rows)
    report={"passed":bool(passed),"cases":rows,"failures":failures,"runtime_seconds":time.perf_counter()-started,
        "parent_qualification_sha256":digest(parent_path),"source_sha256":sources}
    write(output/"root_report.json",report)
    result={**{k:v for k,v in parent.items() if k not in {"source_sha256","rendering_factor","passed"}},
        "passed":bool(passed),"rendering_factor":"adaptive_midpoint_1e-4_max512_precise_lowmem_roots_v1",
        "source_sha256":sources,"root_report_sha256":digest(output/"root_report.json"),
        "parent_qualification_sha256":digest(parent_path)}
    write(output/"qualification.json",result)
    print(json.dumps({"passed":bool(passed),"cases":len(rows),"failures":failures,
        "seconds":time.perf_counter()-started,"worst_expected_l1":max((r['expected_normalized_l1'] for r in rows),default=None)},indent=2))


if __name__=="__main__":
    main()
