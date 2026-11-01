"""Bounded rendering-worker preflight on planned development-role parents.

These images are not a qualified bank and never enter model fitting. Source
labels are allowed source data; target labels are never serialized. No final
parents, pixels or labels are accessed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.generate_robustness_banks_parallel_v1 import render_one,write_native_masks
from scripts.generate_robustness_banks_v1 import source_proposals
from scripts.qualify_benchmark_renderer_v1 import digest,write
from src.simulation.priors import draw_observation_parameters


def main(kind):
    output=Path(f"data/simulations/robustness-bank-worker-preflight-v2-{kind}")
    report=Path(f"results/reports/robustness-bank-worker-preflight-v2/{kind}.json")
    if output.exists() or report.exists():
        raise FileExistsError("Preflight already exists")
    output.mkdir(parents=True)
    write_native_masks(output)
    write(output/"started.json",{"stage":"engineering_preflight_not_training","final_access":False})
    tasks=[]
    for role in (("source_train","source_development") if kind=="source" else ()):
        parents,_=source_proposals(role)
        tasks.extend((role,index,*record,output.as_posix()) for index,record in enumerate(parents[:2]))
    config=yaml.safe_load(Path("configs/simulation_engineering_v1.yaml").read_text())
    for index in (range(2) if kind=="target" else ()):
        tasks.append(("target_adaptation",index,{"parent_system_id":f"target-adapt-v1-{index:04d}"},
            draw_observation_parameters(720502,index,config),None,None,output.as_posix()))
    results=[]
    for task in tasks:
        result=render_one(task)
        if result["failure"] is not None:
            raise ValueError(result["failure"])
        if task[0]=="target_adaptation" and result["label"] is not None:
            raise AssertionError("Target label escaped generator boundary")
        for observation in result["observations"]:
            for field in ("expected_rate","image","variance","psf","invalid_mask","coverage_mask"):
                if field not in observation:
                    continue
                record=observation[field]
                assert digest(output/record["path"])==record["sha256"]
                a=np.load(output/record["path"],allow_pickle=False)
                assert a.ndim==2 and np.isfinite(a).all()
                if "mask" in field:
                    assert a.dtype==bool
        results.append({"role":task[0],"parent":result["identity"]["parent_system_id"],
            "observations":len(result["observations"]),"passed":True,"target_label_returned":task[0]=="target_adaptation" and result["label"] is not None})
        print(json.dumps(results[-1]),flush=True)
    report.parent.mkdir(parents=True,exist_ok=True)
    write(report,{"stage":"engineering_preflight_not_training","cases":results,"passed":True,
        "bank_qualification_granted":False,"cnn_inference_run":False,"final_arrays_accessed":False,
        "generator_sha256":digest("scripts/generate_robustness_banks_parallel_v1.py")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=["source","target"])
    main(parser.parse_args().kind)
