"""Regenerate an explicitly mutable execution snapshot, without final data.

Reads development completion reports and file presence only. It never opens
pilot/final images, labels or predictions and cannot establish superiority.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import yaml


def main():
    config=yaml.safe_load(Path("configs/ml/robustness_development_v4.yaml").read_text())
    banks={}
    for key in ("source_train","source_development","target_gaussian","target_representative"):
        root=Path(config[key]).parent
        complete=root/"complete.json"
        checkpoint=root/"checkpoint.json"
        saved=len(json.loads(checkpoint.read_text())) if checkpoint.exists() else 0
        products=sum(1 for _ in (root/"products").glob("*.npy")) if (root/"products").exists() else 0
        banks[key]={"manifest_exists":Path(config[key]).exists(),"complete":complete.exists(),
            "failed":(root/"failed.json").exists(),"checkpointed_proposals":saved,
            "product_arrays_partial_or_complete":products,"path":str(root)}
        if complete.exists():
            report=json.loads(complete.read_text())
            banks[key].update(parent_count=report["parents"],computed_proposals=report["computed_proposals"],
                physical_rejections=sum(f["approved_physical_single_image_rejection"] for f in report["failures"]),
                runtime_seconds=report["runtime_seconds"])
    root=Path(config["output"])
    fits=[]
    for path in root.rglob("complete.json"):
        report=json.loads(path.read_text())
        if "checkpoint_sha256" in report:
            fits.append({"path":str(path),"arm":report["arm"],"seed":report["seed"],
                "runtime_seconds":report["runtime_seconds"],"optimizer_updates":report["optimizer_updates"]})
    tuning_complete=sum("tuning" in Path(f["path"]).parts for f in fits)
    status={"snapshot_utc":datetime.now(timezone.utc).isoformat(),"mutable_execution_status_not_scientific_result":True,
        "banks":banks,"completed_development_fits":len(fits),"completed_initial_tuning_fits":tuning_complete,
        "planned_initial_tuning_fits":240,"allocation_weights_exist":(root/"allocation/policy.json").exists(),
        "median_completed_fit_seconds":float(np.median([f["runtime_seconds"] for f in fits])) if fits else None,
        "initial_tuning_complete":(root/"tuning_initial_complete.json").exists(),
        "publishable_accuracy_result_established":False,
        "warning":"Snapshot covers development only; final scientific result requires frozen analysis and claim review."}
    output=Path("results/reports/robustness-execution-v4")
    output.mkdir(parents=True,exist_ok=True)
    (output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
