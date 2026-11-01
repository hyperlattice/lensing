"""Two-stage study seals and a one-way target-free prediction/scoring boundary.

Seals are auditable file hashes, not signatures or access-control security.
They detect accidental changes and prevent implicit overwrite/restart through
this workflow. They do not make incomplete experiments scientifically valid.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.data.benchmark_contract_v1 import sha256


def create_once(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return sha256(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_records(paths):
    return {str(Path(p)):sha256(p) for p in paths}


def verify_files(records):
    for path,expected in records.items():
        if sha256(path)!=expected:
            raise ValueError(f"Sealed artifact changed: {path}")


def freeze_methods(path, *, study_id, design, artifacts):
    required={"training_roles","arms","target_information_access","primary_endpoint",
              "secondary_contrasts","alpha_allocation","margins","source_selection_rule",
              "training_budget","noise_and_seed_policy","numerical_qualification",
              "failure_and_recovery_policy","power_planning_rule"}
    if set(design)!=required or set(design["training_roles"])!={"source_train","source_development","target_adaptation"}:
        raise ValueError("Incomplete method design or disallowed training roles")
    if set(design["arms"])!=set(design["target_information_access"]):
        raise ValueError("Every method needs an explicit target-information policy")
    if len(design["arms"])!=len(set(design["arms"])) or not study_id:
        raise ValueError("Invalid study or duplicate method identity")
    return create_once(path,{"schema":"robustness_method_freeze_v1","study_id":study_id,
        "created_utc":timestamp(),"design":design,"artifact_sha256":file_records(artifacts),
        "status":"methods_locked_before_power_pilot"})


def verify_method_freeze(path):
    record=json.loads(Path(path).read_text(encoding="utf-8"))
    if record["schema"]!="robustness_method_freeze_v1":
        raise ValueError("Not a method freeze")
    verify_files(record["artifact_sha256"])
    return record


def freeze_evaluation(path, *, method_freeze, design, checkpoints, pilot_design):
    method=verify_method_freeze(method_freeze)
    required={"parent_count","seed_ids","condition_order","renderer_order","generation_seed",
              "rejection_policy","exclusion_ledger","bootstrap","contrasts","release_directory",
              "prediction_directory","truth_path","ensemble_rule","analysis_recovery_policy"}
    if set(design)!=required or type(design["parent_count"]) is not int or design["parent_count"]<2:
        raise ValueError("Incomplete final-evaluation design")
    if Path(design["release_directory"]).exists() or Path(design["prediction_directory"]).exists():
        raise ValueError("Final identities must be frozen before release/prediction creation")
    truth=Path(design["truth_path"]).resolve()
    vault=(Path(design["release_directory"])/"truth_vault").resolve()
    if not truth.is_relative_to(vault):
        raise ValueError("Truth must belong to this new release's isolated vault")
    seeds=design["seed_ids"]
    if len(seeds)<10 or len(set(seeds))!=len(seeds):
        raise ValueError("Confirmation requires at least ten unique paired seeds")
    if set(checkpoints)!=set(method["design"]["arms"]):
        raise ValueError("Checkpoint methods differ from the frozen method list")
    checkpoint_records={}
    for arm,by_seed in checkpoints.items():
        if {str(s) for s in seeds}!=set(by_seed):
            raise ValueError("Checkpoint seed axes differ")
        checkpoint_records[arm]={s:{"path":str(Path(p)),"sha256":sha256(p)} for s,p in by_seed.items()}
    return create_once(path,{"schema":"robustness_evaluation_freeze_v1","study_id":method["study_id"],
        "created_utc":timestamp(),"method_freeze":{"path":str(Path(method_freeze)),"sha256":sha256(method_freeze)},
        "pilot_design":{"path":str(Path(pilot_design)),"sha256":sha256(pilot_design)},
        "design":design,"checkpoints":checkpoint_records,"status":"locked_before_final_generation"})


def verify_evaluation_freeze(path):
    record=json.loads(Path(path).read_text(encoding="utf-8"))
    if record["schema"]!="robustness_evaluation_freeze_v1":
        raise ValueError("Not an evaluation freeze")
    m=record["method_freeze"]
    verify_files({m["path"]:m["sha256"]})
    verify_method_freeze(m["path"])
    for category in ("scientific_role_inputs_sha256","generation_artifact_sha256","training_evidence_sha256"):
        verify_files(record["design"]["exclusion_ledger"].get(category,{}))
    p=record["pilot_design"]
    verify_files({p["path"]:p["sha256"]})
    for by_seed in record["checkpoints"].values():
        verify_files({r["path"]:r["sha256"] for r in by_seed.values()})
    return record


def begin_predictions(evaluation_freeze, release_manifest):
    frozen=verify_evaluation_freeze(evaluation_freeze)
    release=json.loads(Path(release_manifest).read_text(encoding="utf-8"))
    if release.get("evaluation_freeze_sha256")!=sha256(evaluation_freeze) or release.get("role")!="final":
        raise ValueError("Final release is not bound to the evaluation freeze")
    if Path(release_manifest).parent.resolve()!=Path(frozen["design"]["release_directory"]).resolve():
        raise ValueError("Unexpected final-release identity")
    output=Path(frozen["design"]["prediction_directory"])
    output.mkdir(parents=True,exist_ok=True)
    create_once(output/"predictions_started.json",{"evaluation_freeze_sha256":sha256(evaluation_freeze),
        "release_manifest":{"path":str(Path(release_manifest)),"sha256":sha256(release_manifest)},
        "started_utc":timestamp(),"target_labels_opened":False})
    return output


def seal_predictions(evaluation_freeze, *, prediction_files, validated_axes):
    frozen=verify_evaluation_freeze(evaluation_freeze)
    output=Path(frozen["design"]["prediction_directory"])
    started=json.loads((output/"predictions_started.json").read_text())
    if started["evaluation_freeze_sha256"]!=sha256(evaluation_freeze):
        raise ValueError("Prediction freeze identity changed")
    release=started["release_manifest"]
    verify_files({release["path"]:release["sha256"]})
    expected={"parent_count":frozen["design"]["parent_count"],"seed_ids":frozen["design"]["seed_ids"],
        "condition_order":frozen["design"]["condition_order"],"renderer_order":frozen["design"]["renderer_order"],
        "arms":list(frozen["checkpoints"])}
    if validated_axes!=expected:
        raise ValueError("Prediction axes differ from frozen design")
    manifest=json.loads(Path(release["path"]).read_text())
    parent_ids=manifest["parent_ids"]
    if len(parent_ids)!=expected["parent_count"] or len(set(parent_ids))!=len(parent_ids):
        raise ValueError("Invalid final parent identities")
    if len(prediction_files)!=len(expected["arms"]):
        raise ValueError("Missing prediction methods")
    methods=set()
    fields={"prediction_arcsec","parent_ids","seed_ids","conditions","renderers","arm"}
    shape=(len(expected["seed_ids"]),expected["parent_count"],len(expected["condition_order"]),len(expected["renderer_order"]))
    for path in prediction_files:
        if Path(path).resolve().parent!=output.resolve():
            raise ValueError("Prediction file is outside the frozen output directory")
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!=fields:
                raise ValueError("Unexpected prediction fields; truth must remain separate")
            arm=str(arrays["arm"].item())
            if arm in methods or arm not in expected["arms"]:
                raise ValueError("Duplicate/unknown prediction method")
            methods.add(arm)
            if (arrays["parent_ids"].tolist()!=parent_ids or arrays["seed_ids"].tolist()!=expected["seed_ids"] or
                arrays["conditions"].tolist()!=expected["condition_order"] or arrays["renderers"].tolist()!=expected["renderer_order"] or
                arrays["prediction_arcsec"].shape!=shape or not np.isfinite(arrays["prediction_arcsec"]).all()):
                raise ValueError("Prediction identities, dimensions or finite values differ")
    return create_once(output/"predictions_complete.json",{"evaluation_freeze_sha256":sha256(evaluation_freeze),
        "prediction_sha256":file_records(prediction_files),"validated_axes":validated_axes,
        "completed_utc":timestamp(),"target_labels_opened":False})


def open_truth_after_predictions(evaluation_freeze):
    """Open ONLY this study's declared truth after sealed target-free predictions."""
    frozen=verify_evaluation_freeze(evaluation_freeze)
    output=Path(frozen["design"]["prediction_directory"])
    prediction=json.loads((output/"predictions_complete.json").read_text())
    if prediction["evaluation_freeze_sha256"]!=sha256(evaluation_freeze):
        raise ValueError("Prediction freeze mismatch")
    verify_files(prediction["prediction_sha256"])
    started=json.loads((output/"predictions_started.json").read_text())
    release=started["release_manifest"]
    verify_files({release["path"]:release["sha256"]})
    manifest=json.loads(Path(release["path"]).read_text())
    truth_path=Path(frozen["design"]["truth_path"])
    if sha256(truth_path)!=manifest["truth_sha256"]:
        raise ValueError("Sealed truth artifact changed")
    create_once(output/"analysis_started.json",{"evaluation_freeze_sha256":sha256(evaluation_freeze),
        "predictions_complete_sha256":sha256(output/"predictions_complete.json"),"started_utc":timestamp()})
    truth=json.loads(truth_path.read_text(encoding="utf-8"))
    if set(truth)!=set(manifest["parent_ids"]):
        raise ValueError("Truth identity join differs from final parents")
    return truth
