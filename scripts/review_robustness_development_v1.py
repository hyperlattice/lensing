"""Summarize all source-only trials and run matched finalist budget curves.

Existing fit identities are reused only after complete/checkpoint/settings
verification. Partial or changed runs block execution, never silently restart.
The output is a review artifact; this script does not freeze methods.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_robustness_development_v1 import CONFIG
from src.data.benchmark_contract_v1 import sha256
from src.data.observation_bank_v1 import ObservationBank
from src.ml.robustness_training_v1 import train_arm
from src.ml.source_policy_scoring_v1 import source_error_matrix
from src.ml.study_freeze_v1 import create_once


def summarize(config):
    root=Path(config["output"])
    complete=root/"tuning_initial_complete.json"
    records=json.loads(complete.read_text())["records"]
    grouped=defaultdict(list)
    for record in records:
        relative=record.get("output_relative_path",
            f"tuning/{record['information_access']}/trial-{record['trial']:02d}/seed-{record['seed']}")
        folder=root/relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Tuning record contains an unsafe fit path")
        if sha256(folder/"final.pt")!=record["run"]["checkpoint_sha256"]:
            raise ValueError("Changed tuning checkpoint")
        grouped[(record["information_access"],record["trial"])].append(record)
    expected=set(config["tuning_seeds"])
    trials=defaultdict(list)
    for (label,trial),rows in grouped.items():
        if {r["seed"] for r in rows}!=expected or len(rows)!=len(expected):
            raise ValueError("Incomplete/duplicate paired tuning seeds")
        scores=[r["source_score"]["shared_source_score_arcsec"] for r in rows]
        trials[label].append({"trial":trial,"settings":rows[0]["settings"],
            "source_score_mean_arcsec":float(np.mean(scores)),"scores_by_seed":scores})
    selected={}
    for label,rows in trials.items():
        best=min(rows,key=lambda r:(r["source_score_mean_arcsec"],r["trial"]))
        settings=best["settings"]
        is_mmd="+MMD" in label
        lr_grid=(.0003,.001,.003) if is_mmd else (.0003,.0006,.001,.003)
        secondary_grid=(.003,.01,.1,1.4) if is_mmd else (0.,.0001,.001)
        key="mmd_weight" if is_mmd else "weight_decay"
        selected[label]={**best,"boundary_learning_rate":settings["learning_rate"] in (min(lr_grid),max(lr_grid)),
            "boundary_secondary":settings[key] in (min(secondary_grid),max(secondary_grid)),
            "boundary_note":"zero weight decay is a physical lower boundary; other boundaries require review"}
    # Conservative causal allocation comparison: optimize uniform, share with R.
    causal={k:dict(v) for k,v in selected.items()}
    for candidate,comparator in (("R","G"),("R+MMD","G+MMD"),
            ("R+MMD-representative","G+MMD-representative")):
        causal[candidate]={**selected[comparator],"settings_selected_for":comparator,
                           "independent_candidate_trial":selected[candidate]["trial"]}
    return {"stage":"source_only_review_not_freeze","tuning_summary_sha256":sha256(complete),
        "trials":dict(trials),"independently_selected":selected,"causal_comparison_settings":causal,
        "target_prediction_errors_used":False,"final_images_used":False,
        "requires":"Resolve search boundaries and convergence before Freeze 1"}


def checked_fit(folder,settings,**kwargs):
    folder=Path(folder)
    if folder.exists():
        report=json.loads((folder/"complete.json").read_text())
        started=json.loads((folder/"started.json").read_text())
        if (started["config"]!=settings or started["seed"]!=kwargs["seed"] or started["arm"]!=kwargs["arm"] or
            sha256(folder/"final.pt")!=report["checkpoint_sha256"]):
            raise ValueError("Existing completed fit identity changed")
        expected_banks={str(bank.path):sha256(bank.path) for bank in
            (kwargs["source"],kwargs["development"],kwargs["target"]) if bank is not None}
        if started["input_bank_sha256"]!=expected_banks:
            raise ValueError("Completed fit used different requested banks")
        if (sha256(folder/"started.json")!=report["started_sha256"] or
            sha256(folder/"sampling_controls.json")!=report["sampling_controls_sha256"]):
            raise ValueError("Completed fit provenance or realized schedule changed")
        for path,expected in started["input_bank_sha256"].items():
            if sha256(path)!=expected:
                raise ValueError("Input bank changed since completed fit")
        return report
    return train_arm(settings,output=folder,**kwargs)


def run(action,config_path):
    config=yaml.safe_load(Path(config_path).read_text())
    root=Path(config["output"])
    summary_path=root/"source_tuning_review.json"
    if action=="summarize":
        summary=summarize(config)
        create_once(summary_path,summary)
        print(json.dumps(summary["independently_selected"],indent=2))
        return
    summary=json.loads(summary_path.read_text())
    source=ObservationBank(config["source_train"],expected_role="source_train")
    dev=ObservationBank(config["source_development"],expected_role="source_development")
    gaussian=ObservationBank(config["target_gaussian"],expected_role="target_adaptation")
    representative=ObservationBank(config["target_representative"],expected_role="target_adaptation")
    policy=json.loads((root/"allocation/policy.json").read_text())
    records=[]
    for label,chosen in summary["causal_comparison_settings"].items():
        arm=label.removesuffix("-representative")
        target=(representative if label.endswith("-representative") else gaussian) if "+MMD" in label else None
        for updates in (880,1760,3520):
            settings={**chosen["settings"],"optimizer_updates":updates,"fit_purpose":"development_only"}
            for seed in config["tuning_seeds"]:
                folder=root/"budget_curves"/label/f"updates-{updates}"/f"seed-{seed}"
                report=checked_fit(folder,settings,seed=seed,arm=arm,source=source,development=dev,
                    target=target,probabilities=policy["probabilities"])
                score_path=folder/"curve_source_score.json"
                if score_path.exists():
                    score=json.loads(score_path.read_text())
                    if score["checkpoint_sha256"]!=report["checkpoint_sha256"]:
                        raise ValueError("Curve score identity changed")
                else:
                    _,score=source_error_matrix(folder/"final.pt",dev)
                    create_once(score_path,score)
                records.append({"label":label,"updates":updates,"seed":seed,"score":score,"fit":report})
    create_once(root/"budget_curves_complete.json",{"records":records,"status":"requires_convergence_review_not_a_freeze"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["summarize","curves"])
    parser.add_argument("--config",default=str(CONFIG))
    args=parser.parse_args()
    run(args.action,args.config)
