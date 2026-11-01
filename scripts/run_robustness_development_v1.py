"""Staged source-only development: references, allocation scores, HPO and curves.

This entry point cannot generate/load final data or automatically freeze a
method after viewing target-family outcomes. Source-only HPO selection uses
the same canonical-plus-seven-cell score for every arm.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.benchmark_contract_v1 import sha256
from src.ml.observation_bank_v1 import ObservationBank
from src.ml.robustness_training_v1 import ARMS,train_arm
from src.ml.source_policy_scoring_v1 import source_error_matrix,score_allocation

CONFIG=Path("configs/ml/robustness_development_v1.yaml")


def trial_grid(arm,base):
    if "+MMD" in arm:
        return [{**base,"learning_rate":lr,"mmd_weight":weight} for lr in (.0003,.001,.003) for weight in (.003,.01,.1,1.4)]
    return [{**base,"learning_rate":lr,"weight_decay":wd} for lr in (.0003,.0006,.001,.003) for wd in (0.,.0001,.001)]


def run(action,config_path=CONFIG):
    config=yaml.safe_load(Path(config_path).read_text())
    if config["stage"]!="development_only_no_final_access":
        raise ValueError("This runner is development only")
    output=Path(config["output"])
    output.mkdir(parents=True,exist_ok=True)
    source=ObservationBank(config["source_train"],expected_role="source_train")
    dev=ObservationBank(config["source_development"],expected_role="source_development")
    if action=="reference":
        for seed in config["reference_seeds"]:
            train_arm(config,seed=seed,arm="C",source=source,development=dev,target=None,
                output=output/"reference"/f"seed-{seed}",probabilities=None)
    elif action=="weights":
        checkpoints=[output/"reference"/f"seed-{s}"/"final.pt" for s in config["reference_seeds"]]
        policy=score_allocation(checkpoints,dev,output=output/"allocation")
        print(json.dumps(policy,indent=2))
    elif action=="tune":
        policy_path=output/"allocation/policy.json"
        policy=json.loads(policy_path.read_text())
        if policy["target_images_used"] is not False or policy["final_images_used"] is not False:
            raise ValueError("Allocation provenance is not source only")
        target=ObservationBank(config["target_gaussian"],expected_role="target_adaptation")
        representative=ObservationBank(config["target_representative"],expected_role="target_adaptation")
        experiments=[(arm,None if "+MMD" not in arm else target,arm) for arm in sorted(ARMS)]
        experiments += [("G+MMD",representative,"G+MMD-representative"),("R+MMD",representative,"R+MMD-representative")]
        records=[]
        for arm,adaptation,label in experiments:
            for trial,settings in enumerate(trial_grid(arm,config)):
                for seed in config["tuning_seeds"]:
                    folder=output/"tuning"/label/f"trial-{trial:02d}"/f"seed-{seed}"
                    result=train_arm(settings,seed=seed,arm=arm,source=source,development=dev,target=adaptation,
                        output=folder,probabilities=policy["probabilities"])
                    _,score=source_error_matrix(folder/"final.pt",dev)
                    record={"arm":arm,"information_access":label,"trial":trial,"seed":seed,
                        "settings":settings,"source_score":score,"run":result,
                        "allocation_sha256":sha256(policy_path)}
                    (folder/"source_selection.json").write_text(json.dumps(record,indent=2)+"\n")
                    records.append(record)
                    print(json.dumps({"arm":label,"trial":trial,"seed":seed,"source_score_arcsec":score["shared_source_score_arcsec"]}),flush=True)
        path=output/"tuning_initial_complete.json"
        with path.open("x") as stream:
            json.dump({"records":records,"initial_configs_per_arm":12,"seeds_per_trial":2,
                "status":"development_only_requires_boundary_and_convergence_review"},stream,indent=2)
    else:
        raise ValueError("Unknown development stage")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["reference","weights","tune"])
    parser.add_argument("--config",default=str(CONFIG))
    args=parser.parse_args()
    run(args.action,Path(args.config))
