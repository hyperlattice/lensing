"""Source-data fits at frozen method settings for pilot or confirmation.

The power pilot has separate seeds. Confirmation uses a variance-planned seed
prefix; no fit reads pilot/final images or chooses a checkpoint from them.
"""
import argparse
import json
from pathlib import Path

from scripts.review_robustness_development_v1 import checked_fit
from src.data.benchmark_contract_v1 import sha256
from src.data.observation_bank_v1 import ObservationBank
from src.ml.study_freeze_v1 import verify_method_freeze,create_once


def run(stage,freeze_path,settings_path,output,power_design=None):
    frozen=verify_method_freeze(freeze_path)
    if str(Path(settings_path)) not in frozen["artifact_sha256"]:
        raise ValueError("Training settings must be bound by method freeze")
    settings=json.loads(Path(settings_path).read_text())
    if set(settings["methods"])!=set(frozen["design"]["arms"]):
        raise ValueError("Frozen method identities differ")
    seed_policy=frozen["design"]["noise_and_seed_policy"]
    if stage=="pilot":
        seeds=seed_policy["pilot_training_seeds"]
    elif stage=="confirmation":
        planning=json.loads(Path(power_design).read_text())
        if planning["method_freeze_sha256"]!=sha256(freeze_path):
            raise ValueError("Power design belongs to different methods")
        seeds=seed_policy["confirmatory_seed_candidates"][:planning["selected_seed_count"]]
    else:
        raise ValueError("Unknown fixed-fit stage")
    if set(seed_policy["pilot_training_seeds"])&set(seed_policy["confirmatory_seed_candidates"]):
        raise ValueError("Pilot and confirmation seeds overlap")
    source=ObservationBank(settings["source_train"],expected_role="source_train")
    dev=ObservationBank(settings["source_development"],expected_role="source_development")
    targets={"gaussian_only":ObservationBank(settings["target_gaussian"],expected_role="target_adaptation"),
        "representative_unlabeled_gaussian_and_moffat":ObservationBank(settings["target_representative"],expected_role="target_adaptation")}
    policy=json.loads(Path(settings["policy_path"]).read_text())
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    reports=[]
    for label,method in settings["methods"].items():
        config={**method["config"],"fit_purpose":"power_pilot_fit" if stage=="pilot" else "confirmatory_fit"}
        if config["optimizer_updates"]!=frozen["design"]["training_budget"]["optimizer_updates"]:
            raise ValueError("Fixed-update budget differs from method freeze")
        access=frozen["design"]["target_information_access"][label]
        target=None if access=="none" else targets[access]
        checkpoints[label]={}
        for seed in seeds:
            folder=output/label/f"seed-{seed}"
            report=checked_fit(folder,config,seed=seed,arm=method["arm"],source=source,
                development=dev,target=target,probabilities=policy["probabilities"])
            checkpoints[label][str(seed)]=str(folder/"final.pt")
            reports.append(report)
    create_once(output/"fixed_fits_complete.json",{"stage":stage,"method_freeze_sha256":sha256(freeze_path),
        "seed_ids":seeds,"checkpoints":checkpoints,"reports":reports,"target_labels_used":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["pilot","confirmation"])
    parser.add_argument("method_freeze")
    parser.add_argument("settings")
    parser.add_argument("output")
    parser.add_argument("--power-design")
    args=parser.parse_args()
    run(args.stage,args.method_freeze,args.settings,args.output,args.power_design)
