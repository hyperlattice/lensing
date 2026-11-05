"""Bind actual confirmatory checkpoints and planned N/S before final generation."""
import argparse
import json
from pathlib import Path

import torch

from src.data.benchmark_contract_v1 import sha256
from src.ml.frozen_analysis_v1 import BOOTSTRAP,contrast_specification
from src.ml.study_freeze_v1 import verify_method_freeze,create_once,freeze_evaluation
from src.processing.evaluation_conditions_v1 import CONDITIONS,RENDERERS


def run(method_freeze,settings_path,power_path,fits_path,pilot_release,output):
    method=verify_method_freeze(method_freeze)
    if str(Path(settings_path)) not in method["artifact_sha256"]:
        raise ValueError("Fixed settings are not bound to method freeze")
    settings=json.loads(Path(settings_path).read_text())
    planning=json.loads(Path(power_path).read_text())
    fits=json.loads(Path(fits_path).read_text())
    if (planning["method_freeze_sha256"]!=sha256(method_freeze) or fits["method_freeze_sha256"]!=sha256(method_freeze) or
        fits["stage"]!="confirmation"):
        raise ValueError("Planning/fits do not belong to frozen methods")
    s,n=planning["selected_seed_count"],planning["selected_parent_count"]
    if s not in (10,15,20) or n not in (600,1000,1500,2000):
        raise ValueError("Final N/S outside predeclared planning grid")
    seeds=method["design"]["noise_and_seed_policy"]["confirmatory_seed_candidates"][:s]
    if fits["seed_ids"]!=seeds or set(fits["checkpoints"])!=set(method["design"]["arms"]):
        raise ValueError("Confirmatory fit axes differ")
    pairing={}
    training_evidence={}
    for label,by_seed in fits["checkpoints"].items():
        if set(by_seed)!={str(seed) for seed in seeds}:
            raise ValueError("Missing or extra confirmatory seed")
        for seed in seeds:
            path=Path(by_seed[str(seed)])
            report=json.loads((path.parent/"complete.json").read_text())
            started=json.loads((path.parent/"started.json").read_text())
            checkpoint=torch.load(path,map_location="cpu",weights_only=True)
            expected_config={**settings["methods"][label]["config"],"fit_purpose":"confirmatory_fit"}
            access=method["design"]["target_information_access"][label]
            expected_access=None if access=="none" else access
            if (started["config"]!=expected_config or sha256(path)!=report["checkpoint_sha256"] or
                report["stage"]!="confirmatory_fit" or checkpoint["seed"]!=seed or
                checkpoint["arm"]!=settings["methods"][label]["arm"] or
                checkpoint["target_information_access"]!=expected_access or
                checkpoint["optimizer_updates"]!=method["design"]["training_budget"]["optimizer_updates"]):
                raise ValueError("Actual fit/config/checkpoint differs from frozen method")
            if (sha256(path.parent/"started.json")!=report["started_sha256"] or
                sha256(path.parent/"sampling_controls.json")!=report["sampling_controls_sha256"]):
                raise ValueError("Actual fit provenance or schedule changed")
            for name in ("started.json","complete.json","sampling_controls.json","history.json"):
                evidence_path=path.parent/name
                training_evidence[str(evidence_path)]=sha256(evidence_path)
            controls=(report["initial_weights_sha256"],report["parent_schedule_sha256"],report["source_exposures"])
            if seed in pairing and pairing[seed]!=controls:
                raise ValueError("Actual initialization/parent exposure pairing failed")
            pairing[seed]=controls
    output=Path(output)
    pilot_lineage=Path(pilot_release)/"exclusion_lineage.json"
    pilot_manifest=Path(pilot_release)/"manifest.json"
    if json.loads(pilot_manifest.read_text())["method_freeze_sha256"]!=sha256(method_freeze):
        raise ValueError("Pilot release belongs to different methods")
    generation_path=output/"final_generation_design.json"
    freeze_path=output/"evaluation_freeze.json"
    release="data/simulations/robustness-final-evaluation-v1"
    generation={"role":"final","release_directory":release,"parent_count":n,
        "generation_seed":method["design"]["noise_and_seed_policy"]["final_generation_seed"],
        "method_freeze":str(Path(method_freeze)),"evaluation_freeze":str(freeze_path),
        "exclusion_banks":[settings[k] for k in ("source_train","source_development","target_representative")]+[str(pilot_lineage)],
        "simulation_config":"configs/simulation_engineering_v1.yaml","noise_namespace":720903}
    create_once(generation_path,generation)
    design={"parent_count":n,"seed_ids":seeds,"condition_order":list(CONDITIONS),"renderer_order":list(RENDERERS),
        "generation_seed":generation["generation_seed"],
        "rejection_policy":"Only independently agreed one-image systems excluded; every other failure blocks paired release; first N agreed eligible proposals in fixed pool",
        "exclusion_ledger":{"scientific_role_inputs_sha256":{str(p):sha256(p) for p in generation["exclusion_banks"]},
            "training_evidence_sha256":training_evidence,
            "generation_artifact_sha256":{str(generation_path):sha256(generation_path)},
            "pilot_manifest_sha256":sha256(pilot_manifest)},"bootstrap":BOOTSTRAP,"contrasts":contrast_specification(),
        "release_directory":release,"prediction_directory":str(output/"final-predictions"),
        "truth_path":str(Path(release)/"truth_vault/targets.json"),"ensemble_rule":"none_average_errors",
        "analysis_recovery_policy":"Exclusive start; no silent restart. Hash-verified completed fit recovery only. Scientific correction requires new documented version and affected freeze/pilot."}
    freeze_evaluation(freeze_path,method_freeze=method_freeze,design=design,checkpoints=fits["checkpoints"],pilot_design=power_path)
    print(json.dumps({"evaluation_freeze":str(freeze_path),"sha256":sha256(freeze_path),"parent_count":n,"seeds":len(seeds)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("method_freeze","settings","power_design","fits","pilot_release","output"):
        parser.add_argument(name)
    args=parser.parse_args()
    run(args.method_freeze,args.settings,args.power_design,args.fits,args.pilot_release,args.output)
