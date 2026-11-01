"""Create Freeze 1 only after documented source-only search/convergence review.

This does not infer that a completed search is scientifically adequate. The
review records boundary decisions, selected common budget and their evidence.
No target-family predictive score can enter these choices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.benchmark_contract_v1 import sha256
from src.data.observation_bank_v1 import ObservationBank
from src.ml.frozen_analysis_v1 import BOOTSTRAP,contrast_specification
from src.ml.study_freeze_v1 import create_once,freeze_methods
from src.processing.evaluation_conditions_v1 import CONDITIONS,RENDERERS


def run(config_path,review_path,output):
    config=yaml.safe_load(Path(config_path).read_text())
    review=json.loads(Path(review_path).read_text())
    if (review["status"]!="source_only_boundary_and_convergence_review_complete" or
        review["target_predictive_scores_used"] is not False or review["final_images_used"] is not False or
        review["boundary_review"]["resolved"] is not True or review["convergence_review"]["resolved"] is not True):
        raise ValueError("Search boundaries and convergence need source-only resolution")
    budget=review["optimizer_updates"]
    if type(budget) is not int or budget<880:
        raise ValueError("Invalid reviewed common update budget")
    tuning=json.loads(Path(review["source_tuning_summary"]).read_text())
    if tuning["target_prediction_errors_used"] is not False or tuning["final_images_used"] is not False:
        raise ValueError("Source tuning has unauthorized data access")
    choices=tuning["causal_comparison_settings"]
    required={"C","N","G","T","R","C+MMD","G+MMD","R+MMD","G+MMD-representative","R+MMD-representative"}
    if set(choices)!=required:
        raise ValueError("Required competent comparator matrix is incomplete")
    for choice in choices.values():
        for key in ("model","batch_size","huber_delta","torch_threads"):
            if choice["settings"][key]!=config[key]:
                raise ValueError("Common architecture/objective/batch/thread controls differ")
    costs=[review["training_seconds_per_seed_all_methods"],review["rendering_seconds_per_paired_parent"]]
    if not np.isfinite(costs).all() or min(costs)<=0:
        raise ValueError("Power-planning cost projections require positive measured inputs")
    for a,b in (("R","G"),("R+MMD","G+MMD"),("R+MMD-representative","G+MMD-representative")):
        if choices[a]["settings"]!=choices[b]["settings"]:
            raise ValueError("Allocation contrasts require identical optimizer settings")
    banks=[]
    for key,role in (("source_train","source_train"),("source_development","source_development"),
            ("target_gaussian","target_adaptation"),("target_representative","target_adaptation")):
        banks.append(ObservationBank(config[key],expected_role=role))
    policy_path=Path(config["output"])/"allocation/policy.json"
    policy=json.loads(policy_path.read_text())
    if policy["target_images_used"] is not False or policy["final_images_used"] is not False or len(policy["reference_scores"])<3:
        raise ValueError("Allocation was not derived solely from source development")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    method_path=output/"method_freeze.json"
    settings_path=output/"fixed_training_settings.json"
    access={label:("none" if "+MMD" not in label else
        "representative_unlabeled_gaussian_and_moffat" if label.endswith("-representative") else "gaussian_only") for label in choices}
    settings={k:config[k] for k in ("source_train","source_development","target_gaussian","target_representative")}
    settings.update(policy_path=str(policy_path),methods={label:{"arm":label.removesuffix("-representative"),
        "config":{**choice["settings"],"optimizer_updates":budget}} for label,choice in choices.items()})
    create_once(settings_path,settings)
    pilot_design=output/"pilot_generation_design.json"
    create_once(pilot_design,{"role":"power_pilot","release_directory":"data/simulations/robustness-power-pilot-v1",
        "parent_count":120,"generation_seed":720801,"method_freeze":str(method_path),"evaluation_freeze":None,
        "exclusion_banks":[config[k] for k in ("source_train","source_development","target_representative")],
        "simulation_config":"configs/simulation_engineering_v1.yaml","noise_namespace":720803})
    numerical= banks[0].document["qualification"]
    if any(bank.document["qualification"]!=numerical for bank in banks):
        raise ValueError("Different numerical qualification across training roles")
    artifacts=[str(p) for p in Path("src").rglob("*.py")]
    artifacts += [str(p) for p in Path("scripts").glob("*robustness*v1.py")]
    artifacts += ["scripts/generate_paired_evaluation_v1.py","scripts/predict_paired_evaluation_v1.py",
        "scripts/score_paired_evaluation_v1.py","docs/robustness_development_protocol_v1.md",
        "docs/confirmatory_analysis_protocol_v1.md","configs/simulation_engineering_v1.yaml",
        str(Path(config_path)),str(Path(review_path)),review["source_tuning_summary"],str(settings_path),str(pilot_design),
        str(policy_path),numerical["path"],*[str(bank.path) for bank in banks],
        *review["boundary_review"]["evidence_paths"],*review["convergence_review"]["evidence_paths"]]
    artifacts += ["scripts/generate_robustness_banks_parallel_v3.py","scripts/generate_robustness_banks_v1.py"]
    for bank in banks:
        artifacts += [str(bank.root/name) for name in ("started.json","complete.json","proposal_audit.json","observation_provenance.json")]
    if not review["boundary_review"]["evidence_paths"] or not review["convergence_review"]["evidence_paths"]:
        raise ValueError("Review requires actual search and convergence artifacts")
    design={"training_roles":["source_train","source_development","target_adaptation"],"arms":list(choices),
        "target_information_access":access,"primary_endpoint":{"renderer":"autolens","conditions":["moffat-2.5","moffat-3.5","moffat-4.5"],
            "statistic":"mean single-model absolute error over seeds/parents/conditions","contrast":"R-minus-G"},
        "secondary_contrasts":contrast_specification()[1:],"alpha_allocation":BOOTSTRAP,
        "margins":{"meaningful_superiority_arcsec":-.005,"canonical_ni_arcsec":.005,"tail_p95_ni_arcsec":.010},
        "source_selection_rule":{"review":str(Path(review_path)),"shared_optimizer":"G-selected settings used for paired R",
            "selection_data":"reused source development only","condition_order":list(CONDITIONS),"renderer_order":list(RENDERERS)},
        "training_budget":{"optimizer_updates":budget,"batch_size":config["batch_size"],"checkpoint":"last fixed update"},
        "noise_and_seed_policy":{"pilot_training_seeds":[720811,720812,720813,720814,720815],
            "confirmatory_seed_candidates":list(range(720821,720841)),"pilot_generation_seed":720801,
            "final_generation_seed":720902,"source_schedule":"independent parent/cell/noise streams; seed+100000"},
        "numerical_qualification":numerical,
        "failure_and_recovery_policy":{"algorithmic_failure":"retain/report, no favorable seed replacement",
            "physical_rejection":"independently agreed one-image systems only; report all proposals",
            "numerical_failure":"blocks release; correction requires new documented version and affected freeze/pilot",
            "completed_fit_recovery":"verify completed checkpoint/config/bank hashes; partial fits require explicit new attempt"},
        "power_planning_rule":{"pilot_parent_count":120,"seed_grid":[10,15,20],"parent_grid":[600,1000,1500,2000],
            "target_power":.9,"assumed_true_delta_arcsec":-.010,"margin_arcsec":-.005,"alpha_one_sided":.025,
            "simulation_trials":200,"bootstrap_resamples":10000,"variance_inflation_sensitivity":2.,
            "gaussian_upper_component_confidence_sensitivity":.9,
            "normal_screen_minimum":.85,"monte_carlo_acceptance":"estimated power >= .9; Wilson interval reported",
            "training_seconds_per_seed_all_methods":review["training_seconds_per_seed_all_methods"],
            "rendering_seconds_per_paired_parent":review["rendering_seconds_per_paired_parent"],
            "cost_rule":"minimum estimated training+rendering seconds; tie by smaller S then N",
            "capped_failure":"choose20/2000 and disclose insufficient projected power; do not relax endpoint"}}
    freeze_methods(method_path,study_id="robustness-confirmatory-v1",design=design,artifacts=sorted(set(artifacts)))
    print(json.dumps({"method_freeze":str(method_path),"sha256":sha256(method_path),"status":"locked_before_pilot"}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config");parser.add_argument("source_review");parser.add_argument("output")
    args=parser.parse_args()
    run(args.config,args.source_review,args.output)
