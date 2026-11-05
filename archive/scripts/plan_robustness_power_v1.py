"""Seal pilot predictions, export variance-only summaries and plan final N/S.

No method ranking or observed treatment mean is emitted. Fixed method settings
must already be sealed. This can be expensive: power simulations use the actual
10,000-draw crossed bootstrap, not an asserted normal-theory guarantee.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.predict_paired_evaluation_v1 import checkpoint_predictions
from src.data.benchmark_contract_v1 import sha256
from src.data.paired_evaluation_images_v1 import PairedEvaluationImages
from src.ml.confirmatory_statistics_v1 import paired_variance_components
from src.ml.frozen_analysis_v1 import contrast_specification
from src.ml.power_planning_v1 import normal_power,simulate_crossed_power,gaussian_upper_components
from src.ml.study_freeze_v1 import verify_method_freeze,create_once,verify_files
from src.processing.evaluation_conditions_v1 import CONDITIONS,PRIMARY_CONDITIONS


def pilot_variances(method_freeze,manifest_path,fits_path,output):
    method=verify_method_freeze(method_freeze)
    fits=json.loads(Path(fits_path).read_text())
    if fits["method_freeze_sha256"]!=sha256(method_freeze) or fits["stage"]!="pilot":
        raise ValueError("Pilot fit inventory differs from frozen methods")
    seeds=method["design"]["noise_and_seed_policy"]["pilot_training_seeds"]
    if fits["seed_ids"]!=seeds or set(fits["checkpoints"])!=set(method["design"]["arms"]):
        raise ValueError("Pilot seed/method axes differ")
    images=PairedEvaluationImages(manifest_path,role="power_pilot",freeze_sha256=sha256(method_freeze))
    if len(images.parent_ids)!=method["design"]["power_planning_rule"]["pilot_parent_count"]:
        raise ValueError("Pilot parent count differs from frozen planning rule")
    images.verify_all_products()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    checkpoints={p:sha256(p) for by_seed in fits["checkpoints"].values() for p in by_seed.values()}
    create_once(output/"pilot_predictions_started.json",{"method_freeze_sha256":sha256(method_freeze),
        "manifest":{"path":str(Path(manifest_path)),"sha256":sha256(manifest_path)},"checkpoint_sha256":checkpoints})
    torch.set_num_threads(4)
    predictions={}
    files={}
    for index,(arm,by_seed) in enumerate(fits["checkpoints"].items()):
        predictions[arm]=np.stack([checkpoint_predictions(by_seed[str(seed)],images) for seed in seeds])
        path=output/f"pilot-method-{index:02d}.npz"
        np.savez(path,prediction_arcsec=predictions[arm],arm=np.array(arm),parent_ids=np.array(images.parent_ids),
            seed_ids=np.array(seeds),conditions=np.array(images.document["condition_order"]),renderers=np.array(images.document["renderer_order"]))
        files[str(path)]=sha256(path)
    verify_files(checkpoints)
    create_once(output/"pilot_predictions_complete.json",{"prediction_sha256":files,"target_labels_opened":False})
    truth_path=images.root/"truth_vault/targets.json"
    if sha256(truth_path)!=images.document["truth_sha256"]:
        raise ValueError("Pilot truth changed")
    create_once(output/"pilot_variance_analysis_started.json",{"predictions_complete_sha256":sha256(output/"pilot_predictions_complete.json")})
    truth_mapping=json.loads(truth_path.read_text())
    if set(truth_mapping)!=set(images.parent_ids):
        raise ValueError("Pilot truth identity join differs")
    truth=np.asarray([truth_mapping[p] for p in images.parent_ids])
    indices=[CONDITIONS.index(c) for c in PRIMARY_CONDITIONS]
    errors={arm:abs(v[:,:,indices,1]-truth[None,:,None]) for arm,v in predictions.items()}
    components={}
    for spec in contrast_specification():
        key=spec["candidate"]+"_minus_"+spec["comparator"]
        components[key]=paired_variance_components(errors[spec["candidate"]],errors[spec["comparator"]])
    create_once(output/"variance_components.json",{"method_freeze_sha256":sha256(method_freeze),
        "parent_count":len(truth),"seed_ids":seeds,"paired_variance_components":components,
        "observed_treatment_means_exported":False,"used_for_method_selection":False})


def choose_design(method_freeze,variance_path,output):
    method=verify_method_freeze(method_freeze)
    record=json.loads(Path(variance_path).read_text())
    if record["method_freeze_sha256"]!=sha256(method_freeze):
        raise ValueError("Variance pilot belongs to different methods")
    rule=method["design"]["power_planning_rule"]
    components=record["paired_variance_components"]["R_minus_G"]
    grid=[]
    for seeds in (10,15,20):
        for parents in (600,1000,1500,2000):
            power=normal_power(components,seeds=seeds,parents=parents)
            cost=seeds*rule["training_seconds_per_seed_all_methods"]+parents*rule["rendering_seconds_per_paired_parent"]
            grid.append({"seed_count":seeds,"parent_count":parents,"normal_power":power,
                "estimated_compute_seconds":cost})
    # Screen by analytic approximation only; retain and simulate the maximum
    # design even if no design has promising approximation power.
    candidates=sorted([r for r in grid if r["normal_power"]>=.85],
        key=lambda r:(r["estimated_compute_seconds"],r["seed_count"],r["parent_count"]))
    if not candidates:
        candidates=[next(r for r in grid if (r["seed_count"],r["parent_count"])==(20,2000))]
    simulations=[]
    chosen=None
    for index,candidate in enumerate(candidates):
        result=simulate_crossed_power(components,seeds=candidate["seed_count"],parents=candidate["parent_count"],
            trials=200,resamples=10000,random_seed=720831+index)
        simulations.append(result)
        print(json.dumps(result),flush=True)
        if result["estimated_power"]>=.9:
            chosen=candidate
            break
    if chosen is None:
        chosen=next(r for r in grid if (r["seed_count"],r["parent_count"])==(20,2000))
        if not any(r["seeds"]==20 and r["parents"]==2000 for r in simulations):
            simulations.append(simulate_crossed_power(components,seeds=20,parents=2000,trials=200,resamples=10000,random_seed=720850))
    actual=next(r for r in simulations if r["seeds"]==chosen["seed_count"] and r["parents"]==chosen["parent_count"])
    inflation=simulate_crossed_power(components,seeds=chosen["seed_count"],parents=chosen["parent_count"],
        trials=200,resamples=10000,inflation=2.,random_seed=720851)
    secondary={key:normal_power(value,seeds=chosen["seed_count"],parents=chosen["parent_count"],alpha=.005)
        for key,value in record["paired_variance_components"].items() if key!="R_minus_G"}
    upper=gaussian_upper_components(components)
    create_once(output,{"method_freeze_sha256":sha256(method_freeze),"pilot_variances_sha256":sha256(variance_path),
        "pilot_variances":{"path":str(Path(variance_path)),"sha256":sha256(variance_path)},
        "selected_seed_count":chosen["seed_count"],"selected_parent_count":chosen["parent_count"],
        "planning_power_target_met":actual["estimated_power"]>=.9,"grid":grid,"simulations":simulations,
        "variance_doubling_sensitivity":inflation,"secondary_normal_power_diagnostics":secondary,
        "gaussian_upper_component_sensitivity":{"components":upper,"normal_approximation_power":
            normal_power(upper,seeds=chosen["seed_count"],parents=chosen["parent_count"])},
        "selection_rule":"cheapest analytically promising design with simulated point-estimate power >= .9; otherwise capped 20/2000",
        "limitations":"Five pilot seeds; clipped moment variances; Gaussian effects; Monte Carlo uncertainty; guardrails not jointly powered. No guaranteed final power."})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    pilot=sub.add_parser("pilot")
    pilot.add_argument("method_freeze");pilot.add_argument("manifest");pilot.add_argument("fits");pilot.add_argument("output")
    plan=sub.add_parser("plan")
    plan.add_argument("method_freeze");plan.add_argument("variances");plan.add_argument("output")
    args=parser.parse_args()
    if args.action=="pilot":
        pilot_variances(args.method_freeze,args.manifest,args.fits,args.output)
    else:
        choose_design(args.method_freeze,args.variances,args.output)
