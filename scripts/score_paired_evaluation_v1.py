"""Exactly-once frozen analysis after every prediction method has been sealed.

Produces measured claim gates and descriptive results, including failures.
This script does not choose methods, endpoints, conditions or sample sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ml.confirmatory_statistics_v1 import crossed_paired_difference,paired_variance_components
from src.ml.confirmatory_endpoints_v1 import crossed_tail_difference,seed_t_sensitivity
from src.ml.frozen_analysis_v1 import BOOTSTRAP,contrast_specification
from src.ml.study_freeze_v1 import verify_evaluation_freeze,open_truth_after_predictions,create_once
from src.processing.evaluation_conditions_v1 import CONDITIONS,RENDERERS,PRIMARY_CONDITIONS


def score_contrast(a,b,spec):
    alpha=spec["alpha_one_sided"]
    controls={"resamples":BOOTSTRAP["resamples"],"alpha":alpha,"seed":BOOTSTRAP["seed"]}
    indices=[CONDITIONS.index(c) for c in PRIMARY_CONDITIONS]
    primary_a,primary_b=a[:,:,indices,1],b[:,:,indices,1]
    endpoint=crossed_paired_difference(primary_a,primary_b,**controls)
    canonical={renderer:crossed_paired_difference(a[:,:,[0],r],b[:,:,[0],r],**controls)
        for r,renderer in enumerate(RENDERERS)}
    tail=crossed_tail_difference(primary_a,primary_b,**controls)
    passes={"meaningful_superiority":endpoint["upper_one_sided_arcsec"]<spec["meaningful_superiority_margin_arcsec"],
        "canonical_A":canonical["lenstronomy"]["upper_one_sided_arcsec"]<spec["canonical_ni_margin_arcsec"],
        "canonical_B":canonical["autolens"]["upper_one_sided_arcsec"]<spec["canonical_ni_margin_arcsec"],
        "tail":tail["upper_one_sided_arcsec"]<spec["tail_ni_margin_arcsec"]}
    return {"specification":spec,"endpoint":endpoint,"canonical_guardrails":canonical,"tail_guardrail":tail,
        "seed_conditional_t_sensitivity":seed_t_sensitivity(primary_a,primary_b,alpha=alpha),
        "variance_components":paired_variance_components(primary_a,primary_b),
        "passed_components":passes,"superiority_claim_gate_passed":all(passes.values())}


def descriptive(prediction,truth):
    signed=prediction-truth[None,:,None,None]
    absolute=abs(signed)
    result={"condition_by_renderer_mae_arcsec":absolute.mean((0,1)).tolist(),
        "condition_by_renderer_mean_seed_p95_arcsec":np.quantile(absolute,.95,axis=1,method="linear").mean(0).tolist(),
        "condition_by_renderer_bias_arcsec":signed.mean((0,1)).tolist(),
        "condition_by_renderer_rmse_arcsec":np.sqrt(np.square(signed).mean((0,1))).tolist(),
        "condition_by_renderer_mean_relative_absolute_error":(absolute/truth[None,:,None,None]).mean((0,1)).tolist(),
        "per_seed_condition_renderer_mae_arcsec":absolute.mean(1).tolist(),"radius_bins":[]}
    primary=[CONDITIONS.index(c) for c in PRIMARY_CONDITIONS]
    result["per_seed_primary_B_mae_arcsec"]=absolute[:,:,primary,1].mean((1,2)).tolist()
    result["per_seed_primary_B_p95_arcsec"]=np.quantile(absolute[:,:,primary,1].reshape(len(prediction),-1),.95,axis=1,method="linear").tolist()
    for lower,upper in ((.5,.875),(.875,1.25),(1.25,1.625),(1.625,2.000000001)):
        mask=(truth>=lower)&(truth<upper)
        result["radius_bins"].append({"lower_inclusive_arcsec":lower,"upper_exclusive_arcsec":upper,
            "parent_count":int(mask.sum()),
            "condition_by_renderer_mae_arcsec":absolute[:,mask].mean((0,1)).tolist() if mask.any() else None})
    return result


def run(freeze_path):
    frozen=verify_evaluation_freeze(freeze_path)
    if frozen["design"]["bootstrap"]!=BOOTSTRAP or frozen["design"]["contrasts"]!=contrast_specification():
        raise ValueError("Frozen analysis differs from the implemented pre-outcome specification")
    if (frozen["design"]["condition_order"]!=list(CONDITIONS) or
        frozen["design"]["renderer_order"]!=list(RENDERERS) or frozen["design"]["ensemble_rule"]!="none_average_errors"):
        raise ValueError("Frozen axes/estimand differ")
    output=Path(frozen["design"]["prediction_directory"])
    sealed=json.loads((output/"predictions_complete.json").read_text())
    values={}
    parent_ids=None
    # Identity and hashes are rechecked by the truth-opening gate below.
    for path in sealed["prediction_sha256"]:
        with np.load(path,allow_pickle=False) as arrays:
            if parent_ids is None:
                parent_ids=arrays["parent_ids"].tolist()
            values[str(arrays["arm"].item())]=arrays["prediction_arcsec"].copy()
    truth_mapping=open_truth_after_predictions(freeze_path)
    truth=np.asarray([truth_mapping[p] for p in parent_ids],dtype=float)
    if not np.isfinite(truth).all() or np.any(truth<=0):
        raise ValueError("Invalid positive radius truth")
    errors={arm:abs(v-truth[None,:,None,None]) for arm,v in values.items()}
    contrasts=[]
    for spec in contrast_specification():
        result=score_contrast(errors[spec["candidate"]],errors[spec["comparator"]],spec)
        contrasts.append(result)
        # Checkpoint intermediate analysis results without opening truth again.
        create_once(output/f"contrast-{len(contrasts):02d}.json",result)
    report={"stage":"final_confirmatory_analysis","parent_count":len(truth),
        "seed_ids":frozen["design"]["seed_ids"],"condition_order":list(CONDITIONS),"renderer_order":list(RENDERERS),
        "bootstrap":BOOTSTRAP,"contrasts":contrasts,"descriptive":{arm:descriptive(v,truth) for arm,v in values.items()},
        "primary_claim_gate_passed":contrasts[0]["superiority_claim_gate_passed"],
        "scope":"conditional synthetic two/four-image SIE population, shared observation pipeline, scalar radius point regression",
        "unsupported_claims":["real-data accuracy","posterior calibration","universal paper superiority","exact published model reproduction"]}
    create_once(output/"final_analysis.json",report)
    lines=["# Frozen paired evaluation result","",f"Independent parents: {len(truth)}; paired training seeds: {len(frozen['design']['seed_ids'])}.",
        "Single-model errors averaged; no prediction ensemble. All differences are candidate minus comparator in arcseconds.","",
        "| Contrast | MAE difference | One-sided upper bound | Alpha | All claim gates passed |",
        "|---|---:|---:|---:|---|"]
    for item in contrasts:
        spec=item["specification"]
        e=item["endpoint"]
        lines.append(f"| {spec['candidate']} vs {spec['comparator']} | {e['difference_arcsec']:.6f} | {e['upper_one_sided_arcsec']:.6f} | {spec['alpha_one_sided']:.3f} | {item['superiority_claim_gate_passed']} |")
    lines.extend(["","A superiority claim requires upper bound below -0.005 arcsec and all canonical/tail guardrails.",
        "These are synthetic common-benchmark method ports. The result does not establish real-data or universal paper superiority.",""])
    with (output/"result_overview.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"primary_claim_gate_passed":report["primary_claim_gate_passed"],"report":str(output/"final_analysis.json")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_freeze")
    run(parser.parse_args().evaluation_freeze)
