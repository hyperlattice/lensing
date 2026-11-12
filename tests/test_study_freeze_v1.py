"""Sealed predictions and repeatable analysis on model-free fixtures."""
import json

import numpy as np
import pytest

from src.ml.study_freeze_v1 import (freeze_methods,freeze_evaluation,begin_predictions,
    seal_predictions,open_truth_after_predictions)


def fixture(tmp, *, arms=("G","R"), conditions=("moffat-3.5",), bootstrap=None, contrasts=None):
    settings=tmp/"settings.json"
    settings.write_text('{"source_only":true}')
    design={k:{} for k in ("primary_endpoint","secondary_contrasts","alpha_allocation","margins","source_selection_rule",
        "training_budget","noise_and_seed_policy","numerical_qualification","failure_and_recovery_policy","power_planning_rule")}
    design.update(training_roles=["source_train","source_development","target_adaptation"],
                  arms=list(arms),target_information_access={arm:"none" for arm in arms})
    methods=tmp/"methods.json"
    freeze_methods(methods,study_id="unit-only",design=design,artifacts=[settings])
    checkpoints={}
    for arm in design["arms"]:
        p=tmp/f"{arm}.bin"
        p.write_bytes(arm.encode())
        checkpoints[arm]={str(s):str(p) for s in range(10)}
    pilot=tmp/"pilot.json"
    pilot.write_text('{"variance_only":true}')
    final_design={"parent_count":2,"seed_ids":list(range(10)),"condition_order":list(conditions),
        "renderer_order":["lenstronomy","autolens"],"generation_seed":12345,"rejection_policy":{},
        "exclusion_ledger":{},"bootstrap":bootstrap or {},"contrasts":contrasts or {},"release_directory":str(tmp/"fresh-final"),
        "prediction_directory":str(tmp/"new-predictions"),"truth_path":str(tmp/"fresh-final/truth_vault/targets.json"),
        "ensemble_rule":"none_average_errors","analysis_recovery_policy":"no_implicit_restart"}
    evaluation=tmp/"evaluation.json"
    freeze_evaluation(evaluation,method_freeze=methods,design=final_design,checkpoints=checkpoints,pilot_design=pilot)
    release=tmp/"fresh-final"
    (release/"truth_vault").mkdir(parents=True)
    truth=release/"truth_vault/targets.json"
    truth.write_text(json.dumps({"p1":.9,"p2":1.2}))
    manifest=release/"manifest.json"
    manifest.write_text(json.dumps({"role":"final",
        "parent_ids":["p1","p2"]}))
    axes={k:final_design[k] for k in ("parent_count","seed_ids","condition_order","renderer_order")}
    axes["arms"]=list(arms)
    return evaluation,manifest,axes


def predictions(output,axes,*,extra_truth=False):
    paths=[]
    for arm in axes["arms"]:
        p=output/f"{arm}.npz"
        content={"prediction_arcsec":np.ones((10,2,len(axes["condition_order"]),2),np.float32),"parent_ids":np.array(["p1","p2"]),
            "seed_ids":np.array(axes["seed_ids"]),"conditions":np.array(axes["condition_order"]),
            "renderers":np.array(axes["renderer_order"]),"arm":np.array(arm)}
        if extra_truth:
            content["theta_E_arcsec_true"]=np.array([.9,1.2])
        np.savez(p,**content)
        paths.append(p)
    return paths


def test_final_truth_requires_sealed_predictions_and_allows_repeat_analysis(tmp_path):
    evaluation,manifest,axes=fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        open_truth_after_predictions(evaluation)
    output=begin_predictions(evaluation,manifest)
    with pytest.raises(FileExistsError):
        begin_predictions(evaluation,manifest)
    files=predictions(output,axes)
    seal_predictions(evaluation,prediction_files=files,validated_axes=axes)
    assert open_truth_after_predictions(evaluation)=={"p1":.9,"p2":1.2}
    first_access=(output/"analysis_started.json").read_bytes()
    # Analysis of saved predictions needs neither renderer source nor weights.
    (tmp_path/"settings.json").unlink()
    (tmp_path/"G.bin").unlink()
    assert open_truth_after_predictions(evaluation)=={"p1":.9,"p2":1.2}
    assert (output/"analysis_started.json").read_bytes()==first_access




def test_prediction_truth_field_injection_is_rejected(tmp_path):
    evaluation,manifest,axes=fixture(tmp_path)
    output=begin_predictions(evaluation,manifest)
    files=predictions(output,axes,extra_truth=True)
    with pytest.raises(ValueError,match="truth must remain separate"):
        seal_predictions(evaluation,prediction_files=files,validated_axes=axes)




def test_scoring_saved_predictions_is_repeatable_without_training_artifacts(tmp_path):
    from scripts.score_paired_evaluation_v1 import run
    from src.ml.frozen_analysis_v1 import BOOTSTRAP,contrast_specification
    from src.processing.evaluation_conditions_v1 import CONDITIONS

    contrasts=contrast_specification()
    arms=sorted({spec[key] for spec in contrasts for key in ("candidate","comparator")})
    evaluation,manifest,axes=fixture(tmp_path,arms=arms,conditions=CONDITIONS,
                                    bootstrap=BOOTSTRAP,contrasts=contrasts)
    output=begin_predictions(evaluation,manifest)
    seal_predictions(evaluation,prediction_files=predictions(output,axes),validated_axes=axes)
    run(evaluation)
    paths=[output/"final_analysis.json",output/"result_overview.md",output/"analysis_started.json"]
    before={path:path.read_bytes() for path in paths}
    (tmp_path/"settings.json").unlink()
    for arm in arms:
        (tmp_path/f"{arm}.bin").unlink()
    run(evaluation)
    assert {path:path.read_bytes() for path in paths}==before
