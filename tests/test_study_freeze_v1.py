"""One-way final prediction/label gates on temporary model-free fixtures."""
import json

import numpy as np
import pytest

from src.data.benchmark_contract_v1 import sha256
from src.ml.study_freeze_v1 import (freeze_methods,freeze_evaluation,begin_predictions,
    seal_predictions,open_truth_after_predictions)


def fixture(tmp):
    settings=tmp/"settings.json"
    settings.write_text('{"source_only":true}')
    design={k:{} for k in ("primary_endpoint","secondary_contrasts","alpha_allocation","margins","source_selection_rule",
        "training_budget","noise_and_seed_policy","numerical_qualification","failure_and_recovery_policy","power_planning_rule")}
    design.update(training_roles=["source_train","source_development","target_adaptation"],arms=["G","R"],target_information_access={"G":"none","R":"none"})
    methods=tmp/"methods.json"
    freeze_methods(methods,study_id="unit-only",design=design,artifacts=[settings])
    checkpoints={}
    for arm in design["arms"]:
        p=tmp/f"{arm}.bin"
        p.write_bytes(arm.encode())
        checkpoints[arm]={str(s):str(p) for s in range(10)}
    pilot=tmp/"pilot.json"
    pilot.write_text('{"variance_only":true}')
    final_design={"parent_count":2,"seed_ids":list(range(10)),"condition_order":["moffat-3.5"],
        "renderer_order":["lenstronomy","autolens"],"generation_seed":12345,"rejection_policy":{},
        "exclusion_ledger":{},"bootstrap":{},"contrasts":{},"release_directory":str(tmp/"fresh-final"),
        "prediction_directory":str(tmp/"new-predictions"),"truth_path":str(tmp/"fresh-final/truth_vault/targets.json"),
        "ensemble_rule":"none","analysis_recovery_policy":"no_implicit_restart"}
    evaluation=tmp/"evaluation.json"
    freeze_evaluation(evaluation,method_freeze=methods,design=final_design,checkpoints=checkpoints,pilot_design=pilot)
    release=tmp/"fresh-final"
    (release/"truth_vault").mkdir(parents=True)
    truth=release/"truth_vault/targets.json"
    truth.write_text(json.dumps({"p1":.9,"p2":1.2}))
    manifest=release/"manifest.json"
    manifest.write_text(json.dumps({"evaluation_freeze_sha256":sha256(evaluation),"role":"final",
        "parent_ids":["p1","p2"],"truth_sha256":sha256(truth)}))
    axes={k:final_design[k] for k in ("parent_count","seed_ids","condition_order","renderer_order")}
    axes["arms"]=["G","R"]
    return evaluation,manifest,axes


def predictions(output,axes,*,extra_truth=False):
    paths=[]
    for arm in axes["arms"]:
        p=output/f"{arm}.npz"
        content={"prediction_arcsec":np.ones((10,2,1,2),np.float32),"parent_ids":np.array(["p1","p2"]),
            "seed_ids":np.array(axes["seed_ids"]),"conditions":np.array(axes["condition_order"]),
            "renderers":np.array(axes["renderer_order"]),"arm":np.array(arm)}
        if extra_truth:
            content["theta_E_arcsec_true"]=np.array([.9,1.2])
        np.savez(p,**content)
        paths.append(p)
    return paths


def test_final_truth_requires_complete_sealed_predictions_and_single_analysis(tmp_path):
    evaluation,manifest,axes=fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        open_truth_after_predictions(evaluation)
    output=begin_predictions(evaluation,manifest)
    with pytest.raises(FileExistsError):
        begin_predictions(evaluation,manifest)
    files=predictions(output,axes)
    seal_predictions(evaluation,prediction_files=files,validated_axes=axes)
    assert open_truth_after_predictions(evaluation)=={"p1":.9,"p2":1.2}
    with pytest.raises(FileExistsError):
        open_truth_after_predictions(evaluation)


def test_changed_checkpoint_blocks_final_start(tmp_path):
    evaluation,manifest,_=fixture(tmp_path)
    (tmp_path/"G.bin").write_bytes(b"different-weights")
    with pytest.raises(ValueError,match="artifact changed"):
        begin_predictions(evaluation,manifest)


def test_prediction_truth_field_injection_is_rejected(tmp_path):
    evaluation,manifest,axes=fixture(tmp_path)
    output=begin_predictions(evaluation,manifest)
    files=predictions(output,axes,extra_truth=True)
    with pytest.raises(ValueError,match="truth must remain separate"):
        seal_predictions(evaluation,prediction_files=files,validated_axes=axes)


def test_modified_predictions_block_truth_opening(tmp_path):
    evaluation,manifest,axes=fixture(tmp_path)
    output=begin_predictions(evaluation,manifest)
    files=predictions(output,axes)
    seal_predictions(evaluation,prediction_files=files,validated_axes=axes)
    files[0].write_bytes(files[0].read_bytes()+b"changed")
    with pytest.raises(ValueError,match="artifact changed"):
        open_truth_after_predictions(evaluation)
