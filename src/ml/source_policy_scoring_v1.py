"""Source-development-only scores for allocation and shared HPO selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.ml.augmentation_policy_v1 import CELLS, failure_weights
from src.ml.model import build_model
from src.ml.robustness_training_v1 import grouped_forward
from src.data.benchmark_contract_v1 import sha256


@torch.inference_mode()
def source_error_matrix(checkpoint_path, bank, *, noise_replicates=2, batch_size=16):
    if bank.role!="source_development":
        raise ValueError("Policy/HPO scores require source development, not target or final")
    if type(noise_replicates) is not int or noise_replicates<1:
        raise ValueError("Expected positive noise replicate count")
    started=time.perf_counter()
    torch.set_num_threads(4)
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    model=build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    statistics=checkpoint["statistics"]
    ids=list(bank.by_id)
    conditions=["canonical",*CELLS]
    errors=np.zeros((len(ids),len(conditions)))
    for condition_index,condition in enumerate(conditions):
        for replicate in range(noise_replicates):
            for start in range(0,len(ids),batch_size):
                chosen=ids[start:start+batch_size]
                images=[]
                for parent in chosen:
                    seed=int(np.random.SeedSequence([bank.by_id[parent]["parameter_seed"],720610,condition_index,replicate]).generate_state(1)[0])
                    images.append(bank.source_image(parent,cell="fresh-noise" if condition=="canonical" else condition,noise_seed=seed))
                _,prediction=grouped_forward(model,images,statistics)
                prediction=prediction.numpy()*statistics["target_std"]+statistics["target_mean"]
                truth=np.asarray([bank.targets[p] for p in chosen])
                errors[start:start+len(chosen),condition_index]+=abs(prediction-truth)/noise_replicates
    return errors,{"role":"source_development","parent_ids":ids,"conditions":conditions,
        "noise_replicates":noise_replicates,"noise_seed_namespace":720610,
        "checkpoint_sha256":sha256(checkpoint_path),"bank_sha256":sha256(bank.path),
        "selection_criterion":"mean_absolute_error_canonical_plus_seven_source_cells",
        "canonical_mae_arcsec":float(errors[:,0].mean()),"shared_source_score_arcsec":float(errors.mean()),
        "runtime_seconds":time.perf_counter()-started,"inference_torch_threads":4}


def score_allocation(checkpoints, bank, *, output):
    started=time.perf_counter()
    output=Path(output)
    if output.exists():
        raise FileExistsError("Policy scores already exist; use a new version")
    output.mkdir(parents=True)
    matrices,records=[],[]
    for path in checkpoints:
        errors,record=source_error_matrix(path,bank)
        matrices.append(errors)
        records.append(record)
    if len(matrices)<3:
        raise ValueError("Allocation needs at least three independent reference fits")
    if any(r["parent_ids"]!=records[0]["parent_ids"] or r["conditions"]!=records[0]["conditions"] for r in records):
        raise ValueError("Reference axes differ")
    cube=np.stack(matrices)
    policy=failure_weights(cube[...,0],cube[...,1:],role="source_development",epsilon=.25)
    np.save(output/"absolute_errors.npy",cube)
    policy.update(reference_scores=records,source_development_ids_sha256=hashlib.sha256(json.dumps(records[0]["parent_ids"]).encode()).hexdigest(),
        absolute_errors_sha256=sha256(output/"absolute_errors.npy"),
        stage="source_only_allocation_not_method_freeze",target_images_used=False,final_images_used=False,
        policy_scoring_runtime_seconds=time.perf_counter()-started,
        reference_fit_cost_accounting="Separate three C reference complete.json runtimes plus scoring cost; not included in per-candidate equal source exposures")
    (output/"policy.json").write_text(json.dumps(policy,indent=2,allow_nan=False)+"\n")
    return policy
