"""Write truth-free final predictions for every frozen checkpoint, then seal.

All models and native products are verified before the start marker. A start
marker is exclusive: no implicit retry or checkpoint substitution is allowed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.benchmark_contract_v1 import sha256
from src.data.paired_evaluation_images_v1 import PairedEvaluationImages
from src.ml.model import build_model
from src.ml.robustness_training_v1 import grouped_forward
from src.ml.study_freeze_v1 import verify_evaluation_freeze,begin_predictions,seal_predictions


@torch.inference_mode()
def checkpoint_predictions(checkpoint_path,images,*,batch_size=16):
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    model=build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    statistics=checkpoint["statistics"]
    doc=images.document
    values=np.empty((len(images.parent_ids),len(doc["condition_order"]),len(doc["renderer_order"])),dtype=np.float64)
    for r,renderer in enumerate(doc["renderer_order"]):
        for c,condition in enumerate(doc["condition_order"]):
            for start in range(0,len(images.parent_ids),batch_size):
                ids=images.parent_ids[start:start+batch_size]
                _,prediction=grouped_forward(model,[images.image(p,renderer,condition) for p in ids],statistics)
                values[start:start+len(ids),c,r]=prediction.numpy()*statistics["target_std"]+statistics["target_mean"]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions; retain failed inference")
    return values


def run(freeze_path,manifest_path):
    frozen=verify_evaluation_freeze(freeze_path)
    images=PairedEvaluationImages(manifest_path,role="final",freeze_sha256=sha256(freeze_path))
    if len(images.parent_ids)!=frozen["design"]["parent_count"]:
        raise ValueError("Frozen parent count differs")
    images.verify_all_products()
    output=begin_predictions(freeze_path,manifest_path)
    torch.set_num_threads(4)
    seeds=frozen["design"]["seed_ids"]
    paths=[]
    for index,(arm,checkpoints) in enumerate(frozen["checkpoints"].items()):
        values=[]
        for seed in seeds:
            record=checkpoints[str(seed)]
            values.append(checkpoint_predictions(record["path"],images))
            print(json.dumps({"arm":arm,"seed":seed,"predictions_written":False,"truth_opened":False}),flush=True)
        path=output/f"method-{index:02d}.npz"
        np.savez(path,prediction_arcsec=np.stack(values),parent_ids=np.array(images.parent_ids),seed_ids=np.array(seeds),
            conditions=np.array(images.document["condition_order"]),renderers=np.array(images.document["renderer_order"]),arm=np.array(arm))
        paths.append(path)
    axes={k:frozen["design"][k] for k in ("parent_count","seed_ids","condition_order","renderer_order")}
    axes["arms"]=list(frozen["checkpoints"])
    seal_predictions(freeze_path,prediction_files=paths,validated_axes=axes)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_freeze")
    parser.add_argument("release_manifest")
    args=parser.parse_args()
    run(args.evaluation_freeze,args.release_manifest)
