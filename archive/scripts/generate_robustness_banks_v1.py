"""Generate qualified development banks; final/power roles are not supported.

Source parents are the existing 700/150 train/development identities with new
versioned image formation. Old final products are opened for identity columns
only. Target adaptation labels are neither serialized nor returned. Any
numerical failure is retained in an audit and blocks source bank completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from scripts.qualify_benchmark_renderer_v1 import digest, write
from src.data.benchmark_contract_v1 import validate_lineage
from src.processing.observation_conditions_v1 import condition_for
from src.data.observation_bank_v1 import ObservationBank
from src.simulation.adaptive_integration_v1 import adaptive_clean
from src.simulation.precise_point_renderer_v1 import PreciseSceneEngine
from src.simulation.priors import draw_observation_parameters

QUALIFICATION = Path("results/reports/renderer-qualification-v3/qualification.json")
SOURCE = Path("data/simulations/sim-eng1000-v1")


def identity_ledger():
    columns = {"parent_system_id", "parameter_seed", "noise_seed", "source_template_id", "background_template_id"}
    ledger, hashes = [], {}
    for path in (SOURCE/"splits/test.csv", Path("data/simulations/reliability-blind-holdout-v1/manifest.csv")):
        # Never parse radius/physical labels or read any final array/sidecar.
        frame = pd.read_csv(path, usecols=lambda name: name in columns)
        hashes[path.as_posix()] = digest(path)
        for row in frame.to_dict("records"):
            assets = [str(row[k]) for k in ("source_template_id", "background_template_id") if k in row and pd.notna(row[k])]
            ledger.append({"parent_system_id": str(row["parent_system_id"]), "parameter_seed": int(row["parameter_seed"]),
                "noise_seed": int(row["noise_seed"]), "ancestry_ids": assets, "role": "final"})
    refs = Path("data/simulations/renderer-qualification-v1-complete/references.json")
    numerical = json.loads(refs.read_text())
    hashes[refs.as_posix()] = digest(refs)
    done = set()
    for row in numerical["cases"]:
        if row["index"] in done:
            continue
        done.add(row["index"])
        p=row["parameters"]
        ledger.append({"parent_system_id":f"numerical-v1-{row['index']}","role":"numerical_development",
            "parameter_seed":p["parameter_seed"],"noise_seed":p["noise_seed"],"ancestry_ids":[]})
    proposal=yaml.safe_load(Path("configs/simulation_engineering_v1.yaml").read_text())
    for index in set(range(106))-done:
        p=draw_observation_parameters(720101,index,proposal)
        ledger.append({"parent_system_id":f"numerical-v1-{index}","role":"numerical_development",
            "parameter_seed":p["parameter_seed"],"noise_seed":p["noise_seed"],"ancestry_ids":[]})
    return ledger, hashes


def physical_fingerprint(p):
    keys = ("theta_E_arcsec","q","phi_rad","lens_center_x_arcsec","lens_center_y_arcsec",
            "source_x_arcsec","source_y_arcsec","source_q","source_phi_rad","source_n","source_re_arcsec","source_amp",
            "lens_light_q","lens_light_phi_rad","lens_light_n","lens_light_re_arcsec","lens_light_amp","point_source_amp")
    return hashlib.sha256(json.dumps({k:p[k] for k in keys},sort_keys=True,allow_nan=False).encode()).hexdigest()


def save_array(output, relative, values):
    path = output/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,np.asarray(values,dtype=np.float32))
    return {"path":relative,"sha256":digest(path)}


def source_proposals(role):
    split = "train" if role=="source_train" else "validation"
    allowed = {"sample_id","parent_system_id","parameters_path","source_template_id","background_template_id"}
    path=SOURCE/"splits"/f"{split}.csv"
    frame=pd.read_csv(path,usecols=lambda k:k in allowed)
    tpath=Path("data/simulations/reliability-domain-randomized-v1/splits/train.csv")
    tcols={"sample_id","original_parent_sample_id","pixel_scale","exposure_time","read_noise_electrons","noise_seed","psf_fwhm_arcsec"}
    varied=pd.read_csv(tpath,usecols=lambda k:k in tcols)
    varied=varied[varied.sample_id!=varied.original_parent_sample_id].set_index("original_parent_sample_id")
    rows=[]
    for row in frame.to_dict("records"):
        p=json.loads((SOURCE/row["parameters_path"]).read_text())
        historical={"pixel_scale_arcsec":.05,"exposure_multiplier":1.,"read_noise_multiplier":1.}
        historical_seed=p["noise_seed"]
        if split=="train":
            v=varied.loc[row["sample_id"]]
            if not np.isclose(v.psf_fwhm_arcsec,p["psf_fwhm_arcsec"]):
                raise ValueError("Historical arm has unsupported PSF change")
            historical={"pixel_scale_arcsec":float(v.pixel_scale),"exposure_multiplier":float(v.exposure_time/p["exposure_time"]),
                        "read_noise_multiplier":float(v.read_noise_electrons/p["read_noise_electrons"])}
            historical_seed=int(v.noise_seed)
        rows.append((row,p,historical,historical_seed))
    return rows,{path.as_posix():digest(path),tpath.as_posix():digest(tpath)}


def run(role):
    q=json.loads(QUALIFICATION.read_text())
    if q["passed"] is not True:
        raise ValueError("Numerical qualification failed; no scientific banks generated")
    for path,expected in q["source_sha256"].items():
        if digest(path)!=expected:
            raise ValueError("Qualified rendering source changed")
    name={"source_train":"source-train","source_development":"source-development","target_adaptation":"target-adaptation"}[role]
    output=Path(f"data/simulations/robustness-study-{name}-v1")
    report_path=Path(f"results/reports/robustness-study-banks-v1/{name}.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Use a new bank identity")
    ledger,source_hashes=identity_ledger()
    target=role=="target_adaptation"
    if target:
        proposal=yaml.safe_load(Path("configs/simulation_engineering_v1.yaml").read_text())
        count=500
        proposals=[({"parent_system_id":f"target-adapt-v1-{i:04d}"},
                    draw_observation_parameters(720502,i,proposal),None,None) for i in range(570)]
    else:
        proposals,additional=source_proposals(role)
        source_hashes.update(additional)
        count=len(proposals)
    output.mkdir(parents=True)
    write(output/"started.json",{"role":role,"qualification_sha256":digest(QUALIFICATION),"requested_parents":count,
        "source_identity_sha256":source_hashes,"generator_sha256":digest(__file__)})
    parents,labels,failures,diagnostics=[],{},[],[]
    started=time.perf_counter()
    for index,(row,p,historical,historical_seed) in enumerate(proposals):
        if len(parents)==count:
            break
        parent=row["parent_system_id"]
        try:
            identity={"parent_system_id":parent,"role":role,"parameter_seed":int(p["parameter_seed"]),
                      "noise_seed":int(p["noise_seed"]),"ancestry_ids":[],"latent_sha256":physical_fingerprint(p)}
            validate_lineage([identity],ledger)
            engine=PreciseSceneEngine(p,"autolens" if target else "lenstronomy")
            engine.points()
            canonical=condition_for("fresh-noise")
            if target:
                clean,diag=adaptive_clean(engine,canonical)
                conditions=[canonical,*[{**canonical,"name":f"moffat-{b}","psf_family":"moffat","moffat_beta":b} for b in (2.5,3.5,4.5)]]
                images=[]
                for view,condition in enumerate(conditions):
                    rendered=engine.expected(condition,factor=4,clean=clean)
                    noise=int(np.random.SeedSequence([p["noise_seed"],720503,view]).generate_state(1)[0])
                    observed=engine.observe(condition,factor=4,noise_seed=noise,expected=rendered)
                    images.append(save_array(output,f"products/{parent}-{view}.npy",observed["image"]))
                identity["images"]=images
                diagnostics.append({"parent":parent,"geometry":"canonical",**diag})
            else:
                expected={}
                for cell in ("fresh-noise","pixel-scale-004","pixel-scale-006"):
                    condition=condition_for(cell)
                    clean,diag=adaptive_clean(engine,condition)
                    diagnostics.append({"parent":parent,"geometry":cell,**diag})
                    rendered=engine.expected(condition,factor=4,clean=clean)
                    key="base" if cell=="fresh-noise" else cell
                    expected[key]=save_array(output,f"products/{parent}-{key}.npy",rendered["expected"])
                    if cell=="fresh-noise":
                        for width in ("gaussian-fwhm-075","gaussian-fwhm-125"):
                            shifted=engine.expected(condition_for(width),factor=4,clean=clean)
                            expected[width]=save_array(output,f"products/{parent}-{width}.npy",shifted["expected"])
                identity.update(expected=expected,detector={"exposure_s":p["exposure_time"],"read_noise_electrons":p["read_noise_electrons"],
                    "background_rate":p["background_rate"]},historical_condition=historical,historical_noise_seed=historical_seed)
                labels[parent]=p["theta_E_arcsec"]
            parents.append(identity)
        except Exception as exc:
            approved_single=False
            if target and "found 1 roots" in str(exc):
                from src.simulation.independent_sie import solve_independent_sie_images
                independent=solve_independent_sie_images(source_x_arcsec=p["source_x_arcsec"],source_y_arcsec=p["source_y_arcsec"],
                    theta_E_arcsec=p["theta_E_arcsec"],axis_ratio=p["q"],phi_rad=p["phi_rad"],
                    center_x_arcsec=p["lens_center_x_arcsec"],center_y_arcsec=p["lens_center_y_arcsec"],angular_starts=96)
                approved_single=len(independent)==1
            failures.append({"proposal":index,"parent":parent,"error":type(exc).__name__,"message":str(exc),
                             "approved_physical_single_image_rejection":approved_single})
        if (index+1)%10==0:
            print(json.dumps({"role":role,"attempted":index+1,"accepted":len(parents),"rejections":len(failures),
                              "seconds":time.perf_counter()-started}),flush=True)
    # Source is a fixed existing cohort: no replacements or silent deletions.
    if len(parents)!=count or any(not r["approved_physical_single_image_rejection"] for r in failures):
        write(output/"failed.json",{"requested":count,"accepted":len(parents),"failures":failures})
        raise ValueError("Bank did not meet the declared cohort; preserve failures")
    label_record=None
    if not target:
        write(output/"source_targets.json",labels)
        label_record={"path":"source_targets.json","sha256":digest(output/"source_targets.json")}
    document={"schema":"observation_bank_v1","stage":"qualified_development","role":role,
        "rendering_factor":q["rendering_factor"],"qualification":{"path":QUALIFICATION.as_posix(),"sha256":digest(QUALIFICATION)},
        "exclusion_ledger":ledger,"targets":label_record,"parents":parents,
        "image_semantics":"unnormalized raw rate with background retained",
        "condition_access":"gaussian_only"}
    if target:
        aware={**document,"parents":parents,"condition_access":"representative_unlabeled_gaussian_and_moffat"}
        write(output/"bank-representative.json",aware)
        document={**document,"parents":[{**p,"images":p["images"][:1]} for p in parents]}
    path=output/"bank.json"
    write(path,document)
    bank=ObservationBank(path,expected_role=role)
    report={"role":role,"parents":len(bank.parents),"requested_parents":count,"attempted_proposals":len(parents)+len(failures),
        "failures":failures,"manifest_sha256":digest(path),"qualification_sha256":digest(QUALIFICATION),
        "generator_sha256":digest(__file__),"source_identity_sha256":source_hashes,
        "runtime_seconds":time.perf_counter()-started,"bank_validated":True,"final_arrays_accessed":False,
        "final_identity_columns_only":True,"target_labels_serialized":False,
        "bytes_materialized":sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
    write(output/"integration_diagnostics.json",diagnostics)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    write(report_path,report)
    write(output/"complete.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role",choices=["source_train","source_development","target_adaptation"])
    run(parser.parse_args().role)
