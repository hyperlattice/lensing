"""Deterministic process-parallel rendering of the declared development banks.

Physical settings and bank schema match generate_robustness_banks_v1. Worker
completion order cannot change the cohort: results are consumed in proposal
order. All computed proposals, including unused target-pool members, are
audited; numerical failures anywhere block bank completion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from scripts.generate_robustness_banks_v1 import (SOURCE, identity_ledger,
    physical_fingerprint, save_array, source_proposals)
from scripts.qualify_benchmark_renderer_v1 import digest,write
from src.data.benchmark_contract_v1 import validate_lineage
from src.processing.observation_conditions_v1 import condition_for
from src.data.observation_bank_v1 import ObservationBank
from src.simulation.adaptive_integration_v1 import adaptive_clean
from src.simulation.independent_sie import solve_independent_sie_images
from src.simulation.low_memory_renderer_v2 import LowMemorySceneEngine as PreciseSceneEngine
from src.simulation.priors import draw_observation_parameters

QUALIFICATION=Path("results/reports/renderer-qualification-v5/qualification.json")


def independent_positions(p):
    return solve_independent_sie_images(source_x_arcsec=p["source_x_arcsec"],source_y_arcsec=p["source_y_arcsec"],
        theta_E_arcsec=p["theta_E_arcsec"],axis_ratio=p["q"],phi_rad=p["phi_rad"],
        center_x_arcsec=p["lens_center_x_arcsec"],center_y_arcsec=p["lens_center_y_arcsec"],angular_starts=96)


def write_native_masks(output):
    for cell in ("fresh-noise","pixel-scale-004","pixel-scale-006"):
        shape=tuple(condition_for(cell)["shape"])
        path=Path(output)/f"masks/{shape[0]}x{shape[1]}.npy"
        path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,np.zeros(shape,dtype=bool))


def render_one(task):
    role,index,row,p,historical,historical_seed,output=task
    output=Path(output)
    parent=row["parent_system_id"]
    target=role=="target_adaptation"
    identity={"parent_system_id":parent,"role":role,"parameter_seed":int(p["parameter_seed"]),
        "noise_seed":int(p["noise_seed"]),"ancestry_ids":[],"latent_sha256":physical_fingerprint(p)}
    diagnostics=[]
    observations=[]
    try:
        if any(abs(float(p.get(k,0.)))>0 for k in ("gamma1","gamma2")):
            raise ValueError("This bank is restricted to the declared no-shear SIE prior")
        engine=PreciseSceneEngine(p,"autolens" if target else "lenstronomy")
        positions,_=engine.points()
        if target:
            other=independent_positions(p)
            if len(other)!=len(positions):
                raise ValueError("Independent root multiplicity disagreement")
            distances=np.linalg.norm(positions[:,None]-other[None],axis=2)
            i,j=linear_sum_assignment(distances)
            if distances[i,j].max()>1e-5:
                raise ValueError("Independent root position disagreement")
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
                variance=save_array(output,f"products/{parent}-{view}-variance.npy",observed["variance"])
                psf=save_array(output,f"products/{parent}-{view}-psf.npy",rendered["psf"])
                observations.append({"parent":parent,"condition":condition,"image":images[-1],"variance":variance,"psf":psf,
                    "noise_seed":noise,"exposure_s":p["exposure_time"],"background_rate":p["background_rate"],
                    "read_noise_electrons":p["read_noise_electrons"],"variance_method":"analytic_poisson_plus_read_rate",
                    "mask_semantics":"all_native_pixels_valid; no crop/pad; background retained",
                    "image_unit":"electrons_per_second","variance_unit":"electrons_per_second_squared"})
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
                psf=save_array(output,f"products/{parent}-{key}-psf.npy",rendered["psf"])
                observations.append({"parent":parent,"condition":condition,"expected_rate":expected[key],"psf":psf,
                    "variance_method":"reconstruct from expected rate and explicit detector parameters for each training draw",
                    "mask_semantics":"all_native_pixels_valid; no crop/pad; background retained"})
                if cell=="fresh-noise":
                    for width in ("gaussian-fwhm-075","gaussian-fwhm-125"):
                        shifted=engine.expected(condition_for(width),factor=4,clean=clean)
                        expected[width]=save_array(output,f"products/{parent}-{width}.npy",shifted["expected"])
                        psf=save_array(output,f"products/{parent}-{width}-psf.npy",shifted["psf"])
                        observations.append({"parent":parent,"condition":condition_for(width),"expected_rate":expected[width],"psf":psf,
                            "variance_method":"reconstruct from expected rate and explicit detector parameters for each training draw",
                            "mask_semantics":"all_native_pixels_valid; no crop/pad; background retained"})
            parameter_path=SOURCE/row["parameters_path"]
            identity.update(expected=expected,detector={"exposure_s":p["exposure_time"],"read_noise_electrons":p["read_noise_electrons"],
                "background_rate":p["background_rate"]},historical_condition=historical,historical_noise_seed=historical_seed,
                source_parameter_sha256=digest(parameter_path),source_parameter_path=parameter_path.as_posix())
        for observation in observations:
            h,w=observation["condition"]["shape"]
            relative=f"masks/{h}x{w}.npy"
            mask={"path":relative,"sha256":digest(output/relative)}
            observation.update(invalid_mask=mask,coverage_mask=mask)
        return {"proposal":index,"identity":identity,"label":None if target else p["theta_E_arcsec"],
                "diagnostics":diagnostics,"observations":observations,"failure":None}
    except Exception as exc:
        single_count=("found 1 roots" in str(exc) or "root multiplicity 1" in str(exc))
        approved_single=single_count and len(independent_positions(p))==1
        return {"proposal":index,"identity":identity,"label":None,"diagnostics":diagnostics,"observations":observations,
                "failure":{"proposal":index,"parent":parent,"error":type(exc).__name__,"message":str(exc),
                           "approved_physical_single_image_rejection":bool(approved_single)}}


def run(role,workers):
    if type(workers) is not int or not 1<=workers<=2:
        raise ValueError("Use 1-2 bounded rendering workers; run roles sequentially")
    q=json.loads(QUALIFICATION.read_text())
    if q["passed"] is not True:
        raise ValueError("Numerical qualification failed")
    for path,expected in q["source_sha256"].items():
        if digest(path)!=expected:
            raise ValueError("Qualified rendering source changed")
    name={"source_train":"source-train","source_development":"source-development","target_adaptation":"target-adaptation"}[role]
    output=Path(f"data/simulations/robustness-study-{name}-v4")
    report_path=Path(f"results/reports/robustness-study-banks-v4/{name}.json")
    if report_path.exists() or (output/"complete.json").exists():
        raise FileExistsError("This bank role is already complete")
    ledger,input_hashes=identity_ledger()
    target=role=="target_adaptation"
    if target:
        proposal=yaml.safe_load(Path("configs/simulation_engineering_v1.yaml").read_text())
        count=500
        proposals=[({"parent_system_id":f"target-adapt-v1-{i:04d}"},draw_observation_parameters(720502,i,proposal),None,None) for i in range(570)]
    else:
        proposals,more=source_proposals(role)
        input_hashes.update(more)
        count=len(proposals)
    for row,p,_,_ in proposals:
        validate_lineage([{"parent_system_id":row["parent_system_id"],"role":role,"parameter_seed":int(p["parameter_seed"]),
            "noise_seed":int(p["noise_seed"]),"ancestry_ids":[]}],ledger)
    output.mkdir(parents=True,exist_ok=True)
    if not (output/"started.json").exists():
        write_native_masks(output)
    elif not all((output/f"masks/{h}x{h}.npy").exists() for h in (256,320,213)):
        raise ValueError("Existing v4 checkpoint has incomplete native masks")
    started=time.perf_counter()
    started_record={"role":role,"workers":workers,"requested_parents":count,"computed_proposal_pool":len(proposals),
        "qualification_sha256":digest(QUALIFICATION),"source_identity_sha256":input_hashes,
        "generator_sha256":{__file__:digest(__file__),"scripts/generate_robustness_banks_v1.py":digest("scripts/generate_robustness_banks_v1.py")}}
    if (output/"started.json").exists():
        if json.loads((output/"started.json").read_text()) != started_record:
            raise ValueError("Existing v4 output belongs to different frozen inputs")
    else:
        write(output/"started.json",started_record)
    tasks=[(role,index,*record,output.as_posix()) for index,record in enumerate(proposals)]
    checkpoint=output/"checkpoint.json"
    completed={}
    if checkpoint.exists():
        for result in json.loads(checkpoint.read_text()):
            refs=[]
            def collect_refs(value):
                if isinstance(value,dict):
                    if "path" in value and "sha256" in value:
                        refs.append(value)
                    for child in value.values(): collect_refs(child)
                elif isinstance(value,list):
                    for child in value: collect_refs(child)
            collect_refs(result)
            valid=True
            for ref in refs:
                path=output/ref["path"]
                if not path.exists():
                    valid=False
                    break
                if digest(path)!=ref["sha256"]:
                    raise ValueError(f"Checkpoint product hash mismatch: {ref['path']}")
            if valid:
                completed[int(result["proposal"])]=result
    results=list(completed.values())
    pending=[task for task in tasks if int(task[1]) not in completed]
    since_checkpoint=0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(render_one,pending):
            completed[int(result["proposal"])]=result
            results=[completed[i] for i in sorted(completed)]
            since_checkpoint+=1
            if since_checkpoint>=10:
                tmp=checkpoint.with_suffix(".tmp")
                tmp.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
                tmp.replace(checkpoint)
                since_checkpoint=0
            if len(completed)%10==0 or not pending:
                print(json.dumps({"role":role,"computed":len(results),"pool":len(tasks),
                    "failures":sum(r["failure"] is not None for r in results),"seconds":time.perf_counter()-started}),flush=True)
    results=[completed[i] for i in range(len(tasks))]
    tmp=checkpoint.with_suffix(".tmp")
    tmp.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(checkpoint)
    failures=[r["failure"] for r in results if r["failure"] is not None]
    accepted=[r for r in results if r["failure"] is None]
    # Preserve all proposal diagnostics even if a numerical failure blocks the
    # index. Source pools are filtered only by the declared 2/4-image rule;
    # they are not replenished with new parents or selected by CNN outcomes.
    write(output/"worker_results.json",results)
    requested=count
    if not target:
        count=len(accepted)
    selected=accepted[:count]
    if not count or len(selected)!=count or any(not f["approved_physical_single_image_rejection"] for f in failures):
        write(output/"failed.json",{"requested":count,"accepted":len(accepted),"failures":failures})
        raise ValueError("Bank has unresolved numerical/cohort failures")
    for path,expected in q["source_sha256"].items():
        if digest(path)!=expected:
            raise ValueError("Qualified rendering source changed during generation")
    parents=[r["identity"] for r in selected]
    labels=None
    if not target:
        write(output/"source_targets.json",{r["identity"]["parent_system_id"]:r["label"] for r in selected})
        labels={"path":"source_targets.json","sha256":digest(output/"source_targets.json")}
    document={"schema":"observation_bank_v1","stage":"qualified_development","role":role,
        "rendering_factor":q["rendering_factor"],"qualification":{"path":QUALIFICATION.as_posix(),"sha256":digest(QUALIFICATION)},
        "exclusion_ledger":ledger,"targets":labels,"parents":parents,"image_semantics":"unnormalized raw rate with background retained",
        "condition_access":"gaussian_only"}
    write(output/"observation_provenance.json",[o for r in results for o in r["observations"]])
    document.update(image_unit="electrons_per_second",target_definition="lenstronomy_SIE_circularized_parameter_arcsec",
        observation_provenance={"path":"observation_provenance.json","sha256":digest(output/"observation_provenance.json")})
    if target:
        write(output/"bank-representative.json",{**document,"condition_access":"representative_unlabeled_gaussian_and_moffat"})
        document={**document,"parents":[{**p,"images":p["images"][:1]} for p in parents]}
    write(output/"bank.json",document)
    bank=ObservationBank(output/"bank.json",expected_role=role)
    selected_ids=set(bank.by_id)
    write(output/"proposal_audit.json",[{"proposal":r["proposal"],
        **{k:r["identity"][k] for k in ("parent_system_id","parameter_seed","noise_seed","latent_sha256")},
        "selected":r["identity"]["parent_system_id"] in selected_ids,"failure":r["failure"]} for r in results])
    write(output/"integration_diagnostics.json",[d for r in results for d in r["diagnostics"]])
    report={"role":role,"parents":len(bank.parents),"requested_parents":requested,"computed_proposals":len(results),
        "accepted_proposals":len(accepted),"unused_accepted":len(accepted)-count,"failures":failures,
        "manifest_sha256":digest(output/"bank.json"),"qualification_sha256":digest(QUALIFICATION),
        "runtime_seconds":time.perf_counter()-started,"workers":workers,"bank_validated":True,
        "final_arrays_accessed":False,"final_identity_columns_only":True,"target_labels_serialized":False,
        "source_labels_serialized":not target,"bytes_materialized":sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    write(report_path,report)
    write(output/"complete.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role",choices=["source_train","source_development","target_adaptation"])
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    run(args.role,args.workers)
