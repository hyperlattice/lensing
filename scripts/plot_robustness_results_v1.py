"""Export manuscript figures from the completed frozen analysis only.

No checkpoint inference or truth sidecar is opened. Plots cannot change any
test, training choice or sample size; they display the recorded final results.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.benchmark_contract_v1 import sha256
from src.ml.study_freeze_v1 import verify_evaluation_freeze
from src.processing.evaluation_conditions_v1 import CONDITIONS,PRIMARY_CONDITIONS,RENDERERS


def run(freeze_path,output):
    frozen=verify_evaluation_freeze(freeze_path)
    analysis_path=Path(frozen["design"]["prediction_directory"])/"final_analysis.json"
    report=json.loads(analysis_path.read_text())
    if report["stage"]!="final_confirmatory_analysis":
        raise ValueError("Completed frozen analysis is required before plotting")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    def save(fig,name):
        for suffix in ("png","pdf"):
            fig.savefig(output/f"{name}.{suffix}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    arms=list(frozen["checkpoints"])
    matrices=[np.array([report["descriptive"][arm]["condition_by_renderer_mae_arcsec"] for arm in arms])[:,:,r] for r in range(2)]
    vmax=max(a.max() for a in matrices)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout="constrained",sharey=True)
    for ax,renderer,matrix in zip(axes,RENDERERS,matrices):
        im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap="magma",aspect="auto")
        ax.set_xticks(range(len(CONDITIONS)),CONDITIONS,rotation=70,ha="right",fontsize=8)
        ax.set_yticks(range(len(arms)),arms,fontsize=9)
        ax.set_title(renderer)
    fig.colorbar(im,ax=axes,label="MAE (arcsec)",shrink=.7)
    fig.suptitle(f"{report['parent_count']} paired parents; {len(report['seed_ids'])} paired seeds; single-model errors")
    save(fig,"method_condition_mae")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    labels=[]
    for index,item in enumerate(report["contrasts"]):
        spec=item["specification"]
        e=item["endpoint"]
        labels.append(f"{spec['candidate']} − {spec['comparator']} (alpha={spec['alpha_one_sided']:.3f})")
        color="#27765b" if item["superiority_claim_gate_passed"] else "#b66e26"
        lo,hi=e["lower_one_sided_arcsec"],e["upper_one_sided_arcsec"]
        ax.plot([lo,hi],[index,index],color=color,lw=2)
        ax.scatter(e["difference_arcsec"],index,color=color,zorder=3)
    ax.set_yticks(range(len(labels)),labels,fontsize=9)
    ax.axvline(-.005,color="#ac3f38",linestyle="--",label="Meaningful-superiority boundary")
    ax.axvline(0,color="grey",lw=.8)
    ax.set_xlabel("Candidate minus comparator B-Moffat MAE (arcsec); paired percentile bounds")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title("Green requires all canonical and tail gates; displayed alpha differs by family")
    save(fig,"paired_effects")
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for arm,color in (("G","#32769b"),("R","#ac3f38")):
        values=report["descriptive"][arm]["per_seed_primary_B_mae_arcsec"]
        axes[0].plot(range(len(values)),values,"o-",label=arm,color=color,ms=4)
    axes[0].set(xlabel="Paired training seed index",ylabel="B-Moffat MAE (arcsec)",title="Seed variation")
    axes[0].legend()
    for index,arm in enumerate(arms):
        tail=np.array(report["descriptive"][arm]["per_seed_primary_B_p95_arcsec"])
        axes[1].scatter(np.full(len(tail),index),tail,color="#32769b",alpha=.5,s=12)
        axes[1].scatter(index,tail.mean(),color="black",marker="_",s=100)
    axes[1].set_xticks(range(len(arms)),arms,rotation=75,ha="right",fontsize=7)
    axes[1].set(ylabel="Per-seed P95 error (arcsec)",title="Error tails; dots are seeds")
    primary=[CONDITIONS.index(c) for c in PRIMARY_CONDITIONS]
    for arm,color in (("G","#32769b"),("R","#ac3f38")):
        bins=report["descriptive"][arm]["radius_bins"]
        x=[(b["lower_inclusive_arcsec"]+b["upper_exclusive_arcsec"])/2 for b in bins]
        y=[np.array(b["condition_by_renderer_mae_arcsec"])[primary,1].mean()
            if b["parent_count"] else np.nan for b in bins]
        axes[2].plot(x,y,"o-",color=color,label=arm)
    axes[2].set(xlabel="True-radius bin center (arcsec)",ylabel="B-Moffat MAE (arcsec)",title="Fixed radius-bin diagnostics")
    axes[2].legend()
    save(fig,"seeds_tails_radius")
    (output/"figure_provenance.json").write_text(json.dumps({"analysis":{"path":str(analysis_path),"sha256":sha256(analysis_path)},
        "evaluation_freeze_sha256":sha256(freeze_path),"script_sha256":sha256(__file__),
        "truth_sidecar_opened":False,"checkpoint_inference_run":False,
        "scope":"Displays frozen results; points/seed ranges in diagnostics are not additional confidence claims"},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_freeze");parser.add_argument("output")
    args=parser.parse_args()
    run(args.evaluation_freeze,args.output)
