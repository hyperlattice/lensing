"""Export numerical-control figures; these contain no CNN accuracy results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.benchmark_contract_v1 import sha256


def main():
    inputs={"qualification":Path("results/reports/renderer-qualification-v5/qualification.json"),
        "fine":Path("results/reports/renderer-qualification-v3/fine_reference/report.json"),
        "cross":Path("results/reports/renderer-qualification-v3/cross_engine/report.json"),
        "roots":Path("results/reports/renderer-qualification-v5/root_report.json")}
    reports={k:json.loads(p.read_text()) for k,p in inputs.items()}
    if not all(r["passed"] for r in reports.values()):
        raise ValueError("Figure requires all stated numerical controls to pass")
    arrays=[np.array([r["adaptive_vs_uniform_l1"]["clean"] for r in reports["fine"]["cases"]]),
        np.array([r["cross_engine_l1"]["expected"] for r in reports["cross"]["cases"]]),
        np.array([r["expected_normalized_l1"] for r in reports["roots"]["cases"]])]
    titles=["Adaptive vs. fine uniform integration","Equivalent A/B expected images","Sparse vs. qualified A root search"]
    gates=[.005,1e-5,1e-5]
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout="constrained")
    for ax,values,title,gate in zip(axes,arrays,titles,gates):
        ax.hist(np.log10(np.maximum(values,1e-16)),bins=14,color="#32769b",edgecolor="white")
        ax.axvline(np.log10(gate),color="#ac3f38",linestyle="--",label="Declared gate")
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("log10(relative L1 difference)")
        ax.set_ylabel("View count")
        ax.text(.03,.96,f"{len(values)} views\nmax = {max(values):.3g}",transform=ax.transAxes,va="top",fontsize=9)
        ax.legend(fontsize=8,loc="upper right")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Finite numerical controls; no prediction-performance claim",fontsize=12)
    output=Path("results/figures/renderer-qualification-v5")
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf"):
        path=output/f"numerical_controls.{suffix}"
        if path.exists():
            raise FileExistsError("Figure identity already exists")
        fig.savefig(path,dpi=180)
    plt.close(fig)
    (output/"figure_provenance.json").write_text(json.dumps({"input_sha256":{str(p):sha256(p) for p in inputs.values()},
        "script_sha256":sha256(__file__),"view_counts":[len(a) for a in arrays],"maximum_relative_l1":[float(a.max()) for a in arrays],
        "gates":gates,"parent_count":105,"proposal_count":106,"single_image_rejections":1,
        "scope":"Numerical development only; 315 views are repeated geometries of 105 parents. Fine controls select 56 difficult/additional views. Shared integration/PSF/detector/Jacobian recipe; finite validation is not universal certification.",
        "cnn_inference_run":False},indent=2)+"\n")


if __name__=="__main__":
    main()
