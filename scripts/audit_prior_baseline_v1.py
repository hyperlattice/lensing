"""Inspect pinned code and execute only reviewed AST definitions on CPU fixtures.

No author notebook or top-level source is executed. The audit verifies code
behavior, not the provenance of published models or the validity of a paper.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import r2_score

from src.ml.adaptive_mmd_v1 import adaptive_gaussian_mmd

ROOT = Path("data/external_reference_research/mve-source-17e42ac")
OUTPUT = Path("results/reports/prior-baseline-intake-v1/code_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError("Audit already exists")
    path = ROOT/"src/training/MVEUDA/train_utils.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"MMD_loss", "loss_bnll", "train_loop_mve"}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Unexpected pinned utility definitions")
    namespace = {"torch": torch, "nn": torch.nn, "np": np, "r2_score": r2_score}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    torch.set_num_threads(2)
    torch.manual_seed(720201)
    a, b = torch.randn(8, 4), torch.randn(8, 4)+.5
    author = namespace["MMD_loss"]()(a, b)
    ours = adaptive_gaussian_mmd(a, b)
    difference = abs(float(author-ours))
    torch.testing.assert_close(author, ours, rtol=1e-5, atol=1e-6)
    x = a.clone().requires_grad_()
    adaptive_gaussian_mmd(x, b).backward()
    active_gradient_norm = float(x.grad.norm())

    class FixtureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 2)
        def forward(self, values):
            z = self.encoder(values)
            return torch.nn.functional.softplus(self.head(z))+.1, z

    initial = FixtureModel()
    source_data = [(a, torch.linspace(.6, 1.4, 8))]
    # This label payload is discarded by the reviewed author's train function.
    target_data = [(b, torch.full((8,), float("nan")))]
    states = []
    with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
        for weight in (0., 1.4):
            model = deepcopy(initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            namespace["train_loop_mve"](source_data, target_data, model, torch.nn.MSELoss(),
                namespace["MMD_loss"](), optimizer, weight, .5)
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    update_difference = max(float(abs(states[0][k]-states[1][k]).max()) for k in states[0])
    assert update_difference == 0
    notebook = json.loads((ROOT/"src/training/MVEUDA/MVE_UDA_RunA.ipynb").read_text(encoding="utf-8"))
    cells = [{"index": i, "source": "".join(c.get("source", []))} for i,c in enumerate(notebook["cells"]) if c["cell_type"] == "code"]
    selector_cells = [c["index"] for c in cells if "best_target_R2" in c["source"] and "save_model" in c["source"]]
    report = {"commit": "17e42acb13f0c3111c8a4e665222e215b10d2e7b",
        "utility_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reviewed_definitions_executed": sorted(names), "author_top_level_or_notebooks_executed": False,
        "cpu_fixture_cuda_calls_redirected": True, "fixture_updates_per_weight": 1,
        "weights_compared": [0., 1.4], "max_parameter_difference_after_update": update_difference,
        "mmd_kernel_port_absolute_difference": difference, "port_mmd_feature_gradient_norm": active_gradient_norm,
        "target_label_checkpoint_selector_cells": selector_cells,
        "finding": "Pinned MVE train_loop_mve ignores da_weight in optimized loss; MMD computed under no_grad. Notebook includes target-label-selected checkpoints.",
        "interpretation_limit": "This establishes current pinned code behavior, not which code produced published models. Target forwards may still affect BatchNorm statistics in the author's architecture.",
        "comparison_policy": "Report exact code behavior separately; use functioning, tuned paper-kernel MMD with source-only selection for common-benchmark method comparison.",
        "license": "MIT; copyright 2024 Shrihan Agarwal", "paper_reproduction_complete": False}
    OUTPUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
