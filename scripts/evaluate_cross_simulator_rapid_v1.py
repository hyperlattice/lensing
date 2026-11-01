"""Target-free inference and one-way scoring for the rapid A/B release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.benchmark_contract_v1 import sha256
from src.data.paired_evaluation_images_v1 import PairedEvaluationImages
from src.ml.model import build_model
from src.ml.robustness_training_v1 import grouped_forward
from scripts.audit_cross_simulator_rapid_v1 import audit_release


ROOT = Path("results/ml/cross-simulator-rapid-v2")
METHOD = ROOT / "method_freeze.json"
RELEASE = Path("data/simulations/cross-simulator-rapid-v2/manifest.json")
PREDICTIONS = ROOT / "predictions_before_scoring.csv"
ARMS = ("C", "G", "R")
SEEDS = tuple(730001 + i for i in range(10))
RENDERERS = ("lenstronomy", "autolens")
CONDITIONS = ("canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half",
              "read-noise-double", "moffat-2.5", "moffat-3.5", "moffat-4.5",
              "moffat-3.5-exposure-half")
PRIMARY = ("moffat-2.5", "moffat-3.5", "moffat-4.5")


def _checkpoint(arm, seed):
    return Path(f"results/ml/equal-budget-final-v1/fits/{arm}/seed-{seed}/final.pt")


def predict():
    if PREDICTIONS.exists():
        print(json.dumps({"stage": "predictions_already_sealed", "path": str(PREDICTIONS)}))
        return
    images = PairedEvaluationImages(RELEASE, role="power_pilot", freeze_sha256=sha256(METHOD))
    images.verify_all_products()
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint = torch.load(_checkpoint(arm, seed), map_location="cpu", weights_only=True)
            model = build_model(checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            stats = checkpoint["statistics"]
            for renderer in RENDERERS:
                for condition in CONDITIONS:
                    values = []
                    for start in range(0, len(images.parent_ids), 16):
                        ids = images.parent_ids[start:start + 16]
                        arrays = [images.image(parent, renderer, condition) for parent in ids]
                        _, output = grouped_forward(model, arrays, stats)
                        values.extend((output.detach().numpy() * stats["target_std"] + stats["target_mean"]).tolist())
                    records.extend({"parent_system_id": parent, "renderer": renderer,
                                    "condition": condition, "arm": arm, "seed": seed,
                                    "theta_E_arcsec_predicted": float(value)}
                                   for parent, value in zip(images.parent_ids, values))
    frame = pd.DataFrame.from_records(records)
    expected = len(images.parent_ids) * len(RENDERERS) * len(CONDITIONS) * len(ARMS) * len(SEEDS)
    if len(frame) != expected or not np.isfinite(frame["theta_E_arcsec_predicted"]).all():
        raise ValueError("Prediction dimensions or finite values differ")
    frame.to_csv(PREDICTIONS, index=False)
    seal = {"schema": "cross_simulator_rapid_predictions_v1", "prediction_path": str(PREDICTIONS),
            "prediction_sha256": sha256(PREDICTIONS), "release_manifest_sha256": sha256(RELEASE),
            "rows": len(frame), "truth_opened": False}
    (ROOT / "predictions_complete.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "predictions_sealed", "rows": len(frame), "truth_opened": False}))


def _bootstrap(a, b, seed=620901, draws=10000):
    rng = np.random.default_rng(seed)
    n = a.shape[1]
    indices = rng.integers(0, n, size=(draws, n))
    values = a[:, indices].mean(axis=(0, 2)) - b[:, indices].mean(axis=(0, 2))
    return {"difference_arcsec": float((a - b).mean()),
            "bootstrap_95_low_arcsec": float(np.quantile(values, .025)),
            "bootstrap_95_high_arcsec": float(np.quantile(values, .975)),
            "bootstrap_seed": seed, "bootstrap_resamples": draws}


def score():
    seal = json.loads((ROOT / "predictions_complete.json").read_text(encoding="utf-8"))
    if sha256(seal["prediction_path"]) != seal["prediction_sha256"]:
        raise ValueError("Prediction seal changed")
    pred = pd.read_csv(seal["prediction_path"])
    manifest = json.loads(RELEASE.read_text(encoding="utf-8"))
    if sha256(RELEASE.parent / "truth_vault" / "targets.json") != manifest["truth_sha256"]:
        raise ValueError("Truth vault hash changed")
    truth = json.loads((RELEASE.parent / "truth_vault" / "targets.json").read_text(encoding="utf-8"))
    pred["theta_E_arcsec_true"] = pred["parent_system_id"].map(truth)
    if pred["theta_E_arcsec_true"].isna().any():
        raise ValueError("Incomplete truth join")
    pred["absolute_error_arcsec"] = abs(pred["theta_E_arcsec_predicted"] - pred["theta_E_arcsec_true"])
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {}
        for renderer in RENDERERS:
            metrics[arm][renderer] = {}
            for condition in CONDITIONS:
                subset = pred[(pred.arm == arm) & (pred.renderer == renderer) & (pred.condition == condition)]
                residual = subset.theta_E_arcsec_predicted - subset.theta_E_arcsec_true
                metrics[arm][renderer][condition] = {
                    "mae_arcsec": float(subset.absolute_error_arcsec.mean()),
                    "rmse_arcsec": float(np.sqrt(np.mean(np.square(residual)))),
                    "bias_arcsec": float(residual.mean()), "count": int(len(subset)),
                }
    parents = sorted(truth)
    contrasts = {}
    arrays = {}
    for arm in ARMS:
        for renderer in RENDERERS:
            arr = np.empty((len(SEEDS), len(parents), len(PRIMARY)), dtype=float)
            for si, seed in enumerate(SEEDS):
                sub = pred[(pred.arm == arm) & (pred.renderer == renderer) & (pred.seed == seed)]
                for pi, parent in enumerate(parents):
                    row = sub[sub.parent_system_id == parent]
                    arr[si, pi] = [row[row.condition == c].absolute_error_arcsec.iloc[0] for c in PRIMARY]
            arrays[(arm, renderer)] = arr
    for renderer in RENDERERS:
        for candidate, comparator in (("G", "C"), ("R", "G"), ("R", "C")):
            contrasts[f"{candidate}_vs_{comparator}_{renderer}_primary_moffat"] = _bootstrap(
                arrays[(candidate, renderer)], arrays[(comparator, renderer)])
    transfer = {}
    for arm in ARMS:
        transfer[f"{arm}_autolens_minus_lenstronomy_primary"] = {
            "difference_arcsec": float(arrays[(arm, "autolens")].mean() - arrays[(arm, "lenstronomy")].mean()),
            "autolens_mae_arcsec": float(arrays[(arm, "autolens")].mean()),
            "lenstronomy_mae_arcsec": float(arrays[(arm, "lenstronomy")].mean()),
        }
    renderer_identity = audit_release()
    report = {"schema": "cross_simulator_rapid_report_v1", "stage": "scored_after_target_join",
              "release_manifest_sha256": sha256(RELEASE), "parent_count": len(parents),
              "seed_count": len(SEEDS), "arms": list(ARMS), "renderers": list(RENDERERS),
              "primary_conditions": list(PRIMARY), "metrics": metrics, "contrasts": contrasts,
              "renderer_transfer": transfer, "renderer_identity": renderer_identity,
              "limitations": [
                  "Rapid release uses the existing qualified renderer stack and a new 150-parent AutoLens/Lenstronomy paired cohort.",
                  "Moffat PSF is an engineered unseen observation family, not real telescope data.",
                  "The formal Freeze-2 power-planned workflow was not rerun; item-1 checkpoints were fixed before this evaluation.",
                  "All 1,350 saved Lenstronomy/AutoLens image pairs are byte-identical at float32 precision, so this release is not evidence of a simulator-induced image-domain shift.",
                  "The package implementations use shared project integration, detector and preprocessing; simulator independence is therefore not established.",
              ]}
    (ROOT / "final_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Rapid cross-simulator result", "",
             f"New paired parents: {len(parents)}; paired seeds: {len(SEEDS)}; primary conditions: {', '.join(PRIMARY)}.", "",
             "| Contrast | Lenstronomy difference | AutoLens difference |", "|---|---:|---:|"]
    for candidate, comparator in (("G", "C"), ("R", "G"), ("R", "C")):
        a = contrasts[f"{candidate}_vs_{comparator}_lenstronomy_primary_moffat"]
        b = contrasts[f"{candidate}_vs_{comparator}_autolens_primary_moffat"]
        lines.append(f"| {candidate} vs {comparator} | {a['difference_arcsec']:.6f} "
                     f"[{a['bootstrap_95_low_arcsec']:.6f}, {a['bootstrap_95_high_arcsec']:.6f}] | "
                     f"{b['difference_arcsec']:.6f} [{b['bootstrap_95_low_arcsec']:.6f}, {b['bootstrap_95_high_arcsec']:.6f}] |")
    lines.extend(["", "Renderer identity audit: 1,350/1,350 Lenstronomy/AutoLens image pairs and 40,500/40,500 paired predictions are exactly equal at saved precision.",
                  "Therefore this release is a matched-renderer equivalence control plus an unseen Moffat observation-family shift, not evidence of simulator-induced transfer.",
                  "The Moffat family is engineered synthetic data, not real-data validation.", ""])
    (ROOT / "result_overview.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stage": "scored", "report": str(ROOT / "final_report.json"),
                      "primary_contrasts": {k: v for k, v in contrasts.items() if "primary_moffat" in k}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("predict", "score"))
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.action]()
