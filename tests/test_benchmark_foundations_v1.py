"""Leakage, artifact, paired-budget and adaptation controls for the new study."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from src.data.benchmark_contract_v1 import (BenchmarkImages, load_source_targets,
                                           observation_digest, sha256, validate_lineage)
from src.ml.augmentation_policy_v1 import CELLS, condition_for, failure_weights, paired_policy_schedule
from src.ml.domain_adaptation_v1 import gaussian_mmd, image_features_and_prediction
from src.ml.model import build_model


def write_manifest(root, *, role="numerical_development", stage="engineering"):
    root.mkdir(exist_ok=True)
    arrays = {"image": np.ones((16, 16), dtype=np.float32),
              "variance": np.ones((16, 16), dtype=np.float32),
              "invalid_mask": np.zeros((16, 16), dtype=bool),
              "coverage_mask": np.zeros((16, 16), dtype=bool),
              "psf": np.ones((3, 3), dtype=np.float32)/9}
    products = {}
    for k, a in arrays.items():
        path = root / f"{k}.npy"
        np.save(path, a)
        products[k] = {"path": path.name, "sha256": sha256(path)}
    row = {"sample_id": "view-1", "parent_system_id": "parent-1", "role": role,
           "renderer": "autolens" if role in {"numerical_development", "target_adaptation"} else "lenstronomy",
           "condition": "canonical", "parameter_seed": 12, "noise_seed": 13,
           "ancestry_ids": [], "shape": [16, 16], "pixel_scale_arcsec": .05,
           "psf_family": "gaussian", "products": products}
    manifest = {"schema": "cross_simulator_benchmark_v1", "release_id": "test-v1",
                "stage": stage, "image_unit": "electrons_per_second",
                "variance_unit": "electrons_per_second_squared", "mask_semantics": "true_is_excluded",
                "qualification": None, "observations": [row]}
    if stage == "qualified":
        report = root / "qualification.json"
        report.write_text(json.dumps({"passed": True, "release_id": "test-v1",
                                     "observation_contract_sha256": observation_digest([row])}))
        manifest["qualification"] = {"path": report.name, "sha256": sha256(report)}
    path = root / "benchmark.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_image_only_interface_never_reads_targets_and_detects_changed_bytes(tmp_path):
    path, _ = write_manifest(tmp_path)
    (tmp_path / "targets.json").write_text("intentionally invalid JSON: must never be opened")
    with pytest.raises(ValueError, match="Engineering"):
        BenchmarkImages(path, role="numerical_development")
    images = BenchmarkImages(path, role="numerical_development", allow_engineering=True)
    assert isinstance(images[0], np.ndarray) and images[0].shape == (16, 16)
    with pytest.raises(ValueError, match="unavailable"):
        load_source_targets(tmp_path / "does-not-exist.json", images, role="target_adaptation")
    np.save(tmp_path / "image.npy", np.zeros((16, 16), np.float32))
    with pytest.raises(ValueError, match="changed artifact"):
        images[0]


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m["observations"][0].update(theta_E_arcsec=1.2), "truth belongs"),
    (lambda m: m["observations"][0]["products"]["image"].update(path="../image.npy"), "escapes"),
    (lambda m: m["observations"][0].update(shape=[17, 16]), "geometry"),
    (lambda m: m["observations"][0]["products"].pop("variance"), "Missing required"),
    (lambda m: m["observations"].append(deepcopy(m["observations"][0])), "Duplicate"),
    (lambda m: m["observations"][0].update(role="source_train", renderer="lenstronomy"), "Engineering"),
])
def test_manifest_rejects_leaks_and_malformed_products(tmp_path, mutation, match):
    path, m = write_manifest(tmp_path)
    mutation(m)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=match):
        BenchmarkImages(path, role="numerical_development", allow_engineering=True)


@pytest.mark.parametrize("product,array,match", [
    ("variance", np.zeros((16, 16), np.float32), "Variance"),
    ("invalid_mask", np.zeros((16, 16), np.float32), "Masks"),
    ("psf", np.ones((3, 3), np.float32), "PSF"),
    ("image", np.full((16, 16), np.nan, np.float32), "finite"),
])
def test_product_semantics_not_just_hashes(tmp_path, product, array, match):
    path, m = write_manifest(tmp_path)
    p = tmp_path / f"{product}.npy"
    np.save(p, array)
    m["observations"][0]["products"][product]["sha256"] = sha256(p)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=match):
        BenchmarkImages(path, role="numerical_development", allow_engineering=True)


@pytest.mark.parametrize("shared", ["parent_system_id", "parameter_seed", "noise_seed", "ancestry_ids"])
def test_renaming_parent_does_not_bypass_role_exclusions(tmp_path, shared):
    _, m = write_manifest(tmp_path)
    row = m["observations"][0]
    excluded = {**row, "role": "final", "parent_system_id": "old-parent",
                "parameter_seed": 100, "noise_seed": 101, "ancestry_ids": ["old-stamp"]}
    if shared == "ancestry_ids":
        row[shared] = ["old-stamp"]
    else:
        row[shared] = excluded[shared]
    with pytest.raises(ValueError, match="Cross-role"):
        validate_lineage([row], [excluded])


def test_qualified_source_labels_are_explicit_and_identity_matched(tmp_path):
    path, m = write_manifest(tmp_path, role="source_train", stage="qualified")
    images = BenchmarkImages(path, role="source_train")
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"view-1": 1.2}))
    np.testing.assert_array_equal(load_source_targets(targets, images, role="source_train"), [1.2])
    targets.write_text(json.dumps({"other": 1.2}))
    with pytest.raises(ValueError, match="identities"):
        load_source_targets(targets, images, role="source_train")
    m["observations"][0]["condition"] = "renamed-after-qualification"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="qualification"):
        BenchmarkImages(path, role="source_train")


def test_adaptation_cannot_load_heldout_psf_or_source_domain(tmp_path):
    path, m = write_manifest(tmp_path, role="target_adaptation", stage="qualified")
    images = BenchmarkImages(path, role="target_adaptation")
    with pytest.raises(ValueError, match="unavailable"):
        load_source_targets(tmp_path / "unread.json", images, role="target_adaptation")
    m["observations"][0]["psf_family"] = "moffat"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Held-out"):
        BenchmarkImages(path, role="target_adaptation")


def test_failure_weights_pairing_clipping_exploration_and_role():
    canonical = np.full((3, 9), .1)
    transformed = np.repeat(canonical[..., None], len(CELLS), axis=-1)
    transformed[..., 0] += .2
    transformed[..., 1] -= .05
    r = failure_weights(canonical, transformed, role="source_development")
    p = np.asarray(r["probabilities"])
    assert p[0] == pytest.approx(.75 + .25/7)
    np.testing.assert_allclose(p[1:], .25/7)
    assert sum(p) == pytest.approx(1)
    uniform = failure_weights(canonical, np.zeros_like(transformed), role="source_development")
    np.testing.assert_allclose(uniform["probabilities"], np.ones(7)/7)
    with pytest.raises(ValueError, match="source-development"):
        failure_weights(canonical, transformed, role="final")


def test_policy_changes_do_not_change_parent_exposure_or_noise_stream():
    uniform = np.ones(7)/7
    weighted = np.array([.7, .05, .05, .05, .05, .05, .05])
    kwargs = dict(updates=60, batch_size=16, seed=12071)
    a = paired_policy_schedule(["a", "b", "c"], uniform, **kwargs)
    b = paired_policy_schedule(["a", "b", "c"], weighted, **kwargs)
    assert a == paired_policy_schedule(["a", "b", "c"], uniform, **kwargs)
    assert [x["parents"] for x in a] == [x["parents"] for x in b]
    assert [x["noise_seeds"] for x in a] == [x["noise_seeds"] for x in b]
    assert [x["cell"] for x in a] != [x["cell"] for x in b]
    assert all(len(x["parents"]) == 16 for x in a)
    seeds = [s for x in a for s in x["noise_seeds"]]
    assert len(set(seeds)) == len(seeds)
    for cell in CELLS:
        c = condition_for(cell)
        assert c["psf_family"] == "gaussian"
        assert abs(c["shape"][0]*c["pixel_scale_arcsec"] - 12.8) <= c["pixel_scale_arcsec"]/2


def test_mmd_analytic_estimators_gradient_and_cnn_prediction_parity():
    torch.set_num_threads(2)
    source = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.], [3.], [4.]], dtype=torch.float64, requires_grad=True)
    # Independent scalar loop verifies diagonal exclusion and unequal batches.
    a, b = source.detach().numpy()[:, 0], target.detach().numpy()[:, 0]
    k = lambda u, v: np.exp(-(u-v)**2/2)
    xx = np.mean([k(u,v) for i,u in enumerate(a) for j,v in enumerate(a) if i != j])
    yy = np.mean([k(u,v) for i,u in enumerate(b) for j,v in enumerate(b) if i != j])
    xy = np.mean([k(u,v) for u in a for v in b])
    unbiased = gaussian_mmd(source, target, bandwidths=[1.], estimator="unbiased")
    assert unbiased.item() == pytest.approx(xx+yy-2*xy)
    zero = gaussian_mmd(source, source, bandwidths=[.5, 1., 2.])
    assert zero.item() == pytest.approx(0, abs=1e-12)
    assert gaussian_mmd(source, target, bandwidths=[1.]).item() > 0
    assert torch.autograd.gradcheck(lambda x,y: gaussian_mmd(x,y, bandwidths=[.5, 1.]), (source,target))
    model = build_model({"channels": [2, 2, 2, 2], "pooled_size": 1, "hidden_features": 4, "dropout": 0.})
    model.eval()
    images = torch.randn(3, 1, 32, 32)
    features, prediction = image_features_and_prediction(model, images)
    torch.testing.assert_close(prediction, model(images))
    assert features.shape == (3, 2)
    loss = gaussian_mmd(features, features + .1, bandwidths=[1.]) + prediction.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mmd_synthetic_regression_retains_signal_and_removes_nuisance_shift():
    torch.manual_seed(9401)
    u = torch.linspace(-1, 1, 32)
    source = torch.stack((u, torch.zeros_like(u)), 1)
    target = torch.stack((u, torch.ones_like(u)*2), 1)
    encoder = torch.nn.Linear(2, 2, bias=False)
    head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=.03)
    initial = gaussian_mmd(encoder(source), encoder(target), bandwidths=[.5, 1.]).item()
    for _ in range(180):
        optimizer.zero_grad()
        zs, zt = encoder(source), encoder(target)
        loss = (head(zs).squeeze(-1) - u).square().mean() + gaussian_mmd(zs, zt, bandwidths=[.5, 1.])
        loss.backward()
        optimizer.step()
    final = gaussian_mmd(encoder(source), encoder(target), bandwidths=[.5, 1.]).item()
    assert final < initial / 20
    assert (head(encoder(source)).squeeze(-1) - u).square().mean().item() < 1e-4
    assert (head(encoder(target)).squeeze(-1) - u).square().mean().item() < 1e-3


def test_engineering_import_preserves_source_and_refuses_reuse(tmp_path):
    import csv
    from astropy.io import fits
    from scripts.materialize_benchmark_contract_development_v1 import run

    source = tmp_path / "source"
    source.mkdir()
    folder = source / "view-1"
    folder.mkdir()
    fits.writeto(folder / "raw.fits", np.full((16, 16), .4, np.float32),
                 header=fits.Header({"BUNIT": "ELECTRONS/S"}))
    np.save(folder / "expected_variance.npy", np.full((16, 16), .01, np.float32))
    np.save(folder / "psf.npy", np.ones((3, 3), np.float32)/9)
    (source / "generation_complete.json").write_text(json.dumps({"stage": "development", "cnn_inference_run": False}))
    row = {"sample_id": "view-1", "parent_system_id": "parent-1", "condition": "canonical",
           "raw_path": "view-1/raw.fits", "parameter_seed": 41, "noise_seed": 42,
           "pixel_scale": .05, "shape": "16x16", "simulator_name": "autolens",
           "accuracy_evaluation_eligible": False, "theta_E_arcsec": "not-a-number-must-not-be-parsed"}
    with (source / "development_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    checksums = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
    (source / "checksums.json").write_text(json.dumps(checksums))
    output, report_path = tmp_path / "new-overlay", tmp_path / "report.json"
    report = run(source, output, report_path)
    assert report["contract_validation_passed"] and report["training_role_blocked"]
    assert report["target_sidecars_opened"] == 0 and not report["accuracy_qualified"]
    images = BenchmarkImages(output / "benchmark.json", role="numerical_development", allow_engineering=True)
    np.testing.assert_array_equal(images[0], np.full((16, 16), .4, np.float32))
    assert all(sha256(source / p) == digest for p, digest in checksums.items())
    with pytest.raises(FileExistsError):
        run(source, output, report_path)
    np.save(folder / "expected_variance.npy", np.zeros((16, 16), np.float32))
    with pytest.raises(ValueError, match="Source artifact changed"):
        run(source, tmp_path / "tampered-overlay", tmp_path / "tampered-report.json")
