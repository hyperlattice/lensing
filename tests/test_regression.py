"""Characterize the retained study against the pre-refactor implementation."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import fixture_bank

from src.data.dataset import ObservationBank
from src.ml.analysis import _bootstrap_difference
from src.ml.artifacts import load_config, load_development_config
from src.ml.train import train_arm
from src.simulation.render import SceneEngine

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("arm", ["C", "G", "R"])
def test_training_matches_pre_refactor_checkpoint(tmp_path, arm):
    reference = json.loads((FIXTURES / "training_reference.json").read_text())
    source = ObservationBank(
        fixture_bank(tmp_path / "source", "source_train", 1000),
        expected_role="source_train",
        allow_fixture=True,
    )
    development = ObservationBank(
        fixture_bank(tmp_path / "dev", "source_development", 2000),
        expected_role="source_development",
        allow_fixture=True,
    )
    output = tmp_path / "fit"
    train_arm(
        reference["config"],
        seed=reference["seed"],
        arm=arm,
        source=source,
        development=development,
        target=None,
        output=output,
        probabilities=reference["probabilities"],
        allow_fixture=True,
    )
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    with np.load(FIXTURES / "training_reference.npz") as expected:
        for name, value in checkpoint["model_state"].items():
            np.testing.assert_allclose(
                value.numpy(), expected[f"{arm}__{name}"], rtol=1e-6, atol=1e-7
            )


def test_renderer_matches_pre_refactor_native_products():
    reference = json.loads((FIXTURES / "render_reference.json").read_text())
    engine = SceneEngine(reference["parameters"], "lenstronomy")
    actual = engine.observe(
        reference["condition"], factor=reference["factor"], noise_seed=reference["noise_seed"]
    )
    with np.load(FIXTURES / "render_reference.npz") as expected:
        for name, value in actual.items():
            assert value.shape == expected[name].shape
            assert value.dtype == expected[name].dtype
            # Saved float32 detector products should remain identical locally.
            # Tiny cross-platform changes in nonlinear roots use a tight tolerance.
            np.testing.assert_allclose(value, expected[name], rtol=1e-8, atol=1e-10)


def test_bootstrap_resamples_whole_parents_with_seeds_and_views_together():
    candidate = np.arange(24, dtype=float).reshape(2, 3, 4) / 100
    comparator = np.flip(candidate, axis=1) + 0.05
    seed = 42
    draws = 100
    actual = _bootstrap_difference(candidate, comparator, seed=seed, draws=draws)
    indices = np.random.default_rng(seed).integers(0, 3, size=(draws, 3))
    direct = []
    for parents in indices:
        direct.append(candidate[:, parents, :].mean() - comparator[:, parents, :].mean())
    np.testing.assert_allclose(actual["difference_arcsec"], (candidate - comparator).mean())
    np.testing.assert_allclose(
        [actual["bootstrap_95_low_arcsec"], actual["bootstrap_95_high_arcsec"]],
        np.quantile(direct, [0.025, 0.975]),
        atol=1e-15,
    )


def test_consolidated_config_profiles_preserve_destinations_and_shared_controls():
    study = load_config()
    assert study["output"] == "results/ml/equal-budget-final-v1"
    assert study["policy"] == "results/ml/robustness-development-v4/allocation/policy.json"
    development = load_development_config()
    assert development["output"] == "results/ml/robustness-development-v4"
    assert development["source_train"] == study["source_train"]
    assert development["source_development"] == study["source_development"]
    for key in (
        "optimizer_updates",
        "validation_interval_updates",
        "batch_size",
        "torch_threads",
        "learning_rate",
        "weight_decay",
        "huber_delta",
        "mmd_weight",
        "model",
    ):
        assert development[key] == study["training"][key]

    replication = load_config(profile="seed_replication")
    assert replication["output"] == "results/ml/cgr-seed-replication-v1"
    assert replication["policy"] == "results/ml/cgr-seed-replication-v1/policy/policy.json"
    assert replication["seeds"] == [730011, 730012, 730013, 730014, 730015]
    assert replication["source_train"] == study["source_train"]
    assert replication["source_development"] == study["source_development"]
    assert replication["holdout"] == study["holdout"]
    assert replication["training"]["optimizer_updates"] == study["training"]["optimizer_updates"]
    assert replication["training"]["model"] == study["training"]["model"]

    assert sorted(path.name for path in Path("configs").glob("*.yaml")) == [
        "hst_study.yaml",
        "simulation.yaml",
        "study.yaml",
    ]
    assert Path("data/manifests/hst_compatibility_pilot_v1.json").is_file()
    assert not Path("configs/hst_compatibility_pilot.json").exists()
