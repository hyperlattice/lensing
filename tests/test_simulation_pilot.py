import hashlib

import importlib.util

import pytest

from src.simulation.generate import generate_observation_pilot, generate_sis_controls
from src.testing.validate_simulations import validate_run


def test_p10_generation_is_reproducible_and_complete(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = generate_sis_controls(first, count=2, seed=77, create_preview=True)
    second_summary = generate_sis_controls(second, count=2, seed=77, create_preview=False)

    assert first_summary["count_accepted"] == 2
    assert first_summary["count_rejected"] == 0
    assert second_summary["count_rejected"] == 0
    assert validate_run(first / "manifest.csv")["passed"] is True
    assert (first / "previews" / "contact_sheet.png").exists()
    assert (first / "rejection_log.csv").exists()

    for index in range(1, 3):
        sample = f"sim-p10-v1-{index:04d}"
        for relative in (f"raw/{sample}.fits", f"science/{sample}.npy", f"normalized/{sample}.npy"):
            assert hashlib.sha256((first / relative).read_bytes()).digest() == hashlib.sha256(
                (second / relative).read_bytes()
            ).digest()


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_p60_observation_generation_is_reproducible_and_valid(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = generate_observation_pilot(
        first,
        count=2,
        seed=6001,
        simulation_id="sim-p60-test",
        create_preview=True,
    )
    second_summary = generate_observation_pilot(
        second,
        count=2,
        seed=6001,
        simulation_id="sim-p60-test",
        create_preview=False,
    )

    assert first_summary["count_accepted"] == 2
    assert first_summary["count_rejected"] == 0
    assert second_summary["count_rejected"] == 0
    assert validate_run(first / "manifest.csv")["passed"] is True
    assert (first / "previews" / "contact_sheet.png").exists()

    for index in range(1, 3):
        sample = f"sim-p60-test-{index:04d}"
        for relative in (
            f"raw/{sample}.fits",
            f"science/{sample}.npy",
            f"normalized/{sample}.npy",
            f"variance/{sample}.npy",
            f"expected_variance/{sample}.npy",
            f"scenes/{sample}_expected.npy",
            f"scenes/{sample}_observed_science.npy",
        ):
            assert hashlib.sha256((first / relative).read_bytes()).digest() == hashlib.sha256(
                (second / relative).read_bytes()
            ).digest()
