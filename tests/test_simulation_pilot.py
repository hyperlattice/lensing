
from pathlib import Path

import importlib.util

import pytest

from src.simulation.generate import generate_observation_pilot, generate_sis_controls
from src.testing.validate_simulations import validate_run


@pytest.mark.parametrize("generator,renderer", [
    ("generate_sis_controls", "render_from_parameters"),
    ("generate_sie_extended", "render_sie_extended_source"),
    ("generate_observation_pilot", "render_observation_scene"),
])
@pytest.mark.parametrize("error_type", [RuntimeError, PermissionError, ValueError, FloatingPointError])
def test_generation_distinguishes_execution_errors_from_invalid_samples(
    tmp_path, monkeypatch, generator, renderer, error_type
):
    import src.simulation.generate as generation

    def fail(*args, **kwargs):
        raise error_type("injected failure")

    monkeypatch.setattr(generation, renderer, fail)
    generate = getattr(generation, generator)
    if error_type in (ValueError, FloatingPointError):
        summary = generate(tmp_path / "release", count=1, create_preview=False)
        assert summary["count_accepted"] == 0
        assert summary["count_rejected"] == 1
        assert summary["rejections"][0]["message"] == "injected failure"
    else:
        with pytest.raises(error_type, match="injected failure"):
            generate(tmp_path / "release", count=1, create_preview=False)


def test_p10_generation_is_reproducible_and_complete(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    previews = tmp_path / "previews"
    first_summary = generate_sis_controls(
        first, count=2, seed=77, create_preview=True, preview_root=previews
    )
    second_summary = generate_sis_controls(second, count=2, seed=77, create_preview=False)

    assert first_summary["count_accepted"] == 2
    assert first_summary["count_rejected"] == 0
    assert second_summary["count_rejected"] == 0
    assert validate_run(first / "manifest.csv")["passed"] is True
    assert (previews / "simulated" / "sim-p10-v1" / "contact_sheet.png").exists()
    assert Path(first_summary["preview_index_path"]) == (
        previews / "simulated" / "sim-p10-v1" / "preview_index.csv"
    )
    assert not (first / "previews").exists()
    assert (first / "rejection_log.csv").exists()

    for index in range(1, 3):
        sample = f"sim-p10-v1-{index:04d}"
        for relative in (f"raw/{sample}.fits", f"science/{sample}.npy", f"normalized/{sample}.npy"):
            assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_p60_observation_generation_is_reproducible_and_valid(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    first = tmp_path / "first"
    second = tmp_path / "second"
    previews = tmp_path / "previews"
    first_summary = generate_observation_pilot(
        first,
        count=2,
        seed=6001,
        simulation_id="sim-p60-test",
        create_preview=True,
        preview_root=previews,
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
    assert (previews / "simulated" / "sim-p60-test" / "contact_sheet.png").exists()

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
            assert (first / relative).read_bytes() == (second / relative).read_bytes()
