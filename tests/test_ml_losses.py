import pytest
import torch

from src.ml.losses import build_loss
from src.ml.model import EinsteinRadiusCNN, build_model
from src.ml.metrics import regression_metrics
from src.ml.experiments import load_registered_experiment, validate_suite


@pytest.mark.parametrize("kind", ["huber", "mae", "mse", "log_cosh"])
def test_losses_have_finite_gradients_for_large_residuals(kind):
    pred = torch.tensor([-10000., 0., 10000.], requires_grad=True)
    loss = build_loss({"kind": kind})(pred, torch.zeros(3))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0] < 0 < pred.grad[2]


@pytest.mark.parametrize("config", [{"kind": "unknown"}, {"kind": "huber", "delta": 0},
                                     {"kind": "huber", "delta": float("nan")}, {"kind": "mae", "delta": 1}])
def test_invalid_losses_fail(config):
    with pytest.raises(ValueError):
        build_loss(config)


def test_factory_preserves_baseline_weights_and_scalar_predictions():
    torch.manual_seed(7)
    old = EinsteinRadiusCNN().eval()
    torch.manual_seed(7)
    new = build_model({"kind": "cnn"}).eval()
    assert old.trainable_parameter_count == 248409
    image = torch.ones(2, 1, 64, 64)
    assert torch.equal(old(image), new(image))
    assert new(image).shape == (2,)


def test_prediction_calibration_and_threshold_metrics():
    m = regression_metrics([0.5, 1., 1.5], [0.75, 1., 1.25])
    assert m["prediction_vs_truth_slope"] == pytest.approx(0.5)
    assert m["prediction_vs_truth_intercept_arcsec"] == pytest.approx(0.5)
    assert m["fraction_within_010_arcsec"] == pytest.approx(1/3)
    assert regression_metrics([1.], [1.])["prediction_vs_truth_slope"] is None


def test_ablation_rejects_extra_or_unsupported_changes():
    suite = load_registered_experiment("theta-e-huber-delta-025-v1")
    suite["change"]["learning_rate"] = 0.1
    with pytest.raises(ValueError, match="exactly"):
        validate_suite(suite)
    suite["change"] = {"field": "learning_rate", "value": 0.1}
    with pytest.raises(ValueError, match="only huber_delta"):
        validate_suite(suite)
