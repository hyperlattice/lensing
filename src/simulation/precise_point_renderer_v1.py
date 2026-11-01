"""Fourth-order independent deflection Jacobians for the numerical benchmark.

Positions remain independently solved in each package. Jacobians use each
engine's own deflections with a shared five-point numerical stencil. No
Lenstronomy deflection, derivative or magnification enters the B calculation.
"""
import numpy as np

from src.simulation.benchmark_renderer_v1 import SceneEngine


def fourth_order_amplitudes(engine, positions, *, step=1e-4):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Magnification step must be finite and positive")
    amplitudes = []
    for position in positions:
        derivatives = []
        for axis in range(2):
            offset = np.eye(2)[axis]*step
            grid = np.array([position-2*offset, position-offset, position+offset, position+2*offset])
            alpha = engine.alpha(grid)
            derivatives.append((alpha[0]-8*alpha[1]+8*alpha[2]-alpha[3])/(12*step))
        determinant = np.linalg.det(np.eye(2)-np.column_stack(derivatives))
        amplitudes.append(abs(1/determinant)*engine.p["point_source_amp"])
    amplitudes = np.asarray(amplitudes)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("Invalid fourth-order magnification")
    return amplitudes


class PreciseSceneEngine(SceneEngine):
    def clean(self, condition, *, factor, tile_rows=16):
        width = condition["shape"][1]
        bounded_rows = max(1, min(tile_rows, 262144//(width*factor*factor)))
        return super().clean(condition, factor=factor, tile_rows=bounded_rows)

    def points(self):
        if self._points is None:
            positions, _ = super().points()
            self._points = positions, fourth_order_amplitudes(self, positions)
        return self._points
