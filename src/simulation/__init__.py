"""Simulation backends and project-level lens-model conventions."""

from src.simulation.models import (
    backend_convention_check,
    backend_status,
    e1_e2_to_q_phi,
    ellipticity_from_q_phi,
    q_phi_to_e1_e2,
    sie_kwargs,
    shear_kwargs,
    sis_deflection,
    sis_kwargs,
)
from src.simulation.priors import (
    draw_observation_parameters,
    draw_stress_parameters,
    draw_sie_parameters,
    validate_sie_parameters,
)
from src.simulation.render import (
    render_from_parameters,
    render_sie_extended_source,
    render_sis_extended_source,
    render_sis_point_source,
    sis_image_positions,
)

__all__ = [
    "backend_convention_check",
    "backend_status",
    "e1_e2_to_q_phi",
    "ellipticity_from_q_phi",
    "q_phi_to_e1_e2",
    "sie_kwargs",
    "shear_kwargs",
    "sis_deflection",
    "sis_kwargs",
    "draw_sie_parameters",
    "draw_observation_parameters",
    "draw_stress_parameters",
    "validate_sie_parameters",
    "render_from_parameters",
    "render_sie_extended_source",
    "render_sis_extended_source",
    "render_sis_point_source",
    "sis_image_positions",
]
