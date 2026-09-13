"""Final actuation guards shared by simulator and unit tests."""

from __future__ import annotations

import torch


def filter_residual_action(
    previous: torch.Tensor, desired: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Apply a first-order filter to the executed residual target."""
    if previous.shape != desired.shape:
        raise ValueError("previous and desired residuals must have identical shapes")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("residual filter alpha must be in (0, 1]")
    return torch.lerp(previous, desired, alpha)


def project_joint_velocity(
    velocity: torch.Tensor, velocity_limit: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project measured joint velocity to symmetric hard limits."""
    if velocity.shape != velocity_limit.shape:
        raise ValueError("velocity and velocity_limit must have identical shapes")
    if torch.any(velocity_limit <= 0.0):
        raise ValueError("velocity limits must be positive")
    projected = torch.clamp(velocity, -velocity_limit, velocity_limit)
    return projected, velocity.abs() > velocity_limit


def apply_velocity_torque_guard(
    torque: torch.Tensor,
    velocity: torch.Tensor,
    torque_limit: torch.Tensor,
    velocity_limit: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Replace weak or accelerating torque near a joint velocity limit."""
    if torque.shape != velocity.shape:
        raise ValueError("torque and velocity must have identical shapes")
    if torque_limit.shape != torque.shape or velocity_limit.shape != torque.shape:
        raise ValueError("limit tensors must match the torque shape")
    if margin <= 0.0:
        raise ValueError("velocity guard margin must be positive")

    guard_start = (velocity_limit - margin).clamp_min(0.0)
    activation = ((velocity.abs() - guard_start) / margin).clamp(0.0, 1.0)
    braking = -torch.sign(velocity) * torque_limit * activation
    guarded = torch.where(
        velocity >= 0.0,
        torch.minimum(torque, braking),
        torch.maximum(torque, braking),
    )
    return torch.where(activation > 0.0, guarded, torque)
