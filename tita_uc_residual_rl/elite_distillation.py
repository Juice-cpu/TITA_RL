"""Behavior cloning helpers for successful residual-policy trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .phase_b_learner import PhaseBActorCritic


@dataclass(frozen=True)
class EliteDistillationStats:
    samples: int
    epochs: int
    action_mse_before: float
    action_mse_after: float
    anchor_mse_after: float
    max_policy_shift: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / weights.sum().clamp_min(1e-8)


def distill_elite_actions(
    model: PhaseBActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    source_means: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    anchor_weight: float = 3.0,
) -> EliteDistillationStats:
    """Move the deterministic actor toward safe elite actions.

    Only actor-network weights are updated. The value heads and exploration
    standard deviation are intentionally left untouched for the subsequent
    constrained PPO fine-tune.
    """

    if observations.ndim != 2 or observations.shape[1] != model.policy.network[0].in_features:
        raise ValueError("observations must have shape (N, policy observation dimension)")
    if actions.shape != (observations.shape[0], 8) or source_means.shape != actions.shape:
        raise ValueError("actions and source_means must have shape (N, 8)")
    if observations.shape[0] == 0:
        raise ValueError("at least one elite sample is required")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or anchor_weight < 0.0:
        raise ValueError("distillation hyperparameters are invalid")
    if not (
        torch.isfinite(observations).all()
        and torch.isfinite(actions).all()
        and torch.isfinite(source_means).all()
    ):
        raise ValueError("elite tensors must contain only finite values")

    device = next(model.parameters()).device
    observations = observations.to(device=device, dtype=torch.float32)
    actions = actions.to(device=device, dtype=torch.float32)
    source_means = source_means.to(device=device, dtype=torch.float32)
    weights = (
        torch.ones(observations.shape[0], device=device)
        if sample_weights is None
        else sample_weights.to(device=device, dtype=torch.float32)
    )
    if weights.shape != (observations.shape[0],) or not torch.isfinite(weights).all():
        raise ValueError("sample_weights must be a finite N-vector")
    if torch.any(weights < 0.0) or not torch.any(weights > 0.0):
        raise ValueError("sample_weights must be non-negative with at least one positive value")

    action_limit = (
        model.policy.posture_scales * model.policy.residual_limit
    ).to(device)
    actions = torch.maximum(torch.minimum(actions, action_limit), -action_limit)
    source_means = source_means.detach()

    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    actor_parameters = list(model.policy.network.parameters())
    for parameter in actor_parameters:
        parameter.requires_grad_(True)

    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    model.eval()
    with torch.no_grad():
        initial_mean = model.policy(observations)
        before = _weighted_mean(
            torch.square(initial_mean - actions).mean(dim=1), weights
        )

    count = observations.shape[0]
    for _ in range(epochs):
        order = torch.randperm(count, device=device)
        for start in range(0, count, batch_size):
            batch = order[start : start + batch_size]
            predicted = model.policy(observations[batch])
            action_error = torch.square(predicted - actions[batch]).mean(dim=1)
            anchor_error = torch.square(predicted - source_means[batch]).mean(dim=1)
            batch_weights = weights[batch]
            loss = _weighted_mean(
                action_error + anchor_weight * anchor_error, batch_weights
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()

    with torch.no_grad():
        final_mean = model.policy(observations)
        after = _weighted_mean(torch.square(final_mean - actions).mean(dim=1), weights)
        anchor = _weighted_mean(
            torch.square(final_mean - source_means).mean(dim=1), weights
        )
        shift = torch.abs(final_mean - source_means).max()

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])

    return EliteDistillationStats(
        samples=count,
        epochs=epochs,
        action_mse_before=float(before),
        action_mse_after=float(after),
        anchor_mse_after=float(anchor),
        max_policy_shift=float(shift),
    )
