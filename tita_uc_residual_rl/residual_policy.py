"""Torch residual policy and a small frozen-base controller contract."""

from __future__ import annotations

from typing import Callable, Iterable, Optional
import math

import torch
from torch import nn


class ResidualPolicy(nn.Module):
    """Maps actor observations to an 8-D bounded joint-target residual.

    The posture scale is supplied by the upper-level posture command.  This
    policy never selects LOW/MID/HIGH itself.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        residual_limit: float = 0.20,
        posture_scales: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or residual_limit <= 0:
            raise ValueError("obs_dim and residual_limit must be positive")
        sizes = [int(obs_dim), *[int(v) for v in hidden_sizes], 8]
        layers: list[nn.Module] = []
        for in_size, out_size in zip(sizes[:-2], sizes[1:-1]):
            layers.extend((nn.Linear(in_size, out_size), nn.Tanh()))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)
        # Phase B must begin as the exact frozen official controller. A
        # random residual at iteration zero invalidates that contract.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.residual_limit = float(residual_limit)
        self.log_std = nn.Parameter(torch.full((8,), -3.0))
        scale = torch.ones(8) if posture_scales is None else torch.as_tensor(tuple(posture_scales), dtype=torch.float32)
        if scale.shape != (8,) or torch.any(scale <= 0):
            raise ValueError("posture_scales must contain 8 positive values")
        self.register_buffer("posture_scales", scale)

    def forward(self, observation: torch.Tensor, posture_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if observation.shape[-1] != self.network[0].in_features:
            raise ValueError(f"expected last dimension {self.network[0].in_features}, got {observation.shape[-1]}")
        scale = self.posture_scales if posture_scale is None else posture_scale
        if scale.shape[-1] != 8:
            raise ValueError("posture_scale must end in dimension 8")
        residual = torch.tanh(self.network(observation)) * self.residual_limit
        return residual * scale

    def freeze(self) -> "ResidualPolicy":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def distribution(self, observation: torch.Tensor, posture_scale: Optional[torch.Tensor] = None) -> torch.distributions.Normal:
        """Unbounded Gaussian in residual space for PPO/NP3O rollouts."""
        mean = self.forward(observation, posture_scale)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def configure_exploration(
        self, log_std: float | None = None, *, trainable: bool = True
    ) -> "ResidualPolicy":
        """Configure rollout noise while preserving deterministic actor output."""
        if log_std is not None:
            value = float(log_std)
            if not math.isfinite(value) or not -5.0 <= value <= 1.0:
                raise ValueError("log_std must be finite and within [-5, 1]")
            with torch.no_grad():
                self.log_std.fill_(value)
        self.log_std.requires_grad_(trainable)
        return self

    def sample(self, observation: torch.Tensor, posture_scale: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation, posture_scale)
        raw = distribution.rsample()
        action = raw.clamp(-self.residual_limit, self.residual_limit)
        return action, distribution.log_prob(raw).sum(-1)


class FrozenBaseController:
    """Minimal base-policy adapter used by the prototype and smoke test."""

    def __init__(self, nominal_target: Optional[torch.Tensor] = None) -> None:
        self.nominal_target = torch.zeros(8) if nominal_target is None else nominal_target.detach().clone().float()
        if self.nominal_target.shape != (8,):
            raise ValueError("nominal_target must have shape (8,)")

    def __call__(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.nominal_target.to(device=device).expand(batch_size, -1).clone()


class OfficialBasePolicyAdapter:
    """Wrap an already-loaded official NP3O policy without changing its contract."""

    def __init__(self, policy: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.policy = policy

    @torch.no_grad()
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        target = self.policy(observation)
        if target.shape != (*observation.shape[:-1], 8):
            raise ValueError("official base policy must return (..., 8) joint targets")
        if not torch.isfinite(target).all():
            raise ValueError("official base policy returned non-finite targets")
        return target


def combine_targets(base_target: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Apply a residual without changing the official 8-D target contract."""

    if base_target.shape != residual.shape or base_target.shape[-1] != 8:
        raise ValueError("base_target and residual must have the same (..., 8) shape")
    return base_target + residual
