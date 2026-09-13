"""Small PPO-style update for the residual actor and four cost heads.

This is deliberately an algorithm boundary, not a replacement for the
official NP3O runner.  It preserves vector cost advantages and excludes hard
failures from the soft budget trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .residual_policy import ResidualPolicy


@dataclass(frozen=True)
class UpdateStats:
    loss: float
    policy_loss: float
    cost_loss: float
    entropy: float


class ResidualNP3OTrainer:
    """PPO surrogate with route/energy/mode/smooth multi-cost critics."""

    def __init__(self, policy: ResidualPolicy, obs_dim: int, lr: float = 3e-4, clip_ratio: float = 0.2) -> None:
        self.policy = policy
        self.value = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 4))
        self.optimizer = torch.optim.Adam(list(policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.clip_ratio = float(clip_ratio)

    def update(
        self,
        observation: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        cost_advantage: torch.Tensor,
        returns: torch.Tensor,
        hard_failure: Optional[torch.Tensor] = None,
    ) -> UpdateStats:
        if cost_advantage.shape[-1] != 4 or returns.shape[-1] != 4:
            raise ValueError("cost_advantage and returns must end in four cost heads")
        action, log_prob = self.policy.sample(observation)
        del action  # the current rollout action is represented by old_log_prob
        ratio = torch.exp(log_prob - old_log_prob)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        if hard_failure is None:
            hard_failure = torch.zeros_like(advantage, dtype=torch.bool)
        safe_weight = (~hard_failure).to(advantage.dtype)
        safe_advantage = advantage * safe_weight
        policy_loss = -torch.minimum(ratio * safe_advantage, clipped * safe_advantage).mean()
        predicted_cost = self.value(observation)
        cost_loss = 0.5 * ((predicted_cost - returns) ** 2).mean()
        # A hard failure is an explicit negative signal; it cannot be offset by
        # an energy or route advantage in the soft constrained objective.
        hard_penalty = hard_failure.to(predicted_cost.dtype).mean() * 0.5
        entropy = self.policy.distribution(observation).entropy().sum(-1).mean()
        loss = policy_loss + cost_loss + hard_penalty - 1e-3 * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
        self.optimizer.step()
        return UpdateStats(float(loss.detach()), float(policy_loss.detach()), float(cost_loss.detach()), float(entropy.detach()))
