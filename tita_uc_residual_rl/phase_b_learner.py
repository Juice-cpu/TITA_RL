"""Compact PPO/NP3O-style learner used by the Phase B Isaac Lab runner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_policy import ResidualPolicy


@dataclass(frozen=True)
class PhaseBUpdateStats:
    policy_loss: float
    constraint_loss: float
    reward_value_loss: float
    cost_value_loss: float
    entropy: float
    approximate_kl: float
    cost_mean: tuple[float, ...]
    cost_budget_violation: tuple[float, ...]
    cost_multipliers: tuple[float, ...]


class PhaseBActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 50, residual_limit: float = 0.20, posture_scale: float = 0.75) -> None:
        super().__init__()
        self.policy = ResidualPolicy(
            obs_dim,
            hidden_sizes=(256, 128),
            residual_limit=residual_limit,
            posture_scales=(posture_scale,) * 8,
        )
        self.reward_value = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
        )
        self.cost_value = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 4)
        )

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        return self.policy.distribution(observation)

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(-1),
            self.reward_value(observation).squeeze(-1),
            self.cost_value(observation),
        )


def load_phase_b_state_dict_compatible(
    model: PhaseBActorCritic, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load an older actor-critic after append-only observation expansion."""
    current = model.state_dict()
    adapted = dict(state_dict)
    input_weight_keys = (
        "policy.network.0.weight",
        "reward_value.0.weight",
        "cost_value.0.weight",
    )
    for key in input_weight_keys:
        old = adapted.get(key)
        new = current[key]
        if old is None or old.shape == new.shape:
            continue
        if old.ndim != 2 or old.shape[0] != new.shape[0] or old.shape[1] > new.shape[1]:
            raise ValueError(f"incompatible Phase B input weight for {key}: {old.shape} -> {new.shape}")
        expanded = new.clone()
        expanded.zero_()
        expanded[:, : old.shape[1]].copy_(old)
        adapted[key] = expanded
    model.load_state_dict(adapted)


def generalized_advantage(
    signal: torch.Tensor,
    value: torch.Tensor,
    last_value: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = torch.zeros_like(signal)
    running = torch.zeros_like(last_value)
    for index in reversed(range(signal.shape[0])):
        next_value = last_value if index == signal.shape[0] - 1 else value[index + 1]
        not_done = 1.0 - done[index].to(signal.dtype)
        while not_done.ndim < signal[index].ndim:
            not_done = not_done.unsqueeze(-1)
        delta = signal[index] + gamma * not_done * next_value - value[index]
        running = delta + gamma * lam * not_done * running
        advantage[index] = running
    return advantage, advantage + value


class PhaseBNP3O:
    def __init__(
        self,
        model: PhaseBActorCritic,
        learning_rate: float = 1e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.002,
        cost_multipliers: tuple[float, ...] = (0.40, 0.35, 0.25, 0.05),
        minimum_cost_multipliers: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        cost_budgets: tuple[float, ...] = (0.60, 0.221, 0.266, 0.212),
        dual_learning_rate: float = 0.01,
        cost_multiplier_max: float = 2.0,
    ) -> None:
        if (
            len(cost_multipliers) != 4
            or len(minimum_cost_multipliers) != 4
            or len(cost_budgets) != 4
        ):
            raise ValueError("Phase B requires exactly four cost multipliers, floors, and budgets")
        if any(value <= 0.0 for value in cost_budgets):
            raise ValueError("cost budgets must be positive")
        if any(value < 0.0 for value in minimum_cost_multipliers):
            raise ValueError("minimum cost multipliers must be non-negative")
        if any(
            initial < minimum
            for initial, minimum in zip(cost_multipliers, minimum_cost_multipliers)
        ):
            raise ValueError("initial cost multipliers must not be below their floors")
        if dual_learning_rate < 0.0 or cost_multiplier_max <= 0.0:
            raise ValueError("dual learning rate and multiplier maximum must be valid")
        if any(value > cost_multiplier_max for value in minimum_cost_multipliers):
            raise ValueError("minimum cost multipliers must not exceed the maximum")
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.cost_multipliers = torch.tensor(cost_multipliers, dtype=torch.float32)
        self.minimum_cost_multipliers = torch.tensor(
            minimum_cost_multipliers, dtype=torch.float32
        )
        self.cost_budgets = torch.tensor(cost_budgets, dtype=torch.float32)
        self.dual_learning_rate = dual_learning_rate
        self.cost_multiplier_max = cost_multiplier_max

    def update_dual(self, observed_cost_mean: torch.Tensor) -> torch.Tensor:
        """Projected dual ascent for the four normalized per-step cost budgets."""
        observed = torch.as_tensor(observed_cost_mean, dtype=torch.float32).detach().cpu()
        if observed.shape != self.cost_budgets.shape or not torch.isfinite(observed).all():
            raise ValueError("observed cost mean must be a finite four-vector")
        violation = (observed - self.cost_budgets) / self.cost_budgets.clamp_min(1e-8)
        self.cost_multipliers.add_(self.dual_learning_rate * violation)
        self.cost_multipliers.copy_(
            torch.maximum(self.cost_multipliers, self.minimum_cost_multipliers)
        ).clamp_(
            max=self.cost_multiplier_max
        )
        return violation

    def constraint_state_dict(self) -> dict[str, object]:
        return {
            "cost_multipliers": self.cost_multipliers.tolist(),
            "minimum_cost_multipliers": self.minimum_cost_multipliers.tolist(),
            "cost_budgets": self.cost_budgets.tolist(),
            "dual_learning_rate": self.dual_learning_rate,
            "cost_multiplier_max": self.cost_multiplier_max,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_prob: torch.Tensor,
        reward_advantage: torch.Tensor,
        reward_return: torch.Tensor,
        cost_advantage: torch.Tensor,
        cost_return: torch.Tensor,
        observed_cost_mean: torch.Tensor | None = None,
        epochs: int = 4,
        mini_batches: int = 4,
    ) -> PhaseBUpdateStats:
        flat_obs = observations.flatten(0, 1)
        flat_actions = actions.flatten(0, 1)
        flat_old_log_prob = old_log_prob.flatten()
        flat_reward_advantage = reward_advantage.flatten()
        flat_reward_return = reward_return.flatten()
        flat_cost_advantage = cost_advantage.flatten(0, 1)
        flat_cost_return = cost_return.flatten(0, 1)
        flat_reward_advantage = (flat_reward_advantage - flat_reward_advantage.mean()) / (flat_reward_advantage.std() + 1e-8)
        flat_cost_advantage = (flat_cost_advantage - flat_cost_advantage.mean(0)) / (flat_cost_advantage.std(0) + 1e-8)
        multipliers = self.cost_multipliers.to(flat_obs.device)
        count = flat_obs.shape[0]
        batch_size = max(1, count // mini_batches)
        totals = torch.zeros(6, device=flat_obs.device)
        updates = 0
        for _ in range(epochs):
            indices = torch.randperm(count, device=flat_obs.device)
            for start in range(0, count, batch_size):
                batch = indices[start:start + batch_size]
                distribution = self.model.distribution(flat_obs[batch])
                log_prob = distribution.log_prob(flat_actions[batch]).sum(-1)
                ratio = torch.exp(log_prob - flat_old_log_prob[batch])
                reward_surrogate = torch.minimum(
                    ratio * flat_reward_advantage[batch],
                    ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * flat_reward_advantage[batch],
                )
                cost_surrogate = torch.maximum(
                    ratio[:, None] * flat_cost_advantage[batch],
                    ratio[:, None].clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * flat_cost_advantage[batch],
                ).mean(0)
                constraint_loss = torch.sum(multipliers * cost_surrogate)
                policy_loss = -reward_surrogate.mean() + constraint_loss
                reward_value = self.model.reward_value(flat_obs[batch]).squeeze(-1)
                cost_value = self.model.cost_value(flat_obs[batch])
                reward_value_loss = 0.5 * torch.square(reward_value - flat_reward_return[batch]).mean()
                cost_value_loss = 0.5 * torch.square(cost_value - flat_cost_return[batch]).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss + reward_value_loss + 0.1 * cost_value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                approximate_kl = (flat_old_log_prob[batch] - log_prob).mean().abs()
                totals += torch.stack(
                    (
                        policy_loss.detach(), constraint_loss.detach(),
                        reward_value_loss.detach(), cost_value_loss.detach(),
                        entropy.detach(), approximate_kl.detach(),
                    )
                )
                updates += 1
        values = totals / max(1, updates)
        cost_mean = (
            flat_cost_return.new_zeros(4)
            if observed_cost_mean is None
            else torch.as_tensor(observed_cost_mean, device=flat_cost_return.device)
        )
        violation = (
            cost_mean.detach().cpu() - self.cost_budgets
        ) / self.cost_budgets.clamp_min(1e-8)
        if observed_cost_mean is not None:
            violation = self.update_dual(cost_mean)
        return PhaseBUpdateStats(
            *(float(value) for value in values),
            tuple(float(value) for value in cost_mean.detach().cpu()),
            tuple(float(value) for value in violation),
            tuple(float(value) for value in self.cost_multipliers),
        )
