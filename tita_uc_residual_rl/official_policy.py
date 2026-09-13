"""Frozen official TITA actor and Isaac Lab observation/action adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


OFFICIAL_PROPRIO_DIM = 33
OFFICIAL_HISTORY_LENGTH = 10
OFFICIAL_ACTION_DIM = 8


def swap_leg_blocks(value: torch.Tensor) -> torch.Tensor:
    """Map official policy order to URDF order (and vice versa)."""
    if value.shape[-1] != 8:
        raise ValueError("TITA leg block mapping requires a final dimension of 8")
    return value[..., (4, 5, 6, 7, 0, 1, 2, 3)]


def build_official_proprioception(
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    velocity_command: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    default_joint_pos: torch.Tensor,
    previous_policy_action: torch.Tensor,
) -> torch.Tensor:
    """Reproduce ``LeggedRobot.compute_observations`` for its 33 actor values."""
    batch_shape = joint_pos.shape[:-1]
    expected3 = (*batch_shape, 3)
    expected8 = (*batch_shape, 8)
    for name, value in (
        ("base_ang_vel", base_ang_vel),
        ("projected_gravity", projected_gravity),
        ("velocity_command", velocity_command),
    ):
        if value.shape != expected3:
            raise ValueError(f"{name} must have shape {expected3}, got {tuple(value.shape)}")
    for name, value in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("default_joint_pos", default_joint_pos),
        ("previous_policy_action", previous_policy_action),
    ):
        if value.shape != expected8:
            raise ValueError(f"{name} must have shape {expected8}, got {tuple(value.shape)}")

    # The official environment masks wheel angles before forming the offset.
    masked_pos = joint_pos.clone()
    masked_pos[..., (3, 7)] = 0.0
    command_scale = velocity_command.new_tensor((2.0, 2.0, 0.25))
    proprio = torch.cat(
        (
            base_ang_vel * 0.25,
            projected_gravity,
            velocity_command * command_scale,
            swap_leg_blocks(masked_pos - default_joint_pos),
            swap_leg_blocks(joint_vel) * 0.05,
            previous_policy_action,
        ),
        dim=-1,
    )
    if proprio.shape[-1] != OFFICIAL_PROPRIO_DIM or not torch.isfinite(proprio).all():
        raise ValueError("official proprioception must be finite and 33-dimensional")
    return proprio.clamp(-100.0, 100.0)


class OfficialTitaActor(nn.Module):
    """Exact inference subset of the official ``MlpBarlowTwinsActor``."""

    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU
        self.mlp_encoder = nn.Sequential(
            nn.Linear(330, 512), activation(),
            nn.Linear(512, 256), activation(), nn.LayerNorm(256),
            nn.Linear(256, 128), activation(), nn.LayerNorm(128),
            nn.Linear(128, 26),
        )
        self.actor = nn.Sequential(
            nn.Linear(59, 512), activation(),
            nn.Linear(512, 256), activation(),
            nn.Linear(256, 128), activation(),
            nn.Linear(128, 8),
        )

    def forward(self, proprioception: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if proprioception.shape[-1] != OFFICIAL_PROPRIO_DIM:
            raise ValueError("official actor proprioception must have dimension 33")
        if history.shape != (*proprioception.shape[:-1], OFFICIAL_HISTORY_LENGTH, OFFICIAL_PROPRIO_DIM):
            raise ValueError("official actor history must have shape (..., 10, 33)")
        current_history = torch.cat((history[..., 1:, :], proprioception.unsqueeze(-2)), dim=-2)
        latent = self.mlp_encoder(current_history.flatten(start_dim=-2))
        action = self.actor(torch.cat((latent, proprioception), dim=-1))
        if not torch.isfinite(action).all():
            raise RuntimeError("official TITA actor produced a non-finite action")
        return action

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: torch.device | str = "cpu") -> "OfficialTitaActor":
        payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("official checkpoint must contain model_state_dict")
        source = payload["model_state_dict"]
        prefix = "actor_teacher_backbone."
        state = {
            key[len(prefix):]: value
            for key, value in source.items()
            if key.startswith(prefix + "mlp_encoder.") or key.startswith(prefix + "actor.")
        }
        model = cls()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"official actor state mismatch: missing={missing}, unexpected={unexpected}")
        model.requires_grad_(False).eval().to(device)
        return model


class FrozenOfficialTitaController:
    """Stateful history, reindexing, filtering, and target conversion."""

    def __init__(self, actor: OfficialTitaActor) -> None:
        self.actor = actor
        self.history: torch.Tensor | None = None
        self.previous_policy_action: torch.Tensor | None = None
        self.previous_sim_action: torch.Tensor | None = None
        self.fresh: torch.Tensor | None = None

    def reset(self, batch_size: int, device: torch.device | str) -> None:
        self.history = torch.zeros(batch_size, 10, 33, device=device)
        self.previous_policy_action = torch.zeros(batch_size, 8, device=device)
        self.previous_sim_action = torch.zeros(batch_size, 8, device=device)
        self.fresh = torch.ones(batch_size, dtype=torch.bool, device=device)

    def reset_envs(self, reset_mask: torch.Tensor) -> None:
        if self.history is None or self.previous_policy_action is None or self.previous_sim_action is None or self.fresh is None:
            raise RuntimeError("controller is not initialized")
        self.history[reset_mask] = 0.0
        self.previous_policy_action[reset_mask] = 0.0
        self.previous_sim_action[reset_mask] = 0.0
        self.fresh[reset_mask] = True

    @torch.no_grad()
    def target(self, proprioception: torch.Tensor, default_joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.history is None or self.previous_policy_action is None or self.previous_sim_action is None or self.fresh is None:
            self.reset(proprioception.shape[0], proprioception.device)
        assert self.history is not None and self.previous_policy_action is not None and self.previous_sim_action is not None and self.fresh is not None
        policy_action = self.actor(proprioception, self.history)
        sim_action = swap_leg_blocks(policy_action)
        filtered_action = 0.2 * self.previous_sim_action + 0.8 * sim_action
        target_scale = filtered_action.new_tensor((0.25, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.5))
        joint_target = default_joint_pos + filtered_action * target_scale
        shifted_history = torch.cat((self.history[:, 1:], proprioception.unsqueeze(1)), dim=1)
        repeated_current = proprioception.unsqueeze(1).expand(-1, OFFICIAL_HISTORY_LENGTH, -1)
        self.history = torch.where(self.fresh[:, None, None], repeated_current, shifted_history)
        self.fresh.fill_(False)
        self.previous_policy_action = policy_action
        self.previous_sim_action = sim_action
        return joint_target, policy_action
