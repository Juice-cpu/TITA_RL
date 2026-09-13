#!/usr/bin/env python3
"""Deterministic Phase B evaluation and offline P95 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True)
parser.add_argument("--official-checkpoint", required=True)
parser.add_argument("--residual-checkpoint", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--command-x", type=float, default=0.3)
parser.add_argument("--acceleration-limit", type=float, default=500.0)
parser.add_argument("--residual-filter-alpha", type=float, default=0.25)
parser.add_argument("--objective-p95", type=float, nargs=4, default=(1.0, 1.0, 1.0, 1.0))
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--terrain", choices=("flat", "rough_warmup"), default="flat")
parser.add_argument("--domain-randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from tita_uc_residual_rl.isaaclab_tita_env import (  # noqa: E402
    HARD_REASON_NAMES,
    TitaResidualEnv,
    build_tita_cfg,
)
from tita_uc_residual_rl.official_policy import FrozenOfficialTitaController, OfficialTitaActor  # noqa: E402
from tita_uc_residual_rl.phase_b_learner import PhaseBActorCritic  # noqa: E402


def main() -> None:
    torch.manual_seed(args.seed)
    cfg = build_tita_cfg(
        args.asset,
        num_envs=args.num_envs,
        terrain=args.terrain,
        domain_randomization=args.domain_randomization,
    )
    cfg.seed = args.seed
    cfg.acceleration_limit = args.acceleration_limit
    cfg.residual_filter_alpha = args.residual_filter_alpha
    cfg.enforce_acceleration_limit = True
    cfg.objective_p95 = tuple(args.objective_p95)
    env = TitaResidualEnv(cfg)
    observation, _ = env.reset()
    observation = observation["policy"]
    official = FrozenOfficialTitaController(
        OfficialTitaActor.from_checkpoint(args.official_checkpoint, env.device)
    )
    official.reset(env.num_envs, env.device)
    residual = None
    if args.residual_checkpoint is not None:
        payload = torch.load(
            args.residual_checkpoint.expanduser(), map_location="cpu", weights_only=True
        )
        residual = PhaseBActorCritic(obs_dim=observation.shape[-1]).to(env.device)
        residual.load_state_dict(payload["model_state_dict"])
        residual.eval()

    command = torch.tensor([args.command_x, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    reward_sum = 0.0
    tracking_error_sum = 0.0
    distance_total = 0.0
    energy_total = 0.0
    hard_failures = 0
    projected = 0
    velocity_state_projection_counts = torch.zeros(
        8, dtype=torch.long, device=env.device
    )
    qd_absmax = torch.zeros(8, device=env.device)
    qd_preprojection_absmax = torch.zeros(8, device=env.device)
    hard_reasons = torch.zeros(len(HARD_REASON_NAMES), dtype=torch.long, device=env.device)
    normalized_costs: list[torch.Tensor] = []
    raw_costs: list[torch.Tensor] = []
    smooth_components: list[torch.Tensor] = []
    for _ in range(args.steps):
        env.set_velocity_command(command)
        assert official.previous_policy_action is not None
        proprioception = env.official_proprioception(official.previous_policy_action)
        base_target, _ = official.target(proprioception, env.default_joint_target())
        env.set_base_target(base_target)
        with torch.no_grad():
            action = (
                torch.zeros(env.num_envs, 8, device=env.device)
                if residual is None
                else residual.policy(observation)
            )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        reward_sum += float(reward.sum())
        velocity_error = env.robot.data.root_lin_vel_b[:, :2] - command[:, :2]
        tracking_error_sum += float(torch.linalg.vector_norm(velocity_error, dim=1).sum())
        hard_failures += int(terminated.sum())
        projected += int(env.projected.sum())
        velocity_state_projection_counts += env.velocity_state_projection_flags.sum(0)
        qd_absmax = torch.maximum(qd_absmax, env.joint_velocity_absmax())
        qd_preprojection_absmax = torch.maximum(
            qd_preprojection_absmax,
            env.preprojection_joint_velocity_absmax(),
        )
        hard_reasons += env.hard_reason_flags.sum(0)
        normalized_costs.append(env.soft_costs.clone())
        raw_costs.append(env.raw_soft_costs.clone())
        smooth_components.append(env.smooth_components.clone())
        distance_total += float(env.last_step_distance.sum())
        energy_total += float(env.raw_soft_costs[:, 1].sum())
        official.reset_envs(done)
        observation = next_observation["policy"]

    normalized = torch.cat(normalized_costs, dim=0)
    raw = torch.cat(raw_costs, dim=0)
    smooth = torch.cat(smooth_components, dim=0)
    names = ("route", "energy", "mode", "smooth")
    samples = args.steps * env.num_envs
    result = {
        "controller": "official+residual" if residual is not None else "official",
        "residual_checkpoint": str(args.residual_checkpoint) if args.residual_checkpoint else None,
        "samples": samples,
        "reward_mean": reward_sum / samples,
        "tracking_error_mean": tracking_error_sum / samples,
        "cost_mean": {name: float(normalized[:, i].mean()) for i, name in enumerate(names)},
        "raw_cost_p95": {name: float(torch.quantile(raw[:, i], 0.95)) for i, name in enumerate(names)},
        "smooth_component_mean": {
            name: float(smooth[:, i].mean())
            for i, name in enumerate(
                (
                    "joint_target_change", "residual_rate",
                    "acceleration_ratio", "soft_contact_term",
                )
            )
        },
        "distance_mean_m": distance_total / env.num_envs,
        "energy_mean_j": energy_total / env.num_envs,
        "cot_mech_asset_mass": energy_total / max(29.0254 * 9.81 * distance_total, 1e-8),
        "projection_fraction": projected / samples,
        "velocity_state_projection_counts": velocity_state_projection_counts.tolist(),
        "qd_absmax": qd_absmax.tolist(),
        "qd_preprojection_absmax": qd_preprojection_absmax.tolist(),
        "hard_failures": hard_failures,
        "hard_reason_counts": hard_reasons.tolist(),
        "hard_reason_names": list(HARD_REASON_NAMES),
        "terrain": args.terrain,
        "domain_randomization": args.domain_randomization,
        "acceleration_limit": args.acceleration_limit,
        "residual_filter_alpha": args.residual_filter_alpha,
        "objective_p95": list(args.objective_p95),
        "seed": args.seed,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print("[TITA-UC EVAL] " + encoded)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(encoded + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
