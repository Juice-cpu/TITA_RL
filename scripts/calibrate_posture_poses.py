#!/usr/bin/env python3
"""Find simulation-only LOW/MID/HIGH joint anchors from the official URDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--grid-size", type=int, default=11)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--settle-steps", type=int, default=150)
parser.add_argument("--ramp-steps", type=int, default=150)
parser.add_argument("--joint2-range", type=float, nargs=2, default=(0.2, 1.6))
parser.add_argument("--joint3-range", type=float, nargs=2, default=(-2.5, -0.75))
parser.add_argument("--max-stable-speed", type=float, default=0.20)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from tita_uc_residual_rl.isaaclab_tita_env import TitaResidualEnv, build_tita_cfg  # noqa: E402
from tita_uc_residual_rl.official_policy import (  # noqa: E402
    FrozenOfficialTitaController,
    OfficialTitaActor,
)


def main() -> None:
    if (
        args.grid_size < 2
        or args.steps <= args.settle_steps
        or args.ramp_steps < 1
        or args.max_stable_speed <= 0.0
    ):
        raise ValueError("grid-size must be at least two and steps must exceed settle-steps")
    joint2 = torch.linspace(*args.joint2_range, args.grid_size)
    joint3 = torch.linspace(*args.joint3_range, args.grid_size)
    pairs = torch.cartesian_prod(joint2, joint3)
    cfg = build_tita_cfg(args.asset, num_envs=len(pairs), terrain="flat")
    cfg.enforce_acceleration_limit = False
    cfg.reset_stabilization_steps = 25
    env = TitaResidualEnv(cfg)
    env.reset()
    official = FrozenOfficialTitaController(
        OfficialTitaActor.from_checkpoint(args.checkpoint, env.device)
    )
    official.reset(env.num_envs, env.device)
    default_target = env.default_joint_target().clone()
    target = default_target.clone()
    target[:, 1] = pairs[:, 0].to(env.device)
    target[:, 2] = pairs[:, 1].to(env.device)
    target[:, 5] = pairs[:, 0].to(env.device)
    target[:, 6] = pairs[:, 1].to(env.device)
    zero = torch.zeros(env.num_envs, 8, device=env.device)
    zero_command = torch.zeros(env.num_envs, 3, device=env.device)
    terminated_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    height_sum = torch.zeros(env.num_envs, device=env.device)
    roll_absmax = torch.zeros(env.num_envs, device=env.device)
    pitch_absmax = torch.zeros(env.num_envs, device=env.device)
    speed_absmax = torch.zeros(env.num_envs, device=env.device)
    samples = 0

    for step in range(args.steps):
        env.set_velocity_command(zero_command)
        ramp = min((step + 1) / args.ramp_steps, 1.0)
        assert official.previous_policy_action is not None
        proprioception = env.official_proprioception(official.previous_policy_action)
        official_target, _ = official.target(proprioception, default_target)
        env.set_base_target(official_target + ramp * (target - default_target))
        _, _, terminated, truncated, _ = env.step(zero)
        official.reset_envs(terminated | truncated)
        terminated_count += terminated.long()
        if step >= args.settle_steps:
            gravity = env.robot.data.projected_gravity_b
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
            pitch = torch.atan2(
                -gravity[:, 0],
                torch.sqrt(torch.square(gravity[:, 1]) + torch.square(gravity[:, 2])),
            ).abs()
            speed = torch.linalg.vector_norm(env.robot.data.root_lin_vel_b[:, :2], dim=1)
            height_sum += env.posture_height
            roll_absmax = torch.maximum(roll_absmax, roll)
            pitch_absmax = torch.maximum(pitch_absmax, pitch)
            speed_absmax = torch.maximum(speed_absmax, speed)
            samples += 1

    height = height_sum / max(samples, 1)
    stable = (
        (terminated_count == 0)
        & (roll_absmax <= 0.12)
        & (pitch_absmax <= 0.12)
        & (speed_absmax <= args.max_stable_speed)
        & torch.isfinite(height)
    )
    records = []
    for name, desired_height in zip(("LOW", "MID", "HIGH"), cfg.posture_heights):
        error = torch.abs(height - desired_height)
        error = torch.where(stable, error, torch.full_like(error, torch.inf))
        index = int(torch.argmin(error).item())
        if not torch.isfinite(error[index]):
            records.append(
                {
                    "name": name,
                    "target_height_m": desired_height,
                    "status": "no_stable_candidate",
                }
            )
            continue
        q2 = float(pairs[index, 0])
        q3 = float(pairs[index, 1])
        records.append(
            {
                "name": name,
                "target_height_m": desired_height,
                "measured_height_m": float(height[index]),
                "absolute_error_m": float(error[index]),
                "joint_angles_rad": [0.0, q2, q3, 0.0, 0.0, q2, q3, 0.0],
                "roll_absmax_rad": float(roll_absmax[index]),
                "pitch_absmax_rad": float(pitch_absmax[index]),
                "speed_absmax_mps": float(speed_absmax[index]),
                "termination_count": int(terminated_count[index]),
            }
        )
    result = {
        "source": "Isaac Lab pose sweep using official DDTRobot/tita_rl URDF",
        "hardware_validated": False,
        "grid_size": args.grid_size,
        "candidate_count": len(pairs),
        "stable_candidate_count": int(stable.sum()),
        "max_stable_speed_mps": args.max_stable_speed,
        "minimum_termination_count": int(terminated_count.min()),
        "maximum_termination_count": int(terminated_count.max()),
        "minimum_speed_absmax_mps": float(speed_absmax.min()),
        "poses": records,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(encoded + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
