#!/usr/bin/env python3
"""Scan simulation-only crouch/extend pulses on one measured-height step."""

from __future__ import annotations

import argparse
import itertools
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--residual-checkpoint", type=Path, default=None)
parser.add_argument("--residual-only", action="store_true")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=211)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trigger-distances", type=float, nargs="+", default=(0.08, 0.12, 0.16))
parser.add_argument("--command-speeds", type=float, nargs="+", default=(0.10, 0.20, 0.30))
parser.add_argument("--pulse-command-speeds", type=float, nargs="+", default=())
parser.add_argument("--crouch-command-speeds", type=float, nargs="+", default=())
parser.add_argument("--extension-command-speeds", type=float, nargs="+", default=())
parser.add_argument("--crouch-steps", type=int, nargs="+", default=(6, 10))
parser.add_argument("--extension-steps", type=int, nargs="+", default=(12,))
parser.add_argument("--hip-extension-delay-steps", type=int, nargs="+", default=(0,))
parser.add_argument("--knee-extension-delay-steps", type=int, nargs="+", default=(0,))
parser.add_argument("--hip-crouch-rad", type=float, nargs="+", default=(0.15,))
parser.add_argument("--hip-extension-rad", type=float, nargs="+", default=(0.30,))
parser.add_argument("--knee-crouch-rad", type=float, nargs="+", default=(0.70,))
parser.add_argument("--knee-extension-rad", type=float, nargs="+", default=(0.80,))
parser.add_argument("--summary-only", action="store_true")
parser.add_argument("--stop-after-level", action="store_true")
parser.add_argument("--stable-hold-steps", type=int, default=25)
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
from tita_uc_residual_rl.official_policy import (  # noqa: E402
    FrozenOfficialTitaController,
    OfficialTitaActor,
)
from tita_uc_residual_rl.phase_b_learner import (  # noqa: E402
    PhaseBActorCritic,
    load_phase_b_state_dict_compatible,
)


def main() -> None:
    print("[TITA-UC][probe] building candidate grid", flush=True)
    if args.steps <= 0 or args.stable_hold_steps <= 0:
        raise ValueError("steps and stable hold steps must be positive")
    numeric_grids = (
        args.trigger_distances,
        args.command_speeds,
        args.hip_crouch_rad,
        args.hip_extension_rad,
        args.knee_crouch_rad,
        args.knee_extension_rad,
    )
    if any(value <= 0.0 for grid in numeric_grids for value in grid):
        raise ValueError("motion-grid distances, speeds, and amplitudes must be positive")
    phase_command_grids = (
        args.pulse_command_speeds,
        args.crouch_command_speeds,
        args.extension_command_speeds,
    )
    if any(value < 0.0 for grid in phase_command_grids for value in grid):
        raise ValueError("pulse and phase command speeds must be non-negative")
    if any(value <= 0 for value in (*args.crouch_steps, *args.extension_steps)):
        raise ValueError("motion-grid durations must be positive")
    if any(
        value < 0
        for value in (
            *args.hip_extension_delay_steps,
            *args.knee_extension_delay_steps,
        )
    ):
        raise ValueError("extension delays must be non-negative")
    pulse_command_speeds = args.pulse_command_speeds or (None,)
    crouch_command_speeds = args.crouch_command_speeds or (None,)
    extension_command_speeds = args.extension_command_speeds or (None,)
    candidates = list(
        itertools.product(
            args.trigger_distances,
            args.command_speeds,
            args.crouch_steps,
            args.extension_steps,
            args.hip_crouch_rad,
            args.hip_extension_rad,
            args.knee_crouch_rad,
            args.knee_extension_rad,
            pulse_command_speeds,
            args.hip_extension_delay_steps,
            args.knee_extension_delay_steps,
            crouch_command_speeds,
            extension_command_speeds,
        )
    )
    cfg = build_tita_cfg(args.asset, num_envs=len(candidates), terrain="stairs")
    cfg.seed = args.seed
    cfg.episode_length_s = 30.0
    cfg.stair_step_height_range_m = (0.17, 0.17)
    cfg.stair_tread_range_m = (0.32, 0.32)
    cfg.stair_width_range_m = (1.0, 1.0)
    cfg.stair_num_steps = 1
    cfg.stair_start_x_m = 0.55
    cfg.stair_landing_length_m = 0.80
    cfg.stair_handrail_enabled = True
    cfg.stair_terminate_on_success = False
    if args.residual_only:
        cfg.stair_jump_assist_enabled = True
        cfg.stair_jump_assist_trigger_distance_m = 0.245
        cfg.stair_jump_assist_crouch_steps = 8
        cfg.stair_jump_assist_extension_steps = 12
        cfg.stair_jump_assist_hip_crouch_rad = 0.30
        cfg.stair_jump_assist_hip_extension_rad = 0.60
        cfg.stair_jump_assist_knee_crouch_rad = 1.00
        cfg.stair_jump_assist_knee_extension_rad = 0.90
    cfg.residual_limit = 1.0
    cfg.posture_scale = 1.0
    cfg.residual_filter_alpha = 1.0
    cfg.enforce_acceleration_limit = True
    cfg.acceleration_limit = 500.0
    # Keep failed candidates in the same physical trajectory so an automatic
    # reset cannot be mistaken for jump height. Training still terminates on
    # these failures; this override is only for the feasibility probe.
    cfg.reset_stabilization_steps = 10000
    env = TitaResidualEnv(cfg)
    print("[TITA-UC][probe] environment constructed; resetting", flush=True)
    observation, _ = env.reset()
    print(
        f"[TITA-UC][probe] reset complete; app_running={simulation_app.is_running()}",
        flush=True,
    )
    if not args.headless:
        env.sim.set_camera_view((2.7, 2.7, 1.7), (0.6, 0.0, 0.45))

    print("[TITA-UC][probe] loading frozen official policy", flush=True)
    official = FrozenOfficialTitaController(
        OfficialTitaActor.from_checkpoint(args.checkpoint, env.device)
    )
    official.reset(env.num_envs, env.device)
    residual = None
    if args.residual_checkpoint is not None:
        payload = torch.load(
            args.residual_checkpoint.expanduser(), map_location="cpu", weights_only=True
        )
        policy_config = payload.get("policy_config", {})
        residual = PhaseBActorCritic(
            obs_dim=observation["policy"].shape[-1],
            residual_limit=float(policy_config.get("residual_limit", cfg.residual_limit)),
            posture_scale=float(policy_config.get("posture_scale", cfg.posture_scale)),
        ).to(env.device)
        load_phase_b_state_dict_compatible(residual, payload["model_state_dict"])
        residual.eval()
        print(
            f"[TITA-UC][probe] loading residual checkpoint {args.residual_checkpoint}",
            flush=True,
        )
    high = torch.nn.functional.one_hot(
        torch.full((env.num_envs,), 2, device=env.device), 3
    ).to(torch.float32)
    zero_command = torch.zeros(env.num_envs, 3, device=env.device)
    forward_command = zero_command.clone()
    forward_command[:, 0] = torch.tensor(
        [v[1] for v in candidates], device=env.device
    )
    pulse_command = zero_command.clone()
    pulse_command[:, 0] = torch.tensor(
        [v[1] if v[8] is None else v[8] for v in candidates], device=env.device
    )
    crouch_command = zero_command.clone()
    crouch_command[:, 0] = torch.tensor(
        [
            (v[1] if v[8] is None else v[8]) if v[11] is None else v[11]
            for v in candidates
        ],
        device=env.device,
    )
    extension_command = zero_command.clone()
    extension_command[:, 0] = torch.tensor(
        [
            (v[1] if v[8] is None else v[8]) if v[12] is None else v[12]
            for v in candidates
        ],
        device=env.device,
    )
    zero_action = torch.zeros(env.num_envs, 8, device=env.device)

    print("[TITA-UC][probe] stabilizing", flush=True)
    for stabilization_step in range(150):
        if not simulation_app.is_running():
            raise RuntimeError(
                "Isaac Sim stopped during probe stabilization at "
                f"step {stabilization_step}"
            )
        env.set_velocity_command(zero_command)
        env.set_posture_command(high)
        assert official.previous_policy_action is not None
        proprioception = env.official_proprioception(official.previous_policy_action)
        target, _ = official.target(proprioception, env.default_joint_target())
        env.set_base_target(target)
        observation, _, terminated, truncated, _ = env.step(zero_action)
        official.reset_envs(terminated | truncated)

    baseline_root_z = env.robot.data.root_pos_w[:, 2].clone()
    max_root_z = baseline_root_z.clone()
    max_vertical_speed = torch.zeros(env.num_envs, device=env.device)
    max_local_x = torch.full((env.num_envs,), float("-inf"), device=env.device)
    max_verified_level = torch.zeros(env.num_envs, device=env.device)
    termination_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    raw_collision_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    max_base_contact_force = torch.zeros(env.num_envs, device=env.device)
    first_collision_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    first_collision_x = torch.full((env.num_envs,), float("nan"), device=env.device)
    first_collision_z = torch.full((env.num_envs,), float("nan"), device=env.device)
    first_collision_pitch = torch.full(
        (env.num_envs,), float("nan"), device=env.device
    )
    max_abs_pitch_before_collision = torch.zeros(env.num_envs, device=env.device)
    first_trigger_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    first_trigger_x = torch.full((env.num_envs,), float("nan"), device=env.device)
    first_trigger_z = torch.full((env.num_envs,), float("nan"), device=env.device)
    first_level_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    first_platform_step = torch.full_like(first_level_step, -1)
    collision_steps_before_level = torch.zeros_like(raw_collision_steps)
    collision_steps_before_platform = torch.zeros_like(raw_collision_steps)
    post_level_collision_steps = torch.zeros_like(raw_collision_steps)
    stable_hold_steps = torch.zeros_like(raw_collision_steps)
    max_stable_hold_steps = torch.zeros_like(raw_collision_steps)
    max_abs_roll = torch.zeros(env.num_envs, device=env.device)
    max_abs_pitch = torch.zeros(env.num_envs, device=env.device)
    reason_counts = torch.zeros(
        env.num_envs, len(HARD_REASON_NAMES), dtype=torch.long, device=env.device
    )
    jump_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    trigger_distance = torch.tensor([v[0] for v in candidates], device=env.device)
    crouch_steps = torch.tensor([v[2] for v in candidates], device=env.device)
    extension_steps = torch.tensor([v[3] for v in candidates], device=env.device)
    hip_crouch = torch.tensor([v[4] for v in candidates], device=env.device)
    hip_extension = torch.tensor([v[5] for v in candidates], device=env.device)
    knee_crouch = torch.tensor([v[6] for v in candidates], device=env.device)
    knee_extension = torch.tensor([v[7] for v in candidates], device=env.device)
    hip_extension_delay = torch.tensor([v[9] for v in candidates], device=env.device)
    knee_extension_delay = torch.tensor([v[10] for v in candidates], device=env.device)

    print("[TITA-UC][probe] scanning jump candidates", flush=True)
    for scan_step in range(args.steps):
        if scan_step % 25 == 0:
            print(
                f"[TITA-UC][probe] scan step {scan_step}/{args.steps}; "
                f"active={int(active.sum())}",
                flush=True,
            )
        if not simulation_app.is_running():
            raise RuntimeError(
                f"Isaac Sim stopped during candidate scan at step {scan_step}"
            )
        if not active.any():
            break
        newly_triggered = active & (jump_step < 0) & (
            env._clearance <= trigger_distance
        )
        trigger_local_x = (
            env.robot.data.root_pos_w[:, 0] - env._environment_origins()[:, 0]
        )
        first_trigger_step[newly_triggered] = scan_step
        first_trigger_x[newly_triggered] = trigger_local_x[newly_triggered]
        first_trigger_z[newly_triggered] = env.robot.data.root_pos_w[
            newly_triggered, 2
        ]
        jump_step[newly_triggered] = 0
        action = (
            zero_action.clone()
            if residual is None
            else residual.policy(observation["policy"])
        )
        crouching = active & (jump_step >= 0) & (jump_step < crouch_steps)
        hip_extending = active & (
            jump_step >= crouch_steps + hip_extension_delay
        ) & (
            jump_step < crouch_steps + hip_extension_delay + extension_steps
        )
        knee_extending = active & (
            jump_step >= crouch_steps + knee_extension_delay
        ) & (
            jump_step < crouch_steps + knee_extension_delay + extension_steps
        )
        pulse_end = (
            crouch_steps
            + torch.maximum(hip_extension_delay, knee_extension_delay)
            + extension_steps
        )
        pulsing = active & (jump_step >= 0) & (jump_step < pulse_end)
        extending = hip_extending | knee_extending
        reached_level = first_level_step >= 0
        reached_platform = first_platform_step >= 0
        moving = active & (~reached_platform | (not args.stop_after_level))
        command = torch.where(moving[:, None], forward_command, zero_command)
        command = torch.where((moving & pulsing)[:, None], pulse_command, command)
        command = torch.where((moving & crouching)[:, None], crouch_command, command)
        command = torch.where(
            (moving & extending)[:, None], extension_command, command
        )
        env.set_velocity_command(command)
        env.set_posture_command(high)
        assert official.previous_policy_action is not None
        proprioception = env.official_proprioception(official.previous_policy_action)
        target, _ = official.target(proprioception, env.default_joint_target())
        env.set_base_target(target)

        if torch.any(extending & crouching):
            raise RuntimeError("crouch and extension phases must not overlap")
        if torch.any(hip_extending & ~pulsing) or torch.any(knee_extending & ~pulsing):
            raise RuntimeError("extension phase escaped the pulse command window")
        if not args.residual_only:
            hip = torch.zeros(env.num_envs, device=env.device)
            knee = torch.zeros(env.num_envs, device=env.device)
            hip[crouching] = -hip_crouch[crouching]
            hip[hip_extending] = hip_extension[hip_extending]
            knee[crouching] = -knee_crouch[crouching]
            knee[knee_extending] = knee_extension[knee_extending]
            action[:, 1] += hip
            action[:, 2] += knee
            action[:, 5] += hip
            action[:, 6] += knee

        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        local_x = env.robot.data.root_pos_w[:, 0] - env._environment_origins()[:, 0]
        max_local_x = torch.maximum(max_local_x, local_x)
        max_root_z = torch.maximum(max_root_z, env.robot.data.root_pos_w[:, 2])
        max_vertical_speed = torch.maximum(
            max_vertical_speed, env.robot.data.root_lin_vel_w[:, 2]
        )
        max_verified_level = torch.maximum(max_verified_level, env._verified_stair_level())
        gravity = env.robot.data.projected_gravity_b
        roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
        pitch = torch.atan2(
            -gravity[:, 0],
            torch.sqrt(torch.square(gravity[:, 1]) + torch.square(gravity[:, 2])),
        )
        max_abs_roll = torch.maximum(max_abs_roll, roll.abs())
        max_abs_pitch = torch.maximum(max_abs_pitch, pitch.abs())
        collision_not_seen = first_collision_step < 0
        max_abs_pitch_before_collision = torch.maximum(
            max_abs_pitch_before_collision,
            torch.where(collision_not_seen, pitch.abs(), torch.zeros_like(pitch)),
        )
        termination_counts += terminated.long()
        raw_collision_steps += env._collision.long()
        collision_steps_before_level += (env._collision & ~reached_level).long()
        collision_steps_before_platform += (env._collision & ~reached_platform).long()
        post_level_collision_steps += (env._collision & reached_level).long()
        just_collided = (first_collision_step < 0) & env._collision
        first_collision_step[just_collided] = scan_step
        first_collision_x[just_collided] = local_x[just_collided]
        first_collision_z[just_collided] = env.robot.data.root_pos_w[
            just_collided, 2
        ]
        first_collision_pitch[just_collided] = pitch[just_collided]
        newly_reached_level = (first_level_step < 0) & (
            env._verified_stair_level() >= 1
        )
        first_level_step[newly_reached_level] = scan_step
        platform_entry_x = cfg.stair_start_x_m + cfg.base_collision_half_length_m
        stair_end_x = (
            cfg.stair_start_x_m
            + cfg.stair_num_steps * env.stair_dimensions[:, 1]
            + cfg.stair_landing_length_m
        )
        fully_on_platform = (
            (env._verified_stair_level() >= 1)
            & (local_x >= platform_entry_x)
            & (local_x <= stair_end_x - cfg.base_collision_half_length_m)
        )
        newly_reached_platform = (first_platform_step < 0) & fully_on_platform
        first_platform_step[newly_reached_platform] = scan_step
        planar_speed = torch.linalg.vector_norm(
            env.robot.data.root_lin_vel_w[:, :2], dim=1
        )
        stable = (
            fully_on_platform
            & ~env._collision
            & (planar_speed <= 0.20)
            & (env.robot.data.root_lin_vel_w[:, 2].abs() <= 0.15)
            & (roll.abs() <= 0.25)
            & (pitch.abs() <= 0.25)
        )
        stable_hold_steps = torch.where(
            stable, stable_hold_steps + 1, torch.zeros_like(stable_hold_steps)
        )
        max_stable_hold_steps = torch.maximum(
            max_stable_hold_steps, stable_hold_steps
        )
        net_forces = env._contact_sensor.data.net_forces_w_history
        hard_contact = net_forces[:, :, env._hard_contact_body_ids]
        base_contact_force = torch.linalg.vector_norm(hard_contact, dim=-1).amax(
            dim=(1, 2)
        )
        max_base_contact_force = torch.maximum(
            max_base_contact_force, base_contact_force
        )
        reason_counts += env.hard_reason_flags.long()
        jump_step[(jump_step >= 0) & active] += 1
        active &= ~done
        official.reset_envs(done)

    print("[TITA-UC][probe] candidate scan complete; writing result", flush=True)
    records = []
    for index, candidate in enumerate(candidates):
        strict_success = (
            int(first_platform_step[index]) >= 0
            and int(raw_collision_steps[index]) == 0
            and int(max_stable_hold_steps[index]) >= args.stable_hold_steps
        )
        records.append(
            {
                "trigger_distance_m": candidate[0],
                "command_x_mps": candidate[1],
                "hip_crouch_rad": candidate[4],
                "hip_extension_rad": candidate[5],
                "knee_crouch_rad": candidate[6],
                "knee_extension_rad": candidate[7],
                "pulse_command_x_mps": (
                    candidate[1] if candidate[8] is None else candidate[8]
                ),
                "crouch_command_x_mps": (
                    (candidate[1] if candidate[8] is None else candidate[8])
                    if candidate[11] is None
                    else candidate[11]
                ),
                "extension_command_x_mps": (
                    (candidate[1] if candidate[8] is None else candidate[8])
                    if candidate[12] is None
                    else candidate[12]
                ),
                "hip_extension_delay_steps": candidate[9],
                "knee_extension_delay_steps": candidate[10],
                "crouch_steps": candidate[2],
                "extension_steps": candidate[3],
                "baseline_root_z_m": float(baseline_root_z[index]),
                "root_rise_max_m": float(max_root_z[index] - baseline_root_z[index]),
                "vertical_speed_max_mps": float(max_vertical_speed[index]),
                "local_x_max_m": float(max_local_x[index]),
                "verified_level_max": int(max_verified_level[index]),
                "termination_count": int(termination_counts[index]),
                "raw_collision_steps": int(raw_collision_steps[index]),
                "max_base_contact_force_n": float(max_base_contact_force[index]),
                "first_collision_step": int(first_collision_step[index]),
                "first_collision_local_x_m": float(first_collision_x[index]),
                "first_collision_root_z_m": float(first_collision_z[index]),
                "first_collision_pitch_rad": float(first_collision_pitch[index]),
                "max_abs_pitch_before_collision_rad": float(
                    max_abs_pitch_before_collision[index]
                ),
                "first_trigger_step": int(first_trigger_step[index]),
                "first_trigger_local_x_m": float(first_trigger_x[index]),
                "first_trigger_root_z_m": float(first_trigger_z[index]),
                "first_level_step": int(first_level_step[index]),
                "first_platform_step": int(first_platform_step[index]),
                "collision_steps_before_level": int(
                    collision_steps_before_level[index]
                ),
                "collision_steps_before_platform": int(
                    collision_steps_before_platform[index]
                ),
                "post_level_collision_steps": int(post_level_collision_steps[index]),
                "max_stable_hold_steps": int(max_stable_hold_steps[index]),
                "max_abs_roll_rad": float(max_abs_roll[index]),
                "max_abs_pitch_rad": float(max_abs_pitch[index]),
                "strict_success": strict_success,
                "hard_reason_counts": {
                    name: int(reason_counts[index, reason_index])
                    for reason_index, name in enumerate(HARD_REASON_NAMES)
                },
            }
        )
    result = {
        "step_height_m": 0.17,
        "tread_m": 0.32,
        "platform_end_m": (
            cfg.stair_start_x_m
            + cfg.stair_num_steps * cfg.stair_tread_range_m[0]
            + cfg.stair_landing_length_m
        ),
        "required_stable_hold_steps": args.stable_hold_steps,
        "candidate_count": len(records),
        "successful_candidates": sum(r["verified_level_max"] >= 1 for r in records),
        "collision_free_successful_candidates": sum(
            r["strict_success"]
            for r in records
        ),
        "maximum_root_rise_m": max(r["root_rise_max_m"] for r in records),
        "maximum_vertical_speed_mps": max(r["vertical_speed_max_mps"] for r in records),
        "records": records,
    }
    encoded = json.dumps(result, indent=2)
    if args.summary_only:
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    else:
        print(encoded)
    args.output.expanduser().resolve().write_text(encoded + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
