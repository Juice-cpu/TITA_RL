#!/usr/bin/env python3
"""Run a short TITA-UC Isaac Lab task smoke on a GPU-enabled host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True, help="official 8-DOF TITA USD or URDF")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--checkpoint", default=None, help="official tita_example_10000.pt")
parser.add_argument(
    "--residual-checkpoint",
    default=None,
    help="trained Phase B residual checkpoint; requires --checkpoint",
)
parser.add_argument("--command-x", type=float, default=0.0)
parser.add_argument("--command-y", type=float, default=0.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--posture", choices=("LOW", "MID", "HIGH"), default="MID")
parser.add_argument("--metric-warmup-steps", type=int, default=100)
parser.add_argument("--episode-length-s", type=float, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--acceleration-limit", type=float, default=None)
parser.add_argument("--residual-filter-alpha", type=float, default=0.25)
parser.add_argument("--enforce-acceleration-limit", action="store_true")
parser.add_argument("--disable-projector", action="store_true")
parser.add_argument("--verbose-state", action="store_true")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument(
    "--terrain", choices=("flat", "rough_warmup", "tunnel", "stairs"), default="flat"
)
parser.add_argument("--tunnel-height-range", type=float, nargs=2, default=None)
parser.add_argument("--tunnel-width-range", type=float, nargs=2, default=None)
parser.add_argument("--tunnel-length-range", type=float, nargs=2, default=None)
parser.add_argument("--stair-height-range", type=float, nargs=2, default=None)
parser.add_argument("--stair-tread-range", type=float, nargs=2, default=None)
parser.add_argument("--stair-width-range", type=float, nargs=2, default=None)
parser.add_argument("--stair-num-steps", type=int, default=None)
parser.add_argument("--stair-start-x", type=float, default=None)
parser.add_argument("--stair-landing-length", type=float, default=None)
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
from tita_uc_residual_rl.phase_b_learner import (  # noqa: E402
    PhaseBActorCritic,
    load_phase_b_state_dict_compatible,
)


def main() -> None:
    torch.manual_seed(args.seed)
    cfg = build_tita_cfg(
        args.asset,
        num_envs=args.num_envs,
        terrain=args.terrain,
        domain_randomization=args.domain_randomization,
    )
    cfg.seed = args.seed
    if args.episode_length_s is not None:
        if args.episode_length_s <= 0.0:
            raise ValueError("--episode-length-s must be positive")
        cfg.episode_length_s = args.episode_length_s
    if args.tunnel_height_range is not None:
        cfg.tunnel_height_range_m = tuple(args.tunnel_height_range)
    if args.tunnel_width_range is not None:
        cfg.tunnel_width_range_m = tuple(args.tunnel_width_range)
    if args.tunnel_length_range is not None:
        cfg.tunnel_length_range_m = tuple(args.tunnel_length_range)
    if args.stair_height_range is not None:
        cfg.stair_step_height_range_m = tuple(args.stair_height_range)
    if args.stair_tread_range is not None:
        cfg.stair_tread_range_m = tuple(args.stair_tread_range)
    if args.stair_width_range is not None:
        cfg.stair_width_range_m = tuple(args.stair_width_range)
    if args.stair_num_steps is not None:
        if args.stair_num_steps <= 0:
            raise ValueError("--stair-num-steps must be positive")
        cfg.stair_num_steps = args.stair_num_steps
    if args.stair_start_x is not None:
        if args.stair_start_x <= 0.0:
            raise ValueError("--stair-start-x must be positive")
        cfg.stair_start_x_m = args.stair_start_x
    if args.stair_landing_length is not None:
        if args.stair_landing_length <= 0.0:
            raise ValueError("--stair-landing-length must be positive")
        cfg.stair_landing_length_m = args.stair_landing_length
    if args.acceleration_limit is not None:
        cfg.acceleration_limit = args.acceleration_limit
    cfg.residual_filter_alpha = args.residual_filter_alpha
    cfg.enforce_acceleration_limit = args.enforce_acceleration_limit
    cfg.enable_safety_projector = not args.disable_projector
    env = TitaResidualEnv(cfg)
    obs, _ = env.reset()
    if not args.headless:
        env.sim.set_camera_view((2.5, 2.5, 1.5), (0.0, 0.0, 0.3))
    base = None
    if args.checkpoint:
        actor = OfficialTitaActor.from_checkpoint(args.checkpoint, device=env.device)
        base = FrozenOfficialTitaController(actor)
        base.reset(env.num_envs, env.device)
        print(f"[TITA-UC] loaded frozen official checkpoint: {args.checkpoint}")
    residual = None
    if args.residual_checkpoint:
        if base is None:
            raise ValueError("--residual-checkpoint requires --checkpoint")
        payload = torch.load(
            args.residual_checkpoint, map_location="cpu", weights_only=True
        )
        policy_config = payload.get("policy_config", {})
        residual_limit = float(policy_config.get("residual_limit", cfg.residual_limit))
        posture_scale = float(policy_config.get("posture_scale", cfg.posture_scale))
        cfg.residual_limit = residual_limit
        cfg.posture_scale = posture_scale
        residual = PhaseBActorCritic(
            obs_dim=obs["policy"].shape[-1],
            residual_limit=residual_limit,
            posture_scale=posture_scale,
        ).to(env.device)
        load_phase_b_state_dict_compatible(residual, payload["model_state_dict"])
        residual.eval()
        print(f"[TITA-UC] loaded residual checkpoint: {args.residual_checkpoint}")
    command = torch.tensor(
        [args.command_x, args.command_y, args.command_yaw], device=env.device
    ).repeat(env.num_envs, 1)
    posture_index = ("LOW", "MID", "HIGH").index(args.posture)
    posture_command = torch.nn.functional.one_hot(
        torch.full((env.num_envs,), posture_index, device=env.device), 3
    ).to(torch.float32)
    posture_height_sum = 0.0
    posture_error_sum = 0.0
    posture_samples = 0
    cumulative_terminated = 0
    cumulative_reasons = torch.zeros(
        len(HARD_REASON_NAMES), dtype=torch.long, device=env.device
    )
    cumulative_velocity_joints = torch.zeros(8, dtype=torch.long, device=env.device)
    cumulative_velocity_state_projections = torch.zeros(
        8, dtype=torch.long, device=env.device
    )
    cumulative_infeasible_joints = torch.zeros(8, dtype=torch.long, device=env.device)
    cumulative_infeasible_causes = torch.zeros(4, dtype=torch.long, device=env.device)
    cumulative_projection_joints = torch.zeros(8, dtype=torch.long, device=env.device)
    cumulative_projection_causes = torch.zeros(4, dtype=torch.long, device=env.device)
    velocity_absmax = torch.zeros(8, device=env.device)
    preprojection_velocity_absmax = torch.zeros(8, device=env.device)
    acceleration_absmax = torch.zeros(8, device=env.device)
    base_action_absmax_by_joint = torch.zeros(8, device=env.device)
    clearance_min = float("inf")
    clearance_max = float("-inf")
    soft_contact_force_max = 0.0
    observations_finite = True
    failure_snapshot_printed = False
    safety_hold_steps = 0
    initial_tunnel_successes = int(env.tunnel_success_count.sum().item())
    initial_stair_successes = int(env.stair_success_count.sum().item())
    stair_local_x_max = torch.full((env.num_envs,), float("-inf"), device=env.device)
    stair_root_z_max = torch.full_like(stair_local_x_max, float("-inf"))
    stair_verified_level_max = torch.zeros(env.num_envs, device=env.device)
    raw_soft_cost_samples: list[torch.Tensor] = []
    for step in range(args.steps):
        if not simulation_app.is_running():
            break
        env.set_velocity_command(command)
        env.set_posture_command(posture_command)
        base_action_absmax = 0.0
        if base is not None:
            assert base.previous_policy_action is not None
            proprioception = env.official_proprioception(base.previous_policy_action)
            base_target, policy_action = base.target(proprioception, env.default_joint_target())
            env.set_base_target(base_target)
            base_action_absmax = policy_action.abs().max().item()
            base_action_absmax_by_joint = torch.maximum(
                base_action_absmax_by_joint, policy_action.abs().amax(0)
            )
        with torch.no_grad():
            action = (
                torch.zeros(env.num_envs, 8, device=env.device)
                if residual is None
                else residual.policy(obs["policy"])
            )
        if cfg.stair_enabled:
            stair_local_x = (
                env.robot.data.root_pos_w[:, 0] - env._environment_origins()[:, 0]
            )
            stair_local_x_max = torch.maximum(stair_local_x_max, stair_local_x)
            stair_root_z_max = torch.maximum(
                stair_root_z_max, env.robot.data.root_pos_w[:, 2]
            )
            stair_verified_level_max = torch.maximum(
                stair_verified_level_max, env._verified_stair_level()
            )
        obs, reward, terminated, truncated, _ = env.step(action)
        if step >= args.metric_warmup_steps:
            posture_height_sum += float(env.posture_height.sum())
            posture_error_sum += float(
                torch.abs(env.posture_height - env.posture_target_height).sum()
            )
            posture_samples += env.num_envs
            raw_soft_cost_samples.append(env.raw_soft_costs.clone())
        observations_finite &= bool(torch.isfinite(obs["policy"]).all().item())
        clearance_min = min(clearance_min, float(env.clearance.min().item()))
        clearance_max = max(clearance_max, float(env.clearance.max().item()))
        soft_contact_force_max = max(
            soft_contact_force_max, float(env.soft_contact_force.max().item())
        )
        cumulative_terminated += int(terminated.sum().item())
        safety_hold_steps += int(env.safety_hold.sum().item())
        cumulative_reasons += env.hard_reason_flags.sum(0)
        cumulative_velocity_joints += env.velocity_violation_flags.sum(0)
        cumulative_velocity_state_projections += (
            env.velocity_state_projection_flags.sum(0)
        )
        cumulative_infeasible_joints += env.infeasible_joint_flags.sum(0)
        cumulative_infeasible_causes += env.infeasible_cause_flags.sum((0, 1))
        cumulative_projection_joints += env.projection_joint_flags.sum(0)
        cumulative_projection_causes += env.projection_cause_flags.sum((0, 1))
        velocity_absmax = torch.maximum(velocity_absmax, env.joint_velocity_absmax())
        preprojection_velocity_absmax = torch.maximum(
            preprojection_velocity_absmax,
            env.preprojection_joint_velocity_absmax(),
        )
        acceleration_absmax = torch.maximum(acceleration_absmax, env.joint_acceleration_absmax())
        if base is not None:
            base.reset_envs(terminated | truncated)
        if args.verbose_state and terminated.any() and not failure_snapshot_printed:
            failed = torch.nonzero(terminated, as_tuple=False).flatten()
            terrain_types = getattr(env.terrain, "terrain_types", None)
            terrain_levels = getattr(env.terrain, "terrain_levels", None)
            for env_id in failed.tolist():
                qd = env.robot.data.joint_vel[env_id, env._joint_ids]
                qdd = env.robot.data.joint_acc[env_id, env._joint_ids]
                print(
                    f"[TITA-UC FAILURE] env={env_id} "
                    f"terrain_type={None if terrain_types is None else int(terrain_types[env_id])} "
                    f"terrain_level={None if terrain_levels is None else int(terrain_levels[env_id])} "
                    f"origin={[round(v, 4) for v in env._environment_origins()[env_id].tolist()]} "
                    f"root={[round(v, 4) for v in env.robot.data.root_pos_w[env_id].tolist()]} "
                    f"preview_clearance={float(env.clearance[env_id]):.4f} "
                    f"hard_clearance={float(env.hard_clearance[env_id]):.4f} "
                    f"reasons={env.hard_reason_flags[env_id].tolist()} "
                    f"qd={[round(v, 3) for v in qd.tolist()]} "
                    f"qdd={[round(v, 3) for v in qdd.tolist()]}"
                )
            failure_snapshot_printed = True
        if step == 0 or (step + 1) % 20 == 0:
            print(
                f"[TITA-UC] step={step + 1} reward_mean={reward.mean().item():.6f} "
                f"terminated={terminated.sum().item()} truncated={truncated.sum().item()} "
                f"obs_shape={tuple(obs['policy'].shape)} base_action_absmax={base_action_absmax:.6f} "
                f"projected={env.projected.sum().item()} cumulative_terminated={cumulative_terminated} "
                f"hard_reasons={cumulative_reasons.tolist()} velocity_joints={cumulative_velocity_joints.tolist()} "
                f"velocity_state_projections={cumulative_velocity_state_projections.tolist()} "
                f"infeasible_joints={cumulative_infeasible_joints.tolist()} "
                f"infeasible_causes={cumulative_infeasible_causes.tolist()} "
                f"projection_joints={cumulative_projection_joints.tolist()} "
                f"projection_causes={cumulative_projection_causes.tolist()} "
                f"qd_absmax={[round(value, 3) for value in velocity_absmax.tolist()]} "
                f"qd_preprojection_absmax={[round(value, 3) for value in preprojection_velocity_absmax.tolist()]} "
                f"qdd_absmax={[round(value, 3) for value in acceleration_absmax.tolist()]} "
                f"base_action_by_joint={[round(value, 3) for value in base_action_absmax_by_joint.tolist()]}"
            )
            if args.verbose_state:
                q = env.robot.data.joint_pos[0, env._joint_ids]
                qd = env.robot.data.joint_vel[0, env._joint_ids]
                tau = env.robot.data.applied_torque[0, env._joint_ids]
                root_z = env.robot.data.root_pos_w[:, 2]
                print(
                    f"[TITA-UC STATE] root_z_minmax=({root_z.min().item():.4f},"
                    f"{root_z.max().item():.4f}) q={[round(v, 3) for v in q.tolist()]} "
                    f"qd={[round(v, 3) for v in qd.tolist()]} "
                    f"base_target={[round(v, 3) for v in env._base_target[0].tolist()]} "
                    f"q_cmd={[round(v, 3) for v in env._q_cmd[0].tolist()]} "
                    f"tau={[round(v, 3) for v in tau.tolist()]}"
                )
    final_posture_height = env.posture_height
    stair_local_x_range = (
        (stair_local_x_max.min().item(), stair_local_x_max.max().item())
        if cfg.stair_enabled
        else None
    )
    stair_root_z_range = (
        (stair_root_z_max.min().item(), stair_root_z_max.max().item())
        if cfg.stair_enabled
        else None
    )
    stair_levels = stair_verified_level_max.tolist() if cfg.stair_enabled else None
    raw_costs = (
        torch.cat(raw_soft_cost_samples, dim=0)
        if raw_soft_cost_samples
        else torch.zeros(1, 4, device=env.device)
    )
    cost_names = ("route", "energy", "mode", "smooth")
    summary = {
        "observations_finite": observations_finite,
        "posture": args.posture,
        "posture_height_mean_m": posture_height_sum / max(posture_samples, 1),
        "posture_height_abs_error_mean_m": posture_error_sum / max(posture_samples, 1),
        "cumulative_terminated": cumulative_terminated,
        "tunnel_success_total": int(env.tunnel_success_count.sum().item()) - initial_tunnel_successes,
        "stair_success_total": int(env.stair_success_count.sum().item()) - initial_stair_successes,
        "stair_local_x_max_minmax_m": stair_local_x_range,
        "stair_root_z_max_minmax_m": stair_root_z_range,
        "stair_verified_level_max": stair_levels,
        "stair_geometry": {
            "step_height_range_m": list(cfg.stair_step_height_range_m),
            "tread_range_m": list(cfg.stair_tread_range_m),
            "width_range_m": list(cfg.stair_width_range_m),
            "num_steps": cfg.stair_num_steps,
            "start_x_m": cfg.stair_start_x_m,
            "landing_length_m": cfg.stair_landing_length_m,
        } if cfg.stair_enabled else None,
        "hard_reasons": cumulative_reasons.tolist(),
        "velocity_joint_counts": cumulative_velocity_joints.tolist(),
        "velocity_state_projection_counts": cumulative_velocity_state_projections.tolist(),
        "safety_hold_steps": safety_hold_steps,
        "seed": args.seed,
        "steps": args.steps,
        "episode_length_s": cfg.episode_length_s,
        "residual_checkpoint": args.residual_checkpoint,
        "raw_cost_mean": {
            name: float(raw_costs[:, index].mean())
            for index, name in enumerate(cost_names)
        },
        "raw_cost_p95": {
            name: float(torch.quantile(raw_costs[:, index], 0.95))
            for index, name in enumerate(cost_names)
        },
    }
    print(
        f"[TITA-UC SUMMARY] observations_finite={observations_finite} "
        f"clearance_minmax=({clearance_min:.4f},{clearance_max:.4f}) "
        f"clearance_nonconstant={clearance_max - clearance_min > 1e-6} "
        f"soft_contact_force_max={soft_contact_force_max:.3f} "
        f"safety_hold_steps={safety_hold_steps} "
        f"posture={args.posture} "
        f"posture_height_mean_m={posture_height_sum / max(posture_samples, 1):.6f} "
        f"posture_height_abs_error_mean_m={posture_error_sum / max(posture_samples, 1):.6f} "
        f"posture_height_final_minmax_m=({final_posture_height.min().item():.6f},"
        f"{final_posture_height.max().item():.6f}) "
        f"cumulative_terminated={cumulative_terminated} "
        f"tunnel_success_total={int(env.tunnel_success_count.sum().item()) - initial_tunnel_successes} "
        f"stair_success_total={int(env.stair_success_count.sum().item()) - initial_stair_successes} "
        f"stair_local_x_max_minmax_m={stair_local_x_range} "
        f"stair_root_z_max_minmax_m={stair_root_z_range} "
        f"stair_verified_level_max={stair_levels} "
        f"hard_reasons={cumulative_reasons.tolist()}"
    )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[TITA-UC SUMMARY] output={output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
