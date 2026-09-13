#!/usr/bin/env python3
"""Run frozen-base Phase B residual training in Isaac Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--iterations", type=int, default=2)
parser.add_argument("--rollout-steps", type=int, default=24)
parser.add_argument("--command-x", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--acceleration-limit", type=float, required=True)
parser.add_argument("--residual-filter-alpha", type=float, default=0.25)
parser.add_argument(
    "--terrain", choices=("flat", "rough_warmup", "tunnel", "stairs"), default="flat"
)
parser.add_argument("--domain-randomization", action="store_true")
parser.add_argument("--initial-residual-checkpoint", type=Path, default=None)
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--cost-budgets", type=float, nargs=4, default=None)
parser.add_argument("--initial-cost-multipliers", type=float, nargs=4, default=None)
parser.add_argument("--minimum-cost-multipliers", type=float, nargs=4, default=None)
parser.add_argument("--dual-learning-rate", type=float, default=None)
parser.add_argument("--cost-multiplier-max", type=float, default=None)
parser.add_argument("--warmup-steps", type=int, default=None)
parser.add_argument("--checkpoint-interval", type=int, default=None)
parser.add_argument("--policy-log-std", type=float, default=None)
parser.add_argument("--freeze-policy-std", action="store_true")
parser.add_argument("--posture-curriculum", action="store_true")
parser.add_argument("--fixed-posture", choices=("LOW", "MID", "HIGH"), default=None)
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
    PhaseBNP3O,
    generalized_advantage,
    load_phase_b_state_dict_compatible,
)
from tita_uc_residual_rl.posture_curriculum import (  # noqa: E402
    POSTURE_NAMES,
    PostureCommandScheduler,
    PostureCurriculumConfig,
)
from tita_uc_residual_rl.stair_curriculum import StairCurriculumConfig  # noqa: E402
from tita_uc_residual_rl.tunnel_curriculum import TunnelCurriculumConfig  # noqa: E402


def main() -> None:
    torch.manual_seed(args.seed)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or not (run_dir / "config.json").is_file():
        raise ValueError("run-dir must be initialized by scripts/prepare_rl_run.py")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    p95_config = run_config["objective"]["p95"]
    objective_p95 = tuple(
        float(p95_config[name]) for name in ("route", "energy", "mode", "smooth")
    )
    if any(value <= 0.0 for value in objective_p95):
        raise ValueError("all frozen objective P95 values must be positive")
    constraint_config = run_config.get("constraint_optimizer", {})
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None
        else run_config.get("learning_rate", 1e-4)
    )
    cost_budgets = tuple(
        args.cost_budgets if args.cost_budgets is not None
        else constraint_config.get("cost_budgets", (0.60, 0.221, 0.266, 0.212))
    )
    initial_cost_multipliers = tuple(
        args.initial_cost_multipliers if args.initial_cost_multipliers is not None
        else constraint_config.get(
            "initial_cost_multipliers",
            (
                run_config["objective"]["route_weight"],
                run_config["objective"]["energy_weight"],
                run_config["objective"]["mode_weight"],
                run_config["objective"]["smooth_weight"],
            ),
        )
    )
    minimum_cost_multipliers = tuple(
        args.minimum_cost_multipliers if args.minimum_cost_multipliers is not None
        else constraint_config.get("minimum_cost_multipliers", initial_cost_multipliers)
    )
    dual_learning_rate = float(
        args.dual_learning_rate if args.dual_learning_rate is not None
        else constraint_config.get("dual_learning_rate", 0.01)
    )
    cost_multiplier_max = float(
        args.cost_multiplier_max if args.cost_multiplier_max is not None
        else constraint_config.get("cost_multiplier_max", 2.0)
    )
    warmup_steps = int(
        args.warmup_steps if args.warmup_steps is not None
        else constraint_config.get("warmup_steps", 50)
    )
    checkpoint_interval = int(
        args.checkpoint_interval if args.checkpoint_interval is not None
        else constraint_config.get("checkpoint_interval", 5)
    )
    if warmup_steps < 0 or checkpoint_interval <= 0:
        raise ValueError("warmup steps must be non-negative and checkpoint interval positive")

    cfg = build_tita_cfg(
        args.asset,
        num_envs=args.num_envs,
        terrain=args.terrain,
        domain_randomization=args.domain_randomization,
    )
    cfg.seed = args.seed
    cfg.episode_length_s = float(run_config.get("episode_length_s", cfg.episode_length_s))
    if cfg.episode_length_s <= 0.0:
        raise ValueError("episode length must be positive")
    cfg.acceleration_limit = args.acceleration_limit
    cfg.residual_filter_alpha = args.residual_filter_alpha
    cfg.residual_limit = float(run_config.get("residual_limit", cfg.residual_limit))
    cfg.posture_scale = float(run_config.get("posture_scale", cfg.posture_scale))
    if cfg.residual_limit <= 0.0 or cfg.posture_scale <= 0.0:
        raise ValueError("residual limit and posture scale must be positive")
    cfg.enforce_acceleration_limit = True
    cfg.objective_p95 = objective_p95
    cfg.normalized_cost_clip = float(run_config.get("normalized_cost_clip", float("inf")))
    if cfg.normalized_cost_clip <= 0.0:
        raise ValueError("normalized cost clip must be positive")
    tunnel_config = run_config.get("tunnel_curriculum", {})
    if args.terrain == "tunnel":
        validated_tunnel = TunnelCurriculumConfig(
            height_range_m=tuple(tunnel_config.get("net_height_range_m", (0.34, 0.44))),
            width_range_m=tuple(tunnel_config.get("net_width_range_m", (0.40, 0.65))),
            length_range_m=tuple(tunnel_config.get("length_range_m", (0.60, 1.50))),
            start_x_m=float(tunnel_config.get("start_x_m", 0.75)),
            wall_thickness_m=float(tunnel_config.get("wall_thickness_m", 0.10)),
        )
        cfg.tunnel_height_range_m = validated_tunnel.height_range_m
        cfg.tunnel_width_range_m = validated_tunnel.width_range_m
        cfg.tunnel_length_range_m = validated_tunnel.length_range_m
        cfg.tunnel_start_x_m = validated_tunnel.start_x_m
        cfg.tunnel_wall_thickness_m = validated_tunnel.wall_thickness_m
        cfg.tunnel_preview_distance_m = float(
            tunnel_config.get("preview_distance_m", cfg.tunnel_preview_distance_m)
        )
        cfg.tunnel_clearance_buffer_m = float(
            tunnel_config.get("clearance_buffer_m", cfg.tunnel_clearance_buffer_m)
        )
        cfg.tunnel_success_reward = float(
            tunnel_config.get("success_reward", cfg.tunnel_success_reward)
        )
        if cfg.tunnel_success_reward < 0.0:
            raise ValueError("tunnel success reward must be non-negative")
        cfg.scene.env_spacing = max(
            cfg.scene.env_spacing,
            validated_tunnel.start_x_m + validated_tunnel.length_range_m[1] + 2.0,
        )
    stair_config = run_config.get("stair_curriculum", {})
    if args.terrain == "stairs":
        validated_stairs = StairCurriculumConfig(
            step_height_range_m=tuple(
                stair_config.get("step_height_range_m", (0.025, 0.045))
            ),
            tread_range_m=tuple(stair_config.get("tread_range_m", (0.55, 0.67))),
            width_range_m=tuple(stair_config.get("width_range_m", (0.90, 1.20))),
            num_steps=int(stair_config.get("num_steps", 3)),
            start_x_m=float(stair_config.get("start_x_m", 0.75)),
            landing_length_m=float(stair_config.get("landing_length_m", 0.80)),
        )
        cfg.stair_step_height_range_m = validated_stairs.step_height_range_m
        cfg.stair_tread_range_m = validated_stairs.tread_range_m
        cfg.stair_width_range_m = validated_stairs.width_range_m
        cfg.stair_num_steps = validated_stairs.num_steps
        cfg.stair_start_x_m = validated_stairs.start_x_m
        cfg.stair_landing_length_m = validated_stairs.landing_length_m
        cfg.stair_handrail_enabled = bool(stair_config.get("handrail_enabled", False))
        cfg.stair_handrail_height_m = float(
            stair_config.get("handrail_height_m", cfg.stair_handrail_height_m)
        )
        cfg.stair_handrail_thickness_m = float(
            stair_config.get("handrail_thickness_m", cfg.stair_handrail_thickness_m)
        )
        cfg.stair_handrail_side_offset_m = float(
            stair_config.get("handrail_side_offset_m", cfg.stair_handrail_side_offset_m)
        )
        cfg.stair_handrail_post_spacing_m = float(
            stair_config.get("handrail_post_spacing_m", cfg.stair_handrail_post_spacing_m)
        )
        cfg.stair_handrail_follow_slope = bool(
            stair_config.get("handrail_follow_slope", cfg.stair_handrail_follow_slope)
        )
        if (
            cfg.stair_handrail_enabled
            and (
                cfg.stair_handrail_height_m <= 0.0
                or cfg.stair_handrail_thickness_m <= 0.0
                or cfg.stair_handrail_side_offset_m < 0.0
                or cfg.stair_handrail_post_spacing_m <= 0.0
            )
        ):
            raise ValueError("stair handrail dimensions must be positive")
        cfg.stair_preview_distance_m = float(
            stair_config.get("preview_distance_m", cfg.stair_preview_distance_m)
        )
        cfg.stair_success_reward = float(
            stair_config.get("success_reward", cfg.stair_success_reward)
        )
        cfg.stair_height_progress_reward_scale = float(
            stair_config.get(
                "height_progress_reward_scale", cfg.stair_height_progress_reward_scale
            )
        )
        cfg.stair_step_progress_reward = float(
            stair_config.get("step_progress_reward", cfg.stair_step_progress_reward)
        )
        cfg.stair_level_hold_reward_scale = float(
            stair_config.get(
                "level_hold_reward_scale", cfg.stair_level_hold_reward_scale
            )
        )
        cfg.stair_approach_lift_reward_scale = float(
            stair_config.get(
                "approach_lift_reward_scale", cfg.stair_approach_lift_reward_scale
            )
        )
        cfg.stair_approach_lift_distance_m = float(
            stair_config.get(
                "approach_lift_distance_m", cfg.stair_approach_lift_distance_m
            )
        )
        cfg.stair_collision_penalty = float(
            stair_config.get("collision_penalty", cfg.stair_collision_penalty)
        )
        cfg.stair_collision_terminates = bool(
            stair_config.get("collision_terminates", cfg.stair_collision_terminates)
        )
        cfg.stair_jump_assist_enabled = bool(
            stair_config.get("jump_assist_enabled", False)
        )
        cfg.stair_jump_assist_trigger_distance_m = float(
            stair_config.get(
                "jump_assist_trigger_distance_m",
                cfg.stair_jump_assist_trigger_distance_m,
            )
        )
        cfg.stair_jump_assist_crouch_steps = int(
            stair_config.get(
                "jump_assist_crouch_steps", cfg.stair_jump_assist_crouch_steps
            )
        )
        cfg.stair_jump_assist_extension_steps = int(
            stair_config.get(
                "jump_assist_extension_steps", cfg.stair_jump_assist_extension_steps
            )
        )
        cfg.stair_jump_assist_hip_crouch_rad = float(
            stair_config.get(
                "jump_assist_hip_crouch_rad",
                cfg.stair_jump_assist_hip_crouch_rad,
            )
        )
        cfg.stair_jump_assist_hip_extension_rad = float(
            stair_config.get(
                "jump_assist_hip_extension_rad",
                cfg.stair_jump_assist_hip_extension_rad,
            )
        )
        cfg.stair_jump_assist_knee_crouch_rad = float(
            stair_config.get(
                "jump_assist_knee_crouch_rad",
                cfg.stair_jump_assist_knee_crouch_rad,
            )
        )
        cfg.stair_jump_assist_knee_extension_rad = float(
            stair_config.get(
                "jump_assist_knee_extension_rad",
                cfg.stair_jump_assist_knee_extension_rad,
            )
        )
        cfg.stair_completion_height_tolerance_m = float(
            stair_config.get(
                "completion_height_tolerance_m",
                cfg.stair_completion_height_tolerance_m,
            )
        )
        cfg.stair_start_delay_steps = int(
            stair_config.get("start_delay_steps", cfg.stair_start_delay_steps)
        )
        if (
            cfg.stair_success_reward < 0.0
            or cfg.stair_height_progress_reward_scale < 0.0
            or cfg.stair_step_progress_reward < 0.0
            or cfg.stair_level_hold_reward_scale < 0.0
            or cfg.stair_approach_lift_reward_scale < 0.0
            or cfg.stair_approach_lift_distance_m <= 0.0
            or cfg.stair_collision_penalty < 0.0
            or cfg.stair_jump_assist_trigger_distance_m <= 0.0
            or cfg.stair_jump_assist_crouch_steps <= 0
            or cfg.stair_jump_assist_extension_steps <= 0
            or cfg.stair_jump_assist_hip_crouch_rad < 0.0
            or cfg.stair_jump_assist_hip_extension_rad < 0.0
            or cfg.stair_jump_assist_knee_crouch_rad < 0.0
            or cfg.stair_jump_assist_knee_extension_rad < 0.0
            or cfg.stair_completion_height_tolerance_m < 0.0
            or cfg.stair_start_delay_steps < 0
        ):
            raise ValueError("stair reward scales must be non-negative")
        stair_length = (
            validated_stairs.start_x_m
            + validated_stairs.num_steps * validated_stairs.tread_range_m[1]
            + validated_stairs.landing_length_m
            + 2.0
        )
        cfg.scene.env_spacing = max(cfg.scene.env_spacing, stair_length)
    env = TitaResidualEnv(cfg)
    observation, _ = env.reset()
    observation = observation["policy"]
    if not args.headless:
        env.sim.set_camera_view((2.8, 2.8, 1.7), (0.0, 0.0, 0.35))
    official = FrozenOfficialTitaController(OfficialTitaActor.from_checkpoint(args.checkpoint, env.device))
    official.reset(env.num_envs, env.device)
    model = PhaseBActorCritic(
        obs_dim=observation.shape[-1],
        residual_limit=cfg.residual_limit,
        posture_scale=cfg.posture_scale,
    ).to(env.device)
    if args.initial_residual_checkpoint is not None:
        payload = torch.load(
            args.initial_residual_checkpoint.expanduser(), map_location="cpu", weights_only=True
        )
        load_phase_b_state_dict_compatible(model, payload["model_state_dict"])
        print(f"[TITA-UC TRAIN] warm-start={args.initial_residual_checkpoint}")
    policy_log_std = (
        args.policy_log_std
        if args.policy_log_std is not None
        else run_config.get("policy_log_std")
    )
    freeze_policy_std = bool(
        args.freeze_policy_std or run_config.get("freeze_policy_std", False)
    )
    model.policy.configure_exploration(
        policy_log_std, trainable=not freeze_policy_std
    )
    print(
        "[TITA-UC TRAIN] exploration "
        f"log_std={model.policy.log_std.detach().cpu().tolist()} "
        f"trainable={model.policy.log_std.requires_grad}"
    )
    algorithm = PhaseBNP3O(
        model,
        learning_rate=learning_rate,
        cost_multipliers=initial_cost_multipliers,
        minimum_cost_multipliers=minimum_cost_multipliers,
        cost_budgets=cost_budgets,
        dual_learning_rate=dual_learning_rate,
        cost_multiplier_max=cost_multiplier_max,
    )
    command = torch.tensor([args.command_x, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    posture_config = run_config.get("posture_curriculum", {})
    posture_enabled = bool(args.posture_curriculum or posture_config.get("enabled", False))
    fixed_posture = args.fixed_posture or run_config.get("fixed_posture")
    if fixed_posture is not None and fixed_posture not in POSTURE_NAMES:
        raise ValueError("fixed posture must be LOW, MID, or HIGH")
    if posture_enabled and fixed_posture is not None:
        raise ValueError("posture curriculum and fixed posture are mutually exclusive")
    fixed_posture_index = POSTURE_NAMES.index(fixed_posture) if fixed_posture else None
    fixed_posture_command = None
    if fixed_posture_index is not None:
        fixed_posture_command = torch.nn.functional.one_hot(
            torch.full((env.num_envs,), fixed_posture_index, device=env.device), 3
        ).to(torch.float32)
    posture_scheduler = PostureCommandScheduler(
        env.num_envs,
        env.device,
        PostureCurriculumConfig(
            control_dt=env.step_dt,
            resample_time_s=float(posture_config.get("resample_time_s", 3.0)),
            transition_time_s=float(posture_config.get("transition_time_s", 1.5)),
            switch_hold_time_s=float(posture_config.get("switch_hold_time_s", 0.20)),
            switch_speed_max=float(posture_config.get("switch_speed_max_mps", 0.20)),
            switch_roll_max=float(posture_config.get("switch_roll_max_rad", 0.12)),
            switch_pitch_max=float(posture_config.get("switch_pitch_max_rad", 0.12)),
        ),
    )
    mid_only_fraction = float(posture_config.get("mid_only_fraction", 0.10))
    low_mid_fraction = float(posture_config.get("low_mid_fraction", 0.45))
    posture_speed_limits = torch.tensor(
        posture_config.get("speed_limits_mps", (0.15, 0.30, 0.20)),
        device=env.device,
        dtype=torch.float32,
    )
    if posture_speed_limits.shape != (3,) or torch.any(posture_speed_limits <= 0.0):
        raise ValueError("posture speed limits must be a positive LOW/MID/HIGH vector")
    if not 0.0 <= mid_only_fraction <= low_mid_fraction <= 1.0:
        raise ValueError("posture curriculum fractions must satisfy 0 <= mid_only <= low_mid <= 1")

    for warmup_step in range(warmup_steps):
        if not simulation_app.is_running():
            print("[TITA-UC TRAIN] Isaac Sim window closed during warm-up; stopping.")
            env.close()
            return
        env.set_velocity_command(
            torch.zeros_like(command) if cfg.stair_enabled else command
        )
        if fixed_posture_command is not None:
            env.set_posture_command(fixed_posture_command)
        else:
            env.set_posture_command(
                torch.tensor((0.0, 1.0, 0.0), device=env.device).repeat(env.num_envs, 1)
            )
        assert official.previous_policy_action is not None
        proprioception = env.official_proprioception(official.previous_policy_action)
        base_target, _ = official.target(proprioception, env.default_joint_target())
        env.set_base_target(base_target)
        with torch.no_grad():
            warmup_action = model.policy(observation)
        next_observation, _, terminated, truncated, _ = env.step(warmup_action)
        done = terminated | truncated
        official.reset_envs(done)
        observation = next_observation["policy"]
    print(f"[TITA-UC TRAIN] stabilization warm-up steps={warmup_steps}")

    def save_checkpoint(iteration: int) -> Path:
        output = run_dir / "checkpoints" / f"residual_phase_b_iter_{iteration:05d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": algorithm.optimizer.state_dict(),
                "constraint_state_dict": algorithm.constraint_state_dict(),
                "policy_config": {
                    "residual_limit": cfg.residual_limit,
                    "posture_scale": cfg.posture_scale,
                },
                "iteration": iteration,
            },
            output,
        )
        print(f"[TITA-UC TRAIN] checkpoint={output}")
        return output

    completed_iterations = 0
    stop_requested = False
    save_checkpoint(0)
    for iteration in range(1, args.iterations + 1):
        if not simulation_app.is_running():
            print("[TITA-UC TRAIN] Isaac Sim window closed; stopping training.")
            break
        buffers: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("obs", "action", "log_prob", "reward", "cost", "done", "value", "cost_value")
        }
        tunnel_success_before = env.tunnel_success_count.clone()
        stair_success_before = env.stair_success_count.clone()
        tunnel_ever_exited = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        stair_ever_completed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        base_action_max = 0.0
        hard_failures = 0
        projected_steps = 0
        hard_reason_counts = torch.zeros(
            len(HARD_REASON_NAMES), dtype=torch.long, device=env.device
        )
        velocity_joint_counts = torch.zeros(8, dtype=torch.long, device=env.device)
        velocity_state_projection_counts = torch.zeros(
            8, dtype=torch.long, device=env.device
        )
        projection_joint_counts = torch.zeros(8, dtype=torch.long, device=env.device)
        qd_absmax = torch.zeros(8, device=env.device)
        qd_preprojection_absmax = torch.zeros(8, device=env.device)
        qdd_absmax = torch.zeros(8, device=env.device)
        smooth_component_sum = torch.zeros(4, device=env.device)
        posture_mode_counts = torch.zeros(3, dtype=torch.long, device=env.device)
        posture_height_sum = torch.zeros(3, device=env.device)
        posture_height_error_sum = torch.zeros(3, device=env.device)
        posture_transition_steps = 0
        posture_switch_starts = 0
        posture_pending_steps = 0
        posture_switch_blocked_counts = torch.zeros(3, dtype=torch.long, device=env.device)
        posture_pending_state_absmax = torch.zeros(3, device=env.device)
        stair_local_x_max = torch.full(
            (env.num_envs,), float("-inf"), device=env.device
        )
        stair_root_z_max = torch.full_like(stair_local_x_max, float("-inf"))
        stair_verified_level_max = torch.zeros(env.num_envs, device=env.device)
        progress_fraction = (iteration - 1) / max(args.iterations - 1, 1)
        if not posture_enabled or progress_fraction < mid_only_fraction:
            posture_scheduler.set_allowed_modes((1,))
        elif progress_fraction < low_mid_fraction:
            posture_scheduler.set_allowed_modes((0, 1))
        else:
            posture_scheduler.set_allowed_modes((0, 1, 2))
        for _ in range(args.rollout_steps):
            if not simulation_app.is_running():
                print("[TITA-UC TRAIN] Isaac Sim window closed during rollout; stopping.")
                stop_requested = True
                break
            step_command = command
            if posture_enabled:
                gravity = env.robot.data.projected_gravity_b
                speed = torch.linalg.vector_norm(env.robot.data.root_lin_vel_b[:, :2], dim=1)
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
                pitch = torch.atan2(
                    -gravity[:, 0],
                    torch.sqrt(torch.square(gravity[:, 1]) + torch.square(gravity[:, 2])),
                )
                posture_step = posture_scheduler.step(speed, roll, pitch)
                pending = posture_scheduler.pending
                posture_pending_steps += int(pending.sum().item())
                posture_switch_blocked_counts += torch.stack(
                    (
                        (pending & (speed.abs() > posture_scheduler.cfg.switch_speed_max)).sum(),
                        (pending & (roll.abs() > posture_scheduler.cfg.switch_roll_max)).sum(),
                        (pending & (pitch.abs() > posture_scheduler.cfg.switch_pitch_max)).sum(),
                    )
                )
                if pending.any():
                    pending_state = torch.stack((speed.abs(), roll.abs(), pitch.abs()), dim=1)
                    posture_pending_state_absmax = torch.maximum(
                        posture_pending_state_absmax,
                        pending_state[pending].amax(dim=0),
                    )
                env.set_posture_command(posture_step.command, posture_step.phase)
                step_command = command.clone()
                commanded_speed = torch.linalg.vector_norm(step_command[:, :2], dim=1)
                mode_speed_limit = posture_step.command @ posture_speed_limits
                scale = torch.clamp(
                    mode_speed_limit / commanded_speed.clamp_min(1e-6),
                    max=1.0,
                )
                step_command[:, :2] *= scale[:, None]
                step_command[posture_step.slow_down, :2] = 0.0
                posture_transition_steps += int(posture_step.transition.sum().item())
                posture_switch_starts += int(posture_step.started.sum().item())
            elif fixed_posture_command is not None:
                env.set_posture_command(fixed_posture_command)
                step_command = command.clone()
                fixed_speed_limit = posture_speed_limits[fixed_posture_index]
                commanded_speed = torch.linalg.vector_norm(step_command[:, :2], dim=1)
                scale = torch.clamp(
                    fixed_speed_limit / commanded_speed.clamp_min(1e-6), max=1.0
                )
                step_command[:, :2] *= scale[:, None]
            if cfg.stair_enabled and cfg.stair_start_delay_steps > 0:
                stabilizing_after_reset = (
                    env.episode_length_buf < cfg.stair_start_delay_steps
                )
                step_command = step_command.clone()
                step_command[stabilizing_after_reset] = 0.0
            env.set_velocity_command(step_command)
            assert official.previous_policy_action is not None
            proprioception = env.official_proprioception(official.previous_policy_action)
            base_target, base_action = official.target(proprioception, env.default_joint_target())
            env.set_base_target(base_target)
            action, log_prob, value, cost_value = model.act(observation)
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
            next_observation, reward, terminated, truncated, _ = env.step(action)
            tunnel_ever_exited |= env.tunnel_success_count > tunnel_success_before
            stair_ever_completed |= env.stair_success_count > stair_success_before
            done = terminated | truncated
            posture_indices = env._posture_command.argmax(dim=1)
            posture_mode_counts += posture_indices.bincount(minlength=3)
            posture_error = torch.abs(env.posture_height - env.posture_target_height)
            for posture_index in range(3):
                posture_mask = posture_indices == posture_index
                posture_height_sum[posture_index] += env.posture_height[posture_mask].sum()
                posture_height_error_sum[posture_index] += posture_error[posture_mask].sum()
            buffers["obs"].append(observation)
            buffers["action"].append(action)
            buffers["log_prob"].append(log_prob)
            buffers["reward"].append(reward)
            buffers["cost"].append(env.soft_costs.clone())
            buffers["done"].append(done)
            buffers["value"].append(value)
            buffers["cost_value"].append(cost_value)
            hard_failures += int(terminated.sum().item())
            projected_steps += int(env.projected.sum().item())
            hard_reason_counts += env.hard_reason_flags.sum(0)
            velocity_joint_counts += env.velocity_violation_flags.sum(0)
            velocity_state_projection_counts += (
                env.velocity_state_projection_flags.sum(0)
            )
            projection_joint_counts += env.projection_joint_flags.sum(0)
            qd_absmax = torch.maximum(qd_absmax, env.joint_velocity_absmax())
            qd_preprojection_absmax = torch.maximum(
                qd_preprojection_absmax,
                env.preprojection_joint_velocity_absmax(),
            )
            qdd_absmax = torch.maximum(qdd_absmax, env.joint_acceleration_absmax())
            smooth_component_sum += env.smooth_components.sum(0)
            base_action_max = max(base_action_max, float(base_action.abs().max()))
            official.reset_envs(done)
            if posture_enabled:
                posture_scheduler.reset(done)
            observation = next_observation["policy"]

        if stop_requested:
            break

        with torch.no_grad():
            last_reward_value = model.reward_value(observation).squeeze(-1)
            last_cost_value = model.cost_value(observation)
        tensors = {name: torch.stack(values) for name, values in buffers.items()}
        reward_advantage, reward_return = generalized_advantage(
            tensors["reward"], tensors["value"], last_reward_value, tensors["done"], 0.99, 0.95
        )
        cost_advantage, cost_return = generalized_advantage(
            tensors["cost"], tensors["cost_value"], last_cost_value, tensors["done"], 0.99, 0.95
        )
        stats = algorithm.update(
            tensors["obs"], tensors["action"], tensors["log_prob"],
            reward_advantage, reward_return, cost_advantage, cost_return,
            observed_cost_mean=tensors["cost"].mean(dim=(0, 1)),
        )
        metric = {
            "iteration": iteration,
            "rollout_policy_checkpoint_iteration": iteration - 1,
            "updated_checkpoint_iteration": iteration,
            "reward_mean": float(tensors["reward"].mean()),
            "cost_route": float(tensors["cost"][..., 0].mean()),
            "cost_energy": float(tensors["cost"][..., 1].mean()),
            "cost_mode": float(tensors["cost"][..., 2].mean()),
            "cost_smooth": float(tensors["cost"][..., 3].mean()),
            "smooth_component_mean": (
                smooth_component_sum / (args.rollout_steps * env.num_envs)
            ).tolist(),
            "smooth_component_names": [
                "joint_target_change", "residual_rate",
                "acceleration_ratio", "soft_contact_term"
            ],
            "hard_failures": hard_failures,
            "projection_fraction": projected_steps / (args.rollout_steps * env.num_envs),
            "hard_reason_counts": hard_reason_counts.tolist(),
            "hard_reason_names": list(HARD_REASON_NAMES),
            "velocity_joint_counts": velocity_joint_counts.tolist(),
            "velocity_state_projection_counts": velocity_state_projection_counts.tolist(),
            "terrain": args.terrain,
            "domain_randomization": args.domain_randomization,
            "projection_joint_counts": projection_joint_counts.tolist(),
            "qd_absmax": qd_absmax.tolist(),
            "qd_preprojection_absmax": qd_preprojection_absmax.tolist(),
            "qdd_absmax": qdd_absmax.tolist(),
            "acceleration_limit": args.acceleration_limit,
            "residual_filter_alpha": args.residual_filter_alpha,
            "policy_log_std": model.policy.log_std.detach().cpu().tolist(),
            "policy_std_trainable": model.policy.log_std.requires_grad,
            "posture_curriculum": posture_enabled,
            "fixed_posture": fixed_posture,
            "posture_mode_counts": posture_mode_counts.tolist(),
            "posture_mode_names": list(POSTURE_NAMES),
            "posture_height_mean_m": (
                posture_height_sum / posture_mode_counts.clamp_min(1)
            ).tolist(),
            "posture_height_abs_error_mean_m": (
                posture_height_error_sum / posture_mode_counts.clamp_min(1)
            ).tolist(),
            "posture_transition_steps": posture_transition_steps,
            "posture_switch_starts": posture_switch_starts,
            "posture_pending_steps": posture_pending_steps,
            "posture_switch_blocked_counts": posture_switch_blocked_counts.tolist(),
            "posture_switch_blocked_names": ["speed", "roll", "pitch"],
            "posture_pending_state_absmax": posture_pending_state_absmax.tolist(),
            "objective_p95": list(objective_p95),
            "tunnel_dimensions_min_m": (
                env.tunnel_dimensions.amin(dim=0).tolist() if cfg.tunnel_enabled else None
            ),
            "tunnel_dimensions_max_m": (
                env.tunnel_dimensions.amax(dim=0).tolist() if cfg.tunnel_enabled else None
            ),
            "tunnel_exit_count": int(tunnel_ever_exited.sum().item()),
            "tunnel_success_total": int(
                (env.tunnel_success_count - tunnel_success_before).sum().item()
            ),
            "stair_dimensions_min_m": (
                env.stair_dimensions.amin(dim=0).tolist() if cfg.stair_enabled else None
            ),
            "stair_dimensions_max_m": (
                env.stair_dimensions.amax(dim=0).tolist() if cfg.stair_enabled else None
            ),
            "stair_completion_count": int(stair_ever_completed.sum().item()),
            "stair_success_total": int(
                (env.stair_success_count - stair_success_before).sum().item()
            ),
            "stair_local_x_max_minmax_m": (
                [stair_local_x_max.min().item(), stair_local_x_max.max().item()]
                if cfg.stair_enabled else None
            ),
            "stair_root_z_max_minmax_m": (
                [stair_root_z_max.min().item(), stair_root_z_max.max().item()]
                if cfg.stair_enabled else None
            ),
            "stair_verified_level_histogram": (
                stair_verified_level_max.long().bincount(
                    minlength=cfg.stair_num_steps + 1
                ).tolist()
                if cfg.stair_enabled else None
            ),
            "base_action_absmax": base_action_max,
            **stats.__dict__,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, sort_keys=True) + "\n")
        print("[TITA-UC TRAIN] " + json.dumps(metric, sort_keys=True))
        completed_iterations = iteration
        if iteration % checkpoint_interval == 0:
            save_checkpoint(iteration)

    if completed_iterations == 0 or completed_iterations % checkpoint_interval != 0:
        save_checkpoint(completed_iterations)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
