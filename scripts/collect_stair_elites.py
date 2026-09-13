#!/usr/bin/env python3
"""Collect successful stochastic stair episodes for policy distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True)
parser.add_argument("--official-checkpoint", required=True)
parser.add_argument("--residual-checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2400)
parser.add_argument("--max-success-episodes", type=int, default=16)
parser.add_argument("--command-x", type=float, default=0.15)
parser.add_argument("--episode-length-s", type=float, default=24.0)
parser.add_argument("--acceleration-limit", type=float, default=500.0)
parser.add_argument("--residual-filter-alpha", type=float, default=0.10)
parser.add_argument("--policy-log-std", type=float, default=-3.8)
parser.add_argument("--seed", type=int, default=163)
parser.add_argument("--stair-height-range", type=float, nargs=2, default=(0.005, 0.015))
parser.add_argument("--stair-tread-range", type=float, nargs=2, default=(0.60, 0.75))
parser.add_argument("--stair-width-range", type=float, nargs=2, default=(1.00, 1.30))
parser.add_argument("--stair-num-steps", type=int, default=2)
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
    if args.steps <= 0 or args.max_success_episodes <= 0:
        raise ValueError("steps and max-success-episodes must be positive")
    torch.manual_seed(args.seed)
    cfg = build_tita_cfg(args.asset, num_envs=args.num_envs, terrain="stairs")
    cfg.seed = args.seed
    cfg.episode_length_s = args.episode_length_s
    cfg.acceleration_limit = args.acceleration_limit
    cfg.residual_filter_alpha = args.residual_filter_alpha
    cfg.enforce_acceleration_limit = True
    cfg.stair_step_height_range_m = tuple(args.stair_height_range)
    cfg.stair_tread_range_m = tuple(args.stair_tread_range)
    cfg.stair_width_range_m = tuple(args.stair_width_range)
    cfg.stair_num_steps = args.stair_num_steps
    env = TitaResidualEnv(cfg)
    observation, _ = env.reset()
    observation = observation["policy"]

    official = FrozenOfficialTitaController(
        OfficialTitaActor.from_checkpoint(args.official_checkpoint, env.device)
    )
    official.reset(env.num_envs, env.device)
    payload = torch.load(
        args.residual_checkpoint.expanduser(), map_location="cpu", weights_only=True
    )
    model = PhaseBActorCritic(obs_dim=observation.shape[1]).to(env.device)
    load_phase_b_state_dict_compatible(model, payload["model_state_dict"])
    model.policy.configure_exploration(args.policy_log_std, trainable=False)
    model.eval()

    command = torch.tensor(
        [args.command_x, 0.0, 0.0], device=env.device
    ).repeat(env.num_envs, 1)
    posture = torch.tensor((0.0, 0.0, 1.0), device=env.device).repeat(
        env.num_envs, 1
    )
    episode_fields = ("observations", "actions", "source_means", "sample_weights", "projected")
    active: dict[str, list[list[torch.Tensor]]] = {
        name: [[] for _ in range(env.num_envs)] for name in episode_fields
    }
    elite_episodes: list[dict[str, torch.Tensor]] = []
    success_before = env.stair_success_count.clone()
    hard_reason_counts = torch.zeros(
        len(HARD_REASON_NAMES), dtype=torch.long, device=env.device
    )
    hard_failures = 0
    completed_steps = 0

    for step in range(args.steps):
        if not simulation_app.is_running():
            break
        env.set_velocity_command(command)
        env.set_posture_command(posture)
        assert official.previous_policy_action is not None
        proprioception = env.official_proprioception(official.previous_policy_action)
        base_target, _ = official.target(proprioception, env.default_joint_target())
        env.set_base_target(base_target)
        with torch.no_grad():
            distribution = model.distribution(observation)
            source_mean = distribution.mean
            action = distribution.sample()
        local_x = env.robot.data.root_pos_w[:, 0] - env._environment_origins()[:, 0]
        verified_level = env._verified_stair_level()
        sample_weight = (
            1.0
            + 2.0 * (local_x >= cfg.stair_start_x_m - 0.25).float()
            + verified_level / max(cfg.stair_num_steps, 1)
        )
        next_observation, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        projected = env.projection_joint_flags.any(dim=1)
        success_after = env.stair_success_count.clone()
        succeeded = success_after > success_before
        success_before = success_after

        cpu_values = {
            "observations": observation.detach().cpu(),
            "actions": action.detach().cpu(),
            "source_means": source_mean.detach().cpu(),
            "sample_weights": sample_weight.detach().cpu(),
            "projected": projected.detach().cpu(),
        }
        for env_id in range(env.num_envs):
            for name in episode_fields:
                active[name][env_id].append(cpu_values[name][env_id])
            if bool(done[env_id]):
                if bool(succeeded[env_id]):
                    elite_episodes.append(
                        {
                            name: torch.stack(active[name][env_id])
                            for name in episode_fields
                        }
                    )
                for name in episode_fields:
                    active[name][env_id].clear()

        hard_failures += int(terminated.sum())
        hard_reason_counts += env.hard_reason_flags.sum(dim=0)
        official.reset_envs(done)
        observation = next_observation["policy"]
        completed_steps = step + 1
        if completed_steps == 1 or completed_steps % 200 == 0:
            print(
                f"[TITA-UC ELITES] step={completed_steps}/{args.steps} "
                f"success_episodes={len(elite_episodes)} hard_failures={hard_failures}"
            )
        if len(elite_episodes) >= args.max_success_episodes:
            break

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if elite_episodes:
        lengths = [episode["observations"].shape[0] for episode in elite_episodes]
        dataset = {
            name: torch.cat([episode[name] for episode in elite_episodes], dim=0)
            for name in episode_fields
        }
        dataset["episode_ids"] = torch.cat(
            [torch.full((length,), index, dtype=torch.long) for index, length in enumerate(lengths)]
        )
    else:
        lengths = []
        dataset = {
            "observations": torch.empty((0, observation.shape[1])),
            "actions": torch.empty((0, 8)),
            "source_means": torch.empty((0, 8)),
            "sample_weights": torch.empty((0,)),
            "projected": torch.empty((0,), dtype=torch.bool),
            "episode_ids": torch.empty((0,), dtype=torch.long),
        }
    metadata = {
        "source_checkpoint": str(args.residual_checkpoint.expanduser().resolve()),
        "seed": args.seed,
        "completed_steps": completed_steps,
        "success_episodes": len(elite_episodes),
        "episode_lengths": lengths,
        "samples": int(dataset["observations"].shape[0]),
        "projected_samples": int(dataset["projected"].sum()),
        "hard_failures": hard_failures,
        "hard_reason_counts": hard_reason_counts.tolist(),
        "hard_reason_names": list(HARD_REASON_NAMES),
        "policy_log_std": args.policy_log_std,
        "command_x": args.command_x,
        "stair_height_range": list(args.stair_height_range),
        "stair_tread_range": list(args.stair_tread_range),
        "stair_width_range": list(args.stair_width_range),
        "stair_num_steps": args.stair_num_steps,
    }
    torch.save({**dataset, "metadata": metadata}, output)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[TITA-UC ELITES] " + json.dumps(metadata, sort_keys=True))
    print(f"[TITA-UC ELITES] dataset={output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
