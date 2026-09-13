"""Short local smoke run for the TITA residual-control core."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .config import DEFAULT_CONFIG
from .residual_policy import FrozenBaseController, ResidualPolicy, combine_targets
from .toy_env import ToyTitaEnv


def run(steps: int = 200, seed: int = 7) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ToyTitaEnv(DEFAULT_CONFIG, horizon=steps, seed=seed)
    first_obs = env.reset()
    obs = torch.from_numpy(first_obs.flatten()).float().unsqueeze(0)
    policy = ResidualPolicy(obs.shape[-1], residual_limit=DEFAULT_CONFIG.residual_limit,
                            posture_scales=DEFAULT_CONFIG.posture_scales)
    base = FrozenBaseController()
    total_reward = 0.0
    projected = 0
    holds = 0
    stops = 0
    last_info: dict[str, object] = {}
    for _ in range(steps):
        with torch.no_grad():
            residual = policy(obs)
            raw = combine_targets(base(1), residual).squeeze(0).numpy()
        result = env.step(raw)
        total_reward += result.reward
        projected += int(bool(result.info.get("projected", False)))
        holds += int(result.info.get("status") == "HOLD")
        stops += int(result.info.get("status") == "STOP")
        last_info = result.info
        obs = torch.from_numpy(result.observation.flatten()).float().unsqueeze(0)
        if result.done:
            break
    cot = env.objective.mechanical_cot(env.energy, env.distance)
    return {
        "steps": env.t,
        "reward": total_reward,
        "energy_mech_j": env.energy,
        "distance_m": env.distance,
        "cot_mech": cot,
        "projected_steps": projected,
        "hold_steps": holds,
        "stop_steps": stops,
        "last_info": last_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args.steps, args.seed), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
