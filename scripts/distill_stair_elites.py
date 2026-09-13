#!/usr/bin/env python3
"""Distill successful stochastic stair actions into a deterministic actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tita_uc_residual_rl.elite_distillation import distill_elite_actions
from tita_uc_residual_rl.phase_b_learner import (
    PhaseBActorCritic,
    load_phase_b_state_dict_compatible,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=3.0)
    parser.add_argument("--include-projected", action="store_true")
    parser.add_argument("--seed", type=int, default=163)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    datasets = [
        torch.load(path.expanduser(), map_location="cpu", weights_only=True)
        for path in args.dataset
    ]
    checkpoint = torch.load(
        args.checkpoint.expanduser(), map_location="cpu", weights_only=True
    )
    observations = torch.cat(
        [dataset["observations"].float() for dataset in datasets], dim=0
    )
    actions = torch.cat([dataset["actions"].float() for dataset in datasets], dim=0)
    source_means = torch.cat(
        [dataset["source_means"].float() for dataset in datasets], dim=0
    )
    weights = torch.cat(
        [dataset["sample_weights"].float() for dataset in datasets], dim=0
    )
    projected = torch.cat(
        [dataset["projected"].bool() for dataset in datasets], dim=0
    )
    keep = torch.ones_like(projected) if args.include_projected else ~projected
    if not torch.any(keep):
        raise ValueError("elite dataset has no eligible samples")

    model = PhaseBActorCritic(obs_dim=observations.shape[1])
    load_phase_b_state_dict_compatible(model, checkpoint["model_state_dict"])
    stats = distill_elite_actions(
        model,
        observations[keep],
        actions[keep],
        source_means[keep],
        sample_weights=weights[keep],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        anchor_weight=args.anchor_weight,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    payload["model_state_dict"] = model.state_dict()
    payload["distillation"] = {
        **stats.to_dict(),
        "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "elite_datasets": [str(path.expanduser().resolve()) for path in args.dataset],
        "projected_samples_excluded": int((~keep).sum()),
        "anchor_weight": args.anchor_weight,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    torch.save(payload, output)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(payload["distillation"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[TITA-UC DISTILL] " + json.dumps(payload["distillation"], sort_keys=True))
    print(f"[TITA-UC DISTILL] checkpoint={output}")


if __name__ == "__main__":
    main()
