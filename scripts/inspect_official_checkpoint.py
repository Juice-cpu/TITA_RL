#!/usr/bin/env python3
"""Inspect the official TITA checkpoint without importing Isaac Gym."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    # The checkpoint is inspected as tensors only; do not execute arbitrary
    # pickle objects while reading a file supplied outside this package.
    payload = torch.load(args.checkpoint.expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("checkpoint must contain model_state_dict")
    state = payload["model_state_dict"]
    required = {
        "actor_output_dim": tuple(state["actor_teacher_backbone.actor.6.bias"].shape)[0],
        "proprioception_dim": tuple(state["history_encoder.encoder.0.weight"].shape)[1],
        "scan_dim": tuple(state["scan_encoder.0.weight"].shape)[1],
        "history_length": 10,
        "history_flat_dim": tuple(state["actor_teacher_backbone.mlp_encoder.0.weight"].shape)[1],
        "history_latent_dim": tuple(state["actor_teacher_backbone.mlp_encoder.8.bias"].shape)[0],
        "iteration": int(payload.get("iter", -1)),
    }
    if required["actor_output_dim"] != 8:
        raise ValueError(f"official checkpoint action dimension is {required['actor_output_dim']}, expected 8")
    if required["history_flat_dim"] != required["proprioception_dim"] * required["history_length"]:
        raise ValueError("checkpoint history dimensions are inconsistent")
    print(json.dumps(required, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
