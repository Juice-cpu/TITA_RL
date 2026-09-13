"""Strict aggregation for deterministic measured-stair feasibility probes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


MOTION_FIELDS = (
    "trigger_distance_m",
    "command_x_mps",
    "pulse_command_x_mps",
    "crouch_command_x_mps",
    "extension_command_x_mps",
    "crouch_steps",
    "extension_steps",
    "hip_extension_delay_steps",
    "knee_extension_delay_steps",
    "hip_crouch_rad",
    "hip_extension_rad",
    "knee_crouch_rad",
    "knee_extension_rad",
)


def motion_key(record: dict[str, Any]) -> tuple[float | int, ...]:
    """Return the deterministic identity of one open-loop stair motion."""
    normalized = dict(record)
    normalized.setdefault("pulse_command_x_mps", record["command_x_mps"])
    normalized.setdefault("crouch_command_x_mps", normalized["pulse_command_x_mps"])
    normalized.setdefault("extension_command_x_mps", normalized["pulse_command_x_mps"])
    normalized.setdefault("hip_extension_delay_steps", 0)
    normalized.setdefault("knee_extension_delay_steps", 0)
    return tuple(normalized[field] for field in MOTION_FIELDS)


def aggregate_strict_probe_results(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Find motions that strictly succeed in every supplied probe result."""
    if not payloads:
        raise ValueError("at least one stair probe result is required")
    for payload in payloads:
        if payload.get("step_height_m") != 0.17 or payload.get("tread_m") != 0.32:
            raise ValueError("all probes must use the measured 0.17 m x 0.32 m stair")
        if not isinstance(payload.get("records"), list):
            raise ValueError("each stair probe result must contain records")

    record_maps = [
        {motion_key(record): record for record in payload["records"]}
        for payload in payloads
    ]
    common_keys = set(record_maps[0]).intersection(*(set(records) for records in record_maps[1:]))
    strict_keys = [
        key
        for key in common_keys
        if all(bool(records[key].get("strict_success", False)) for records in record_maps)
    ]
    strict_keys.sort(
        key=lambda key: (
            -min(int(records[key]["max_stable_hold_steps"]) for records in record_maps),
            max(float(records[key]["max_base_contact_force_n"]) for records in record_maps),
            key,
        )
    )
    candidates = []
    for key in strict_keys:
        candidate = dict(zip(MOTION_FIELDS, key, strict=True))
        candidate["per_probe"] = [records[key] for records in record_maps]
        candidates.append(candidate)
    return {
        "probe_count": len(payloads),
        "common_candidate_count": len(common_keys),
        "strict_multiseed_candidate_count": len(candidates),
        "strict_candidates": candidates,
    }
