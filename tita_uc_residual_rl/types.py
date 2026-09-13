"""Shared typed data structures for policy, safety, and objective modules."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Posture(str, Enum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


@dataclass
class ControlState:
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    q_cmd_prev: Optional[np.ndarray] = None

    def validate(self, dof: int = 8) -> None:
        for name in ("q", "qd", "tau"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (dof,):
                raise ValueError(f"{name} must have shape ({dof},), got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")


@dataclass
class SafetyContext:
    speed: float = 0.0
    clearance: float = np.inf
    roll: float = 0.0
    pitch: float = 0.0
    posture_transition: bool = False
    hold_elapsed: float = 1.0
    sensor_valid: bool = True
    collision: bool = False
    swept_collision: bool = False
    path_version: int = 0
    expected_path_version: int = 0
    goal_age: float = 0.0
    goal_valid: bool = True


@dataclass(frozen=True)
class RouteSample:
    local_length: float = 0.0
    corridor_deviation: float = 0.0
    min_clearance: float = np.inf
    curvature: float = 0.0


@dataclass(frozen=True)
class PostureSample:
    target_height: float = 0.0
    actual_height: float = 0.0
    height_rate: float = 0.0
    transition_smoothness: float = 0.0


@dataclass(frozen=True)
class Observation:
    """Actor-visible input. Privileged simulator quantities stay out of it."""

    proprioception: np.ndarray
    action_history: np.ndarray
    velocity_command: np.ndarray
    posture_command: np.ndarray
    posture_actual: np.ndarray
    posture_phase: float
    route_preview: np.ndarray
    clearance_preview: np.ndarray
    curvature_preview: np.ndarray
    braking_margin: float
    energy_accumulator: float
    path_version: float
    goal_age: float

    def flatten(self) -> np.ndarray:
        parts = [
            self.proprioception, self.action_history, self.velocity_command,
            self.posture_command, self.posture_actual, np.asarray([self.posture_phase]),
            self.route_preview, self.clearance_preview, self.curvature_preview,
            np.asarray([self.braking_margin, self.energy_accumulator,
                        self.path_version, self.goal_age]),
        ]
        out = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
        if not np.all(np.isfinite(out)):
            raise ValueError("observation contains non-finite values")
        return out
