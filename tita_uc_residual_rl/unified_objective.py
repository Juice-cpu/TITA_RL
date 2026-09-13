"""v1.0 route, mechanical-energy, and posture objective calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import ControllerConfig
from .types import ControlState, PostureSample, RouteSample


@dataclass(frozen=True)
class ObjectiveTerms:
    route: float
    energy: float
    mode: float
    smooth: float

    def as_dict(self) -> dict[str, float]:
        return {"route": self.route, "energy": self.energy, "mode": self.mode, "smooth": self.smooth}


@dataclass(frozen=True)
class HardConstraintReport:
    violations: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return not self.violations


class UnifiedObjective:
    """Computes fixed-scale objective increments and the 8-DOF energy proxy."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def route_cost(self, sample: RouteSample) -> float:
        clearance_risk = 0.0 if not np.isfinite(sample.min_clearance) else max(0.0, 1.0 - sample.min_clearance / 0.30)
        return max(0.0, sample.local_length) + abs(sample.corridor_deviation) + clearance_risk + abs(sample.curvature)

    def energy_cost(self, tau: np.ndarray, qd: np.ndarray, dt: float) -> float:
        tau = np.asarray(tau, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        if tau.shape != (8,) or qd.shape != (8,):
            raise ValueError("mechanical energy requires exactly 8 joint values")
        if not np.all(np.isfinite(tau)) or not np.all(np.isfinite(qd)):
            raise ValueError("non-finite torque or velocity")
        return float(np.sum(np.abs(tau * qd)) * dt)

    def mechanical_cot(self, energy_mech: float, distance: float) -> float:
        if distance <= 0:
            return 0.0
        return float(energy_mech / (self.config.mass_kg * self.config.gravity * distance))

    def electrical_cot(self, power_left: np.ndarray, power_right: np.ndarray, dt: float, distance: float) -> float:
        """Separate rosbag/deployment metric; never relabel mechanical energy."""
        left = np.asarray(power_left, dtype=np.float64)
        right = np.asarray(power_right, dtype=np.float64)
        if left.shape != right.shape or left.ndim != 1:
            raise ValueError("left/right electrical power must be matching 1-D arrays")
        if dt <= 0 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("invalid electrical power sample")
        energy_elec = float(np.sum(np.maximum(left, 0.0) + np.maximum(right, 0.0)) * dt)
        return self.mechanical_cot(energy_elec, distance)

    def mode_cost(self, sample: PostureSample) -> float:
        return abs(sample.height_rate) + abs(sample.target_height - sample.actual_height) + abs(sample.transition_smoothness)

    def terms(self, route: RouteSample, energy: float, mode: PostureSample, smooth: float) -> ObjectiveTerms:
        norm = self.config.normalization
        return ObjectiveTerms(
            route=norm.normalized(self.route_cost(route), "route"),
            energy=norm.normalized(energy, "energy"),
            mode=norm.normalized(self.mode_cost(mode), "mode"),
            smooth=norm.normalized(max(0.0, smooth), "smooth"),
        )

    def scalar_cost(self, terms: ObjectiveTerms) -> float:
        weights = self.config.weights
        return float(weights.route * terms.route + weights.energy * terms.energy
                     + weights.mode * terms.mode + weights.smooth * terms.smooth)

    def reward_increment(self, terms: ObjectiveTerms) -> float:
        return -self.scalar_cost(terms)

    @staticmethod
    def hard_constraints(
        *,
        clearance: float,
        safe_clearance: float,
        tau: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qd_max: np.ndarray,
        qdd_max: np.ndarray,
        tau_max: np.ndarray,
        unsafe_posture_switch: bool = False,
        stale_sensor: bool = False,
        invalid_path: bool = False,
    ) -> HardConstraintReport:
        violations: list[str] = []
        if clearance < safe_clearance:
            violations.append("clearance_violation")
        if np.any(q < q_min) or np.any(q > q_max):
            violations.append("joint_limit")
        if np.any(np.abs(qd) > qd_max + 1e-9):
            violations.append("velocity_limit")
        if np.any(np.abs(qdd) > qdd_max + 1e-9):
            violations.append("acceleration_limit")
        if np.any(np.abs(tau) > tau_max + 1e-9):
            violations.append("torque_limit")
        if unsafe_posture_switch:
            violations.append("unsafe_posture_switch")
        if stale_sensor:
            violations.append("stale_sensor")
        if invalid_path:
            violations.append("invalid_goal_or_path_version")
        return HardConstraintReport(tuple(violations))
