"""Deterministic command-space safety projection for the low-level target."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np

from .config import ControllerConfig
from .types import ControlState, SafetyContext


class SafetyStatus(str, Enum):
    ACCEPT = "ACCEPT"
    HOLD = "HOLD"
    STOP = "STOP"


@dataclass(frozen=True)
class SafetyResult:
    q_cmd: np.ndarray
    status: SafetyStatus
    reasons: tuple[str, ...] = ()
    projected: bool = False
    estimated_tau: np.ndarray = field(default_factory=lambda: np.zeros(8))
    estimated_qd: np.ndarray = field(default_factory=lambda: np.zeros(8))
    estimated_qdd: np.ndarray = field(default_factory=lambda: np.zeros(8))


class SafetyProjector:
    """Project a raw joint target and apply non-compensable hard constraints."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        (
            self.q_min, self.q_max, self.qd_max, self.qdd_max,
            self.tau_max, self.target_gain, self.position_gain, self.damping,
        ) = config.safety_limits.arrays()
        self.position_controlled = self.position_gain > 0.0
        self.dt = config.safety.dt
        if self.dt <= 0:
            raise ValueError("safety dt must be positive")

    def _hard_status(self, context: SafetyContext) -> tuple[SafetyStatus, List[str]]:
        reasons: List[str] = []
        if not context.sensor_valid:
            reasons.append("stale_sensor")
        if not context.goal_valid:
            reasons.append("invalid_goal")
        if context.path_version != context.expected_path_version:
            reasons.append("invalid_path_version")
        if context.goal_age > self.config.safety.goal_max_age:
            reasons.append("stale_goal")
        if context.collision or context.swept_collision:
            reasons.append("collision")
        if reasons:
            return SafetyStatus.STOP, reasons

        safe_clearance = self.config.safety.safe_clearance(context.speed)
        if context.clearance < safe_clearance:
            reasons.append("braking_distance_violation")
            return SafetyStatus.HOLD, reasons

        if context.posture_transition:
            limits = self.config.safety
            if abs(context.speed) > limits.switch_speed_max:
                reasons.append("unsafe_posture_switch_speed")
            if abs(context.roll) > limits.switch_roll_max:
                reasons.append("unsafe_posture_switch_roll")
            if abs(context.pitch) > limits.switch_pitch_max:
                reasons.append("unsafe_posture_switch_pitch")
            if context.hold_elapsed < limits.switch_hold_time:
                reasons.append("posture_hold_not_satisfied")
            if reasons:
                return SafetyStatus.HOLD, reasons
        return SafetyStatus.ACCEPT, reasons

    def project(self, q_ref_raw: np.ndarray, state: ControlState, context: SafetyContext) -> SafetyResult:
        state.validate(self.config.dof)
        raw = np.asarray(q_ref_raw, dtype=np.float64)
        if raw.shape != (self.config.dof,) or not np.all(np.isfinite(raw)):
            return SafetyResult(state.q.copy(), SafetyStatus.STOP, ("invalid_action",))

        status, reasons = self._hard_status(context)
        if status == SafetyStatus.STOP:
            return SafetyResult(state.q.copy(), status, tuple(reasons), False, state.tau.copy())

        # Intersect target-space intervals. Wheels use the official torque
        # proxy and therefore have no target-position velocity/acceleration
        # interpretation.
        lower = self.q_min.copy()
        upper = self.q_max.copy()
        if self.config.safety.project_target_kinematics:
            velocity_lower = np.where(
                self.position_controlled, state.q - self.qd_max * self.dt, -np.inf
            )
            velocity_upper = np.where(
                self.position_controlled, state.q + self.qd_max * self.dt, np.inf
            )
            lower = np.maximum(lower, velocity_lower)
            upper = np.minimum(upper, velocity_upper)
        torque_lower = (
            -self.tau_max + self.position_gain * state.q + self.damping * state.qd
        ) / self.target_gain
        torque_upper = (
            self.tau_max + self.position_gain * state.q + self.damping * state.qd
        ) / self.target_gain
        lower = np.maximum(lower, torque_lower)
        upper = np.minimum(upper, torque_upper)
        # qdd = ((q_cmd-q)/dt - qd)/dt, so bound target velocity first.
        if self.config.safety.project_target_kinematics:
            qd_lower = state.qd - self.qdd_max * self.dt
            qd_upper = state.qd + self.qdd_max * self.dt
            acceleration_lower = np.where(
                self.position_controlled, state.q + qd_lower * self.dt, -np.inf
            )
            acceleration_upper = np.where(
                self.position_controlled, state.q + qd_upper * self.dt, np.inf
            )
            lower = np.maximum(lower, acceleration_lower)
            upper = np.minimum(upper, acceleration_upper)

        if np.any(lower > upper + 1e-12):
            return SafetyResult(state.q.copy(), SafetyStatus.STOP, ("infeasible_safety_intersection",), False, state.tau.copy())
        q_cmd = np.clip(raw, lower, upper)
        qd_cmd = (q_cmd - state.q) / self.dt
        qdd_cmd = (qd_cmd - state.qd) / self.dt
        tau_cmd = (
            self.target_gain * q_cmd
            - self.position_gain * state.q
            - self.damping * state.qd
        )
        projected = not np.allclose(q_cmd, raw, rtol=0.0, atol=1e-10)
        return SafetyResult(q_cmd, status, tuple(reasons), projected, tau_cmd, qd_cmd, qdd_cmd)
