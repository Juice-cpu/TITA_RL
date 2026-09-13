"""Configuration with explicit, frozen normalization and safety parameters."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


DOF = 8
TITA_JOINT_NAMES = (
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_right_leg_4",
)


@dataclass(frozen=True)
class SafetyLimits:
    """Limits used by the command-space projector.

    The defaults are conservative placeholders until the TITA calibration is
    supplied.  They are intentionally visible rather than silently inferred.
    """

    q_min: Tuple[float, ...] = (
        -0.785398, -1.919862, -2.670354, -999999.0,
        -0.785398, -1.919862, -2.670354, -999999.0,
    )
    q_max: Tuple[float, ...] = (
        0.785398, 3.490659, -0.698132, 999999.0,
        0.785398, 3.490659, -0.698132, 999999.0,
    )
    # Asset-level defaults from the official URDF. Replace these with
    # measured hardware limits before deployment.
    qd_max: Tuple[float, ...] = (25.0, 25.0, 25.0, 20.0, 25.0, 25.0, 25.0, 20.0)
    qdd_max: Tuple[float, ...] = (80.0,) * DOF
    tau_max: Tuple[float, ...] = (60.0, 60.0, 60.0, 15.0, 60.0, 60.0, 60.0, 15.0)
    # Official controller law is
    # tau = target_gain*q_cmd - position_gain*q - damping*q_dot.
    # leg_4 is a wheel torque proxy, not a position-controlled joint.
    target_gain: Tuple[float, ...] = (40.0, 40.0, 40.0, 10.0, 40.0, 40.0, 40.0, 10.0)
    position_gain: Tuple[float, ...] = (40.0, 40.0, 40.0, 0.0, 40.0, 40.0, 40.0, 0.0)
    damping: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5)

    def arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(v, dtype=np.float64) for v in (
            self.q_min, self.q_max, self.qd_max, self.qdd_max,
            self.tau_max, self.target_gain, self.position_gain, self.damping,
        ))


@dataclass(frozen=True)
class SafetyConfig:
    dt: float = 0.02
    # A position target is not a one-step kinematic command. Keep this off
    # unless a calibrated target-to-state dynamics model is supplied.
    project_target_kinematics: bool = False
    clearance_d0: float = 0.10
    reaction_time: float = 0.15
    braking_accel: float = 1.5
    clearance_margin: float = 0.03
    switch_speed_max: float = 0.20
    switch_roll_max: float = 0.12
    switch_pitch_max: float = 0.12
    switch_hold_time: float = 0.20
    goal_max_age: float = 0.50

    def safe_clearance(self, speed: float) -> float:
        speed = abs(float(speed))
        return (self.clearance_d0 + self.reaction_time * speed
                + speed * speed / (2.0 * self.braking_accel)
                + self.clearance_margin)


@dataclass(frozen=True)
class ObjectiveNormalization:
    """Offline P95 scales; these values never depend on the current batch."""

    route: float = 1.0
    energy: float = 1.0
    mode: float = 1.0
    smooth: float = 1.0

    def normalized(self, value: float, name: str) -> float:
        scale = float(getattr(self, name))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"normalization scale for {name!r} must be positive")
        return float(value) / scale


@dataclass(frozen=True)
class ObjectiveWeights:
    route: float = 1.0
    energy: float = 0.15
    mode: float = 0.20
    smooth: float = 0.05


@dataclass(frozen=True)
class ControllerConfig:
    dof: int = DOF
    joint_names: Tuple[str, ...] = TITA_JOINT_NAMES
    posture_scales: Tuple[float, ...] = (0.75,) * DOF
    residual_limit: float = 0.20
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    normalization: ObjectiveNormalization = field(default_factory=ObjectiveNormalization)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    # Nominal URDF link-mass sum. Real COT requires a complete loaded-robot
    # weighing before this can be treated as a hardware parameter.
    mass_kg: float = 29.0254
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.dof != DOF:
            raise ValueError("TITA controller contract requires exactly 8 DOF")
        for name, values in (("joint_names", self.joint_names), ("posture_scales", self.posture_scales)):
            if len(values) != self.dof:
                raise ValueError(f"{name} must contain exactly {self.dof} values")
        if self.mass_kg <= 0 or self.gravity <= 0:
            raise ValueError("mass and gravity must be positive")


DEFAULT_CONFIG = ControllerConfig()
