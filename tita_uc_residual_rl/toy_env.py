"""Deterministic toy task for exercising the controller without a simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ControllerConfig, DEFAULT_CONFIG
from .safety_projector import SafetyProjector, SafetyStatus
from .types import ControlState, Observation, PostureSample, RouteSample, SafetyContext
from .unified_objective import UnifiedObjective


@dataclass(frozen=True)
class ToyStep:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, object]


class ToyTitaEnv:
    """Small deterministic proxy, useful for smoke tests and CI only."""

    def __init__(self, config: ControllerConfig = DEFAULT_CONFIG, horizon: int = 200, seed: int = 0) -> None:
        self.config = config
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.projector = SafetyProjector(config)
        self.objective = UnifiedObjective(config)
        self.reset()

    def reset(self) -> Observation:
        self.t = 0
        self.distance = 0.0
        self.energy = 0.0
        self.q = np.array((0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0), dtype=np.float64)
        self.qd = np.zeros(8, dtype=np.float64)
        self.last_cmd = self.q.copy()
        self.path_version = 0
        self.nominal_target = self.q.copy()
        return self._observation()

    def _observation(self) -> Observation:
        speed = 0.35
        safe = self.config.safety.safe_clearance(speed)
        proprio = np.concatenate((self.q, self.qd, np.zeros(3), np.array([1.0, 0.0, 0.0])))
        return Observation(
            proprioception=proprio,
            action_history=self.last_cmd.copy(),
            velocity_command=np.array([speed, 0.0], dtype=np.float64),
            posture_command=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            posture_actual=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            posture_phase=0.0,
            route_preview=np.array([0.15, 0.0, 0.0], dtype=np.float64),
            clearance_preview=np.array([safe + 0.15, safe + 0.15], dtype=np.float64),
            curvature_preview=np.array([0.0, 0.0], dtype=np.float64),
            braking_margin=0.15,
            energy_accumulator=self.energy,
            path_version=float(self.path_version),
            goal_age=0.0,
        )

    def step(self, q_ref_raw: np.ndarray, *, context: Optional[SafetyContext] = None) -> ToyStep:
        if context is None:
            speed = 0.35
            safe = self.config.safety.safe_clearance(speed)
            context = SafetyContext(speed=speed, clearance=safe + 0.15,
                                   path_version=self.path_version,
                                   expected_path_version=self.path_version)
        state = ControlState(q=self.q.copy(), qd=self.qd.copy(),
                             tau=np.zeros(8, dtype=np.float64),
                             q_cmd_prev=self.last_cmd.copy())
        result = self.projector.project(q_ref_raw, state, context)
        if result.status == SafetyStatus.STOP:
            info = {"status": result.status.value, "reasons": result.reasons, "hard_failure": True}
            return ToyStep(self._observation(), -10.0, True, info)

        # A stable first-order plant: it does not pretend to model TITA dynamics.
        qd_next = np.clip(0.85 * self.qd + 0.15 * result.estimated_qd,
                          -np.asarray(self.config.safety_limits.qd_max),
                          np.asarray(self.config.safety_limits.qd_max))
        q_next = self.q + qd_next * self.config.safety.dt
        limits = self.config.safety_limits
        tau = (
            np.asarray(limits.target_gain) * result.q_cmd
            - np.asarray(limits.position_gain) * self.q
            - np.asarray(limits.damping) * self.qd
        )
        dt = self.config.safety.dt
        energy_step = self.objective.energy_cost(tau, qd_next, dt)
        self.energy += energy_step
        self.distance += 0.35 * dt
        smooth = float(np.linalg.norm(result.q_cmd - self.last_cmd))
        terms = self.objective.terms(
            RouteSample(local_length=0.35 * dt, corridor_deviation=float(np.linalg.norm(q_next)) * 0.01,
                         min_clearance=context.clearance, curvature=0.0),
            energy_step,
            PostureSample(), smooth,
        )
        self.q, self.qd, self.last_cmd = q_next, qd_next, result.q_cmd.copy()
        self.t += 1
        done = self.t >= self.horizon
        reward = self.objective.reward_increment(terms)
        if result.status == SafetyStatus.HOLD:
            reward -= 0.5
        info = {
            "status": result.status.value,
            "reasons": result.reasons,
            "projected": result.projected,
            "terms": terms.as_dict(),
            "energy_step": energy_step,
            "distance": self.distance,
            "hard_failure": False,
        }
        return ToyStep(self._observation(), reward, done, info)
