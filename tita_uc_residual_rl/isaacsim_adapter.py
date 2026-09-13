"""Import-safe integration boundary for Isaac Sim 4.5/Isaac Lab migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TITA_JOINT_NAMES


@dataclass
class IsaacSimArticulationAdapter:
    """Native Articulation adapter placeholder.

    The core package intentionally has no import-time dependency on `omni`.
    A real task should bind `articulation`, joint ordering, and simulator time
    here, while retaining `SafetyProjector.project` as the final command gate.
    """

    articulation: Any = None
    joint_names: tuple[str, ...] = TITA_JOINT_NAMES
    decimation: int = 4

    @staticmethod
    def availability() -> dict[str, bool]:
        result: dict[str, bool] = {}
        for module in ("isaacsim", "isaaclab", "isaaclab_tasks", "omni.isaac.lab", "isaacgym", "rl_games", "rsl_rl"):
            try:
                __import__(module)
                result[module] = True
            except ImportError:
                result[module] = False
        return result

    def validate_joint_order(self, simulator_names: list[str] | tuple[str, ...]) -> None:
        if tuple(simulator_names) != self.joint_names:
            raise ValueError(
                "joint order mismatch: configure the official TITA order explicitly; "
                f"expected {self.joint_names}, got {tuple(simulator_names)}"
            )

    def apply_joint_targets(self, q_cmd: Any) -> None:
        if self.articulation is None:
            raise RuntimeError("Isaac Sim articulation is not bound")
        if getattr(q_cmd, "shape", None) is None or q_cmd.shape[-1] != 8:
            raise ValueError("q_cmd must end in 8 joints")
        # This is deliberately an explicit hook instead of guessing Isaac Lab APIs.
        raise NotImplementedError("bind the Isaac Sim 4.5 Articulation API in the task adapter")
