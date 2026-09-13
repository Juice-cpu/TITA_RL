"""Deterministic simulation-only stair dimension curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StairCurriculumConfig:
    step_height_range_m: tuple[float, float] = (0.025, 0.045)
    tread_range_m: tuple[float, float] = (0.55, 0.67)
    width_range_m: tuple[float, float] = (0.90, 1.20)
    num_steps: int = 3
    start_x_m: float = 0.75
    landing_length_m: float = 0.80

    def __post_init__(self) -> None:
        for name, value in (
            ("step height", self.step_height_range_m),
            ("tread", self.tread_range_m),
            ("width", self.width_range_m),
        ):
            if len(value) != 2 or value[0] <= 0.0 or value[1] < value[0]:
                raise ValueError(f"stair {name} range must be positive and ordered")
        if self.num_steps <= 0:
            raise ValueError("stair num_steps must be positive")
        if self.start_x_m <= 0.0 or self.landing_length_m <= 0.0:
            raise ValueError("stair start and landing length must be positive")


def sample_stair_dimensions(
    num_envs: int,
    seed: int,
    config: StairCurriculumConfig | None = None,
) -> np.ndarray:
    """Return step height, tread length, and width for each environment."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    cfg = config or StairCurriculumConfig()
    rng = np.random.default_rng(int(seed))
    low = np.array(
        (cfg.step_height_range_m[0], cfg.tread_range_m[0], cfg.width_range_m[0]),
        dtype=np.float32,
    )
    high = np.array(
        (cfg.step_height_range_m[1], cfg.tread_range_m[1], cfg.width_range_m[1]),
        dtype=np.float32,
    )
    return rng.uniform(low, high, size=(num_envs, 3)).astype(np.float32)
