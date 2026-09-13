"""Deterministic simulation-only tunnel dimension curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TunnelCurriculumConfig:
    height_range_m: tuple[float, float] = (0.34, 0.44)
    width_range_m: tuple[float, float] = (0.40, 0.65)
    length_range_m: tuple[float, float] = (0.60, 1.50)
    start_x_m: float = 0.75
    wall_thickness_m: float = 0.10

    def __post_init__(self) -> None:
        for name, value in (
            ("height", self.height_range_m),
            ("width", self.width_range_m),
            ("length", self.length_range_m),
        ):
            if len(value) != 2 or value[0] <= 0.0 or value[1] < value[0]:
                raise ValueError(f"tunnel {name} range must be positive and ordered")
        if self.start_x_m <= 0.0 or self.wall_thickness_m <= 0.0:
            raise ValueError("tunnel start and wall thickness must be positive")


def sample_tunnel_dimensions(
    num_envs: int,
    seed: int,
    config: TunnelCurriculumConfig | None = None,
) -> np.ndarray:
    """Return height, width, and length for each vectorized environment."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    cfg = config or TunnelCurriculumConfig()
    rng = np.random.default_rng(int(seed))
    low = np.array(
        (cfg.height_range_m[0], cfg.width_range_m[0], cfg.length_range_m[0]),
        dtype=np.float32,
    )
    high = np.array(
        (cfg.height_range_m[1], cfg.width_range_m[1], cfg.length_range_m[1]),
        dtype=np.float32,
    )
    return rng.uniform(low, high, size=(num_envs, 3)).astype(np.float32)
