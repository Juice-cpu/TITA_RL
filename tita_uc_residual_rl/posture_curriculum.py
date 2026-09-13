"""Upper-level LOW/MID/HIGH command curriculum for residual training.

The scheduler owns posture selection.  The low-level residual policy only
observes the resulting command and transition phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


POSTURE_NAMES = ("LOW", "MID", "HIGH")


@dataclass(frozen=True)
class PostureCurriculumConfig:
    control_dt: float = 0.02
    resample_time_s: float = 3.0
    transition_time_s: float = 1.5
    switch_hold_time_s: float = 0.20
    switch_speed_max: float = 0.20
    switch_roll_max: float = 0.12
    switch_pitch_max: float = 0.12
    initial_mode: int = 1

    def __post_init__(self) -> None:
        if self.control_dt <= 0.0 or self.resample_time_s <= 0.0:
            raise ValueError("control and posture resampling times must be positive")
        if self.transition_time_s <= 0.0 or self.switch_hold_time_s < 0.0:
            raise ValueError("posture transition times are invalid")
        if min(self.switch_speed_max, self.switch_roll_max, self.switch_pitch_max) <= 0.0:
            raise ValueError("posture switch limits must be positive")
        if self.initial_mode not in range(len(POSTURE_NAMES)):
            raise ValueError("initial posture mode must be LOW=0, MID=1, or HIGH=2")


@dataclass(frozen=True)
class PostureCommandStep:
    command: torch.Tensor
    phase: torch.Tensor
    slow_down: torch.Tensor
    transition: torch.Tensor
    started: torch.Tensor
    completed: torch.Tensor


class PostureCommandScheduler:
    """Vectorized posture state machine with deterministic safety gating."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        config: PostureCurriculumConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = config or PostureCurriculumConfig()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.current_mode = torch.full(
            (num_envs,), self.cfg.initial_mode, dtype=torch.long, device=self.device
        )
        self.requested_mode = self.current_mode.clone()
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.transition = torch.zeros_like(self.pending)
        self.stable_elapsed = torch.zeros(num_envs, device=self.device)
        self.phase = torch.zeros(num_envs, device=self.device)
        self.time_until_request = torch.full(
            (num_envs,), self.cfg.resample_time_s, device=self.device
        )
        self.allowed_modes = (self.cfg.initial_mode,)

    def set_allowed_modes(self, modes: tuple[int, ...]) -> None:
        normalized = tuple(sorted(set(int(mode) for mode in modes)))
        if not normalized or any(mode not in range(len(POSTURE_NAMES)) for mode in normalized):
            raise ValueError("allowed posture modes must contain LOW=0, MID=1, or HIGH=2")
        self.allowed_modes = normalized

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if mask.shape != (self.num_envs,):
            raise ValueError("posture reset mask has the wrong shape")
        self.current_mode[mask] = self.cfg.initial_mode
        self.requested_mode[mask] = self.cfg.initial_mode
        self.pending[mask] = False
        self.transition[mask] = False
        self.stable_elapsed[mask] = 0.0
        self.phase[mask] = 0.0
        self.time_until_request[mask] = self.cfg.resample_time_s

    def _request_new_modes(self, mask: torch.Tensor) -> None:
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        allowed = torch.tensor(self.allowed_modes, dtype=torch.long, device=self.device)
        if allowed.numel() == 1:
            sampled = allowed.expand(env_ids.numel())
        else:
            choices = torch.randint(allowed.numel(), (env_ids.numel(),), device=self.device)
            sampled = allowed[choices]
            same = sampled == self.current_mode[env_ids]
            choices[same] = (choices[same] + 1) % allowed.numel()
            sampled = allowed[choices]
        self.requested_mode[env_ids] = sampled
        changed = sampled != self.current_mode[env_ids]
        self.pending[env_ids] = changed
        self.stable_elapsed[env_ids] = 0.0
        self.time_until_request[env_ids] = self.cfg.resample_time_s

    def step(
        self,
        speed: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
    ) -> PostureCommandStep:
        for name, value in (("speed", speed), ("roll", roll), ("pitch", pitch)):
            if value.shape != (self.num_envs,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector with one value per environment")

        idle = ~(self.pending | self.transition)
        self.time_until_request[idle] -= self.cfg.control_dt
        self._request_new_modes(idle & (self.time_until_request <= 0.0))

        stable = (
            (torch.abs(speed) <= self.cfg.switch_speed_max)
            & (torch.abs(roll) <= self.cfg.switch_roll_max)
            & (torch.abs(pitch) <= self.cfg.switch_pitch_max)
        )
        self.stable_elapsed = torch.where(
            self.pending & stable,
            self.stable_elapsed + self.cfg.control_dt,
            torch.zeros_like(self.stable_elapsed),
        )
        started = self.pending & (self.stable_elapsed >= self.cfg.switch_hold_time_s)
        self.pending[started] = False
        self.transition[started] = True
        self.phase[started] = 0.0

        active_before_advance = self.transition.clone()
        self.phase[active_before_advance] += self.cfg.control_dt / self.cfg.transition_time_s
        completed = active_before_advance & (self.phase >= 1.0)
        self.current_mode[completed] = self.requested_mode[completed]
        self.transition[completed] = False
        self.phase[completed] = 0.0
        self.time_until_request[completed] = self.cfg.resample_time_s

        commanded_mode = torch.where(self.transition, self.requested_mode, self.current_mode)
        command = torch.nn.functional.one_hot(commanded_mode, len(POSTURE_NAMES)).to(torch.float32)
        slow_down = self.pending | self.transition
        return PostureCommandStep(
            command=command,
            phase=self.phase[:, None].clone(),
            slow_down=slow_down.clone(),
            transition=self.transition.clone(),
            started=started,
            completed=completed,
        )
