"""Cost-vector interface used by an NP3O-style multi-cost learner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .unified_objective import ObjectiveTerms


@dataclass(frozen=True)
class CostVector:
    route: float
    energy: float
    mode: float
    smooth: float

    @property
    def values(self) -> np.ndarray:
        return np.asarray([self.route, self.energy, self.mode, self.smooth], dtype=np.float64)

    @classmethod
    def from_terms(cls, terms: ObjectiveTerms) -> "CostVector":
        return cls(terms.route, terms.energy, terms.mode, terms.smooth)


@dataclass
class NP3OCostAccumulator:
    """Minimal rollout accumulator; hard constraints remain a separate channel."""

    total: np.ndarray = None  # type: ignore[assignment]
    hard_violations: int = 0
    steps: int = 0

    def __post_init__(self) -> None:
        if self.total is None:
            self.total = np.zeros(4, dtype=np.float64)

    def add(self, costs: CostVector, hard_violation: bool = False) -> None:
        values = costs.values
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("cost vector must be finite and have four components")
        self.total += values
        self.steps += 1
        self.hard_violations += int(hard_violation)

    def mean(self) -> CostVector:
        values = self.total / max(1, self.steps)
        return CostVector(*[float(v) for v in values])

    def critic_target(self, bootstrap: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        """Return a vector target for a multi-head cost critic.

        A real NP3O runner can replace this method with its constrained GAE;
        keeping the vector shape here prevents accidental scalarization.
        """
        bootstrap = np.asarray(bootstrap, dtype=np.float64)
        if bootstrap.shape != (4,):
            raise ValueError("bootstrap must have shape (4,)")
        return self.total + float(gamma) ** self.steps * bootstrap
