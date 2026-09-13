"""TITA unified-objective residual-control prototype."""

from .config import ControllerConfig, DEFAULT_CONFIG
from .safety_projector import SafetyProjector, SafetyResult, SafetyStatus
from .unified_objective import UnifiedObjective
from .trainer import ResidualNP3OTrainer
from .residual_policy import OfficialBasePolicyAdapter, ResidualPolicy
from .official_policy import FrozenOfficialTitaController, OfficialTitaActor

__all__ = [
    "ControllerConfig",
    "DEFAULT_CONFIG",
    "SafetyProjector",
    "SafetyResult",
    "SafetyStatus",
    "UnifiedObjective",
    "ResidualNP3OTrainer",
    "ResidualPolicy",
    "OfficialBasePolicyAdapter",
    "OfficialTitaActor",
    "FrozenOfficialTitaController",
]
