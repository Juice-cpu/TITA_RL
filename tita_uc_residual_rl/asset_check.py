"""Asset checks used before creating an Isaac Lab Articulation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .config import TITA_JOINT_NAMES


def urdf_joint_names(path: str | Path, include_fixed: bool = False) -> tuple[str, ...]:
    """Return URDF joints in file order, excluding fixed joints by default."""
    root = ET.parse(Path(path)).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if include_fixed or joint_type != "fixed":
            names.append(joint.attrib["name"])
    return tuple(names)


def require_eight_dof(path: str | Path, expected: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Fail early instead of silently mapping a non-8-DOF asset to 8 actions."""
    names = urdf_joint_names(path)
    if len(names) != 8:
        raise ValueError(
            f"{path} exposes {len(names)} non-fixed URDF joints, but TITA-UC requires exactly 8; "
            "provide the official 8-DOF asset or an explicit reduction map"
        )
    if expected is not None and names != expected:
        raise ValueError(f"URDF joint order mismatch: expected {expected}, got {names}")
    return names


def require_official_tita_urdf(path: str | Path) -> tuple[str, ...]:
    """Validate the exact official 8-joint order used by ``tita_rl``."""
    return require_eight_dof(path, expected=TITA_JOINT_NAMES)
