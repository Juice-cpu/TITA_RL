#!/usr/bin/env python3
"""Validate that an asset matches the fixed 8-DOF controller contract."""

from __future__ import annotations

import argparse

from tita_uc_residual_rl.asset_check import require_official_tita_urdf, urdf_joint_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf")
    args = parser.parse_args()
    names = urdf_joint_names(args.urdf)
    print(f"non-fixed joints ({len(names)}):")
    for index, name in enumerate(names):
        print(f"  {index}: {name}")
    require_official_tita_urdf(args.urdf)
    print("official TITA 8-DOF order: OK")


if __name__ == "__main__":
    main()
