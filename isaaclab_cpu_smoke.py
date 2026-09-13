"""Minimal Isaac Lab 2.0/Isaac Sim 4.5 smoke with an explicit CPU device."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=0.01, device="cpu"))
    sim.reset()
    for index in range(5):
        sim.step()
        print(f"[TITA-UC] Isaac Lab CPU step {index + 1}/5")


if __name__ == "__main__":
    main()
    simulation_app.close()
