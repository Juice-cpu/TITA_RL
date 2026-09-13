"""Isaac Lab 2.0 DirectRLEnv for an 8-DOF TITA residual controller.

The file is intentionally not imported by the simulator-independent package
tests.  Launch it through ``isaaclab.sh -p`` after supplying an 8-DOF USD or
URDF asset with ``TITA_USD_PATH``/``TITA_URDF_PATH``.  The local ROS URDF is
not silently accepted because it exposes a different joint contract.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import trimesh

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from .asset_check import require_official_tita_urdf
from .config import TITA_JOINT_NAMES
from .official_policy import build_official_proprioception
from .safety_actuation import (
    apply_velocity_torque_guard,
    filter_residual_action,
    project_joint_velocity,
)
from .stair_curriculum import StairCurriculumConfig, sample_stair_dimensions
from .tunnel_curriculum import TunnelCurriculumConfig, sample_tunnel_dimensions


HARD_REASON_NAMES = (
    "infeasible_projection",
    "velocity",
    "acceleration",
    "torque",
    "collision",
    "height",
    "attitude",
    "clearance",
)


@configclass
class TitaDomainRandomizationCfg:
    """Conservative simulation-only Phase B2 randomization defaults."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )


@configclass
class TitaResidualEnvCfg(DirectRLEnvCfg):
    """Configuration preserving the official 8-D joint-target interface."""

    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 8
    # root lin vel (3), root ang vel (3), gravity (3), q (8), qd (8),
    # previous residual (8), command (3), route preview (4), posture command
    # and actual (6), phase (1), energy/path/goal state (3), stair height (1).
    observation_space = 51
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        device="cuda:0",
        disable_contact_processing=True,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )
    height_scanner: RayCasterCfg | None = None
    perception_preview_mesh = False
    tunnel_enabled = False
    tunnel_height_range_m = (0.34, 0.44)
    tunnel_width_range_m = (0.40, 0.65)
    tunnel_length_range_m = (0.60, 1.50)
    tunnel_start_x_m = 0.75
    tunnel_wall_thickness_m = 0.10
    tunnel_preview_distance_m = 1.50
    tunnel_clearance_buffer_m = 0.01
    tunnel_success_reward = 10.0
    stair_enabled = False
    stair_step_height_range_m = (0.025, 0.045)
    stair_tread_range_m = (0.55, 0.67)
    stair_width_range_m = (0.90, 1.20)
    stair_num_steps = 3
    stair_start_x_m = 0.75
    stair_landing_length_m = 0.80
    stair_handrail_enabled = False
    stair_handrail_height_m = 0.90
    stair_handrail_thickness_m = 0.05
    stair_handrail_side_offset_m = 0.06
    stair_handrail_post_spacing_m = 0.90
    stair_handrail_follow_slope = True
    stair_preview_distance_m = 1.50
    stair_success_reward = 12.0
    stair_terminate_on_success = True
    stair_height_progress_reward_scale = 4.0
    stair_step_progress_reward = 3.0
    stair_level_hold_reward_scale = 0.0
    stair_approach_lift_reward_scale = 0.0
    stair_approach_lift_distance_m = 0.30
    stair_collision_penalty = 0.0
    stair_collision_terminates = True
    stair_jump_assist_enabled = False
    stair_jump_assist_trigger_distance_m = 0.08
    stair_jump_assist_crouch_steps = 6
    stair_jump_assist_extension_steps = 12
    stair_jump_assist_hip_crouch_rad = 0.15
    stair_jump_assist_hip_extension_rad = 0.30
    stair_jump_assist_knee_crouch_rad = 0.70
    stair_jump_assist_knee_extension_rad = 0.80
    stair_completion_height_tolerance_m = 0.005
    stair_start_delay_steps = 75
    stair_height_preview_reference_m = 0.10
    base_collision_half_length_m = 0.235
    base_collision_half_width_m = 0.15
    base_collision_half_height_m = 0.095
    base_collision_center_z_m = 0.017

    residual_limit = 0.20
    posture_scale = 0.75
    residual_filter_alpha = 0.25
    joint_target_change_reference = 0.50
    tracking_reward_scale = 2.0
    route_reward_scale = 0.40
    energy_cost_scale = 0.35
    mode_cost_scale = 0.25
    smooth_cost_scale = 0.05
    route_length_weight = 0.40
    route_clearance_weight = 0.35
    route_curvature_weight = 0.25
    route_reference_length = 5.0
    clearance_sigma = 0.30
    corridor_half_width = 0.50
    posture_heights = (0.10, 0.20, 0.30)
    # Simulation-only pose anchors calibrated by scripts/calibrate_posture_poses.py
    # with the frozen official balancing policy, including a 0.20 m/s stability gate.
    posture_joint_anchors = (
        (0.0, 0.76, -2.150, 0.0, 0.0, 0.76, -2.150, 0.0),
        (0.0, 0.90, -1.800, 0.0, 0.0, 0.90, -1.800, 0.0),
        (0.0, 0.90, -1.100, 0.0, 0.0, 0.90, -1.100, 0.0),
    )
    posture_feedforward_transition_s = 1.5
    enable_posture_feedforward = True
    wheel_radius = 0.0925
    height_rate_reference = 0.10
    mode_height_rate_weight = 0.50
    mode_hold_weight = 0.50
    # Frozen offline P95 values. Keep at one only while collecting the
    # baseline calibration set, then write the measured values to config.
    objective_p95 = (0.0007925402, 1.9832050, 0.6782783, 0.06604067)
    normalized_cost_clip = float("inf")
    hard_failure_cost = 10.0
    clearance_default = 1.0
    safe_clearance = 0.20
    clearance_hold_margin = 0.0
    clearance_release_hysteresis = 0.10
    terrain_step_threshold = 0.055
    terrain_drop_threshold = 0.075
    terrain_preview_threshold = 0.002
    hard_clearance_half_width = 0.25
    collision_force_threshold = 20.0
    soft_contact_cost_weight = 0.10
    reset_stabilization_steps = 10
    # Asset limits from the official TITA URDF.
    leg_torque_limit = 60.0
    wheel_torque_limit = 15.0
    leg_velocity_limit = 25.0
    wheel_velocity_limit = 20.0
    velocity_safety_margin = 1.0
    # This remains disabled until an official-policy simulation baseline is
    # measured. The URDF does not define an acceleration limit.
    acceleration_limit = 80.0
    acceleration_filter_time_constant = 0.10
    enforce_acceleration_limit = False
    enable_safety_projector = True
    project_target_kinematics = False
    # Official leg PD and wheel torque-proxy gains.
    leg_kp = 40.0
    leg_kd = 1.0
    wheel_target_gain = 10.0
    wheel_damping = 0.5
    wheel_hold_brake_torque = 3.0

    robot: ArticulationCfg = None  # type: ignore[assignment]


def build_tita_cfg(
    asset_path: str | None = None,
    *,
    num_envs: int = 1024,
    terrain: str = "flat",
    domain_randomization: bool = False,
) -> TitaResidualEnvCfg:
    """Build a task configuration from an explicit 8-DOF USD/URDF path."""
    asset = asset_path or os.environ.get("TITA_USD_PATH")
    if not asset:
        raise ValueError("set TITA_USD_PATH to the converted official 8-DOF TITA USD")
    asset_path_obj = Path(asset).expanduser()
    if not asset_path_obj.exists():
        raise FileNotFoundError(asset_path_obj)
    cfg = TitaResidualEnvCfg()
    # Isaac Lab normally runs on CUDA, but keeping this overridable makes
    # diagnostics reproducible on machines where only the CPU backend is
    # available (for example CI or a driver-recovery session).
    cfg.sim.device = os.environ.get("TITA_SIM_DEVICE", cfg.sim.device)
    cfg.scene.num_envs = int(num_envs)
    if terrain not in ("flat", "rough_warmup", "tunnel", "stairs"):
        raise ValueError("terrain must be 'flat', 'rough_warmup', 'tunnel', or 'stairs'")
    if terrain == "rough_warmup":
        cfg.perception_preview_mesh = True
        cfg.clearance_hold_margin = 0.35
        cfg.height_scanner = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 2.2)),
            max_distance=3.0,
            debug_vis=False,
            mesh_prim_paths=["/World/perception_preview/terrain"],
        )
    elif terrain == "tunnel":
        cfg.tunnel_enabled = True
        cfg.scene.env_spacing = 4.5
    elif terrain == "stairs":
        cfg.stair_enabled = True
        cfg.scene.env_spacing = 6.0
    if domain_randomization:
        cfg.events = TitaDomainRandomizationCfg()
    if asset_path_obj.suffix.lower() == ".urdf":
        require_official_tita_urdf(asset_path_obj)
        spawn = sim_utils.UrdfFileCfg(
            asset_path=str(asset_path_obj),
            activate_contact_sensors=True,
            fix_base=False,
            merge_fixed_joints=False,
            make_instanceable=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                )
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        )
    else:
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(asset_path_obj), activate_contact_sensors=True
        )
    cfg.robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn,
        # Match the official TITA RL default pose.  In particular, leg_3 has
        # a negative range in the URDF and cannot be initialized at zero.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.30),
            joint_pos={
                "joint_left_leg_1": 0.0,
                "joint_left_leg_2": 0.8,
                "joint_left_leg_3": -1.5,
                "joint_left_leg_4": 0.0,
                "joint_right_leg_1": 0.0,
                "joint_right_leg_2": 0.8,
                "joint_right_leg_3": -1.5,
                "joint_right_leg_4": 0.0,
            },
        ),
        actuators={
            "tita_legs": ImplicitActuatorCfg(
                joint_names_expr=["joint_.*_leg_[123]"], stiffness=0.0, damping=0.0,
                effort_limit=cfg.leg_torque_limit,
                velocity_limit=cfg.leg_velocity_limit - cfg.velocity_safety_margin,
            ),
            "tita_wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_.*_leg_4"], stiffness=0.0, damping=0.0,
                effort_limit=cfg.wheel_torque_limit,
                velocity_limit=cfg.wheel_velocity_limit - cfg.velocity_safety_margin,
            ),
        },
    )
    return cfg


class TitaResidualEnv(DirectRLEnv):
    """Vectorized Isaac Lab task with target-space residual and hard gating."""

    cfg: TitaResidualEnvCfg

    def __init__(self, cfg: TitaResidualEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.robot is None:
            raise ValueError("cfg.robot is not configured; call build_tita_cfg first")
        super().__init__(cfg, render_mode, **kwargs)
        sampled_tunnels = getattr(
            self,
            "_sampled_tunnel_dimensions",
            np.zeros((self.num_envs, 3), dtype=np.float32),
        )
        self._tunnel_dimensions = torch.as_tensor(
            sampled_tunnels, dtype=torch.float32, device=self.device
        )
        sampled_stairs = getattr(
            self,
            "_sampled_stair_dimensions",
            np.zeros((self.num_envs, 3), dtype=np.float32),
        )
        self._stair_dimensions = torch.as_tensor(
            sampled_stairs, dtype=torch.float32, device=self.device
        )
        self._joint_ids, resolved_names = self.robot.find_joints(list(TITA_JOINT_NAMES), preserve_order=True)
        if tuple(resolved_names) != TITA_JOINT_NAMES:
            raise ValueError(f"Isaac Lab joint order mismatch: expected {TITA_JOINT_NAMES}, got {resolved_names}")
        self._actions = torch.zeros(self.num_envs, 8, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._base_target = self.robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._q_cmd = self._base_target.clone()
        self._previous_q_cmd = self._q_cmd.clone()
        self._hard_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tunnel_success_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._tunnel_success_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stair_success_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stair_success_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._distance = torch.zeros(self.num_envs, device=self.device)
        self._last_step_distance = torch.zeros_like(self._distance)
        self._energy = torch.zeros_like(self._distance)
        self._previous_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._previous_root_height = self.robot.data.root_pos_w[:, 2].clone()
        self._velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._soft_costs = torch.zeros(self.num_envs, 4, device=self.device)
        self._raw_soft_costs = torch.zeros_like(self._soft_costs)
        self._smooth_components = torch.zeros(self.num_envs, 4, device=self.device)
        self._projected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._safety_hold = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._posture_command = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._posture_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._posture_offset = torch.zeros(self.num_envs, 8, device=self.device)
        self._path_version = torch.zeros(self.num_envs, 1, device=self.device)
        self._goal_age = torch.zeros(self.num_envs, 1, device=self.device)
        self._hard_reason_flags = torch.zeros(
            self.num_envs, len(HARD_REASON_NAMES), dtype=torch.bool, device=self.device
        )
        self._last_hard_reason_flags = torch.zeros_like(self._hard_reason_flags)
        self._last_velocity_violation_flags = torch.zeros(self.num_envs, 8, dtype=torch.bool, device=self.device)
        self._last_velocity_state_projection_flags = torch.zeros_like(
            self._last_velocity_violation_flags
        )
        self._last_joint_velocity_abs = torch.zeros(self.num_envs, 8, device=self.device)
        self._last_preprojection_joint_velocity_abs = torch.zeros_like(
            self._last_joint_velocity_abs
        )
        self._last_joint_acceleration_abs = torch.zeros(self.num_envs, 8, device=self.device)
        self._filtered_joint_acceleration = torch.zeros(
            self.num_envs, 8, device=self.device
        )
        self._previous_joint_velocity = self.robot.data.joint_vel[
            :, self._joint_ids
        ].clone()
        self._last_infeasible_joint_flags = torch.zeros(
            self.num_envs, 8, dtype=torch.bool, device=self.device
        )
        self._last_infeasible_cause_flags = torch.zeros(
            self.num_envs, 8, 4, dtype=torch.bool, device=self.device
        )
        self._last_projection_joint_flags = torch.zeros(
            self.num_envs, 8, dtype=torch.bool, device=self.device
        )
        self._last_projection_cause_flags = torch.zeros(
            self.num_envs, 8, 4, dtype=torch.bool, device=self.device
        )
        self._previous_posture_height = (
            self.robot.data.root_pos_w[:, 2] - self.cfg.wheel_radius
        ).clone()
        self._collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._soft_contact_force = torch.zeros(self.num_envs, device=self.device)
        self._clearance = torch.full(
            (self.num_envs,), self.cfg.clearance_default, device=self.device
        )
        self._hard_clearance = torch.full_like(
            self._clearance, self.cfg.clearance_default
        )
        self._stair_height_preview = torch.zeros(self.num_envs, device=self.device)
        self._stair_approach_root_height = self.robot.data.root_pos_w[:, 2].clone()
        self._stair_approach_height_frozen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_verified_stair_level = torch.zeros(
            self.num_envs, device=self.device
        )
        self._stair_jump_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._stair_jump_level = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._hard_contact_body_ids, hard_names = self._contact_sensor.find_bodies(
            "base_link"
        )
        self._soft_contact_body_ids, soft_names = self._contact_sensor.find_bodies(
            ".*leg_3"
        )
        if len(self._hard_contact_body_ids) != 1:
            raise RuntimeError("failed to resolve the TITA base_link hard-contact body")
        if len(self._soft_contact_body_ids) != 2:
            raise RuntimeError("failed to resolve the two TITA leg_3 soft-contact bodies")
        print(f"[TITA-UC] hard contact bodies: {hard_names}")
        print(f"[TITA-UC] soft contact bodies: {soft_names}")
        if self.cfg.tunnel_enabled:
            minimum = self._tunnel_dimensions.amin(dim=0).tolist()
            maximum = self._tunnel_dimensions.amax(dim=0).tolist()
            print(
                "[TITA-UC] tunnel dimensions height/width/length "
                f"min={minimum} max={maximum}"
            )
        if self.cfg.stair_enabled:
            minimum = self._stair_dimensions.amin(dim=0).tolist()
            maximum = self._stair_dimensions.amax(dim=0).tolist()
            print(
                "[TITA-UC] stair dimensions height/tread/width "
                f"min={minimum} max={maximum} steps={self.cfg.stair_num_steps}"
            )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        if self.cfg.perception_preview_mesh:
            self._create_perception_preview_mesh()
        if self.cfg.height_scanner is not None:
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.tunnel_enabled:
            self._create_tunnel_curriculum()
        if self.cfg.stair_enabled:
            self._create_stair_curriculum()
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_tunnel_curriculum(self) -> None:
        tunnel_cfg = TunnelCurriculumConfig(
            height_range_m=tuple(self.cfg.tunnel_height_range_m),
            width_range_m=tuple(self.cfg.tunnel_width_range_m),
            length_range_m=tuple(self.cfg.tunnel_length_range_m),
            start_x_m=self.cfg.tunnel_start_x_m,
            wall_thickness_m=self.cfg.tunnel_wall_thickness_m,
        )
        seed = int(self.cfg.seed if self.cfg.seed is not None else 0) + 4100
        dimensions = sample_tunnel_dimensions(self.cfg.scene.num_envs, seed, tunnel_cfg)
        self._sampled_tunnel_dimensions = dimensions
        thickness = tunnel_cfg.wall_thickness_m
        wall_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.27, 0.29))
        ceiling_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.46, 0.49, 0.50))
        for env_index, (height, width, length) in enumerate(dimensions):
            # Children authored below env_0 are inherited by Isaac Lab's
            # cloned env prims even when the shape spawner itself is not
            # cloned.  Put each static tunnel in an independent world-space
            # namespace and offset it by the corresponding environment origin.
            env_origin = self.terrain.env_origins[env_index].cpu().tolist()
            center_x = env_origin[0] + tunnel_cfg.start_x_m + float(length) / 2.0
            center_y = env_origin[1]
            wall_size = (float(length), thickness, float(height) + thickness)
            for side_name, side in (("Left", 1.0), ("Right", -1.0)):
                wall = sim_utils.CuboidCfg(
                    size=wall_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=wall_material,
                )
                # The public spawner is decorated with Isaac Lab's ``clone``
                # helper.  Even a literal env_0 path is expanded across the
                # cloned environment hierarchy, which would give every env
                # env_0's dimensions and collide with the following loop
                # iteration.  Call the wrapped single-prim spawner so each
                # vectorized environment keeps its own sampled geometry.
                wall.func.__wrapped__(
                    f"/World/tunnels/env_{env_index}/{side_name}Wall",
                    wall,
                    translation=(
                        center_x,
                        center_y + side * (float(width) + thickness) / 2.0,
                        (float(height) + thickness) / 2.0,
                    ),
                )
            ceiling = sim_utils.CuboidCfg(
                size=(float(length), float(width) + 2.0 * thickness, thickness),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=ceiling_material,
            )
            ceiling.func.__wrapped__(
                f"/World/tunnels/env_{env_index}/Ceiling",
                ceiling,
                translation=(
                    center_x,
                    center_y,
                    float(height) + thickness / 2.0,
                ),
            )

    def _create_stair_curriculum(self) -> None:
        stair_cfg = StairCurriculumConfig(
            step_height_range_m=tuple(self.cfg.stair_step_height_range_m),
            tread_range_m=tuple(self.cfg.stair_tread_range_m),
            width_range_m=tuple(self.cfg.stair_width_range_m),
            num_steps=self.cfg.stair_num_steps,
            start_x_m=self.cfg.stair_start_x_m,
            landing_length_m=self.cfg.stair_landing_length_m,
        )
        seed = int(self.cfg.seed if self.cfg.seed is not None else 0) + 5100
        dimensions = sample_stair_dimensions(self.cfg.scene.num_envs, seed, stair_cfg)
        self._sampled_stair_dimensions = dimensions
        materials = (
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.36, 0.38)),
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.50, 0.52)),
        )
        for env_index, (step_height, tread, width) in enumerate(dimensions):
            env_origin = self.terrain.env_origins[env_index].cpu().tolist()
            for step_index in range(stair_cfg.num_steps):
                segment_length = float(tread)
                if step_index == stair_cfg.num_steps - 1:
                    segment_length += stair_cfg.landing_length_m
                top_height = float(step_height) * (step_index + 1)
                center_x = (
                    env_origin[0]
                    + stair_cfg.start_x_m
                    + step_index * float(tread)
                    + segment_length / 2.0
                )
                step = sim_utils.CuboidCfg(
                    size=(segment_length, float(width), top_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=materials[step_index % len(materials)],
                )
                step.func.__wrapped__(
                    f"/World/stairs/env_{env_index}/Step_{step_index + 1}",
                    step,
                    translation=(
                        center_x,
                        env_origin[1],
                        top_height / 2.0,
                    ),
                )
            if self.cfg.stair_handrail_enabled:
                self._create_stair_handrails(
                    env_index,
                    env_origin,
                    float(step_height),
                    float(tread),
                    float(width),
                    stair_cfg,
                )

    def _create_stair_handrails(
        self,
        env_index: int,
        env_origin: list[float],
        step_height: float,
        tread: float,
        width: float,
        stair_cfg: StairCurriculumConfig,
    ) -> None:
        """Create side handrails as static collision geometry for one stairway."""
        thickness = float(self.cfg.stair_handrail_thickness_m)
        side_y = float(width) / 2.0 + float(self.cfg.stair_handrail_side_offset_m)
        rail_height = float(self.cfg.stair_handrail_height_m)
        total_run = stair_cfg.num_steps * tread
        post_spacing = max(float(self.cfg.stair_handrail_post_spacing_m), thickness)
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.17, 0.18))
        for side_name, side in (("Left", 1.0), ("Right", -1.0)):
            # Posts are placed on the first riser, at each step boundary, and
            # at the end landing.  They remain outside the clear stair width.
            post_count = max(2, int(math.ceil(total_run / post_spacing)) + 1)
            for post_index in range(post_count):
                along = min(total_run, post_index * total_run / (post_count - 1))
                level = min(stair_cfg.num_steps, int(math.floor(along / tread)) + 1)
                base_z = step_height * level
                post = sim_utils.CuboidCfg(
                    size=(thickness, thickness, rail_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=material,
                )
                post.func.__wrapped__(
                    f"/World/stairs/env_{env_index}/{side_name}RailPost_{post_index + 1}",
                    post,
                    translation=(
                        env_origin[0] + stair_cfg.start_x_m + along,
                        env_origin[1] + side * side_y,
                        base_z + rail_height / 2.0,
                    ),
                )
            # The measured staircase has a continuous rail following the
            # stair slope, then a level segment over the top landing.  A
            # single sloped cuboid avoids artificial horizontal ledges at
            # every riser while retaining a collidable rail envelope.
            total_run = stair_cfg.num_steps * tread
            total_rise = stair_cfg.num_steps * step_height
            slope_length = math.hypot(total_run, total_rise)
            slope_angle = math.atan2(total_rise, total_run)
            if self.cfg.stair_handrail_follow_slope:
                rail = sim_utils.CuboidCfg(
                    size=(slope_length, thickness, thickness),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=material,
                )
                rail.func.__wrapped__(
                    f"/World/stairs/env_{env_index}/{side_name}RailSlope",
                    rail,
                    translation=(
                        env_origin[0] + stair_cfg.start_x_m + total_run / 2.0,
                        env_origin[1] + side * side_y,
                        step_height + rail_height + total_rise / 2.0,
                    ),
                    # Isaac uses (w, x, y, z); negative Y rotates +X toward +Z.
                    orientation=(
                        math.cos(slope_angle / 2.0),
                        0.0,
                        -math.sin(slope_angle / 2.0),
                        0.0,
                    ),
                )
                landing = sim_utils.CuboidCfg(
                    size=(stair_cfg.landing_length_m, thickness, thickness),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=material,
                )
                landing.func.__wrapped__(
                    f"/World/stairs/env_{env_index}/{side_name}RailLanding",
                    landing,
                    translation=(
                        env_origin[0]
                        + stair_cfg.start_x_m
                        + total_run
                        + stair_cfg.landing_length_m / 2.0,
                        env_origin[1] + side * side_y,
                        total_rise + rail_height,
                    ),
                )
            else:
                # Compatibility mode for old checkpoints and geometry tests.
                for step_index in range(stair_cfg.num_steps):
                    segment_length = tread
                    if step_index == stair_cfg.num_steps - 1:
                        segment_length += stair_cfg.landing_length_m
                    top_height = step_height * (step_index + 1)
                    rail = sim_utils.CuboidCfg(
                        size=(segment_length, thickness, thickness),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=material,
                    )
                    rail.func.__wrapped__(
                        f"/World/stairs/env_{env_index}/{side_name}Rail_{step_index + 1}",
                        rail,
                        translation=(
                            env_origin[0]
                            + stair_cfg.start_x_m
                            + step_index * tread
                            + segment_length / 2.0,
                            env_origin[1] + side * side_y,
                            top_height + rail_height,
                        ),
                    )

    def _create_perception_preview_mesh(self) -> None:
        """Create a collision-free geometry layer for the first perception curriculum."""
        origins = self.terrain.env_origins[:, :2].cpu().numpy()
        xy_min = origins.min(axis=0) - 2.0
        xy_max = origins.max(axis=0) + 2.0
        center = 0.5 * (xy_min + xy_max)
        extent = xy_max - xy_min
        meshes = [trimesh.creation.box(extents=(extent[0], extent[1], 0.002))]
        meshes[0].apply_translation((center[0], center[1], -0.001))
        for index, origin in enumerate(origins):
            obstacle = trimesh.creation.box(extents=(0.12, 0.12, 0.08))
            lateral = 1.00 if index % 2 == 0 else -1.00
            forward = 0.75 + 0.10 * (index % 3)
            obstacle.apply_translation((origin[0] + forward, origin[1] + lateral, 0.04))
            meshes.append(obstacle)
        preview_mesh = trimesh.util.concatenate(meshes)
        create_prim_from_mesh("/World/perception_preview/terrain", preview_mesh)
        sim_utils.define_collision_properties(
            "/World/perception_preview/terrain/mesh",
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_q_cmd.copy_(self._q_cmd)
        self._previous_actions = self._actions.clone()
        residual_limit = self.cfg.residual_limit * self.cfg.posture_scale
        desired_actions = actions.clamp(-residual_limit, residual_limit)
        self._actions = filter_residual_action(
            self._previous_actions, desired_actions, self.cfg.residual_filter_alpha
        )
        if self.cfg.enable_posture_feedforward:
            anchors = self._base_target.new_tensor(self.cfg.posture_joint_anchors)
            desired_anchor = self._posture_command @ anchors
            default_anchor = self.robot.data.default_joint_pos[:, self._joint_ids]
            feedforward_alpha = min(
                self.step_dt / self.cfg.posture_feedforward_transition_s, 1.0
            )
            self._posture_offset.lerp_(
                desired_anchor - default_anchor, feedforward_alpha
            )
        else:
            self._posture_offset.zero_()
        raw = self._base_target + self._posture_offset + self._actions
        if self.cfg.stair_enabled and self.cfg.stair_jump_assist_enabled:
            raw = raw + self._stair_jump_assist()
        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self.robot.data.joint_vel[:, self._joint_ids]
        qmin = self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        qmax = self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        qdmax = q.new_tensor(
            (
                self.cfg.leg_velocity_limit, self.cfg.leg_velocity_limit,
                self.cfg.leg_velocity_limit, self.cfg.wheel_velocity_limit,
                self.cfg.leg_velocity_limit, self.cfg.leg_velocity_limit,
                self.cfg.leg_velocity_limit, self.cfg.wheel_velocity_limit,
            )
        ).expand_as(q)
        taumax = q.new_tensor(
            (
                self.cfg.leg_torque_limit, self.cfg.leg_torque_limit,
                self.cfg.leg_torque_limit, self.cfg.wheel_torque_limit,
                self.cfg.leg_torque_limit, self.cfg.leg_torque_limit,
                self.cfg.leg_torque_limit, self.cfg.wheel_torque_limit,
            )
        ).expand_as(q)
        target_gain = q.new_tensor(
            (
                self.cfg.leg_kp, self.cfg.leg_kp, self.cfg.leg_kp, self.cfg.wheel_target_gain,
                self.cfg.leg_kp, self.cfg.leg_kp, self.cfg.leg_kp, self.cfg.wheel_target_gain,
            )
        ).expand_as(q)
        position_gain = q.new_tensor(
            (self.cfg.leg_kp, self.cfg.leg_kp, self.cfg.leg_kp, 0.0,
             self.cfg.leg_kp, self.cfg.leg_kp, self.cfg.leg_kp, 0.0)
        ).expand_as(q)
        damping = q.new_tensor(
            (self.cfg.leg_kd, self.cfg.leg_kd, self.cfg.leg_kd, self.cfg.wheel_damping,
             self.cfg.leg_kd, self.cfg.leg_kd, self.cfg.leg_kd, self.cfg.wheel_damping)
        ).expand_as(q)
        position_controlled = position_gain > 0.0
        infinity = torch.full_like(q, torch.inf)

        # Deterministic vectorized equivalent of SafetyProjector for Isaac Lab.
        lower = qmin.clone()
        upper = qmax.clone()
        causes = torch.zeros(self.num_envs, 8, 4, dtype=torch.bool, device=self.device)
        projection_causes = torch.zeros_like(causes)
        causes[:, :, 0] = lower > upper
        projection_causes[:, :, 0] = (raw < lower) | (raw > upper)

        if self.cfg.project_target_kinematics:
            velocity_lower = torch.where(position_controlled, q - qdmax * self.step_dt, -infinity)
            velocity_upper = torch.where(position_controlled, q + qdmax * self.step_dt, infinity)
            projection_causes[:, :, 1] = (raw < velocity_lower) | (raw > velocity_upper)
            lower = torch.maximum(lower, velocity_lower)
            upper = torch.minimum(upper, velocity_upper)
            causes[:, :, 1] = (lower > upper) & ~causes.any(dim=2)

        torque_lower = (-taumax + position_gain * q + damping * qd) / target_gain
        torque_upper = (taumax + position_gain * q + damping * qd) / target_gain
        projection_causes[:, :, 2] = (raw < torque_lower) | (raw > torque_upper)
        lower = torch.maximum(lower, torque_lower)
        upper = torch.minimum(upper, torque_upper)
        causes[:, :, 2] = (lower > upper) & ~causes.any(dim=2)

        if self.cfg.project_target_kinematics and self.cfg.enforce_acceleration_limit:
            qd_lower = qd - self.cfg.acceleration_limit * self.step_dt
            qd_upper = qd + self.cfg.acceleration_limit * self.step_dt
            acceleration_lower = torch.where(
                position_controlled, q + qd_lower * self.step_dt, -infinity
            )
            acceleration_upper = torch.where(
                position_controlled, q + qd_upper * self.step_dt, infinity
            )
            projection_causes[:, :, 3] = (
                (raw < acceleration_lower) | (raw > acceleration_upper)
            )
            lower = torch.maximum(lower, acceleration_lower)
            upper = torch.minimum(upper, acceleration_upper)
            causes[:, :, 3] = (lower > upper) & ~causes.any(dim=2)

        infeasible_joints = lower > upper
        self._hard_failure = torch.any(infeasible_joints, dim=1)
        projected_cmd = torch.minimum(torch.maximum(raw, lower), upper)
        zero_torque_hold = (position_gain * q + damping * qd) / target_gain
        zero_torque_hold = torch.minimum(torch.maximum(zero_torque_hold, qmin), qmax)
        if self.cfg.enable_safety_projector:
            self._q_cmd = torch.where(self._hard_failure[:, None], zero_torque_hold, projected_cmd)
        else:
            self._q_cmd = raw
            self._hard_failure.zero_()
        speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        braking_clearance = 0.08 + 0.15 * speed + torch.square(speed) / 3.0 + 0.03
        hold_threshold = braking_clearance + self.cfg.clearance_hold_margin
        release_threshold = hold_threshold + self.cfg.clearance_release_hysteresis
        activate_hold = self._hard_clearance <= hold_threshold
        keep_hold = self._safety_hold & (self._hard_clearance <= release_threshold)
        self._safety_hold.copy_(activate_hold | keep_hold)
        if self.cfg.enable_safety_projector:
            hold_command = self._q_cmd.clone()
            wheel_velocity = qd[:, (3, 7)]
            brake_torque = -torch.sign(wheel_velocity) * self.cfg.wheel_hold_brake_torque
            hold_command[:, (3, 7)] = (
                brake_torque + self.cfg.wheel_damping * wheel_velocity
            ) / self.cfg.wheel_target_gain
            self._q_cmd = torch.where(
                self._safety_hold[:, None], hold_command, self._q_cmd
            )
        projection_joints = torch.abs(self._q_cmd - raw) > 1e-7
        self._projected = torch.any(projection_joints, dim=1)
        self._last_projection_joint_flags.copy_(projection_joints)
        self._last_projection_cause_flags.copy_(projection_causes)
        self._last_infeasible_joint_flags.copy_(infeasible_joints)
        self._last_infeasible_cause_flags.copy_(causes)

    def _apply_action(self):
        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self.robot.data.joint_vel[:, self._joint_ids]
        tau = torch.empty_like(q)
        leg_ids = (0, 1, 2, 4, 5, 6)
        wheel_ids = (3, 7)
        tau[:, leg_ids] = (
            self.cfg.leg_kp * (self._q_cmd[:, leg_ids] - q[:, leg_ids])
            - self.cfg.leg_kd * qd[:, leg_ids]
        )
        tau[:, wheel_ids] = (
            self.cfg.wheel_target_gain * self._q_cmd[:, wheel_ids]
            - self.cfg.wheel_damping * qd[:, wheel_ids]
        )
        limits = tau.new_tensor(
            (self.cfg.leg_torque_limit, self.cfg.leg_torque_limit, self.cfg.leg_torque_limit,
             self.cfg.wheel_torque_limit, self.cfg.leg_torque_limit, self.cfg.leg_torque_limit,
             self.cfg.leg_torque_limit, self.cfg.wheel_torque_limit)
        ).expand_as(tau)
        velocity_limits = tau.new_tensor(
            (self.cfg.leg_velocity_limit, self.cfg.leg_velocity_limit,
             self.cfg.leg_velocity_limit, self.cfg.wheel_velocity_limit,
             self.cfg.leg_velocity_limit, self.cfg.leg_velocity_limit,
             self.cfg.leg_velocity_limit, self.cfg.wheel_velocity_limit)
        ).expand_as(tau)
        tau = apply_velocity_torque_guard(
            tau, qd, limits, velocity_limits, self.cfg.velocity_safety_margin
        )
        self.robot.set_joint_effort_target(
            torch.clamp(tau, -limits, limits), joint_ids=self._joint_ids
        )

    def _stair_jump_assist(self) -> torch.Tensor:
        """Return the simulation-tested symmetric crouch/extend pulse."""
        level = self._verified_stair_level().long()
        advanced = level > self._stair_jump_level
        self._stair_jump_step[advanced] = -1
        self._stair_jump_level.copy_(level)

        newly_triggered = (self._stair_jump_step < 0) & (
            self._clearance <= float(self.cfg.stair_jump_assist_trigger_distance_m)
        )
        self._stair_jump_step[newly_triggered] = 0
        crouch_steps = int(self.cfg.stair_jump_assist_crouch_steps)
        extension_steps = int(self.cfg.stair_jump_assist_extension_steps)
        crouching = (self._stair_jump_step >= 0) & (
            self._stair_jump_step < crouch_steps
        )
        extending = (self._stair_jump_step >= crouch_steps) & (
            self._stair_jump_step < crouch_steps + extension_steps
        )

        hip_offset = self._actions.new_zeros(self.num_envs)
        knee_offset = self._actions.new_zeros(self.num_envs)
        hip_offset[crouching] = -float(self.cfg.stair_jump_assist_hip_crouch_rad)
        hip_offset[extending] = float(self.cfg.stair_jump_assist_hip_extension_rad)
        knee_offset[crouching] = -float(self.cfg.stair_jump_assist_knee_crouch_rad)
        knee_offset[extending] = float(self.cfg.stair_jump_assist_knee_extension_rad)
        assist = self._actions.new_zeros(self.num_envs, 8)
        assist[:, 1] = hip_offset
        assist[:, 2] = knee_offset
        assist[:, 5] = hip_offset
        assist[:, 6] = knee_offset
        active = self._stair_jump_step >= 0
        self._stair_jump_step[active] += 1
        return assist

    def set_base_target(self, target: torch.Tensor) -> None:
        """Set the frozen official policy target in URDF joint order."""
        if target.shape != (self.num_envs, 8) or not torch.isfinite(target).all():
            raise ValueError(f"base target must be finite with shape ({self.num_envs}, 8)")
        self._base_target.copy_(target)

    def default_joint_target(self) -> torch.Tensor:
        return self.robot.data.default_joint_pos[:, self._joint_ids]

    @property
    def soft_costs(self) -> torch.Tensor:
        """Route, mechanical-energy, mode, and smooth costs for NP3O."""
        return self._soft_costs

    @property
    def raw_soft_costs(self) -> torch.Tensor:
        return self._raw_soft_costs

    @property
    def smooth_components(self) -> torch.Tensor:
        """Target-change, residual-rate, acceleration, and contact terms."""
        return self._smooth_components

    @property
    def energy_mech(self) -> torch.Tensor:
        return self._energy

    @property
    def distance(self) -> torch.Tensor:
        return self._distance

    @property
    def last_step_distance(self) -> torch.Tensor:
        return self._last_step_distance

    @property
    def hard_failure(self) -> torch.Tensor:
        return self._hard_failure

    @property
    def projected(self) -> torch.Tensor:
        return self._projected

    @property
    def safety_hold(self) -> torch.Tensor:
        return self._safety_hold

    @property
    def tunnel_dimensions(self) -> torch.Tensor:
        """Per-environment net height, net width, and length in meters."""
        return self._tunnel_dimensions

    @property
    def tunnel_exited(self) -> torch.Tensor:
        if not self.cfg.tunnel_enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        local_x = self.robot.data.root_pos_w[:, 0] - self._environment_origins()[:, 0]
        return local_x > self.cfg.tunnel_start_x_m + self._tunnel_dimensions[:, 2]

    @property
    def tunnel_success_count(self) -> torch.Tensor:
        """Cumulative successful exits per environment; preserved across resets."""
        return self._tunnel_success_count

    @property
    def stair_dimensions(self) -> torch.Tensor:
        """Per-environment step height, tread length, and width in meters."""
        return self._stair_dimensions

    @property
    def stair_completed(self) -> torch.Tensor:
        if not self.cfg.stair_enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        local_x = self.robot.data.root_pos_w[:, 0] - self._environment_origins()[:, 0]
        tread = self._stair_dimensions[:, 1]
        completion_x = (
            self.cfg.stair_start_x_m
            + self.cfg.stair_num_steps * tread
            + 0.25 * self.cfg.stair_landing_length_m
        )
        return (local_x > completion_x) & (
            self._verified_stair_level() >= self.cfg.stair_num_steps
        )

    def _stair_support_height(self) -> torch.Tensor:
        """Height of the stair surface supporting the robot base center."""
        if not self.cfg.stair_enabled:
            return torch.zeros(self.num_envs, device=self.device)
        origins = self._environment_origins()
        local_x = self.robot.data.root_pos_w[:, 0] - origins[:, 0]
        local_y = self.robot.data.root_pos_w[:, 1] - origins[:, 1]
        height, tread, width = self._stair_dimensions.unbind(dim=1)
        level = (torch.floor((local_x - self.cfg.stair_start_x_m) / tread) + 1.0).clamp(
            min=0.0, max=float(self.cfg.stair_num_steps)
        )
        on_stair = (
            (local_x >= self.cfg.stair_start_x_m)
            & (
                local_x
                <= self.cfg.stair_start_x_m
                + self.cfg.stair_num_steps * tread
                + self.cfg.stair_landing_length_m
            )
            & (torch.abs(local_y) <= width / 2.0)
        )
        return torch.where(on_stair, level * height, torch.zeros_like(level))

    def _verified_stair_level(self) -> torch.Tensor:
        if not self.cfg.stair_enabled:
            return torch.zeros(self.num_envs, device=self.device)
        origins = self._environment_origins()
        local_x = self.robot.data.root_pos_w[:, 0] - origins[:, 0]
        local_y = self.robot.data.root_pos_w[:, 1] - origins[:, 1]
        height, tread, width = self._stair_dimensions.unbind(dim=1)
        geometric_level = (torch.floor((local_x - self.cfg.stair_start_x_m) / tread) + 1.0).clamp(
            min=0.0, max=float(self.cfg.stair_num_steps)
        )
        lateral_ok = torch.abs(local_y) <= (
            width / 2.0 - self.cfg.base_collision_half_width_m
        ).clamp_min(0.0)
        stair_end = (
            self.cfg.stair_start_x_m
            + self.cfg.stair_num_steps * tread
            + self.cfg.stair_landing_length_m
        )
        longitudinal_ok = (local_x >= self.cfg.stair_start_x_m) & (
            local_x <= stair_end
        )
        required_root_height = (
            self._stair_approach_root_height
            + geometric_level * height
            - self.cfg.stair_completion_height_tolerance_m
        )
        height_ok = self.robot.data.root_pos_w[:, 2] >= required_root_height
        return torch.where(
            longitudinal_ok & lateral_ok & height_ok,
            geometric_level,
            torch.zeros_like(geometric_level),
        )

    @property
    def stair_success_count(self) -> torch.Tensor:
        """Cumulative successful stair completions; preserved across resets."""
        return self._stair_success_count

    @property
    def clearance(self) -> torch.Tensor:
        """Nearest terrain feature from the local ray-cast preview."""
        return self._clearance

    @property
    def hard_clearance(self) -> torch.Tensor:
        """Nearest non-traversable feature used by the braking constraint."""
        return self._hard_clearance

    @property
    def posture_height(self) -> torch.Tensor:
        """Wheel-axis to body-center height used by the official command interface."""
        return (
            self.robot.data.root_pos_w[:, 2]
            - self.cfg.wheel_radius
            - self._stair_support_height()
        )

    @property
    def posture_target_height(self) -> torch.Tensor:
        levels = self._posture_command.new_tensor(self.cfg.posture_heights)
        return torch.sum(self._posture_command * levels, dim=1)

    @property
    def soft_contact_force(self) -> torch.Tensor:
        """Peak leg_3 contact force; diagnostic and soft-cost input only."""
        return self._soft_contact_force

    @property
    def hard_reason_flags(self) -> torch.Tensor:
        """Flags ordered according to :data:`HARD_REASON_NAMES`."""
        return self._last_hard_reason_flags

    @property
    def velocity_violation_flags(self) -> torch.Tensor:
        return self._last_velocity_violation_flags

    def joint_velocity_absmax(self) -> torch.Tensor:
        return self._last_joint_velocity_abs.amax(0)

    @property
    def velocity_state_projection_flags(self) -> torch.Tensor:
        return self._last_velocity_state_projection_flags

    def preprojection_joint_velocity_absmax(self) -> torch.Tensor:
        return self._last_preprojection_joint_velocity_abs.amax(0)

    def joint_acceleration_absmax(self) -> torch.Tensor:
        return self._last_joint_acceleration_abs.amax(0)

    @property
    def infeasible_joint_flags(self) -> torch.Tensor:
        return self._last_infeasible_joint_flags

    @property
    def infeasible_cause_flags(self) -> torch.Tensor:
        """Per-joint position, velocity, torque, acceleration conflict flags."""
        return self._last_infeasible_cause_flags

    @property
    def projection_joint_flags(self) -> torch.Tensor:
        return self._last_projection_joint_flags

    @property
    def projection_cause_flags(self) -> torch.Tensor:
        """Per-joint position, velocity, torque, acceleration activation flags."""
        return self._last_projection_cause_flags

    def set_velocity_command(self, command: torch.Tensor) -> None:
        if command.shape != (self.num_envs, 3) or not torch.isfinite(command).all():
            raise ValueError(f"velocity command must be finite with shape ({self.num_envs}, 3)")
        if self.cfg.stair_enabled:
            starts_moving = torch.linalg.vector_norm(command[:, :2], dim=1) > 1e-5
            freeze = starts_moving & ~self._stair_approach_height_frozen
            self._stair_approach_root_height[freeze] = torch.maximum(
                self._stair_approach_root_height[freeze],
                self.robot.data.root_pos_w[freeze, 2],
            )
            self._stair_approach_height_frozen[freeze] = True
        self._velocity_command.copy_(command)

    def set_posture_command(self, command: torch.Tensor, phase: torch.Tensor | None = None) -> None:
        """Set an upper-level one-hot LOW/MID/HIGH command without selecting it in-policy."""
        if command.shape != (self.num_envs, 3) or not torch.isfinite(command).all():
            raise ValueError(f"posture command must be finite with shape ({self.num_envs}, 3)")
        if not torch.allclose(command.sum(dim=1), torch.ones(self.num_envs, device=self.device), atol=1e-5):
            raise ValueError("posture command rows must sum to one")
        if torch.any((command < 0.0) | (command > 1.0)):
            raise ValueError("posture command must be a one-hot probability vector")
        self._posture_command.copy_(command)
        if phase is not None:
            if phase.shape != (self.num_envs, 1) or not torch.isfinite(phase).all():
                raise ValueError(f"posture phase must have shape ({self.num_envs}, 1)")
            self._posture_phase.copy_(phase.clamp(0.0, 1.0))

    def _environment_origins(self) -> torch.Tensor:
        if self.cfg.terrain.terrain_type == "generator":
            return self.terrain.env_origins
        return self.scene.env_origins

    def _update_perception(self) -> None:
        if self.cfg.height_scanner is None:
            self._clearance.fill_(self.cfg.clearance_default)
            self._hard_clearance.fill_(self.cfg.clearance_default)
        else:
            hits_z = self._height_scanner.data.ray_hits_w[..., 2]
            ray_xy = self._height_scanner.ray_starts[0, :, :2]
            center = (torch.abs(ray_xy[:, 0]) <= 0.10) & (torch.abs(ray_xy[:, 1]) <= 0.15)
            reference_height = torch.median(hits_z[:, center], dim=1).values
            elevation = hits_z - reference_height[:, None]
            forward = ray_xy[:, 0] > 0.05
            valid_hit = torch.isfinite(hits_z) & torch.isfinite(elevation)
            swept_corridor = torch.abs(ray_xy[:, 1]) <= self.cfg.hard_clearance_half_width
            hard_hazard = valid_hit & forward[None, :] & swept_corridor[None, :] & (
                (elevation > self.cfg.terrain_step_threshold)
                | (elevation < -self.cfg.terrain_drop_threshold)
            )
            preview_hazard = valid_hit & forward[None, :] & (
                torch.abs(elevation) > self.cfg.terrain_preview_threshold
            )
            forward_distance = ray_xy[:, 0].expand(self.num_envs, -1)
            fallback = torch.full_like(forward_distance, self.cfg.clearance_default)
            preview_candidates = torch.where(preview_hazard, forward_distance, fallback)
            hard_candidates = torch.where(hard_hazard, forward_distance, fallback)
            preview_clearance = preview_candidates.amin(dim=1).clamp_min(0.0)
            hard_clearance = hard_candidates.amin(dim=1).clamp_min(0.0)
            self._clearance.copy_(preview_clearance)
            self._hard_clearance.copy_(hard_clearance)

        if self.cfg.tunnel_enabled:
            origins = self._environment_origins()
            local_x = self.robot.data.root_pos_w[:, 0] - origins[:, 0]
            local_y = self.robot.data.root_pos_w[:, 1] - origins[:, 1]
            height, width, length = self._tunnel_dimensions.unbind(dim=1)
            start = self.cfg.tunnel_start_x_m
            end = start + length
            near_tunnel = (
                (local_x >= start - self.cfg.tunnel_preview_distance_m)
                & (local_x <= end)
            )
            body_top = (
                self.robot.data.root_pos_w[:, 2]
                + self.cfg.base_collision_center_z_m
                + self.cfg.base_collision_half_height_m
            )
            vertical_margin = height - body_top
            lateral_margin = (
                width / 2.0
                - torch.abs(local_y)
                - self.cfg.base_collision_half_width_m
            )
            envelope_margin = torch.minimum(vertical_margin, lateral_margin)
            preview_clearance = envelope_margin.clamp_min(0.0)
            self._clearance.copy_(
                torch.where(
                    near_tunnel,
                    preview_clearance,
                    torch.full_like(preview_clearance, self.cfg.clearance_default),
                )
            )
            unsafe_envelope = envelope_margin < self.cfg.tunnel_clearance_buffer_m
            distance_to_entry = (start - local_x).clamp_min(0.0)
            hard_clearance = torch.where(
                near_tunnel & unsafe_envelope,
                distance_to_entry,
                torch.full_like(distance_to_entry, self.cfg.clearance_default),
            )
            self._hard_clearance.copy_(hard_clearance)

        if self.cfg.stair_enabled:
            origins = self._environment_origins()
            local_x = self.robot.data.root_pos_w[:, 0] - origins[:, 0]
            tread = self._stair_dimensions[:, 1]
            relative_x = local_x - self.cfg.stair_start_x_m
            step_index = torch.floor(relative_x / tread).clamp(
                min=-1, max=self.cfg.stair_num_steps - 1
            )
            next_riser = self.cfg.stair_start_x_m + (step_index + 1.0) * tread
            before_first = local_x < self.cfg.stair_start_x_m
            approach_height = torch.maximum(
                self._stair_approach_root_height,
                self.robot.data.root_pos_w[:, 2],
            )
            update_reference = before_first & ~self._stair_approach_height_frozen
            self._stair_approach_root_height.copy_(
                torch.where(
                    update_reference,
                    approach_height,
                    self._stair_approach_root_height,
                )
            )
            next_riser = torch.where(
                before_first,
                torch.full_like(next_riser, self.cfg.stair_start_x_m),
                next_riser,
            )
            before_landing = step_index < self.cfg.stair_num_steps - 1
            near_stairs = (
                (local_x >= self.cfg.stair_start_x_m - self.cfg.stair_preview_distance_m)
                & before_landing
            )
            preview_clearance = (next_riser - local_x).clamp_min(0.0)
            self._clearance.copy_(
                torch.where(
                    near_stairs,
                    preview_clearance,
                    torch.full_like(preview_clearance, self.cfg.clearance_default),
                )
            )
            # Stair risers are traversable features. They affect route preview,
            # but must not trigger the non-traversable braking-distance HOLD.
            self._hard_clearance.fill_(self.cfg.clearance_default)
            self._stair_height_preview.copy_(
                self._stair_dimensions[:, 0] / self.cfg.stair_height_preview_reference_m
            )
        else:
            self._stair_height_preview.zero_()

        net_forces = self._contact_sensor.data.net_forces_w_history
        hard_contact = net_forces[:, :, self._hard_contact_body_ids]
        peak_force = torch.linalg.vector_norm(hard_contact, dim=-1).amax(dim=1)
        self._collision.copy_(
            torch.any(peak_force > self.cfg.collision_force_threshold, dim=1)
        )
        soft_contact = net_forces[:, :, self._soft_contact_body_ids]
        self._soft_contact_force.copy_(
            torch.linalg.vector_norm(soft_contact, dim=-1).amax(dim=(1, 2))
        )

    def official_proprioception(self, previous_policy_action: torch.Tensor) -> torch.Tensor:
        return build_official_proprioception(
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self._velocity_command,
            self.robot.data.joint_pos[:, self._joint_ids],
            self.robot.data.joint_vel[:, self._joint_ids],
            self.robot.data.default_joint_pos[:, self._joint_ids],
            previous_policy_action,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_perception()
        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self.robot.data.joint_vel[:, self._joint_ids]
        root_lin = self.robot.data.root_lin_vel_b
        root_ang = self.robot.data.root_ang_vel_b
        gravity = self.robot.data.projected_gravity_b
        speed = torch.linalg.vector_norm(root_lin[:, :2], dim=1)
        braking_distance = 0.08 + 0.15 * speed + torch.square(speed) / 3.0 + 0.03
        corridor_deviation = torch.abs(
            self.robot.data.root_pos_w[:, 1] - self._environment_origins()[:, 1]
        )
        curvature = (torch.abs(root_ang[:, 2]) / (speed + 0.10)).clamp(max=5.0)
        clearance = self._clearance
        preview = torch.stack((clearance, corridor_deviation, curvature, clearance - braking_distance), dim=1)
        roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
        pitch = torch.atan2(-gravity[:, 0], torch.sqrt(torch.square(gravity[:, 1]) + torch.square(gravity[:, 2])))
        posture_height = self.posture_height
        posture_actual = torch.stack((posture_height, roll, pitch), dim=1)
        obs = torch.cat(
            (
                root_lin, root_ang, gravity, q, qd, self._actions, self._velocity_command,
                preview, self._posture_command, posture_actual, self._posture_phase,
                self._energy[:, None], self._path_version, self._goal_age,
                self._stair_height_preview[:, None],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self.robot.data.joint_vel[:, self._joint_ids]
        tau = self.robot.data.applied_torque[:, self._joint_ids]
        xy = self.robot.data.root_pos_w[:, :2]
        delta_xy = xy - self._previous_xy
        local_length = torch.linalg.vector_norm(delta_xy, dim=1).clamp(max=0.1)
        self._last_step_distance.copy_(local_length.detach())
        progress = delta_xy[:, 0].clamp(-0.1, 0.1)
        self._previous_xy = xy.detach().clone()
        root_height = self.robot.data.root_pos_w[:, 2]
        height_progress = (root_height - self._previous_root_height).clamp(-0.05, 0.05)
        self._previous_root_height.copy_(root_height.detach())
        self._distance += local_length
        energy_step = torch.sum(torch.abs(tau * qd), dim=1) * self.step_dt
        self._energy += energy_step
        speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        corridor_deviation = torch.abs(xy[:, 1] - self._environment_origins()[:, 1])
        clearance = self._clearance
        clearance_risk = torch.exp(-clearance / self.cfg.clearance_sigma)
        curvature = (torch.abs(self.robot.data.root_ang_vel_b[:, 2]) / (speed + 0.10)).clamp(max=5.0)
        corridor_half_width = torch.full_like(
            corridor_deviation, self.cfg.corridor_half_width
        )
        if self.cfg.stair_enabled:
            corridor_half_width = (
                self._stair_dimensions[:, 2] / 2.0
                - self.cfg.base_collision_half_width_m
            ).clamp_min(0.0)
        corridor_violation = torch.relu(corridor_deviation - corridor_half_width)
        forward_progress = progress.clamp_min(0.0)
        detour_length = torch.relu(local_length - forward_progress)
        commanded_progress = torch.linalg.vector_norm(self._velocity_command[:, :2], dim=1) * self.step_dt
        progress_deficit = torch.relu(commanded_progress - forward_progress)
        route_cost = (
            self.cfg.route_length_weight
            * (detour_length + progress_deficit) / self.cfg.route_reference_length
            + self.cfg.route_clearance_weight * clearance_risk * local_length / self.cfg.route_reference_length
            + self.cfg.route_curvature_weight * self.cfg.route_reference_length
            * torch.square(curvature) * local_length
            + corridor_violation
        )
        posture_height = self.posture_height
        height_rate = (posture_height - self._previous_posture_height) / self.step_dt
        self._previous_posture_height.copy_(posture_height.detach())
        posture_levels = posture_height.new_tensor(self.cfg.posture_heights)
        target_height = torch.sum(self._posture_command * posture_levels, dim=1)
        mode_cost = (
            self.cfg.mode_height_rate_weight
            * torch.square(height_rate / self.cfg.height_rate_reference)
            + self.cfg.mode_hold_weight
            * torch.square((posture_height - target_height) / self.cfg.height_rate_reference)
        )
        residual_rate = torch.mean(
            torch.abs(self._actions - self._previous_actions), dim=1
        ) / max(self.cfg.residual_limit * self.cfg.posture_scale, 1e-6)
        joint_target_change = torch.mean(
            torch.abs(self._q_cmd - self._previous_q_cmd), dim=1
        ) / self.cfg.joint_target_change_reference
        acceleration_ratio = torch.mean(
            self._last_joint_acceleration_abs / self.cfg.acceleration_limit, dim=1
        )
        soft_contact_ratio = (
            self._soft_contact_force / self.cfg.collision_force_threshold
        ).clamp(max=5.0)
        soft_contact_term = self.cfg.soft_contact_cost_weight * soft_contact_ratio
        self._smooth_components = torch.stack(
            (
                joint_target_change,
                residual_rate,
                acceleration_ratio,
                soft_contact_term,
            ),
            dim=1,
        )
        smooth_cost = self._smooth_components.sum(dim=1)
        self._raw_soft_costs = torch.stack(
            (route_cost, energy_step, mode_cost, smooth_cost), dim=1
        )
        p95 = self._raw_soft_costs.new_tensor(self.cfg.objective_p95).clamp_min(1e-8)
        self._soft_costs = (self._raw_soft_costs / p95).clamp(
            min=0.0, max=self.cfg.normalized_cost_clip
        )
        linear_error = torch.sum(
            torch.square(self.robot.data.root_lin_vel_b[:, :2] - self._velocity_command[:, :2]), dim=1
        )
        yaw_error = torch.square(self.robot.data.root_ang_vel_b[:, 2] - self._velocity_command[:, 2])
        tracking_reward = torch.exp(-linear_error / 0.25) + 0.5 * torch.exp(-yaw_error / 0.25)
        normalized_route, normalized_energy, normalized_mode, normalized_smooth = self._soft_costs.unbind(1)
        stair_step_progress = torch.zeros(self.num_envs, device=self.device)
        stair_approach_lift = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.stair_enabled:
            verified_level = self._verified_stair_level()
            verified_max = torch.maximum(
                verified_level, self._previous_verified_stair_level
            )
            stair_step_progress = verified_max - self._previous_verified_stair_level
            self._previous_verified_stair_level.copy_(verified_max)
            near_next_riser = (
                (self._clearance <= self.cfg.stair_approach_lift_distance_m)
                & (verified_max < self.cfg.stair_num_steps)
            )
            stair_approach_lift = (
                height_progress.clamp_min(0.0) * near_next_riser.float()
            )
        reward = (
            self.cfg.tracking_reward_scale * tracking_reward
            + self.cfg.tracking_reward_scale * progress
            - self.cfg.route_reward_scale * normalized_route
            - self.cfg.energy_cost_scale * normalized_energy
            - self.cfg.mode_cost_scale * normalized_mode
            - self.cfg.smooth_cost_scale * normalized_smooth
            - self.cfg.hard_failure_cost * self._hard_failure.float()
            + self.cfg.tunnel_success_reward * self._tunnel_success_step.float()
            + self.cfg.stair_success_reward * self._stair_success_step.float()
            + self.cfg.stair_height_progress_reward_scale
            * height_progress.clamp_min(0.0)
            * float(self.cfg.stair_enabled)
            + self.cfg.stair_approach_lift_reward_scale * stair_approach_lift
            + self.cfg.stair_step_progress_reward * stair_step_progress
            + self.cfg.stair_level_hold_reward_scale
            * (self._previous_verified_stair_level / max(self.cfg.stair_num_steps, 1))
            * float(self.cfg.stair_enabled)
            - self.cfg.stair_collision_penalty
            * self._collision.float()
            * float(self.cfg.stair_enabled)
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._joint_ids]
        measured_qd = self.robot.data.joint_vel[:, self._joint_ids]
        tau = self.robot.data.applied_torque[:, self._joint_ids]
        limits = q.new_tensor(
            (self.cfg.leg_torque_limit, self.cfg.leg_torque_limit, self.cfg.leg_torque_limit,
             self.cfg.wheel_torque_limit, self.cfg.leg_torque_limit, self.cfg.leg_torque_limit,
             self.cfg.leg_torque_limit, self.cfg.wheel_torque_limit)
        ).expand_as(q)
        velocity_limits = q.new_tensor(
            (
                self.cfg.leg_velocity_limit, self.cfg.leg_velocity_limit,
                self.cfg.leg_velocity_limit, self.cfg.wheel_velocity_limit,
                self.cfg.leg_velocity_limit, self.cfg.leg_velocity_limit,
                self.cfg.leg_velocity_limit, self.cfg.wheel_velocity_limit,
            )
        ).expand_as(q)
        self._last_preprojection_joint_velocity_abs.copy_(torch.abs(measured_qd))
        qd, velocity_state_projection = project_joint_velocity(
            measured_qd, velocity_limits
        )
        if self.cfg.enable_safety_projector and velocity_state_projection.any():
            affected_envs = torch.any(velocity_state_projection, dim=1).nonzero().flatten()
            self.robot.write_joint_state_to_sim(
                q[affected_envs], qd[affected_envs],
                joint_ids=self._joint_ids, env_ids=affected_envs
            )
        else:
            qd = measured_qd
            velocity_state_projection.zero_()
        self._last_velocity_state_projection_flags.copy_(velocity_state_projection)
        velocity_by_joint = torch.abs(qd) > velocity_limits
        self._last_joint_velocity_abs.copy_(torch.abs(qd))
        measured_acceleration = (qd - self._previous_joint_velocity) / self.step_dt
        filter_alpha = self.step_dt / (
            self.cfg.acceleration_filter_time_constant + self.step_dt
        )
        self._filtered_joint_acceleration.lerp_(measured_acceleration, filter_alpha)
        self._previous_joint_velocity.copy_(qd)
        self._last_joint_acceleration_abs.copy_(
            torch.abs(self._filtered_joint_acceleration)
        )
        self._last_velocity_violation_flags.copy_(velocity_by_joint)
        velocity = torch.any(velocity_by_joint, dim=1)
        acceleration = (
            torch.any(self._last_joint_acceleration_abs > self.cfg.acceleration_limit, dim=1)
            if self.cfg.enforce_acceleration_limit
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        torque = torch.any(torch.abs(tau) > limits, dim=1)
        height = self.robot.data.root_pos_w[:, 2] < 0.15
        attitude = self.robot.data.projected_gravity_b[:, 2] > -0.3
        speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        braking_clearance = (
            0.10 + 0.15 * speed + torch.square(speed) / 3.0 + 0.03
        )
        # Reaching the braking envelope activates HOLD in the projector.  It is
        # only a hard violation if the robot still crosses the zero-speed
        # clearance floor after that intervention.
        clearance = self._hard_clearance < (0.08 + 0.03)
        stabilized = self.episode_length_buf >= self.cfg.reset_stabilization_steps
        velocity &= stabilized
        acceleration &= stabilized
        torque &= stabilized
        collision = self._collision & stabilized
        if self.cfg.stair_enabled and not self.cfg.stair_collision_terminates:
            collision = torch.zeros_like(collision)
        height &= stabilized
        attitude &= stabilized
        clearance &= stabilized
        self._hard_reason_flags = torch.stack(
            (
                self._hard_failure,
                velocity,
                acceleration,
                torque,
                collision,
                height,
                attitude,
                clearance,
            ),
            dim=1,
        )
        self._last_hard_reason_flags.copy_(self._hard_reason_flags)
        hard = torch.any(self._hard_reason_flags, dim=1)
        self._hard_failure |= hard
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.tunnel_enabled:
            self._tunnel_success_step.copy_(self.tunnel_exited & ~hard)
            self._tunnel_success_count += self._tunnel_success_step.long()
            timeout |= self._tunnel_success_step
        else:
            self._tunnel_success_step.zero_()
        if self.cfg.stair_enabled:
            self._stair_success_step.copy_(self.stair_completed & ~hard)
            self._stair_success_count += self._stair_success_step.long()
            if self.cfg.stair_terminate_on_success:
                timeout |= self._stair_success_step
        else:
            self._stair_success_step.zero_()
        return hard, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        default_q = self.robot.data.default_joint_pos[env_ids][:, self._joint_ids]
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self._environment_origins()[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            default_q, torch.zeros_like(default_q), joint_ids=self._joint_ids, env_ids=env_ids
        )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._base_target[env_ids] = default_q
        self._q_cmd[env_ids] = default_q
        self._previous_q_cmd[env_ids] = default_q
        self._hard_failure[env_ids] = False
        self._tunnel_success_step[env_ids] = False
        self._stair_success_step[env_ids] = False
        self._projected[env_ids] = False
        self._safety_hold[env_ids] = False
        self._hard_reason_flags[env_ids] = False
        self._last_infeasible_joint_flags[env_ids] = False
        self._last_infeasible_cause_flags[env_ids] = False
        self._last_projection_joint_flags[env_ids] = False
        self._last_projection_cause_flags[env_ids] = False
        self._last_joint_acceleration_abs[env_ids] = 0.0
        self._last_velocity_state_projection_flags[env_ids] = False
        self._last_preprojection_joint_velocity_abs[env_ids] = 0.0
        self._filtered_joint_acceleration[env_ids] = 0.0
        self._previous_joint_velocity[env_ids] = 0.0
        self._distance[env_ids] = 0.0
        self._last_step_distance[env_ids] = 0.0
        self._energy[env_ids] = 0.0
        self._soft_costs[env_ids] = 0.0
        self._raw_soft_costs[env_ids] = 0.0
        self._smooth_components[env_ids] = 0.0
        self._previous_xy[env_ids] = default_root[:, :2]
        self._previous_root_height[env_ids] = default_root[:, 2]
        self._velocity_command[env_ids] = 0.0
        self._posture_command[env_ids] = self._posture_command.new_tensor((0.0, 1.0, 0.0))
        self._posture_phase[env_ids] = 0.0
        self._posture_offset[env_ids] = 0.0
        self._path_version[env_ids] = 0.0
        self._goal_age[env_ids] = 0.0
        self._previous_posture_height[env_ids] = default_root[:, 2] - self.cfg.wheel_radius
        self._collision[env_ids] = False
        self._soft_contact_force[env_ids] = 0.0
        self._clearance[env_ids] = self.cfg.clearance_default
        self._hard_clearance[env_ids] = self.cfg.clearance_default
        self._stair_height_preview[env_ids] = 0.0
        self._stair_approach_root_height[env_ids] = default_root[:, 2]
        self._stair_approach_height_frozen[env_ids] = False
        self._previous_verified_stair_level[env_ids] = 0.0
        self._stair_jump_step[env_ids] = -1
        self._stair_jump_level[env_ids] = 0
