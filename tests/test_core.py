import numpy as np
import pytest
import torch

from tita_uc_residual_rl.config import ControllerConfig, DEFAULT_CONFIG, SafetyLimits
from tita_uc_residual_rl.asset_check import require_eight_dof, require_official_tita_urdf, urdf_joint_names
from tita_uc_residual_rl.np3o_costs import CostVector, NP3OCostAccumulator
from tita_uc_residual_rl.residual_policy import ResidualPolicy
from tita_uc_residual_rl.safety_projector import SafetyProjector, SafetyStatus
from tita_uc_residual_rl.safety_actuation import (
    apply_velocity_torque_guard,
    filter_residual_action,
    project_joint_velocity,
)
from tita_uc_residual_rl.toy_env import ToyTitaEnv
from tita_uc_residual_rl.trainer import ResidualNP3OTrainer
from tita_uc_residual_rl.types import ControlState, PostureSample, RouteSample, SafetyContext
from tita_uc_residual_rl.unified_objective import UnifiedObjective
from tita_uc_residual_rl.official_policy import (
    FrozenOfficialTitaController,
    OfficialTitaActor,
    build_official_proprioception,
    swap_leg_blocks,
)
from tita_uc_residual_rl.phase_b_learner import (
    PhaseBActorCritic,
    PhaseBNP3O,
    generalized_advantage,
    load_phase_b_state_dict_compatible,
)
from tita_uc_residual_rl.elite_distillation import distill_elite_actions
from tita_uc_residual_rl.posture_curriculum import (
    PostureCommandScheduler,
    PostureCurriculumConfig,
)
from tita_uc_residual_rl.tunnel_curriculum import (
    TunnelCurriculumConfig,
    sample_tunnel_dimensions,
)
from tita_uc_residual_rl.stair_curriculum import (
    StairCurriculumConfig,
    sample_stair_dimensions,
)
from tita_uc_residual_rl.stair_validation import aggregate_strict_probe_results


def make_state():
    q = np.array((0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0))
    return ControlState(q, np.zeros(8), np.zeros(8))


def test_policy_is_bounded_eight_dimensional():
    policy = ResidualPolicy(12, residual_limit=0.2)
    output = policy(torch.zeros(4, 12))
    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()
    assert torch.max(torch.abs(output)) <= 0.2 + 1e-6
    assert torch.count_nonzero(output) == 0


def test_policy_exploration_can_be_reduced_and_frozen():
    policy = ResidualPolicy(12).configure_exploration(-4.5, trainable=False)
    distribution = policy.distribution(torch.zeros(2, 12))

    torch.testing.assert_close(
        distribution.stddev,
        torch.full((2, 8), float(np.exp(-4.5))),
    )
    assert not policy.log_std.requires_grad
    with pytest.raises(ValueError, match="log_std"):
        policy.configure_exploration(-6.0)


def test_torque_projection_uses_p_control():
    limits = SafetyLimits(
        q_min=(-100.0,) * 8, q_max=(100.0,) * 8,
        qd_max=(100.0,) * 8, qdd_max=(10000.0,) * 8,
    )
    projector = SafetyProjector(ControllerConfig(safety_limits=limits))
    state = make_state()
    result = projector.project(np.full(8, 10.0), state, SafetyContext(clearance=1.0))
    assert result.projected
    assert np.all(np.abs(result.estimated_tau) <= np.asarray(limits.tau_max) + 1e-8)
    expected = np.asarray(limits.tau_max)
    np.testing.assert_allclose(result.estimated_tau, expected, atol=1e-8)


def test_wheel_projection_uses_official_torque_proxy():
    limits = SafetyLimits(qd_max=(100.0,) * 8, qdd_max=(10000.0,) * 8)
    projector = SafetyProjector(ControllerConfig(safety_limits=limits))
    state = ControlState(np.zeros(8), np.full(8, 2.0), np.zeros(8))
    result = projector.project(np.full(8, 10.0), state, SafetyContext(clearance=1.0))
    wheel_ids = np.array([3, 7])
    np.testing.assert_allclose(result.estimated_tau[wheel_ids], 15.0, atol=1e-8)
    np.testing.assert_allclose(result.q_cmd[wheel_ids], 1.6, atol=1e-8)


def test_velocity_torque_guard_brakes_before_hard_limit():
    torque = torch.tensor([[5.0, 10.0, -10.0, -12.0]])
    velocity = torch.tensor([[19.0, 19.75, -19.75, 19.75]])
    torque_limit = torch.full_like(torque, 15.0)
    velocity_limit = torch.full_like(torque, 20.0)

    guarded = apply_velocity_torque_guard(
        torque, velocity, torque_limit, velocity_limit, margin=0.5
    )

    assert guarded[0, 0].item() == pytest.approx(5.0)
    assert guarded[0, 1].item() == pytest.approx(-7.5)
    assert guarded[0, 2].item() == pytest.approx(7.5)
    assert guarded[0, 3].item() == pytest.approx(-12.0)


def test_velocity_torque_guard_uses_full_brake_beyond_limit():
    torque = torch.tensor([[15.0, -15.0]])
    velocity = torch.tensor([[20.1, -20.1]])
    torque_limit = torch.full_like(torque, 15.0)
    velocity_limit = torch.full_like(torque, 20.0)

    guarded = apply_velocity_torque_guard(
        torque, velocity, torque_limit, velocity_limit, margin=0.5
    )

    torch.testing.assert_close(guarded, torch.tensor([[-15.0, 15.0]]))


def test_joint_velocity_projection_reports_and_clamps_overshoot():
    velocity = torch.tensor([[19.0, 20.25, -25.5]])
    limits = torch.tensor([[20.0, 20.0, 25.0]])

    projected, flags = project_joint_velocity(velocity, limits)

    torch.testing.assert_close(projected, torch.tensor([[19.0, 20.0, -25.0]]))
    assert flags.tolist() == [[False, True, True]]


def test_residual_filter_limits_step_change():
    previous = torch.tensor([[0.0, 0.10]])
    desired = torch.tensor([[0.20, -0.10]])

    filtered = filter_residual_action(previous, desired, alpha=0.25)

    torch.testing.assert_close(filtered, torch.tensor([[0.05, 0.05]]))
    with pytest.raises(ValueError, match="alpha"):
        filter_residual_action(previous, desired, alpha=0.0)


def test_clearance_and_collision_are_hard_statuses():
    projector = SafetyProjector(DEFAULT_CONFIG)
    state = make_state()
    speed = 0.8
    context = SafetyContext(speed=speed, clearance=DEFAULT_CONFIG.safety.safe_clearance(speed) - 1e-3)
    result = projector.project(np.zeros(8), state, context)
    assert result.status == SafetyStatus.HOLD
    assert "braking_distance_violation" in result.reasons
    result = projector.project(np.zeros(8), state, SafetyContext(collision=True))
    assert result.status == SafetyStatus.STOP


def test_posture_transition_requires_stable_hold():
    projector = SafetyProjector(DEFAULT_CONFIG)
    context = SafetyContext(clearance=1.0, posture_transition=True, hold_elapsed=0.0)
    result = projector.project(np.zeros(8), make_state(), context)
    assert result.status == SafetyStatus.HOLD
    assert "posture_hold_not_satisfied" in result.reasons


def test_energy_requires_exactly_eight_joints_and_cot():
    objective = UnifiedObjective(DEFAULT_CONFIG)
    energy = objective.energy_cost(np.ones(8), np.arange(8), 0.1)
    assert energy == pytest.approx(2.8)
    denominator = DEFAULT_CONFIG.mass_kg * DEFAULT_CONFIG.gravity
    assert objective.mechanical_cot(energy, 1.0) == pytest.approx(energy / denominator)
    assert objective.electrical_cot(np.array([2.0, -1.0]), np.array([3.0, 4.0]), 0.5, 1.0) == pytest.approx(4.5 / denominator)
    with pytest.raises(ValueError):
        objective.energy_cost(np.ones(7), np.ones(7), 0.1)


def test_normalization_is_fixed_and_hard_constraints_not_scalarized():
    objective = UnifiedObjective(DEFAULT_CONFIG)
    terms_a = objective.terms(RouteSample(local_length=1.0), 2.0, PostureSample(), 0.0)
    terms_b = objective.terms(RouteSample(local_length=1.0), 2.0, PostureSample(), 0.0)
    assert terms_a == terms_b
    report = objective.hard_constraints(
        clearance=0.0, safe_clearance=0.2, tau=np.zeros(8), q=np.zeros(8), qd=np.zeros(8),
        qdd=np.zeros(8), q_min=-np.ones(8), q_max=np.ones(8), qd_max=np.ones(8),
        qdd_max=np.ones(8), tau_max=np.ones(8),
    )
    assert not report.safe
    assert "clearance_violation" in report.violations


def test_toy_env_runs_and_accumulates_vector_costs():
    env = ToyTitaEnv(horizon=5)
    obs = env.reset()
    assert obs.flatten().ndim == 1
    for _ in range(5):
        step = env.step(np.zeros(8))
        if step.done:
            break
    assert env.t == 5
    assert np.isfinite(env.energy)
    acc = NP3OCostAccumulator()
    acc.add(CostVector(1, 2, 3, 4), hard_violation=True)
    assert acc.mean().values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert acc.hard_violations == 1


def test_residual_np3o_update_keeps_four_cost_heads():
    torch.manual_seed(0)
    policy = ResidualPolicy(12)
    trainer = ResidualNP3OTrainer(policy, obs_dim=12)
    observation = torch.zeros(6, 12)
    with torch.no_grad():
        _, old_log_prob = policy.sample(observation)
    stats = trainer.update(
        observation, old_log_prob.detach(), torch.ones(6), torch.ones(6, 4),
        torch.ones(6, 4), hard_failure=torch.tensor([False, True, False, False, False, False]),
    )
    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.cost_loss)


def test_asset_check_rejects_the_local_non_eight_dof_urdf():
    urdf = "/home/ky/tita_low_level_control/titati_locomotion/titati_bringup/urdf/titati_robot_2.urdf"
    names = urdf_joint_names(urdf)
    assert len(names) == 16
    with pytest.raises(ValueError, match="exactly 8"):
        require_eight_dof(urdf)


def test_official_tita_asset_has_the_expected_eight_joint_order():
    urdf = "/home/ky/Documents/Codex/2026-09-11/new-chat/work/tita_rl_official/resources/tita/urdf/tita_description.urdf"
    names = require_official_tita_urdf(urdf)
    assert len(names) == 8
    assert names[0] == "joint_left_leg_1"
    assert names[-1] == "joint_right_leg_4"


def test_official_proprioception_contract_and_reindex():
    zeros3 = torch.zeros(2, 3)
    q = torch.arange(8, dtype=torch.float32).repeat(2, 1)
    qd = q + 10.0
    previous = torch.ones(2, 8)
    proprio = build_official_proprioception(
        zeros3, torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1), zeros3,
        q, qd, torch.zeros_like(q), previous,
    )
    assert proprio.shape == (2, 33)
    assert swap_leg_blocks(q)[0].tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert proprio[0, -8:].tolist() == [1.0] * 8


def test_official_checkpoint_actor_is_frozen_and_finite():
    checkpoint = "/home/ky/tita_uc_residual_rl/tita_rl_official/tita_example_10000.pt"
    actor = OfficialTitaActor.from_checkpoint(checkpoint)
    output = actor(torch.zeros(3, 33), torch.zeros(3, 10, 33))
    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    controller = FrozenOfficialTitaController(actor)
    target, action = controller.target(torch.zeros(3, 33), torch.zeros(3, 8))
    assert target.shape == action.shape == (3, 8)


def test_phase_b_update_uses_reward_and_four_cost_values():
    torch.manual_seed(3)
    model = PhaseBActorCritic(obs_dim=50)
    algorithm = PhaseBNP3O(model)
    observations = torch.randn(4, 6, 50)
    flat = observations.flatten(0, 1)
    with torch.no_grad():
        distribution = model.distribution(flat)
        actions = distribution.sample().view(4, 6, 8)
        log_prob = distribution.log_prob(actions.flatten(0, 1)).sum(-1).view(4, 6)
        values = model.reward_value(flat).view(4, 6)
        cost_values = model.cost_value(flat).view(4, 6, 4)
    dones = torch.zeros(4, 6, dtype=torch.bool)
    reward_advantage, reward_return = generalized_advantage(
        torch.ones(4, 6), values, torch.zeros(6), dones, 0.99, 0.95
    )
    cost_advantage, cost_return = generalized_advantage(
        torch.ones(4, 6, 4), cost_values, torch.zeros(6, 4), dones, 0.99, 0.95
    )
    stats = algorithm.update(
        observations, actions, log_prob, reward_advantage, reward_return,
        cost_advantage, cost_return, epochs=1, mini_batches=2,
    )
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.reward_value_loss)
    assert np.isfinite(stats.cost_value_loss)


def test_phase_b_dual_multipliers_follow_budget_violations_and_bounds():
    model = PhaseBActorCritic(obs_dim=50)
    algorithm = PhaseBNP3O(
        model,
        cost_multipliers=(0.1, 0.1, 0.1, 0.1),
        minimum_cost_multipliers=(0.0, 0.0, 0.0, 0.0),
        cost_budgets=(1.0, 1.0, 1.0, 1.0),
        dual_learning_rate=0.2,
        cost_multiplier_max=0.25,
    )
    violation = algorithm.update_dual(torch.tensor((2.0, 0.5, 1.0, 3.0)))
    assert violation.tolist() == pytest.approx((1.0, -0.5, 0.0, 2.0))
    assert algorithm.cost_multipliers.tolist() == pytest.approx((0.25, 0.0, 0.1, 0.25))
    algorithm.update_dual(torch.zeros(4))
    assert algorithm.cost_multipliers.tolist() == pytest.approx((0.05, 0.0, 0.0, 0.05))


def test_phase_b_dual_multipliers_preserve_unified_objective_floors():
    model = PhaseBActorCritic(obs_dim=50)
    floors = (0.4, 0.35, 0.25, 0.05)
    algorithm = PhaseBNP3O(
        model,
        cost_multipliers=floors,
        minimum_cost_multipliers=floors,
        cost_budgets=(1.0, 1.0, 1.0, 1.0),
        dual_learning_rate=0.2,
    )
    algorithm.update_dual(torch.zeros(4))
    assert algorithm.cost_multipliers.tolist() == pytest.approx(floors)
    state = algorithm.constraint_state_dict()
    assert state["minimum_cost_multipliers"] == pytest.approx(floors)


def test_phase_b_constraint_configuration_requires_four_positive_budgets():
    model = PhaseBActorCritic(obs_dim=50)
    with pytest.raises(ValueError, match="exactly four"):
        PhaseBNP3O(model, cost_budgets=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        PhaseBNP3O(model, cost_budgets=(1.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="floors"):
        PhaseBNP3O(model, minimum_cost_multipliers=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="below"):
        PhaseBNP3O(model, minimum_cost_multipliers=(0.5, 0.0, 0.0, 0.0))


def test_phase_b_checkpoint_supports_append_only_observation_expansion():
    original = PhaseBActorCritic(obs_dim=50)
    expanded = PhaseBActorCritic(obs_dim=51)

    load_phase_b_state_dict_compatible(expanded, original.state_dict())

    for key in (
        "policy.network.0.weight",
        "reward_value.0.weight",
        "cost_value.0.weight",
    ):
        torch.testing.assert_close(
            expanded.state_dict()[key][:, :50], original.state_dict()[key]
        )
        assert torch.count_nonzero(expanded.state_dict()[key][:, 50]) == 0


def test_elite_distillation_updates_only_deterministic_actor_network():
    torch.manual_seed(19)
    model = PhaseBActorCritic(obs_dim=51)
    observations = torch.randn(96, 51)
    with torch.no_grad():
        source_means = model.policy(observations).clone()
    actions = source_means + 0.03
    reward_value_before = {
        key: value.clone() for key, value in model.reward_value.state_dict().items()
    }
    cost_value_before = {
        key: value.clone() for key, value in model.cost_value.state_dict().items()
    }
    log_std_before = model.policy.log_std.clone()

    stats = distill_elite_actions(
        model,
        observations,
        actions,
        source_means,
        epochs=12,
        batch_size=32,
        learning_rate=1e-3,
        anchor_weight=1.0,
    )

    assert stats.action_mse_after < stats.action_mse_before
    assert stats.max_policy_shift > 0.0
    torch.testing.assert_close(model.policy.log_std, log_std_before)
    for key, value in reward_value_before.items():
        torch.testing.assert_close(model.reward_value.state_dict()[key], value)
    for key, value in cost_value_before.items():
        torch.testing.assert_close(model.cost_value.state_dict()[key], value)


def test_posture_curriculum_waits_for_safe_hold_and_completes_transition():
    scheduler = PostureCommandScheduler(
        2,
        "cpu",
        PostureCurriculumConfig(
            control_dt=0.1,
            resample_time_s=0.1,
            transition_time_s=0.2,
            switch_hold_time_s=0.2,
        ),
    )
    scheduler.set_allowed_modes((0, 1))
    unsafe = scheduler.step(
        torch.tensor((0.3, 0.3)), torch.zeros(2), torch.zeros(2)
    )
    assert unsafe.slow_down.all()
    assert not unsafe.started.any()
    assert unsafe.command.argmax(1).tolist() == [1, 1]

    first_hold = scheduler.step(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert not first_hold.started.any()
    started = scheduler.step(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert started.started.all()
    assert started.transition.all()
    completed = scheduler.step(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert completed.completed.all()
    assert completed.command.argmax(1).tolist() == [0, 0]


def test_tunnel_dimensions_are_reproducible_and_bounded():
    cfg = TunnelCurriculumConfig(
        height_range_m=(0.34, 0.44),
        width_range_m=(0.40, 0.65),
        length_range_m=(0.60, 1.50),
    )
    first = sample_tunnel_dimensions(32, 53, cfg)
    second = sample_tunnel_dimensions(32, 53, cfg)
    assert first.shape == (32, 3)
    assert torch.from_numpy(first).equal(torch.from_numpy(second))
    assert ((first[:, 0] >= 0.34) & (first[:, 0] <= 0.44)).all()
    assert ((first[:, 1] >= 0.40) & (first[:, 1] <= 0.65)).all()
    assert ((first[:, 2] >= 0.60) & (first[:, 2] <= 1.50)).all()


def test_stair_dimensions_are_reproducible_and_bounded():
    cfg = StairCurriculumConfig(
        step_height_range_m=(0.025, 0.045),
        tread_range_m=(0.55, 0.67),
        width_range_m=(0.90, 1.20),
        num_steps=3,
    )
    first = sample_stair_dimensions(32, 163, cfg)
    second = sample_stair_dimensions(32, 163, cfg)
    assert first.shape == (32, 3)
    assert torch.from_numpy(first).equal(torch.from_numpy(second))
    assert ((first[:, 0] >= 0.025) & (first[:, 0] <= 0.045)).all()
    assert ((first[:, 1] >= 0.55) & (first[:, 1] <= 0.67)).all()
    assert ((first[:, 2] >= 0.90) & (first[:, 2] <= 1.20)).all()


def test_stair_curriculum_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="num_steps"):
        StairCurriculumConfig(num_steps=0)
    with pytest.raises(ValueError, match="step height"):
        StairCurriculumConfig(step_height_range_m=(0.0, 0.04))


def test_strict_stair_probe_aggregation_requires_every_seed_to_pass():
    motion = {
        "trigger_distance_m": 0.25,
        "command_x_mps": 0.45,
        "crouch_steps": 8,
        "extension_steps": 12,
        "hip_crouch_rad": 0.30,
        "hip_extension_rad": 0.60,
        "knee_crouch_rad": 1.00,
        "knee_extension_rad": 1.00,
        "max_stable_hold_steps": 30,
        "max_base_contact_force_n": 0.0,
    }
    first = {"step_height_m": 0.17, "tread_m": 0.32, "records": [{**motion, "strict_success": True}]}
    second = {"step_height_m": 0.17, "tread_m": 0.32, "records": [{**motion, "strict_success": False}]}

    rejected = aggregate_strict_probe_results((first, second))
    assert rejected["common_candidate_count"] == 1
    assert rejected["strict_multiseed_candidate_count"] == 0

    second["records"][0]["strict_success"] = True
    accepted = aggregate_strict_probe_results((first, second))
    assert accepted["strict_multiseed_candidate_count"] == 1
    assert accepted["strict_candidates"][0]["command_x_mps"] == 0.45
    assert accepted["strict_candidates"][0]["pulse_command_x_mps"] == 0.45
    assert accepted["strict_candidates"][0]["crouch_command_x_mps"] == 0.45
    assert accepted["strict_candidates"][0]["extension_command_x_mps"] == 0.45
    assert accepted["strict_candidates"][0]["hip_extension_delay_steps"] == 0
    assert accepted["strict_candidates"][0]["knee_extension_delay_steps"] == 0


def test_strict_stair_probe_aggregation_distinguishes_pulse_timing():
    motion = {
        "trigger_distance_m": 0.245,
        "command_x_mps": 0.45,
        "pulse_command_x_mps": 0.15,
        "crouch_steps": 8,
        "extension_steps": 12,
        "hip_extension_delay_steps": 2,
        "knee_extension_delay_steps": 0,
        "hip_crouch_rad": 0.30,
        "hip_extension_rad": 0.60,
        "knee_crouch_rad": 1.00,
        "knee_extension_rad": 0.90,
        "max_stable_hold_steps": 30,
        "max_base_contact_force_n": 0.0,
        "strict_success": True,
    }
    other_timing = {
        **motion,
        "pulse_command_x_mps": 0.0,
        "hip_extension_delay_steps": 0,
        "knee_extension_delay_steps": 2,
    }
    first = {"step_height_m": 0.17, "tread_m": 0.32, "records": [motion]}
    second = {"step_height_m": 0.17, "tread_m": 0.32, "records": [other_timing]}

    result = aggregate_strict_probe_results((first, second))
    assert result["common_candidate_count"] == 0
    assert result["strict_multiseed_candidate_count"] == 0


def test_strict_stair_probe_aggregation_distinguishes_phase_commands():
    motion = {
        "trigger_distance_m": 0.28,
        "command_x_mps": 0.45,
        "pulse_command_x_mps": 0.15,
        "crouch_command_x_mps": 0.0,
        "extension_command_x_mps": 0.45,
        "crouch_steps": 8,
        "extension_steps": 12,
        "hip_extension_delay_steps": 2,
        "knee_extension_delay_steps": 2,
        "hip_crouch_rad": 0.30,
        "hip_extension_rad": 0.60,
        "knee_crouch_rad": 1.00,
        "knee_extension_rad": 0.90,
        "max_stable_hold_steps": 30,
        "max_base_contact_force_n": 0.0,
        "strict_success": True,
    }
    other_phase_command = {**motion, "extension_command_x_mps": 0.30}
    first = {"step_height_m": 0.17, "tread_m": 0.32, "records": [motion]}
    second = {
        "step_height_m": 0.17,
        "tread_m": 0.32,
        "records": [other_phase_command],
    }

    result = aggregate_strict_probe_results((first, second))
    assert result["common_candidate_count"] == 0
    assert result["strict_multiseed_candidate_count"] == 0


def test_strict_stair_probe_aggregation_rejects_wrong_geometry():
    with pytest.raises(ValueError, match="measured"):
        aggregate_strict_probe_results(
            ({"step_height_m": 0.16, "tread_m": 0.32, "records": []},)
        )
