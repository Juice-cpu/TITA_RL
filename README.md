# TITA-UC-Residual-NP3O

Isaac Lab implementation of an 8-DOF safety residual controller for TITA.
The frozen official TITA policy produces the base joint target and a learned
8-D residual adjusts it before a deterministic safety projector applies the
command.

中文入门训练教程（包含启动、调参、严格评估、日志记录和启动脚本逐行说明）：
[docs/RL_TRAINING_BEGINNER_GUIDE_CN.md](docs/RL_TRAINING_BEGINNER_GUIDE_CN.md)

The implementation is intentionally limited to simulation and ROS2 shadow
validation. It must not send motion commands to hardware at this stage.

## Current status

- Isaac Sim 4.5 and Isaac Lab v2.0.0 run on the local RTX 4070.
- The official 8-DOF URDF and `tita_example_10000.pt` actor are loaded.
- Official mixed control semantics are preserved: six leg joints use PD
  position targets and the two `leg_4` wheel joints use velocity-like effort
  targets.
- The Phase B base actor is frozen. Only the 8-D residual actor and its reward
  and four cost critics are optimized.
- The 50-D residual observation contains proprioception, previous residual,
  velocity command, route/clearance preview, posture state, and objective
  state.
- Route, mechanical energy, posture mode, and smoothness costs use frozen
  offline P95 normalization.
- The selected Phase B5 simulation candidate passed a corrected three-seed,
  384,000-sample-per-controller comparison with zero hard failures.
- The selected runtime residual filter alpha is 0.10.
- The simulator-independent test suite has 23 tests.
- A staged posture curriculum now uses the official LOW/MID/HIGH command
  heights (0.10/0.20/0.30 m). It starts from MID, opens LOW/MID transitions,
  then opens all three modes while enforcing the v1.0 switch preconditions.

The current learner is a compact PPO/NP3O-style implementation with four cost
critics and projected dual updates against fixed offline budgets. The unified
objective weights are multiplier floors, so adaptive updates can strengthen a
constraint but cannot remove route, energy, posture, or smoothness pressure.
It is not a line-for-line reproduction of the official NP3O trainer.

## Visual training

Omit `--headless` to open the Isaac Sim viewport. For an observable run, use
16 environments; the viewport camera follows the first environment. Closing
the Isaac window stops the run after discarding any incomplete rollout and
saves the most recently completed iteration.

First initialize a unique run directory, then launch training:

```bash
/home/ky/GBU_tita/.miniforge3/envs/habitat_ros/bin/python \
  scripts/prepare_rl_run.py --config configs/phase_b2_rough.json \
  --stage phase_b_residual --run-id phase_b2_visual_NEW

PYTHONUNBUFFERED=1 PYTHONPATH=. TERM=xterm \
./IsaacLab/isaaclab.sh -p scripts/train_phase_b.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --checkpoint ./tita_rl_official/tita_example_10000.pt \
  --initial-residual-checkpoint ./rl_runs/phase_b_residual/phase_b5_floor_np3o_iter020_selected/checkpoints/residual_phase_b_selected.pt \
  --run-dir ./rl_runs/phase_b_residual/phase_b2_visual_NEW \
  --num-envs 16 --iterations 100 --rollout-steps 24 \
  --command-x 0.3 --seed 7 --acceleration-limit 500 \
  --residual-filter-alpha 0.10 --terrain rough_warmup \
  --domain-randomization --device cuda:0
```

Visual training runs are experimental and must pass deterministic evaluation
before replacing the selected checkpoint.

For the first visible posture-curriculum smoke run, initialize a new run from
`configs/phase_b3_posture_curriculum.json` and add `--posture-curriculum` to
the training command. This stage remains on a flat collision plane; it
validates safe posture switching before physical tunnel and stair curricula.
Training metrics include per-mode measured height and absolute height error;
`scripts/run_tita_residual_lab.py --posture LOW|MID|HIGH` provides a fixed-mode
check for a saved checkpoint.

`scripts/calibrate_posture_poses.py` performs a simulation-only symmetric
leg-angle sweep against the official URDF. Its output may seed posture
feed-forward targets, but it is never treated as hardware calibration.
The speed-gated sweep found stable wheel-axis/body-center heights of 0.1238 m,
0.1979 m, and 0.2929 m for LOW/MID/HIGH, with maximum planar speeds of
0.1094/0.1949/0.1678 m/s under the frozen official controller. The environment
smoothly blends these simulation-only anchors underneath the learned residual.
During posture requests the velocity command is zero until the transition is
complete. LOW/MID/HIGH curriculum speeds are capped at 0.15/0.30/0.20 m/s.
Normalized training costs are winsorized at 10 after frozen-P95 division so a
single corridor excursion cannot numerically destabilize the value critics.
The 30-iteration `phase_b3_posture_speed_gated_visual_002` smoke run sampled
LOW/MID/HIGH 4630/4824/2066 times, started 48 transitions, and recorded zero
hard failures and zero safety projections. It is a smoke result, not a selected
checkpoint; the Phase B5 flat-ground candidate remains unchanged.

## Randomized LOW tunnel curriculum

`--terrain tunnel` creates independent collision geometry for every vectorized
environment. Height, width, and length are sampled deterministically from the
run seed. The curricula are staged around the measured clearance boundary:

```text
bootstrap: height 0.48-0.52 m, width 0.70-0.80 m, length 0.40-0.60 m
bridge:    height 0.46-0.50 m, width 0.66-0.74 m, length 0.45-0.70 m
bridge-low: height 0.44-0.48 m, width 0.64-0.72 m, length 0.50-0.75 m
easy: height 0.40-0.44 m, width 0.58-0.65 m, length 0.60-0.90 m
full: height 0.34-0.44 m, width 0.40-0.65 m, length 0.60-1.50 m
```

The 150-iteration bootstrap checkpoint was rejected after paired evaluation:
it produced 35 versus 37 baseline exits on bootstrap seed 79 and increased
easy-range collisions from 21/14/19 to 32/32/32 on seeds 71/79/83, with zero
easy exits for both controllers. New tunnel training therefore starts from the
selected flat checkpoint with `configs/phase_b4_tunnel_low_bridge.json`. A
seed-89 boundary sweep produced 19 exits and 4 collisions in this bridge
range. The bridge checkpoint must improve paired multi-seed evaluation before
it initializes the bridge-low stage. Easy and full-range training remain gated
on stable performance at each preceding level.

Bridge-low training must use long rollouts. A 60-iteration experiment with
24-step rollouts was rejected because the actor was updated repeatedly before
the robot reached the tunnel: seed-109 deterministic performance regressed
from 24 exits / 6 collisions to 0 / 36 at both iterations 10 and 20. The
replacement uses 192-step rollouts, a 5e-6 learning rate, and checkpoints every
iteration. Because the long-rollout smoke still exposed a stochastic-policy
versus deterministic-policy gap, bridge-low fine-tuning also fixes
`policy_log_std` at -4.5 (about 0.011 rad) instead of learning with the inherited
roughly 0.05 rad exploration noise.

## Controller contract

The official joint order is:

```text
joint_left_leg_1
joint_left_leg_2
joint_left_leg_3
joint_left_leg_4
joint_right_leg_1
joint_right_leg_2
joint_right_leg_3
joint_right_leg_4
```

The official default position is:

```text
[0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0]
```

Control is computed as:

```text
q_raw = q_base + delta_q
q_cmd = SafetyProject(q_raw)

leg joints:  tau = 40 * (q_cmd - q) - 1.0 * qd
wheel joints: tau = 10 * q_cmd - 0.5 * qd
```

URDF limits are 60 Nm and 25 rad/s for leg joints, and 15 Nm and 20
rad/s for wheel joints. The actuator uses a 0.5 rad/s internal velocity
margin. The frozen acceleration threshold is 500 rad/s^2, calibrated from an
Isaac Sim official-policy baseline whose observed maximum was 402.911
rad/s^2. This is a simulation threshold, not a hardware-validated limit.

Hard-reason counters always use this order:

```text
[
  infeasible_projection,
  velocity,
  acceleration,
  torque,
  collision,
  height,
  attitude,
  clearance
]
```

## Unified objective

The primary objective weights are route 0.40, mechanical energy 0.35, and
posture mode 0.25. Smoothness is an additional cost channel with multiplier
0.05. Frozen P95 values are:

```text
route  = 0.0007925402
energy = 1.9832050
mode   = 0.6782783
smooth = 0.2076107264
```

The `smooth_v2` channel is the sum of final joint-target change, residual
change, filtered joint-acceleration ratio, and soft-contact cost. Its frozen
P95 was calibrated with the official frozen policy using seed 11, 128
environments for 200 steps, rough warm-up terrain, and domain randomization.
The selected residual filter alpha is 0.10; the official baseline emits zero
residual, so its smoothness calibration is independent of this filter value.

Mechanical energy uses `sum(abs(tau_i * qd_i)) * dt` over all eight joints.
The reported mechanical COT uses the nominal URDF link mass of 29.0254 kg;
neither this mass nor the mechanical-to-electrical correlation has been
validated on a loaded physical robot.

## Reproduce the latest evaluation

Run from `/home/ky/tita_uc_residual_rl`:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. TERM=xterm \
./IsaacLab/isaaclab.sh -p scripts/evaluate_phase_b.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --official-checkpoint ./tita_rl_official/tita_example_10000.pt \
  --residual-checkpoint ./rl_runs/phase_b_residual/phase_b5_floor_np3o_iter020_selected/checkpoints/residual_phase_b_selected.pt \
  --num-envs 128 --steps 1000 --command-x 0.3 \
  --acceleration-limit 500 \
  --residual-filter-alpha 0.10 \
  --objective-p95 0.0007925402 1.9832050 0.6782783 0.2076107264 \
  --seed 11 --terrain rough_warmup --domain-randomization \
  --output ./rl_runs/phase_b_residual/phase_b5_floor_np3o_iter020_selected/residual_eval_seed11_1000steps.json \
  --headless --device cuda:0
```

The paired official baseline is reproduced by removing
`--residual-checkpoint` and writing to `official_eval_seed11.json`. The current
comparison is recorded in
`rl_runs/phase_b_residual/phase_b5_floor_np3o_iter020_selected/RESULTS.md`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
/home/ky/GBU_tita/.miniforge3/envs/habitat_ros/bin/python \
-m pytest -q tests/test_core.py -s
```

## Known validation gaps

- The rough warm-up uses a stable plane with non-colliding RayCaster preview
  geometry; physical obstacle contacts, slopes, and stairs are not validated.
- Clearance/corridor preview is simulated and is not connected to live ROS2
  obstacle data yet.
- LOW/HIGH posture curriculum passed one flat-plane smoke run, but its new
  checkpoint has not passed deterministic multi-seed evaluation.
- Domain randomization currently covers base mass and contact material; sensor
  noise, actuator delay, and broader dynamics randomization are not validated.
- ONNX/TensorRT numerical equivalence is not tested.
- ROS2/NeuPAN shadow validation is not completed.
- Electrical COT and its correlation with simulated mechanical COT are not
  measured.

These gaps prevent claiming production or hardware readiness. The latest
result is a successful flat-plane Phase B validation, not a completed sim2real
controller.
