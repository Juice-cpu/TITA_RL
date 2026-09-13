# Phase B2 state-projected warm-up acceptance

Date: 2026-09-12

Evaluation contract:

```text
Isaac Sim 4.5 / Isaac Lab v2.0.0 / CUDA 0 / RTX 4070
official 8-DOF TITA URDF and tita_example_10000.pt
128 environments x 200 steps, seed 11
forward command 0.3 m/s, MID posture
native PhysX support plane plus non-colliding RayCaster preview geometry
base-mass and contact-material domain randomization
acceleration threshold 500 rad/s^2
joint velocity limits 25 rad/s (legs), 20 rad/s (wheels)
```

| Metric | Official | Official + residual | Change |
| --- | ---: | ---: | ---: |
| reward mean | 2.545951 | 2.559234 | +0.522% |
| tracking error mean | 0.063844 | 0.063790 | -0.083% |
| distance mean (m) | 1.239863 | 1.262723 | +1.844% |
| mechanical energy mean (J) | 87.563034 | 87.302292 | -0.298% |
| mechanical COT (nominal URDF mass) | 0.248028 | 0.242812 | -2.103% |
| normalized route cost | 0.596091 | 0.564709 | -5.265% |
| normalized energy cost | 0.220761 | 0.220104 | -0.298% |
| normalized mode cost | 0.269791 | 0.264489 | -1.965% |
| normalized smooth cost | 0.086673 | 0.101735 | +17.379% |
| hard failures | 0 | 0 | pass |
| evaluation velocity state projections | 0 | 0 | pass |

Training acceptance:

- All five iterations completed with zero hard failures and zero post-projector
  velocity violations.
- The first stochastic rollout produced one right-wheel PhysX overshoot:
  20.257 rad/s before projection and 20.000 rad/s after projection.
- No velocity state projection was required in iterations 2-5.
- All measured filtered joint accelerations remained below 500 rad/s^2.

Decision: pass the Phase B2 perception/domain-randomization warm-up. The
residual policy improves reward, progress, mechanical COT, route cost, posture
cost, and tracking while preserving the measured hard constraints. Smoothness
regresses by 17.379% and remains a required optimization target.

This is not acceptance of physical rough terrain, collision-bearing obstacles,
LOW/HIGH posture transitions, electrical energy, ROS2 shadow, or hardware.
The preview geometry is a non-colliding sensing layer above a stable native
PhysX support plane because wheel-triangle mesh contacts were unstable in the
current Isaac Sim 4.5 setup.

Source artifacts:

```text
config.json
metadata.json
metrics.jsonl
official_eval.json
residual_eval.json
checkpoints/residual_phase_b_iter_00005.pt
```
