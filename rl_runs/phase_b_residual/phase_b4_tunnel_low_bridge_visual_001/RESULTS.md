# Phase B4 LOW Bridge

This visible run trained for 100 iterations from the protected selected
flat-ground residual. The official 8-DOF base policy remained frozen.

## Training

- Geometry: height 0.46-0.50 m, width 0.66-0.74 m, length 0.45-0.70 m
- Seed: 89
- Environments: 16
- Cumulative exits: 125
- Final-iteration hard failures: 0
- Final-iteration velocity violations: 0

Iteration 100 was not selected because deterministic performance had already
started to regress. Iteration 20 was selected for early stopping.

## Paired deterministic evaluation

Each run used 16 environments, 800 steps, a 0.15 m/s LOW command, residual
filter alpha 0.10, and the Safety Projector.

| Seed | Baseline exits / collisions | Iteration-20 exits / collisions |
| --- | ---: | ---: |
| 97 | 12 / 8 | 34 / 0 |
| 101 | 8 / 11 | 33 / 0 |
| 103 | 12 / 9 | 35 / 0 |
| **Total** | **32 / 28** | **102 / 0** |

Iteration 50 also achieved 34 / 0 on seed 97. Iteration 100 regressed to
29 / 2, so it was rejected. On the lower 0.44-0.48 m seed-107 range, the
selected bridge candidate achieved 30 / 0 versus the flat baseline's 1 / 15.

Selected checkpoint:
`checkpoints/residual_phase_b_selected.pt`

SHA-256:
`127910ee39068757dff909ac84bab66f64bd532e2adc81626900995ff5635b1e`

This checkpoint is a simulation curriculum candidate only. It is not approved
for full-range tunnels, ROS2 actuation, or hardware deployment.
