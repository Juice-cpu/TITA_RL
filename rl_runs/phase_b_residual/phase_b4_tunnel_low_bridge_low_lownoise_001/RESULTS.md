# Phase B4 LOW Bridge-Low Low-Noise Selection

This run warm-started the selected bridge controller and trained for six
iterations with 192 rollout steps per iteration on randomized tunnels with:

```text
net height 0.44-0.48 m
net width  0.64-0.72 m
length     0.50-0.75 m
```

The residual policy used a `5e-6` learning rate and fixed exploration
`log_std=-4.5`. Training recorded 52 cumulative tunnel exits. Only iteration 4
had one collision; the other five iterations had no hard failure, and all six
iterations had zero joint-velocity violations.

Deterministic evaluation used 16 environments, 800 steps, seed-matched tunnel
dimensions, a 0.15 m/s command, LOW posture, the acceleration projector at
500 rad/s^2, and residual filter alpha 0.10.

| Seed | Input bridge: exits/collisions | Iteration 1: exits/collisions |
| --- | ---: | ---: |
| 109 | 24 / 6 | 24 / 5 |
| 113 | 28 / 2 | 28 / 2 |
| 127 | 27 / 3 | 27 / 3 |
| Total | 79 / 11 | 79 / 10 |

Iteration 3 was also checked on seed 109 and returned 24 exits / 6 collisions,
so it did not improve on the input controller. Iteration 1 is selected because
it preserves all 79 exits, removes one collision in aggregate, and does not
regress on either new seed. No evaluated run reported velocity, acceleration,
torque, joint-limit, height, attitude, or clearance hard failures; the reported
hard failures were tunnel collisions.

Selected checkpoint:

```text
checkpoints/residual_phase_b_selected.pt
```

Its SHA-256 is:

```text
eed4394f83430be00c2f20486e8831650e79b0fdd9731f62e62f85e9e70b0cba
```

This result is simulation-only. The tunnel dimensions and LOW pose anchors are
not hardware-calibrated, and the checkpoint is not approved for deployment.
