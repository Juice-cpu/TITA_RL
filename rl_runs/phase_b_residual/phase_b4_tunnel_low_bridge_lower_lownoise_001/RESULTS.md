# Phase B4 LOW Bridge-Lower Low-Noise Selection

This run inserted an overlapping curriculum level after the direct easy-range
pre-training gate failed. It warm-started the selected bridge-low controller
and trained for six iterations with 192 rollout steps per iteration on:

```text
net height 0.42-0.46 m
net width  0.62-0.70 m
length     0.55-0.82 m
```

The residual policy used a `5e-6` learning rate and fixed exploration
`log_std=-4.5`. Training recorded 41 cumulative tunnel exits. Per-iteration
collision counts were 0, 1, 1, 2, 1, and 1. Every iteration had zero joint
velocity violations.

Deterministic evaluation used 16 environments, 800 steps, seed-matched tunnel
dimensions, a 0.15 m/s command, LOW posture, the acceleration projector at
500 rad/s^2, and residual filter alpha 0.10.

| Seed | Input bridge-low: exits/collisions | Iteration 1: exits/collisions |
| --- | ---: | ---: |
| 137 | 17 / 9 | 17 / 8 |
| 149 | 17 / 10 | 17 / 10 |
| 157 | 18 / 10 | 18 / 10 |
| Total | 52 / 29 | 52 / 28 |

The retention check on the previous 0.44-0.48 m range used seed 113. The input
and iteration 1 both returned 28 exits / 2 collisions. Iteration 1 is selected
because it preserves all exits, removes one collision in aggregate on the new
range, and does not regress on the previous range. All evaluated hard failures
were tunnel collisions; no velocity or other hard-failure category was seen.

Selected checkpoint:

```text
checkpoints/residual_phase_b_selected.pt
```

Its SHA-256 is:

```text
eafa1e20b548a024b02648aef5e33d30fe4509a1618363074f779bec69e5d353
```

This result is simulation-only. The tunnel dimensions and LOW pose anchors are
not hardware-calibrated, and the checkpoint is not approved for deployment.
