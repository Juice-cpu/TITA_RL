# Phase B4 LOW Bootstrap 150

This visible Isaac Sim run trained the 8-D residual actor for 150 iterations
from the selected flat-ground checkpoint. The official base policy stayed
frozen. Geometry was randomized independently per vector environment.

## Training

- Geometry: height 0.48-0.52 m, width 0.70-0.80 m, length 0.40-0.60 m
- Seed: 73
- Vectorized environments: 16
- Rollout steps per iteration: 24
- Cumulative tunnel exits: 174
- Final-iteration hard failures: 0
- Final-iteration velocity violations: 0
- Final checkpoint: `checkpoints/residual_phase_b_iter_00150.pt`

## Deterministic evaluation

All evaluations used 16 environments, 800 steps, a 0.15 m/s LOW command,
residual filter alpha 0.10, and the Safety Projector.

| Geometry / seed | Selected flat baseline | Bootstrap-150 | Decision |
| --- | ---: | ---: | --- |
| bootstrap / 79, successes | 37 | 35 | regression |
| bootstrap / 79, collisions | 0 | 0 | tie |
| easy / 71, successes | 0 | 0 | tie |
| easy / 71, collisions | 21 | 32 | regression |
| easy / 79, successes | 0 | 0 | tie |
| easy / 79, collisions | 14 | 32 | regression |
| easy / 83, successes | 0 | 0 | tie |
| easy / 83, collisions | 19 | 32 | regression |

The checkpoint is rejected for promotion: it did not improve bootstrap
success and increased easy-range collisions on every paired seed. It does not
replace the selected flat-ground checkpoint and must not initialize easy or
full-range training.

An additional seed-89 boundary sweep found 2 successes / 14 collisions at
0.44-0.48 m and 19 successes / 4 collisions at 0.46-0.50 m. The latter is the
next bridge curriculum. All results are simulation-only and not deployable.
