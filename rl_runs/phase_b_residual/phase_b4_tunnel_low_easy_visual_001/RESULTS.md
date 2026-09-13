# Phase B4 LOW Easy Tunnel Visual Smoke

This run warm-started the 8-D residual actor from the selected flat-ground
checkpoint. The official base policy remained frozen.

## Geometry

- Configured net height: 0.40-0.44 m
- Configured net width: 0.58-0.65 m
- Configured length: 0.60-0.90 m
- Sampled seed-67 minimum: 0.4007 x 0.5870 x 0.6200 m
- Sampled seed-67 maximum: 0.4387 x 0.6493 x 0.8696 m

## Smoke result

- Iterations: 50
- Vectorized environments: 16
- First observed tunnel exit: iteration 26
- Environments observed exiting: 1
- Hard failures across rollouts: 38
- Iterations with zero hard failures: 28/50
- Joint velocity violation count: 0
- Final checkpoint: `checkpoints/residual_phase_b_iter_00050.pt`

This is an experimental smoke checkpoint. It is not selected for deployment
and does not replace the flat-ground candidate. The next gate is deterministic
multi-seed evaluation on the easy range before medium/full tunnel training.
