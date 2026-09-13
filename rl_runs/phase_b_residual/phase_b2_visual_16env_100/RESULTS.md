# Phase B2 visual training run

Date: 2026-09-12

This run opened the Isaac Sim viewport and trained 16 parallel TITA
environments for 100 complete iterations. It warm-started from the selected
Phase B2 checkpoint and used rough warm-up preview geometry, domain
randomization, residual filter alpha 0.20, and the deterministic Safety
Projector.

Training completed with:

```text
iterations: 100
hard failures: 0
joint velocity violations: 0
velocity state projections: 0
checkpoint: checkpoints/residual_phase_b_iter_00100.pt
```

Seed-11 deterministic evaluation against the official policy:

| Metric | Change |
| --- | ---: |
| reward mean | -1.196% |
| tracking error | +0.417% |
| distance | -1.593% |
| mechanical COT | -0.350% |
| route cost | +11.006% |
| energy cost | -1.938% |
| mode cost | +2.787% |
| smooth_v2 cost | -2.366% |
| hard failures | 0 |

Decision: the visual training operation passed, but this checkpoint is not a
model candidate. It improves smoothness and energy while regressing route,
posture, tracking, progress, and total reward. The selected Phase B2
checkpoint remains unchanged.
