# Phase B2 smooth_v2 selected simulation candidate

Date: 2026-09-12

The selected controller is the frozen official 8-DOF TITA policy plus
`checkpoints/residual_phase_b_selected.pt`, residual filter alpha 0.20, and
the deterministic Safety Projector. The selected checkpoint is byte-identical
to the accepted state-projected warm-up source; its SHA-256 is
`5f31cc09d5889242a5a6b5032d6cedb62322b826322f32d59c4261d3c168dacf`.

Evaluation contract:

```text
Isaac Sim 4.5 / Isaac Lab v2.0.0 / CUDA 0 / RTX 4070
official 8-DOF TITA URDF and tita_example_10000.pt
seeds 11, 17, 23; 128 environments x 200 steps per seed
76,800 samples per controller
forward command 0.3 m/s, MID posture
rough_warmup RayCaster preview and domain randomization
acceleration threshold 500 rad/s^2
smooth_v2 frozen P95 0.2076107264
```

| Metric | Official | Official + selected residual | Change |
| --- | ---: | ---: | ---: |
| reward mean | 2.539849 | 2.548499 | +0.341% |
| tracking error mean | 0.063418 | 0.063010 | -0.644% |
| distance mean (m) | 1.236553 | 1.258716 | +1.792% |
| mechanical energy mean (J) | 87.557404 | 86.995778 | -0.641% |
| mechanical COT (nominal URDF mass) | 0.248677 | 0.242731 | -2.391% |
| normalized route cost | 0.599413 | 0.581641 | -2.965% |
| normalized energy cost | 0.220747 | 0.219331 | -0.641% |
| normalized mode cost | 0.265978 | 0.262552 | -1.288% |
| normalized smooth_v2 cost | 0.212141 | 0.211950 | -0.090% |
| hard failures | 0 | 0 | pass |

Decision: accept this combination as the current Phase B2 simulation
candidate. All aggregate objectives improve and no hard failure occurred.
Seed 23 had a small smoothness increase of about 0.10%, so the aggregate
smoothness gain is modest and must be rechecked on broader terrain and command
distributions before Phase C.

The 5-iteration alpha-0.25 and 20-iteration alpha-0.20 filter-aware
checkpoints are retained as experiment artifacts but are not selected: both
regressed smooth_v2 in deterministic evaluation. This indicates that the
compact fixed-multiplier NP3O update needs budget/penalty refinement before
additional long training.

This result is simulation-only. It does not validate physical obstacles,
LOW/HIGH transitions, electrical energy, ROS2 shadow operation, or hardware.
