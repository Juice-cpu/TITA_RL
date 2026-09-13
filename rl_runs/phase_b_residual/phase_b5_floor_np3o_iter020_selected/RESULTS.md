# Phase B5 fixed-floor NP3O selected simulation candidate

This candidate is the frozen official 8-DOF TITA actor plus the iteration-20
residual from `phase_b5_floor_np3o_warm_050`, residual filter alpha 0.10, and
the deterministic Safety Projector. The unified objective weights
`[0.40, 0.35, 0.25, 0.05]` are also the adaptive multiplier floors.

Evaluation used Isaac Sim 4.5, Isaac Lab 2.0.0, RTX 4070, seeds 11/17/23,
128 environments, 1000 steps per seed, a 0.3 m/s MID-posture command,
rough-warmup preview terrain, and domain randomization. Each controller was
evaluated over 384,000 samples.

| Metric | Official | Official + residual | Change |
| --- | ---: | ---: | ---: |
| tracking error mean | 0.019627 | 0.018635 | -5.052% |
| distance mean (m) | 5.959040 | 6.072757 | +1.908% |
| mechanical energy mean (J) | 109.714819 | 107.732948 | -1.806% |
| mechanical COT | 0.064662 | 0.062304 | -3.646% |
| normalized route cost | 46.770959 | 4.540864 | -90.291% |
| normalized energy cost | 0.055322 | 0.054323 | -1.806% |
| normalized mode cost | 0.093559 | 0.092813 | -0.798% |
| normalized smooth cost | 0.052278 | 0.051892 | -0.740% |
| hard failures | 0 | 0 | pass |

Decision: accept iteration 20 as the current simulation candidate. Every
acceptance metric improves for each tested seed and in aggregate, with no
hard failure. The iteration-100 visual-training checkpoint was rejected due
to tracking and posture regression and is not used here.

This is simulation-only. Physical obstacles, LOW/HIGH transitions,
electrical energy, ONNX/TensorRT equivalence, ROS2/NeuPAN shadow operation,
and hardware actuation remain unvalidated.
