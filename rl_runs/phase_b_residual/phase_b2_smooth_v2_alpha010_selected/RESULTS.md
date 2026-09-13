# Phase B2 alpha 0.10 selected simulation candidate

The selected controller combines the frozen official 8-DOF TITA policy,
`checkpoints/residual_phase_b_selected.pt`, residual filter alpha 0.10, and
the deterministic Safety Projector. The checkpoint SHA-256 is
`5f31cc09d5889242a5a6b5032d6cedb62322b826322f32d59c4261d3c168dacf`.

Evaluation used Isaac Sim 4.5, Isaac Lab 2.0.0, the RTX 4070, seeds 11/17/23,
128 environments, 1000 steps per seed, a 0.3 m/s MID-posture command,
rough-warmup preview terrain, and domain randomization. Each controller was
evaluated over 384,000 samples.

| Metric | Official | Official + residual | Change |
| --- | ---: | ---: | ---: |
| tracking error mean | 0.019627 | 0.018757 | -4.429% |
| distance mean (m) | 5.959040 | 6.041549 | +1.385% |
| mechanical energy mean (J) | 109.714819 | 107.776486 | -1.767% |
| mechanical COT | 0.064662 | 0.062652 | -3.109% |
| normalized route cost | 46.770959 | 33.280173 | -28.844% |
| normalized energy cost | 0.055322 | 0.054345 | -1.767% |
| normalized mode cost | 0.093559 | 0.093532 | -0.028% |
| normalized smooth cost | 0.052278 | 0.052052 | -0.433% |
| hard failures | 0 | 0 | pass |

Decision: accept alpha 0.10 as the Phase B2 simulation candidate. All
aggregate acceptance metrics improve and there are no hard failures. Seed 17
has a 0.018% mode-cost regression, so posture robustness remains a specific
validation target for broader commands and LOW/HIGH transitions.

This is simulation-only. It does not authorize hardware actuation and does
not validate physical obstacles, LOW/HIGH posture transitions, electrical
energy, ONNX/TensorRT equivalence, or the ROS2/NeuPAN shadow path.
