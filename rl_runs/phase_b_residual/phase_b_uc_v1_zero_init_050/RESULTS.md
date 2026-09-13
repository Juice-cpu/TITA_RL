# Phase B zero-init iteration 50 acceptance

Date: 2026-09-12

Evaluation contract:

```text
Isaac Sim 4.5 / Isaac Lab v2.0.0 / CUDA 0
official 8-DOF TITA URDF and tita_example_10000.pt
128 environments x 200 steps, seed 11
forward command 0.3 m/s, MID posture, flat plane
acceleration threshold 500 rad/s^2
```

| Metric | Official | Official + residual | Change |
| --- | ---: | ---: | ---: |
| reward mean | 2.534741 | 2.548588 | +0.546% |
| tracking error mean | 0.055019 | 0.054157 | -1.567% |
| distance mean (m) | 1.220739 | 1.248829 | +2.301% |
| mechanical energy mean (J) | 90.054619 | 88.770058 | -1.426% |
| mechanical COT (nominal URDF mass) | 0.259081 | 0.249641 | -3.644% |
| normalized route cost | 0.623697 | 0.592754 | -4.961% |
| normalized energy cost | 0.227043 | 0.223805 | -1.426% |
| normalized mode cost | 0.255346 | 0.251520 | -1.498% |
| normalized smooth cost | 0.294566 | 0.308044 | +4.576% |
| projection fraction | 0.020000 | 0.020000 | unchanged |
| hard failures | 0 | 0 | pass |

Decision: pass the first flat-plane Phase B acceptance. The residual policy
keeps all seven measured hard-failure counters at zero while improving COT,
distance, tracking, route, and posture metrics. Smoothness regresses by 4.576%
and remains an explicit optimization target; this result does not pass later
terrain, transition, domain-randomization, export, ROS2 shadow, or hardware
acceptance stages.

Source artifacts:

```text
official_eval.json
residual_eval.json
checkpoints/residual_phase_b_iter_00050.pt
metrics.jsonl
```
