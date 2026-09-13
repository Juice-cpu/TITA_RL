# Phase B5 HIGH two-step v4 results

Date: 2026-09-13

## Purpose

v4 extended each rollout to the full 24 s episode (1200 simulation steps) so
the approximately 1.95 m route could receive terminal completion credit at a
0.15 m/s command speed. It used 32 environments, two updates, learning rate
1e-7, frozen log standard deviation -3.8, and the fixed stair-specific P95.

## Corrected rollout results

Success counts below are new successes in each rollout, not cumulative counts.

| Rollout | Policy checkpoint | Strict completions | Hard failures |
|---:|---:|---:|---:|
| 1 | 0 | 1 | 7 (6 collision, 1 attitude) |
| 2 | 1 | 1 | 7 (3 collision, 4 attitude) |

## Deterministic seed-163 evaluation

| Candidate | Strict completions | Hard terminations | Decision |
|---|---:|---:|---|
| flat selected baseline | 2 | 4 | reference |
| iter 1 | 0 | 4 (3 collision, 1 attitude) | rejected |
| iter 2 | 0 | 3 collision | rejected |

Full-horizon credit assignment produced sparse stochastic successes but did not
move the deterministic mean to a successful policy. No selected checkpoint was
created. The next justified curriculum shortens the approach, tread and landing
while retaining the same two-step height range and strict safety gate.
