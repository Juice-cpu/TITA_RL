# Phase B5 HIGH two-step v6 medium-tread results

Date: 2026-09-13

## Configuration

- HIGH posture, two ultra-shallow steps (0.005-0.015 m)
- tread 0.35-0.45 m, width 1.00-1.30 m, start x 0.45 m, landing 0.50 m
- 32 environments, 900-step rollouts, 3 updates, learning rate 1e-7
- Warm start: v5 short-geometry iter 3
- Frozen policy standard deviation -3.8 and acceleration projector 500 rad/s^2

## Training rollouts

| Rollout | Policy checkpoint | Strict completions | Hard failures |
|---:|---:|---:|---:|
| 1 | 0 | 29 | 0 |
| 2 | 1 | 29 | 1 collision |
| 3 | 2 | 26 | 1 collision |

## Deterministic paired evaluation

| Seed | Flat selected baseline | v6 iter 2 | Candidate hard failures |
|---:|---:|---:|---:|
| 163 | 23 | 23 | 0 |
| 167 | 16 | 19 | 0 |
| 173 | 15 | 18 | 0 |

The candidate did not reduce completions or increase hard failures on any seed.
The HIGH flat retention test on seed 163 ran 1000 steps with zero hard terminations
and zero joint-velocity violations. Iteration 2 is selected for this simulation-only
two-step medium-tread course.

This checkpoint is not validated for three steps, normal-height stairs, domain
randomization, hardware, or deployment.
