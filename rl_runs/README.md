# TITA-UC RL runs

This directory is the output boundary for experiments.  Source code and
frozen experiment inputs stay outside it.

## Stages

- `phase_a_baseline/`: Isaac Lab reproduction of the official 8-DOF control
  contract.  No residual policy is trained here.
- `phase_b_residual/`: official policy frozen; only the 8-D joint-target
  residual is trained.
- `phase_c_joint_finetune/`: only the final official policy layers may be
  unfrozen, with a lower learning rate.
- `phase_d_shadow/`: sim2sim and ROS2 shadow validation.  No real actuator
  command is allowed from this directory.

Each initialized run contains:

```text
config.json       copied training configuration
metadata.json     git/asset/device/runtime information
metrics.jsonl     one JSON object per evaluation interval
checkpoints/      policy and optimizer checkpoints
videos/           optional Isaac Lab recordings
official_eval.json paired deterministic official-policy result
residual_eval.json deterministic residual-policy result
RESULTS.md         acceptance decision and metric comparison
```

The directories are intentionally empty in the repository.  Use
`scripts/prepare_rl_run.py` before launching a training job.

Evaluation results are only comparable when the asset, command, environment
count, step count, seed, acceleration limit, and frozen P95 values match.
Training rollout rewards are diagnostics and must not be used as a substitute
for the paired deterministic evaluation.
