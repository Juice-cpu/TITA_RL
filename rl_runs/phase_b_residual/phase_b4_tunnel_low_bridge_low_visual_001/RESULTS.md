# Phase B4 LOW Bridge-Low Short-Rollout Rejection

This run warm-started the selected bridge candidate and trained for 60
iterations with 24 rollout steps per iteration on the 0.44-0.48 m height,
0.64-0.72 m width, and 0.50-0.75 m length range.

Training reached 15 cumulative exits by iteration 10, then never recorded
another exit. Collisions appeared from iteration 19 and the final iteration
recorded three. Velocity violations remained zero.

On deterministic seed-109 evaluation (16 environments, 800 steps), the input
bridge candidate achieved 24 exits / 6 collisions. Both iteration 10 and
iteration 20 regressed to 0 exits / 36 collisions. No checkpoint from this run
is selected.

The failure is consistent with a delayed-reward rollout mismatch: at 0.15 m/s
the robot needs roughly 250 control steps to reach the tunnel at 0.75 m, but
the policy was updated every 24 steps. The next experiment uses 192-step
rollouts, a 5e-6 learning rate, and per-iteration checkpoints so updates include
the obstacle interaction and can be stopped early.

All results are simulation-only and not approved for deployment.
