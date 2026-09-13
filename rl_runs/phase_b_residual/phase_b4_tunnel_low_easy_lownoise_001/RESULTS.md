# Phase B4 LOW Easy Pre-Training Gate

The selected bridge-low controller was evaluated before any optimization on
the 0.40-0.44 m height, 0.58-0.65 m width, and 0.60-0.90 m length range.

Deterministic seed 131 used 16 environments for 800 steps at 0.15 m/s. It
recorded 0 tunnel exits and 32 collisions, with zero joint-velocity violations
and no other hard-failure category.

Training was not started because the all-failure distribution creates an
abrupt curriculum step and provides no successful trajectories to preserve.
No checkpoint is selected from this run. The next experiment inserts an
overlapping 0.42-0.46 m transition range.

This result is simulation-only and is not approved for deployment.
