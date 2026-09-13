# Phase B4 visual alpha 0.10 run

This experimental run warm-started from the selected Phase B2 checkpoint and
completed 16 environments x 100 iterations in the Isaac Sim viewport. All
training iterations reported zero hard failures.

The deterministic seed-11 evaluation used 128 environments x 1000 steps.
Iteration 100 remained collision-free but regressed against the official
baseline in posture mode cost (0.100688 versus 0.095301) and tracking error
(0.030200 versus 0.019426). It is rejected and does not replace the selected
Phase B2 alpha 0.10 checkpoint.

Diagnosis: the dual optimizer allowed the energy and posture multipliers to
decay to zero. The learner now supports fixed per-cost multiplier floors, and
the unified objective weights are configured as those floors for subsequent
runs.
