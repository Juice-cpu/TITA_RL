#!/usr/bin/env bash
set -uo pipefail

repo_root=/home/ky/tita_uc_residual_rl
run_dir="$repo_root/rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001"
log_file="$run_dir/training.log"
status_file="$run_dir/training.status"
pid_file="$run_dir/training.pid"
launcher_log="$run_dir/training.launcher.log"

cd "$repo_root"

if [[ "${1:-}" != "--worker" ]]; then
  rm -f "$launcher_log"
  setsid -f "$0" --worker </dev/null >>"$launcher_log" 2>&1
  printf 'Detached visual stair training launched. Status: %s\n' "$status_file"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export TERM=xterm

rm -f "$status_file" "$pid_file"
printf '%s\n' "$$" >"$pid_file"
printf 'state=running\npid=%s\nstarted_at=%s\n' "$$" "$(date -Iseconds)" >"$status_file"

finalize() {
  exit_code=$?
  latest_checkpoint=$(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'residual_phase_b_iter_*.pt' -printf '%f\n' | sort | tail -1)
  if [[ -n "$latest_checkpoint" && $exit_code -eq 0 ]]; then
    state=completed
  else
    state=failed
  fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\nlatest_checkpoint=%s\nlog=%s\n' \
    "$state" "$exit_code" "$(date -Iseconds)" "$latest_checkpoint" "$log_file" >"$status_file"
  rm -f "$pid_file"
}
trap finalize EXIT

./IsaacLab/isaaclab.sh -p scripts/train_phase_b.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --checkpoint ./tita_rl_official/tita_example_10000.pt \
  --initial-residual-checkpoint ./rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt \
  --run-dir "$run_dir" \
  --num-envs 16 \
  --iterations 60 \
  --rollout-steps 500 \
  --command-x 0.45 \
  --seed 211 \
  --acceleration-limit 500 \
  --residual-filter-alpha 0.10 \
  --terrain stairs \
  --fixed-posture HIGH \
  --device cuda:0 2>&1 | tee "$log_file"

exit "${PIPESTATUS[0]}"
