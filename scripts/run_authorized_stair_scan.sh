#!/usr/bin/env bash
set -uo pipefail

repo_root=/home/ky/tita_uc_residual_rl
diagnostics_dir="$repo_root/rl_runs/diagnostics"
result_file="$diagnostics_dir/trained_policy_seed211.json"
log_file="$diagnostics_dir/trained_policy_seed211.log"
status_file="$diagnostics_dir/trained_policy_seed211.status"
pid_file="$diagnostics_dir/trained_policy_seed211.pid"
launcher_log="$diagnostics_dir/trained_policy_seed211.launcher.log"

mkdir -p "$diagnostics_dir"
cd "$repo_root"

if [[ "${1:-}" != "--worker" ]]; then
  rm -f "$launcher_log"
  setsid -f "$0" --worker </dev/null >>"$launcher_log" 2>&1
  printf 'Detached stair scan launched. Status: %s\n' "$status_file"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export TERM=xterm

rm -f "$result_file" "$status_file" "$pid_file"
printf '%s\n' "$$" >"$pid_file"
printf 'state=running\npid=%s\nstarted_at=%s\n' "$$" "$(date -Iseconds)" >"$status_file"

finalize() {
  exit_code=$?
  if [[ -s "$result_file" && $exit_code -eq 0 ]]; then
    state=completed
  else
    state=failed
  fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\nresult=%s\nlog=%s\n' \
    "$state" "$exit_code" "$(date -Iseconds)" "$result_file" "$log_file" >"$status_file"
  rm -f "$pid_file"
}
trap finalize EXIT

./IsaacLab/isaaclab.sh -p scripts/probe_stair_jump_feasibility.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --checkpoint ./tita_rl_official/tita_example_10000.pt \
  --residual-checkpoint ./rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt \
  --steps 250 \
  --output "$result_file" \
  --seed 211 \
  --trigger-distances 0.245 \
  --command-speeds 0.45 \
  --pulse-command-speeds 0.45 \
  --crouch-command-speeds 0.45 \
  --extension-command-speeds 0.45 \
  --crouch-steps 8 \
  --extension-steps 12 \
  --hip-extension-delay-steps 0 \
  --knee-extension-delay-steps 0 \
  --hip-crouch-rad 0.30 \
  --hip-extension-rad 0.60 \
  --knee-crouch-rad 1.00 \
  --knee-extension-rad 0.90 \
  --residual-only \
  --summary-only \
  --device cuda:0 2>&1 | tee "$log_file"

exit "${PIPESTATUS[0]}"
