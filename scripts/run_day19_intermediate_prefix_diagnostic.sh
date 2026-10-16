#!/usr/bin/env bash
set -euo pipefail

# Run in the user's activated lerobot-xpu environment.
DAY19_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DAY19_PROJECT_ROOT="$(cd -- "$DAY19_SCRIPT_DIR/.." && pwd)"
cd -- "$DAY19_PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-day19_initial7_capped_then20_step21_v2}"
DAY19_MODEL="outputs/act_panda_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000/checkpoints/014000/pretrained_model"
DAY19_TRACE="results/${RUN_TAG}_grasp_trace_v1.jsonl"
DAY19_LOG="results/${RUN_TAG}_eval.log"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
  echo "Usage: bash scripts/run_day19_intermediate_prefix_diagnostic.sh [--check]" >&2
  exit 2
fi
if [[ -e "$DAY19_TRACE" || -e "$DAY19_LOG" ]]; then
  echo "Output exists; choose a new RUN_TAG before running." >&2
  printf 'Log: %s\nTrace: %s\n' "$DAY19_LOG" "$DAY19_TRACE" >&2
  exit 2
fi
for DAY19_REQUIRED in \
  scripts/evaluate_panda_capped_prefix.py \
  scripts/panda_capped_action_prefix.py \
  scripts/evaluate_panda_act.py \
  scripts/panda_action_prefix.py \
  configs/day11_home_x_centered.json \
  configs/day19_intermediate_prefix_diagnostic_positions.json \
  "$DAY19_MODEL/config.json" "$DAY19_MODEL/model.safetensors"; do
  if [[ ! -f "$DAY19_REQUIRED" ]]; then
    printf 'Required file is missing: %s\nNo evaluation started.\n' "$DAY19_REQUIRED" >&2
    exit 2
  fi
done
command -v python >/dev/null

DAY19_COMMAND=(python scripts/evaluate_panda_capped_prefix.py
  --checkpoint "$DAY19_MODEL" --device xpu
  --task PandaPickCubeKeyboardRandomViewerLong-v0
  --episodes 6 --max-steps 150 --fps 10
  --n-action-steps 7 --later-n-action-steps 20 --prefix-switch-step 21
  --gripper-mode policy
  --home-pose configs/day11_home_x_centered.json --min-tcp-z 0.008
  --position-schedule configs/day19_intermediate_prefix_diagnostic_positions.json
  --debug-geometry --trace-output "$DAY19_TRACE")

printf 'Run tag: %s\nInitial prefixes: 7, 7, 6; then 20 from step 21.\n' "$RUN_TAG"
if [[ "${1:-}" == "--check" ]]; then
  printf 'Command:'
  printf ' %q' "${DAY19_COMMAND[@]}"
  printf '\nPreflight passed; no outputs created and no evaluation started.\n'
  exit 0
fi

mkdir -p results
# Reserve the new log exclusively, then append evaluator output to it.
(set -o noclobber; : > "$DAY19_LOG")
"${DAY19_COMMAND[@]}" 2>&1 | tee -a "$DAY19_LOG"
