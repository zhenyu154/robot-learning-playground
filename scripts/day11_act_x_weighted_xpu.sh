#!/usr/bin/env bash

set -euo pipefail

# Day11: fixed-Y X-axis training with active-X weighted ACT loss.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_REPO_ID="${DATASET_REPO_ID:-wusanggg/panda_pick_cube_day11_x_axis_v1}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/robotics/data/panda_pick_cube_day11_x_axis_v1}"
ACTIVE_WEIGHT="${ACTIVE_WEIGHT:-4}"
CLOSE_WEIGHT="${CLOSE_WEIGHT:-1}"
STEPS="${STEPS:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
PROGRESS_MINITERS="${PROGRESS_MINITERS:-50}"
SAVE_FREQ="${SAVE_FREQ:-$STEPS}"
SAVE_LOG="${SAVE_LOG:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/act_panda_day11_x_weighted_w${ACTIVE_WEIGHT}_xpu_v1}"
JOB_NAME="${JOB_NAME:-act_panda_day11_x_weighted_w${ACTIVE_WEIGHT}_xpu_v1}"

if [[ "$SAVE_LOG" != "0" && "$SAVE_LOG" != "1" ]]; then
  echo "SAVE_LOG must be 0 or 1" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" || ( "$SAVE_LOG" == "1" && -e "${OUTPUT_DIR}.log" ) ]]; then
  echo "Refusing to overwrite existing output or log: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "Local dataset not found: $DATASET_ROOT" >&2
  exit 2
fi
python - <<'PY_CHECK'
import torch
assert torch.xpu.is_available(), "Intel XPU is unavailable in this Python environment"
print(f"Using {torch.__version__} on {torch.xpu.get_device_name(0)}")
PY_CHECK

mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
TRAIN_CMD=(python "$SCRIPT_DIR/day8_train_y_weighted.py" \
  --active-axis=x \
  --active-weight="$ACTIVE_WEIGHT" \
  --close-weight="$CLOSE_WEIGHT" \
  --progress-miniters="$PROGRESS_MINITERS" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=xpu \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --batch_size=8 \
  --num_workers=0 \
  --steps="$STEPS" \
  --seed=1000 \
  --log_freq="$LOG_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME")

if [[ "$SAVE_LOG" == "1" ]]; then
  "${TRAIN_CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}.log"
else
  "${TRAIN_CMD[@]}"
fi
