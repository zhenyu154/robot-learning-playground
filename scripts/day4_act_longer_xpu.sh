#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Override DATASET_REPO_ID, DATASET_ROOT, or OUTPUT_DIR when running elsewhere.

# XPU counterpart to day4_act_longer.sh. The original CPU experiment remains intact.
python - <<'PY_CHECK'
import torch
assert torch.xpu.is_available(), 'Intel XPU is unavailable in this Python environment'
print(f'Using {torch.__version__} on {torch.xpu.get_device_name(0)}')
PY_CHECK

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/act_panda_longer_xpu_v1}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 2
fi

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID:-wusanggg/panda_pick_cube_day1}" \
  --dataset.root="${DATASET_ROOT:-${HOME}/robotics/data/panda_pick_cube_day1}" \
  --policy.type=act \
  --policy.device=xpu \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --batch_size=8 \
  --num_workers=0 \
  --steps=2000 \
  --seed=1000 \
  --log_freq=100 \
  --save_freq=200 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_panda_longer_xpu_v1
