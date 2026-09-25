#!/usr/bin/env bash

set -euo pipefail

# Run from the lerobot-xpu environment. Existing CPU scripts are unchanged.
python - <<'PY_CHECK'
import torch
assert torch.xpu.is_available(), 'Intel XPU is unavailable in this Python environment'
print(f'Using {torch.__version__} on {torch.xpu.get_device_name(0)}')
PY_CHECK

OUTPUT_DIR=/home/wusanggg/robotics/robot-learning-playground/outputs/act_panda_xpu_smoke_v1
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 2
fi

lerobot-train \
  --dataset.repo_id=wusanggg/panda_pick_cube_day1 \
  --dataset.root=/home/wusanggg/robotics/data/panda_pick_cube_day1 \
  --policy.type=act \
  --policy.device=xpu \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --batch_size=2 \
  --num_workers=0 \
  --steps=5 \
  --log_freq=1 \
  --save_freq=0 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_panda_xpu_smoke_v1
