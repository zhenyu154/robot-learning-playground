#!/usr/bin/env bash

set -euo pipefail

# Day 5: train on demonstrations with longer, clearer gripper-close supervision.
# Run this only after the new dataset has passed the collection QA checks.

python - <<'PY_CHECK'
import torch
assert torch.xpu.is_available(), "Intel XPU is unavailable in this Python environment"
print(f"Using {torch.__version__} on {torch.xpu.get_device_name(0)}")
PY_CHECK

OUTPUT_DIR=/home/wusanggg/robotics/robot-learning-playground/outputs/act_panda_day5_gripper_xpu_v1
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 2
fi

lerobot-train \
  --dataset.repo_id=wusanggg/panda_pick_cube_day5_gripper_v1 \
  --dataset.root=/home/wusanggg/robotics/data/panda_pick_cube_day5_gripper_v1 \
  --policy.type=act \
  --policy.device=xpu \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --batch_size=8 \
  --num_workers=0 \
  --steps=500 \
  --seed=1000 \
  --log_freq=25 \
  --save_freq=100 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_panda_day5_gripper_xpu_v1
