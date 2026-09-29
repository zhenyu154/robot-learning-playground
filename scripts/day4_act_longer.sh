#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Override DATASET_REPO_ID, DATASET_ROOT, or OUTPUT_DIR when running elsewhere.

# Day 4 controlled experiment:
# Keep the Day 2 dataset and ACT configuration fixed, and change only
# the total number of training steps from 500 to 2000.

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID:-wusanggg/panda_pick_cube_day1}" \
  --dataset.root="${DATASET_ROOT:-${HOME}/robotics/data/panda_pick_cube_day1}" \
  --policy.type=act \
  --policy.device=cpu \
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
  --output_dir="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/act_panda_longer_v1}" \
  --job_name=act_panda_longer_v1
