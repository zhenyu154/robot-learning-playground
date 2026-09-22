#!/usr/bin/env bash

set -e

lerobot-train \
  --dataset.repo_id=wusanggg/panda_pick_cube_day1 \
  --dataset.root=/home/wusanggg/robotics/data/panda_pick_cube_day1 \
  --policy.type=act \
  --policy.device=cpu \
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
  --output_dir=/home/wusanggg/robotics/robot-learning-playground/outputs/act_panda_smoke_v1 \
  --job_name=act_panda_smoke_v1