#!/usr/bin/env bash
# MahutKol PickCube ortamında toplanan gösterimler ile SmolVLA Eğitimi.
#
# Kullanım:
#   bash scripts/train_smolvla.sh [steps] [batch_size]
set -e
STEPS=${1:-5000}
BATCH=${2:-8}
OUT=~/vla_runs/smolvla_mahut_kol
rm -rf "$OUT"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --dataset.repo_id=local/pick_cube_arm \
  --dataset.root="$HOME/vla_data/pick_cube_arm" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=1000 \
  --log_freq=50 \
  --num_workers=2 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUT"
