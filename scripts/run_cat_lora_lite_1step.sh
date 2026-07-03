#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/342/hyellim/projects/CAT-LoRA}"
CONFIG="${CONFIG:-$PROJECT_ROOT/music_infuser/configs/cat_lora_lite_1step.yaml}"
SAFE_T5="${SAFE_T5:-/home/ubuntu/342/hyellim/checkpoints/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db68a3ef122daf6e605701de53f766d671c19aa}"

export MOCHI_T5_MODEL="${MOCHI_T5_MODEL:-$SAFE_T5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT"
python music_infuser/train_cat_lora.py --config-path "$CONFIG"
