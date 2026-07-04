#!/usr/bin/env bash
set -euo pipefail
export AUDIO_SUITE="${AUDIO_SUITE:-/home/ubuntu/342/hyellim/projects/CAT-LoRA/assets/counterfactual_suites/moon_10s}"
export MODEL_NAME="${MODEL_NAME:-cat_lora_lite_train500_3000step_10s}"
export MUSICINFUSER_PATH="${MUSICINFUSER_PATH:-finetunes/cat_lora_lite_train500_3000step/model_3000.cat_lora.safetensors}"
export NUM_FRAMES="${NUM_FRAMES:-300}"
exec "$(dirname "$0")/run_counterfactual_inference.sh"
