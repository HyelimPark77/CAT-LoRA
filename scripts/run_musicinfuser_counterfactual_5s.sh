#!/usr/bin/env bash
set -euo pipefail
export AUDIO_SUITE="${AUDIO_SUITE:-/home/ubuntu/342/hyellim/projects/CAT-LoRA/assets/counterfactual_suites/moon_5s}"
export MODEL_NAME="${MODEL_NAME:-musicinfuser_original_5s}"
export MUSICINFUSER_PATH="${MUSICINFUSER_PATH:-weights/musicinfuser.safetensors}"
export NUM_FRAMES="${NUM_FRAMES:-150}"
exec "$(dirname "$0")/run_counterfactual_inference.sh"
