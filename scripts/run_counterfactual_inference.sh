#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/342/hyellim/projects/CAT-LoRA}"
AUDIO_SUITE="${AUDIO_SUITE:?Set AUDIO_SUITE to a counterfactual suite directory}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME, e.g. musicinfuser_original or cat_lora_lite_train500_3000step}"
MUSICINFUSER_PATH="${MUSICINFUSER_PATH:?Set MUSICINFUSER_PATH to a safetensors checkpoint}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/counterfactual/$MODEL_NAME}"
PROMPT="${PROMPT:-a dancer moving to the rhythm of the music}"
NUM_FRAMES="${NUM_FRAMES:-150}"
SEED="${SEED:-42}"
MOCHI_DIR="${MOCHI_DIR:-weights}"
SAFE_T5="${SAFE_T5:-/home/ubuntu/342/hyellim/checkpoints/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db68a3ef122daf6e605701de53f766d671c19aa}"

export MOCHI_T5_MODEL="${MOCHI_T5_MODEL:-$SAFE_T5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT"
while IFS= read -r audio; do
  stem="$(basename "$audio" .wav)"
  python inference.py \
    --musicinfuser-path "$MUSICINFUSER_PATH" \
    --mochi-dir "$MOCHI_DIR" \
    --output-dir "$OUT_ROOT/$stem" \
    --input-file "$audio" \
    --prompt "$PROMPT" \
    --num-frames "$NUM_FRAMES" \
    --seed "$SEED"
done < <(python - <<'PY'
import json
import os
from pathlib import Path
suite = Path(os.environ["AUDIO_SUITE"])
meta = json.loads((suite / "suite_manifest.json").read_text(encoding="utf-8"))
for row in meta["entries"]:
    print(row["audio"])
PY
)
