#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/342/hyellim/projects/CAT-LoRA}"
MANIFEST="${1:-$PROJECT_ROOT/music_infuser/configs/counterfactual_eval_manifest.json}"
OUT_CSV="${2:-$PROJECT_ROOT/outputs/metrics/counterfactual_metrics.csv}"

cd "$PROJECT_ROOT"
python -m music_infuser.cat_lora.evaluate_counterfactual_suite \
  --manifest "$MANIFEST" \
  --output-csv "$OUT_CSV"
