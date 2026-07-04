#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/342/hyellim/projects/CAT-LoRA}"
cd "$PROJECT_ROOT"
mkdir -p manifests outputs/metrics

PYTHONPATH="$PROJECT_ROOT" python -m music_infuser.cat_lora.build_counterfactual_manifest \
  --audio-suite "$PROJECT_ROOT/assets/counterfactual_suites/moon_5s" \
  --model "musicinfuser_original_5s=$PROJECT_ROOT/outputs/counterfactual/musicinfuser_original_5s" \
  --model "cat_lora_lite_train500_3000step_5s=$PROJECT_ROOT/outputs/counterfactual/cat_lora_lite_train500_3000step_5s" \
  --output "$PROJECT_ROOT/manifests/moon_counterfactual_5s.json"

PYTHONPATH="$PROJECT_ROOT" python -m music_infuser.cat_lora.evaluate_counterfactual_suite \
  --manifest "$PROJECT_ROOT/manifests/moon_counterfactual_5s.json" \
  --output-csv "$PROJECT_ROOT/outputs/metrics/moon_counterfactual_5s.csv"

PYTHONPATH="$PROJECT_ROOT" python -m music_infuser.cat_lora.build_counterfactual_manifest \
  --audio-suite "$PROJECT_ROOT/assets/counterfactual_suites/moon_10s" \
  --model "musicinfuser_original_10s=$PROJECT_ROOT/outputs/counterfactual/musicinfuser_original_10s" \
  --model "cat_lora_lite_train500_3000step_10s=$PROJECT_ROOT/outputs/counterfactual/cat_lora_lite_train500_3000step_10s" \
  --output "$PROJECT_ROOT/manifests/moon_counterfactual_10s.json"

PYTHONPATH="$PROJECT_ROOT" python -m music_infuser.cat_lora.evaluate_counterfactual_suite \
  --manifest "$PROJECT_ROOT/manifests/moon_counterfactual_10s.json" \
  --output-csv "$PROJECT_ROOT/outputs/metrics/moon_counterfactual_10s.csv"

PYTHONPATH="$PROJECT_ROOT" python -m music_infuser.cat_lora.summarize_counterfactual_metrics \
  --csv "$PROJECT_ROOT/outputs/metrics/moon_counterfactual_5s.csv" \
  --csv "$PROJECT_ROOT/outputs/metrics/moon_counterfactual_10s.csv" \
  --output-dir "$PROJECT_ROOT/outputs/metrics/summary"
