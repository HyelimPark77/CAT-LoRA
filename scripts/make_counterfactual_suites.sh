#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/342/hyellim/projects/CAT-LoRA}"
INPUT_AUDIO="${INPUT_AUDIO:-/home/ubuntu/342/hyellim/projects/listen-align-m2v/assets/audio/original/moon.mp3}"
PREFIX="${PREFIX:-moon}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/assets/counterfactual_suites}"
REQUIRE_MISMATCH="${REQUIRE_MISMATCH:-1}"

cd "$PROJECT_ROOT"

if [[ "$REQUIRE_MISMATCH" == "1" ]]; then
  if [[ -z "${MISMATCH_SAME_AUDIO:-}" || -z "${MISMATCH_DIFF_AUDIO:-}" ]]; then
    echo "MISMATCH_SAME_AUDIO and MISMATCH_DIFF_AUDIO are required for paper-grade suites." >&2
    echo "Set REQUIRE_MISMATCH=0 only for a quick diagnostic suite." >&2
    exit 2
  fi
fi

for dur in 5 10; do
  args=(
    --input-audio "$INPUT_AUDIO"
    --output-dir "$OUT_ROOT/${PREFIX}_${dur}s"
    --prefix "$PREFIX"
    --duration-sec "$dur"
  )
  if [[ -n "${MISMATCH_SAME_AUDIO:-}" ]]; then
    args+=(--mismatch-same-audio "$MISMATCH_SAME_AUDIO")
  fi
  if [[ -n "${MISMATCH_DIFF_AUDIO:-}" ]]; then
    args+=(--mismatch-diff-audio "$MISMATCH_DIFF_AUDIO")
  fi
  python -m music_infuser.cat_lora.make_counterfactual_audio_suite "${args[@]}"
done
