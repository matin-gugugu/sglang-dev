#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
PHASE_ROOT="${PHASE_ROOT:-$REPO_ROOT/experiment-results/phase14c}"
GENERIC_RUNNER="$REPO_ROOT/scripts/run_cross_model_multiscale_timing_dataset.sh"
SMOKE_ROOT="$PHASE_ROOT/smoke"
DEEPSEEK_ROOT="$PHASE_ROOT/deepseek_tp_extension"
DECODE_ROOT="$PHASE_ROOT/decode_extension"
ANALYSIS_ROOT="$PHASE_ROOT/extended_dataset_analysis"
MODEL_ROOT="$PHASE_ROOT/tp_phase_no_backend_analysis"
DECODE_PROFILES="uniform_b4 uniform_b16 long_tail"

cd "$REPO_ROOT"

run_smoke() {
  TP=8 \
  OUTPUT_ROOT="$SMOKE_ROOT/tp8" \
  REPEAT_IDS="0" \
  MIXED_PROFILES="uniform_b16 long_tail" \
    bash "$GENERIC_RUNNER" mixed deepseek

  TP=8 \
  OUTPUT_ROOT="$SMOKE_ROOT/tp8" \
  REPEAT_IDS="0" \
  CHUNK_SIZES="1024" \
  CHUNK_BATCH_SIZES="1" \
  CHUNK_INPUT_LENS_1024="1025" \
    bash "$GENERIC_RUNNER" chunked deepseek
}

run_deepseek_extension() {
  local tp
  for tp in 4 8; do
    TP="$tp" \
    OUTPUT_ROOT="$DEEPSEEK_ROOT/tp$tp" \
    REPEAT_IDS="0 1 2" \
    MIXED_PROFILES="balanced staircase bimodal" \
    CHUNK_SIZES="1024 4096" \
    CHUNK_BATCH_SIZES="1 4" \
    CHUNK_INPUT_LENS_1024="1023 1024 1025" \
    CHUNK_INPUT_LENS_4096="4095 4096 4097" \
      bash "$GENERIC_RUNNER" all deepseek
  done
}

run_decode_extension() {
  local tp model
  for tp in 2 4 8; do
    for model in qwen qwen30 deepseek; do
      TP="$tp" \
      OUTPUT_ROOT="$DECODE_ROOT/tp$tp" \
      REPEAT_IDS="0 1 2" \
      MIXED_PROFILES="$DECODE_PROFILES" \
        bash "$GENERIC_RUNNER" mixed "$model"
    done
  done
}

run_formal() {
  run_deepseek_extension
  run_decode_extension
}

run_analysis() {
  python scripts/build_phase14c_extended_dataset.py \
    --output-dir "$ANALYSIS_ROOT"
  python scripts/analyze_tp_phase_no_backend.py \
    --input-csv "$ANALYSIS_ROOT/aggregated_configurations.csv" \
    --output-dir "$MODEL_ROOT"
}

case "${1:-all}" in
  smoke)
    run_smoke
    ;;
  formal)
    run_formal
    ;;
  analyze)
    run_analysis
    ;;
  formal-analyze)
    run_formal
    run_analysis
    ;;
  all)
    run_smoke
    run_formal
    run_analysis
    ;;
  *)
    printf 'usage: %s [smoke|formal|analyze|formal-analyze|all]\n' "$0" >&2
    exit 64
    ;;
esac
