#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
PHASE_ROOT="${PHASE_ROOT:-$REPO_ROOT/experiment-results/phase14}"
GENERIC_RUNNER="$REPO_ROOT/scripts/run_cross_model_multiscale_timing_dataset.sh"
SMOKE_ROOT="$PHASE_ROOT/tp_group_size_timing_smoke"
FORMAL_ROOT="$PHASE_ROOT/tp_group_size_timing_ground_truth"
ANALYSIS_ROOT="$PHASE_ROOT/tp_group_size_timing_analysis"
read -r -a TP_LIST <<<"${PHASE14_TPS:-4 8}"
read -r -a MODEL_LIST <<<"${PHASE14_MODELS:-qwen qwen30}"

cd "$REPO_ROOT"

run_matrix() {
  local output_root="$1"
  local repeats="$2"
  local profiles="$3"
  local chunks="$4"
  local batches="$5"
  local inputs_1024="$6"
  local inputs_4096="$7"
  local tp model

  for tp in "${TP_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      TP="$tp" \
      OUTPUT_ROOT="$output_root/tp$tp" \
      REPEAT_IDS="$repeats" \
      MIXED_PROFILES="$profiles" \
      CHUNK_SIZES="$chunks" \
      CHUNK_BATCH_SIZES="$batches" \
      CHUNK_INPUT_LENS_1024="$inputs_1024" \
      CHUNK_INPUT_LENS_4096="$inputs_4096" \
        bash "$GENERIC_RUNNER" all "$model"
    done
  done
}

run_smoke() {
  run_matrix \
    "$SMOKE_ROOT" \
    "0" \
    "balanced" \
    "1024" \
    "1" \
    "1025" \
    ""
}

run_formal() {
  run_matrix \
    "$FORMAL_ROOT" \
    "0 1 2" \
    "balanced staircase bimodal" \
    "1024 4096" \
    "1 4" \
    "1023 1024 1025" \
    "4095 4096 4097"
}

run_analysis() {
  python scripts/analyze_tp_group_size_timing.py \
    --phase14-root "$FORMAL_ROOT" \
    --output-dir "$ANALYSIS_ROOT"
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
