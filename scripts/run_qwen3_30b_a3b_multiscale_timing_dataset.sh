#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
PHASE_ROOT="${PHASE_ROOT:-$REPO_ROOT/experiment-results/phase13}"
GENERIC_RUNNER="$REPO_ROOT/scripts/run_cross_model_multiscale_timing_dataset.sh"
PHASE11_INPUT="$REPO_ROOT/experiment-results/phase11/multiscale_timing_ground_truth"
SMOKE_ROOT="$PHASE_ROOT/qwen3_30b_a3b_multiscale_timing_smoke"
FORMAL_ROOT="$PHASE_ROOT/multiscale_timing_ground_truth"
ANALYSIS_ROOT="$PHASE_ROOT/three_model_multiscale_timing_analysis"

cd "$REPO_ROOT"

run_smoke() {
  OUTPUT_ROOT="$SMOKE_ROOT" \
  REPEAT_IDS=0 \
  MIXED_PROFILES=balanced \
  CHUNK_SIZES=1024 \
    bash "$GENERIC_RUNNER" all qwen30
}

run_formal() {
  OUTPUT_ROOT="$FORMAL_ROOT" \
    bash "$GENERIC_RUNNER" all qwen30
}

run_analysis() {
  python scripts/analyze_multiscale_timing.py \
    --input-dir "$PHASE11_INPUT" \
    --input-dir "$FORMAL_ROOT" \
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
  all)
    run_smoke
    run_formal
    run_analysis
    ;;
  *)
    printf 'usage: %s [smoke|formal|analyze|all]\n' "$0" >&2
    exit 64
    ;;
esac
