#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase10/multiscale_pattern_demand}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"
TP=2
GPU_START="${GPU_START:-0}"
VISIBLE_DEVICES="$GPU_START,$((GPU_START + 1))"
read -r -a REPEAT_ID_LIST <<<"${REPEAT_IDS:-0 1 2}"
read -r -a CHUNK_SIZE_LIST <<<"${CHUNK_SIZES:-1024 2048 4096}"

cd "$REPO_ROOT"

check_gpus_idle() {
  local index=0
  local failed=0
  local free_mib utilization
  while IFS=, read -r free_mib utilization; do
    free_mib="${free_mib//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if ((
      index >= GPU_START
      && index < GPU_START + TP
      && (free_mib < MIN_FREE_MIB || utilization > MAX_IDLE_UTIL)
    )); then
      printf 'GPU %d busy: free=%s MiB util=%s%%\n' \
        "$index" "$free_mib" "$utilization" >&2
      failed=1
    fi
    ((index += 1))
  done < <(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if ((failed)); then
    printf 'Refusing to run on busy GPUs.\n' >&2
    return 2
  fi
}

start_telemetry() {
  local output="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used \
    --format=csv \
    --loop=1 \
    >"$output" 2>&1 &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID:-}" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill "$TELEMETRY_PID"
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

set_model() {
  case "$1" in
    qwen)
      MODEL_NAME="qwen3-8b"
      MODEL_PATH="/media/ssd1/Qwen3-8B"
      ;;
    deepseek)
      MODEL_NAME="deepseek-v2-lite"
      MODEL_PATH="/media/ssd1/DeepSeek-V2-Lite"
      ;;
    *)
      printf 'unknown model selector: %s\n' "$1" >&2
      return 64
      ;;
  esac
  [[ -s "$MODEL_PATH/config.json" ]] || {
    printf 'model missing: %s\n' "$MODEL_PATH" >&2
    return 2
  }
}

set_profile() {
  case "$1" in
    balanced)
      OUTPUT_LENS=(16 16 16 16 32 32 64 64)
      ;;
    staircase)
      OUTPUT_LENS=(16 16 16 24 32 40 48 64)
      ;;
    bimodal)
      OUTPUT_LENS=(8 8 16 16 16 64 64 64)
      ;;
    *)
      printf 'unknown output profile: %s\n' "$1" >&2
      return 64
      ;;
  esac
}

run_mixed_case() {
  local profile="$1"
  local repeat="$2"
  local directory="$OUTPUT_ROOT/$MODEL_NAME/mixed_same_coarse/$profile/r$repeat"
  local result="$directory/result.jsonl"
  set_profile "$profile"

  if [[ -s "$directory/DONE" ]]; then
    python scripts/validate_multiscale_pattern_result.py \
      --result "$result" \
      --mode mixed-decode \
      --model "$MODEL_NAME" \
      --tp "$TP" \
      --expected-rows 1 \
      --output-lens "${OUTPUT_LENS[@]}"
    printf 'skip completed %s mixed %s r%s\n' "$MODEL_NAME" "$profile" "$repeat"
    return
  fi

  check_gpus_idle
  mkdir -p "$directory"
  rm -f \
    "$result" \
    "$directory/run.log" \
    "$directory/telemetry.csv" \
    "$directory/validate.log" \
    "$directory/DONE"
  printf 'start %s mixed profile=%s repeat=%s\n' \
    "$MODEL_NAME" "$profile" "$repeat"
  start_telemetry "$directory/telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$TP" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      --batch-size 8 \
      --input-len 512 \
      --output-len 64 \
      --output-lens-per-request "${OUTPUT_LENS[@]}" \
      --profile-stage decode \
      --warmup-each-workload \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --run-name "$MODEL_NAME-tp2-same-coarse-$profile-r$repeat" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM
  python scripts/validate_multiscale_pattern_result.py \
    --result "$result" \
    --mode mixed-decode \
    --model "$MODEL_NAME" \
    --tp "$TP" \
    --expected-rows 1 \
    --output-lens "${OUTPUT_LENS[@]}" \
    | tee "$directory/validate.log"
  printf 'complete\n' >"$directory/DONE"
}

set_chunk_inputs() {
  case "$1" in
    1024)
      INPUT_LENS=(1023 1024 1025 2047 2048 2049)
      ;;
    2048)
      INPUT_LENS=(2047 2048 2049 4095 4096 4097)
      ;;
    4096)
      INPUT_LENS=(4095 4096 4097 8191 8192 8193)
      ;;
    *)
      printf 'unsupported chunk size: %s\n' "$1" >&2
      return 64
      ;;
  esac
}

run_chunked_case() {
  local chunk_size="$1"
  local repeat="$2"
  local directory="$OUTPUT_ROOT/$MODEL_NAME/chunked_prefill/c$chunk_size/r$repeat"
  local result="$directory/result.jsonl"
  set_chunk_inputs "$chunk_size"

  if [[ -s "$directory/DONE" ]]; then
    python scripts/validate_multiscale_pattern_result.py \
      --result "$result" \
      --mode chunked-prefill \
      --model "$MODEL_NAME" \
      --tp "$TP" \
      --expected-rows 12 \
      --chunk-size "$chunk_size"
    printf 'skip completed %s chunk=%s r%s\n' \
      "$MODEL_NAME" "$chunk_size" "$repeat"
    return
  fi

  check_gpus_idle
  mkdir -p "$directory"
  rm -f \
    "$result" \
    "$directory/run.log" \
    "$directory/telemetry.csv" \
    "$directory/validate.log" \
    "$directory/DONE"
  printf 'start %s chunk=%s repeat=%s\n' \
    "$MODEL_NAME" "$chunk_size" "$repeat"
  start_telemetry "$directory/telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$TP" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      --batch-size 1 4 \
      --input-len "${INPUT_LENS[@]}" \
      --output-len 8 \
      --prefill-chunk-size "$chunk_size" \
      --profile-stage prefill \
      --warmup-each-workload \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --run-name "$MODEL_NAME-tp2-chunk$chunk_size-r$repeat" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM
  python scripts/validate_multiscale_pattern_result.py \
    --result "$result" \
    --mode chunked-prefill \
    --model "$MODEL_NAME" \
    --tp "$TP" \
    --expected-rows 12 \
    --chunk-size "$chunk_size" \
    | tee "$directory/validate.log"
  printf 'complete\n' >"$directory/DONE"
}

run_mixed_suite() {
  local repeat profile
  for repeat in "${REPEAT_ID_LIST[@]}"; do
    for profile in balanced staircase bimodal; do
      run_mixed_case "$profile" "$repeat"
    done
  done
}

run_chunked_suite() {
  local repeat chunk_size
  for repeat in "${REPEAT_ID_LIST[@]}"; do
    for chunk_size in "${CHUNK_SIZE_LIST[@]}"; do
      run_chunked_case "$chunk_size" "$repeat"
    done
  done
}

MODE_SELECTOR="${2:-all}"
case "$MODE_SELECTOR" in
  all)
    MODELS=(qwen deepseek)
    ;;
  qwen|deepseek)
    MODELS=("$MODE_SELECTOR")
    ;;
  *)
    printf 'usage: %s [all|mixed|chunked] [all|qwen|deepseek]\n' "$0" >&2
    exit 64
    ;;
esac

for model_selector in "${MODELS[@]}"; do
  set_model "$model_selector"
  case "${1:-all}" in
    all)
      run_mixed_suite
      run_chunked_suite
      ;;
    mixed)
      run_mixed_suite
      ;;
    chunked)
      run_chunked_suite
      ;;
    *)
      printf 'usage: %s [all|mixed|chunked] [all|qwen|deepseek]\n' "$0" >&2
      exit 64
      ;;
  esac
done
