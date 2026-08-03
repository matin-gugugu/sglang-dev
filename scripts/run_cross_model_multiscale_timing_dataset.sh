#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase11/multiscale_timing_ground_truth}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"
TP=2
GPU_START="${GPU_START:-0}"
VISIBLE_DEVICES="$GPU_START,$((GPU_START + 1))"
KEEP_TRACES="${KEEP_TRACES:-0}"
read -r -a REPEAT_ID_LIST <<<"${REPEAT_IDS:-0 1 2}"
read -r -a CHUNK_SIZE_LIST <<<"${CHUNK_SIZES:-1024 2048 4096}"
read -r -a MIXED_PROFILE_LIST <<<"${MIXED_PROFILES:-balanced staircase bimodal}"

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
    printf 'Refusing to run timing labels on busy GPUs.\n' >&2
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
    qwen30)
      MODEL_NAME="qwen3-30b-a3b"
      MODEL_PATH="/media/ssd1/Qwen3-30B-A3B"
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
  if [[ "$MODEL_NAME" == "qwen3-30b-a3b" ]]; then
    [[ -s "$REPO_ROOT/experiment-results/phase12/qwen3_30b_a3b_admission/tp2/DONE" \
      && -s "$REPO_ROOT/experiment-results/phase12/qwen3_30b_a3b_admission/tp8/DONE" ]] || {
      printf 'Qwen3-30B-A3B TP=2/8 admission is incomplete\n' >&2
      return 2
    }
  fi
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

extract_all_rank() {
  local phase="$1"
  local repeat="$2"
  local directory="$3"
  local output="$directory/all_rank_ground_truth.jsonl"
  local first=1
  local rank_zero_trace
  local rank
  local trace
  local -a rank_zero_traces traces args

  rm -f "$output"
  mapfile -t rank_zero_traces < <(
    find "$directory/traces" -maxdepth 1 \
      -name "trace_rank0_*_${phase}.trace.json.gz" -type f | sort
  )
  if ((${#rank_zero_traces[@]} == 0)); then
    printf 'no rank-0 %s traces found in %s\n' "$phase" "$directory" >&2
    return 1
  fi

  for rank_zero_trace in "${rank_zero_traces[@]}"; do
    traces=()
    for ((rank = 0; rank < TP; rank++)); do
      traces+=("${rank_zero_trace/_rank0_/_rank${rank}_}")
    done
    args=(
      python scripts/extract_all_rank_comm_trace.py
      --result "$directory/result.jsonl"
      --output "$output"
      --repeat-id "$repeat"
    )
    for trace in "${traces[@]}"; do
      [[ -s "$trace" ]] || {
        printf 'missing rank trace: %s\n' "$trace" >&2
        return 1
      }
      args+=(--trace "$trace")
    done
    if ((first)); then
      args+=(--overwrite)
      first=0
    fi
    "${args[@]}"
  done
}

remove_raw_traces() {
  local directory="$1"
  if ((KEEP_TRACES)); then
    return
  fi
  find "$directory/traces" -maxdepth 1 -type f \
    -name '*.trace.json.gz' -delete
  printf 'Raw profiler traces were deleted after validated label extraction.\n' \
    >"$directory/TRACES_REMOVED"
}

run_profiled_case() {
  local mode="$1"
  local label="$2"
  local repeat="$3"
  local expected="$4"
  local directory="$OUTPUT_ROOT/$MODEL_NAME/$mode/$label/r$repeat"
  shift 4
  local -a workload_args=("$@")
  local phase validate_mode
  local -a validate_args profile_window_args

  if [[ "$mode" == "mixed_same_coarse" ]]; then
    phase="decode"
    validate_mode="mixed-decode"
    set_profile "$label"
    validate_args=(--output-lens "${OUTPUT_LENS[@]}")
    # output_len=64 executes 63 Decode forwards. Profile the complete draining
    # batch so the measured kernel sequence and full multi-support histogram
    # align exactly without window scaling.
    profile_window_args=(--profile-start-step 0 --profile-steps 63)
  else
    phase="prefill"
    validate_mode="chunked-prefill"
    validate_args=(--chunk-size "${label#c}")
    profile_window_args=()
  fi

  if [[ -s "$directory/DONE" ]]; then
    python scripts/validate_multiscale_timing_result.py \
      --result "$directory/result.jsonl" \
      --ground-truth "$directory/all_rank_ground_truth.jsonl" \
      --mode "$validate_mode" \
      --model "$MODEL_NAME" \
      --tp "$TP" \
      --expected-rows "$expected" \
      "${validate_args[@]}"
    printf 'skip completed %s %s/%s r%s\n' \
      "$MODEL_NAME" "$mode" "$label" "$repeat"
    return
  fi

  check_gpus_idle
  mkdir -p "$directory/traces"
  rm -f \
    "$directory/result.jsonl" \
    "$directory/all_rank_ground_truth.jsonl" \
    "$directory/run.log" \
    "$directory/extract.log" \
    "$directory/validate.log" \
    "$directory/telemetry.csv" \
    "$directory/TRACES_REMOVED" \
    "$directory/DONE"
  find "$directory/traces" -maxdepth 1 -type f \
    -name '*.trace.json.gz' -delete

  printf 'start %s %s/%s r%s expected=%s GPU=%s\n' \
    "$MODEL_NAME" "$mode" "$label" "$repeat" "$expected" "$VISIBLE_DEVICES"
  start_telemetry "$directory/telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  timeout 3600s env CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$TP" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      "${workload_args[@]}" \
      --profile-stage "$phase" \
      "${profile_window_args[@]}" \
      --warmup-each-workload \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --profile \
      --profile-all-ranks \
      --profile-activities GPU \
      --profile-prefix "$directory/traces/trace" \
      --run-name "$MODEL_NAME-tp2-timing-$mode-$label-r$repeat" \
      --result-filename "$directory/result.jsonl" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM

  extract_all_rank "$phase" "$repeat" "$directory" \
    >"$directory/extract.log" 2>&1
  python scripts/validate_multiscale_timing_result.py \
    --result "$directory/result.jsonl" \
    --ground-truth "$directory/all_rank_ground_truth.jsonl" \
    --mode "$validate_mode" \
    --model "$MODEL_NAME" \
    --tp "$TP" \
    --expected-rows "$expected" \
    "${validate_args[@]}" \
    | tee "$directory/validate.log"
  remove_raw_traces "$directory"
  printf 'complete\n' >"$directory/DONE"
}

run_mixed_suite() {
  local repeat profile
  for repeat in "${REPEAT_ID_LIST[@]}"; do
    for profile in "${MIXED_PROFILE_LIST[@]}"; do
      set_profile "$profile"
      run_profiled_case mixed_same_coarse "$profile" "$repeat" 1 \
        --batch-size 8 \
        --input-len 512 \
        --output-len 64 \
        --output-lens-per-request "${OUTPUT_LENS[@]}"
    done
  done
}

run_chunked_suite() {
  local repeat chunk_size
  for repeat in "${REPEAT_ID_LIST[@]}"; do
    for chunk_size in "${CHUNK_SIZE_LIST[@]}"; do
      set_chunk_inputs "$chunk_size"
      run_profiled_case chunked_prefill "c$chunk_size" "$repeat" 12 \
        --batch-size 1 4 \
        --input-len "${INPUT_LENS[@]}" \
        --output-len 8 \
        --prefill-chunk-size "$chunk_size"
    done
  done
}

MODE="${1:-all}"
MODEL_SELECTOR="${2:-qwen}"
set_model "$MODEL_SELECTOR"

case "$MODE" in
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
    printf 'usage: %s [all|mixed|chunked] [qwen|qwen30|deepseek]\n' "$0" >&2
    exit 64
    ;;
esac
