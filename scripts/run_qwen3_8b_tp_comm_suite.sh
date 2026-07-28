#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/Qwen3-8B}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"

cd "$REPO_ROOT"

visible_devices() {
  local tp="$1"
  local devices=""
  local index
  for ((index = 0; index < tp; index++)); do
    if [[ -n "$devices" ]]; then
      devices+=","
    fi
    devices+="$index"
  done
  printf '%s' "$devices"
}

check_gpus_idle() {
  local tp="$1"
  local index=0
  local failed=0
  local free_mib utilization
  while IFS=, read -r free_mib utilization; do
    free_mib="${free_mib//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if ((index < tp)); then
      if ((free_mib < MIN_FREE_MIB || utilization > MAX_IDLE_UTIL)); then
        printf 'GPU %d is busy: free=%s MiB util=%s%%\n' \
          "$index" "$free_mib" "$utilization" >&2
        failed=1
      fi
    fi
    ((index += 1))
  done < <(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if ((failed)); then
    printf 'Refusing to run a timing experiment on shared/busy GPUs.\n' >&2
    return 2
  fi
}

extract_trace() {
  local trace="$1"
  local result="$2"
  local output="$3"
  local repeat="$4"
  python scripts/extract_inference_comm_trace.py \
    --trace "$trace" \
    --result "$result" \
    --output "$output" \
    --repeat-id "$repeat" \
    --overwrite
}

run_decode_case() {
  local tp="$1"
  local label="$2"
  local repeat="$3"
  local max_output="$4"
  local profile_steps="$5"
  shift 5
  local output_lens=("$@")
  local directory="$REPO_ROOT/experiment-results/phase1/qwen3_8b_tp${tp}_inference_comm/representative/decode_equal_payload/$label/r$repeat"
  local result="$directory/result.jsonl"
  local ground_truth="$directory/comm_ground_truth.jsonl"
  local trace="$directory/trace_batch8_input512_output${max_output}_decode.trace.json.gz"

  if [[ -s "$ground_truth" && -s "$trace" && -s "$result" ]]; then
    printf 'skip completed TP=%s %s r%s\n' "$tp" "$label" "$repeat"
    return
  fi

  check_gpus_idle "$tp"
  mkdir -p "$directory"
  rm -f "$result" "$ground_truth" "$directory"/*.trace.json.gz
  CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$tp" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      --batch-size 8 \
      --input-len 512 \
      --output-len "$max_output" \
      --output-lens-per-request "${output_lens[@]}" \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --profile \
      --profile-activities GPU \
      --profile-stage decode \
      --profile-start-step 0 \
      --profile-steps "$profile_steps" \
      --profile-prefix "$directory/trace" \
      --run-name "qwen3-8b-tp${tp}-equal-payload-${label}-r${repeat}" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1
  extract_trace "$trace" "$result" "$ground_truth" "$repeat"
}

run_prefill_8mib_case() {
  local repeat="$1"
  local directory="$REPO_ROOT/experiment-results/phase1/qwen3_8b_tp2_inference_comm/representative/prefill_payload_curve/r$repeat"
  local result="$directory/result.jsonl"
  local ground_truth="$directory/comm_ground_truth.jsonl"
  local trace="$directory/trace_batch1_input1024_output8_prefill.trace.json.gz"

  if [[ -s "$ground_truth" && -s "$trace" && -s "$result" ]]; then
    printf 'skip completed TP=2 prefill-8mib r%s\n' "$repeat"
    return
  fi

  check_gpus_idle 2
  mkdir -p "$directory"
  rm -f "$result" "$ground_truth" "$directory"/*.trace.json.gz
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp 2 \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      --batch-size 1 \
      --input-len 1024 \
      --output-len 8 \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --profile \
      --profile-activities GPU \
      --profile-stage prefill \
      --profile-prefix "$directory/trace" \
      --run-name "qwen3-8b-tp2-prefill-8mib-r${repeat}" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1
  extract_trace "$trace" "$result" "$ground_truth" "$repeat"
}

run_tp2_stability() {
  local repeat
  for repeat in {3..9}; do
    run_prefill_8mib_case "$repeat"
    run_decode_case 2 mixed "$repeat" 36 35 4 4 8 8 16 16 36 36
    run_decode_case 2 longtail "$repeat" 52 51 4 4 4 4 4 4 52 52
  done
}

run_cross_tp() {
  local tp="$1"
  local repeat
  for repeat in {0..9}; do
    run_decode_case "$tp" uniform "$repeat" 16 15 16 16 16 16 16 16 16 16
    run_decode_case "$tp" mixed "$repeat" 36 35 4 4 8 8 16 16 36 36
    run_decode_case "$tp" longtail "$repeat" 52 51 4 4 4 4 4 4 52 52
  done
}

case "${1:-all}" in
  tp2-stability)
    run_tp2_stability
    ;;
  tp4)
    run_cross_tp 4
    ;;
  tp8)
    run_cross_tp 8
    ;;
  all)
    run_tp2_stability
    run_cross_tp 4
    run_cross_tp 8
    ;;
  *)
    printf 'usage: %s [tp2-stability|tp4|tp8|all]\n' "$0" >&2
    exit 64
    ;;
esac
