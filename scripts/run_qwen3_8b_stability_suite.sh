#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase5/qwen3_8b_stability}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"
DECODE_PROFILE_START="${DECODE_PROFILE_START:-4}"
DECODE_PROFILE_STEPS="${DECODE_PROFILE_STEPS:-8}"

cd "$REPO_ROOT"

visible_devices() {
  local tp="$1"
  local devices=""
  local index
  for ((index = 0; index < tp; index++)); do
    [[ -n "$devices" ]] && devices+=","
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
    if ((index < tp && (free_mib < MIN_FREE_MIB || utilization > MAX_IDLE_UTIL))); then
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
  ((failed == 0))
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

extract_group() {
  local phase="$1"
  local repeat="$2"
  local directory="$3"
  local output="$directory/comm_ground_truth.jsonl"
  local first=1
  local trace
  rm -f "$output"
  while IFS= read -r trace; do
    local args=(
      python scripts/extract_inference_comm_trace.py
      --trace "$trace"
      --result "$directory/result.jsonl"
      --output "$output"
      --repeat-id "$repeat"
    )
    if [[ "$phase" == "decode" ]]; then
      args+=(
        --profile-start-step "$DECODE_PROFILE_START"
        --profile-end-step "$((DECODE_PROFILE_START + DECODE_PROFILE_STEPS - 1))"
      )
    fi
    if ((first)); then
      args+=(--overwrite)
      first=0
    fi
    "${args[@]}"
  done < <(
    find "$directory/traces" -maxdepth 1 \
      -name "*_${phase}.trace.json.gz" -type f | sort
  )
}

validate_group() {
  local phase="$1"
  local directory="$2"
  local expected="$3"
  python - "$phase" "$directory" "$expected" <<'PY'
import json
import sys
from pathlib import Path

phase, directory, expected = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
results = [
    json.loads(line)
    for line in (directory / "result.jsonl").read_text().splitlines()
    if line.strip()
]
ground_truth = [
    json.loads(line)
    for line in (directory / "comm_ground_truth.jsonl").read_text().splitlines()
    if line.strip()
]
assert len(results) == expected, (len(results), expected)
assert len(ground_truth) == expected, (len(ground_truth), expected)
assert all(row["phase"] == phase for row in ground_truth)
assert all(
    row["alignment"]["exact_one_kernel_per_call"] for row in ground_truth
)
keys = {
    (
        row["workload"]["batch_size"],
        row["workload"]["input_len"],
        row["workload"]["output_len"],
    )
    for row in ground_truth
}
assert len(keys) == expected, (len(keys), expected)
print(f"validated {directory}: {expected} {phase} workloads")
PY
}

run_group() {
  local tp="$1"
  local repeat="$2"
  local phase="$3"
  local label="$4"
  local expected="$5"
  shift 5
  local directory="$OUTPUT_ROOT/tp${tp}/r${repeat}/${phase}/${label}"

  if [[ -s "$directory/DONE" ]]; then
    validate_group "$phase" "$directory" "$expected"
    printf 'skip TP=%s r=%s %s/%s\n' "$tp" "$repeat" "$phase" "$label"
    return
  fi

  check_gpus_idle "$tp"
  mkdir -p "$directory/traces"
  rm -f \
    "$directory/result.jsonl" \
    "$directory/comm_ground_truth.jsonl" \
    "$directory/run.log" \
    "$directory/extract.log" \
    "$directory/validate.log" \
    "$directory/telemetry.csv"
  find "$directory/traces" -maxdepth 1 -type f \
    -name '*.trace.json.gz' -delete

  local -a profile_args=(--profile-stage "$phase")
  if [[ "$phase" == "decode" ]]; then
    profile_args+=(
      --profile-start-step "$DECODE_PROFILE_START"
      --profile-steps "$DECODE_PROFILE_STEPS"
    )
  fi

  printf 'start TP=%s r=%s %s/%s expected=%s\n' \
    "$tp" "$repeat" "$phase" "$label" "$expected"
  start_telemetry "$directory/telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$tp" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      "$@" \
      "${profile_args[@]}" \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --profile \
      --profile-activities GPU \
      --profile-prefix "$directory/traces/trace" \
      --run-name "qwen3-8b-stability-tp${tp}-${phase}-${label}-r${repeat}" \
      --result-filename "$directory/result.jsonl" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM

  extract_group "$phase" "$repeat" "$directory" \
    >"$directory/extract.log" 2>&1
  validate_group "$phase" "$directory" "$expected" \
    | tee "$directory/validate.log"
  printf 'complete\n' >"$directory/DONE"
}

run_repeat() {
  local repeat="$1"

  run_group 2 "$repeat" decode b1_l128_m32_128 2 \
    --batch-size 1 --input-len 128 --output-len 32 128
  run_group 2 "$repeat" decode b1_l2048_m128 1 \
    --batch-size 1 --input-len 2048 --output-len 128
  run_group 2 "$repeat" prefill b8_l128 1 \
    --batch-size 8 --input-len 128 --output-len 8

  run_group 4 "$repeat" decode b1_l128_8192_m32_128_512 6 \
    --batch-size 1 --input-len 128 8192 --output-len 32 128 512
  run_group 4 "$repeat" decode b4_l2048_m128 1 \
    --batch-size 4 --input-len 2048 --output-len 128

  run_group 8 "$repeat" decode b1_l128_8192_m32_512 4 \
    --batch-size 1 --input-len 128 8192 --output-len 32 512
  run_group 8 "$repeat" decode b2_l128_8192_m32_128 4 \
    --batch-size 2 --input-len 128 8192 --output-len 32 128
  run_group 8 "$repeat" decode b4_l2048_8192_m32_128_512 6 \
    --batch-size 4 --input-len 2048 8192 --output-len 32 128 512
  run_group 8 "$repeat" decode b8_l8192_m32_512 2 \
    --batch-size 8 --input-len 8192 --output-len 32 512
  run_group 8 "$repeat" decode b16_l128_m128 1 \
    --batch-size 16 --input-len 128 --output-len 128
  run_group 8 "$repeat" prefill b1_l2048 1 \
    --batch-size 1 --input-len 2048 --output-len 8
}

for repeat in 3 4 5 6 7 8 9; do
  run_repeat "$repeat"
done
