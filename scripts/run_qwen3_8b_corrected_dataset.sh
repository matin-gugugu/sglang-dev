#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase6/qwen3_8b_corrected_all_rank}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"
DECODE_PROFILE_START="${DECODE_PROFILE_START:-4}"
DECODE_PROFILE_STEPS="${DECODE_PROFILE_STEPS:-8}"
KEEP_TRACES="${KEEP_TRACES:-1}"

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
  if ((failed)); then
    printf 'Refusing to run a timing experiment on shared/busy GPUs.\n' >&2
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

extract_directory() {
  local tp="$1"
  local phase="$2"
  local repeat="$3"
  local directory="$4"
  local output="$directory/all_rank_ground_truth.jsonl"
  local first=1
  local rank_zero_trace
  rm -f "$output"
  while IFS= read -r rank_zero_trace; do
    local -a args=(
      python scripts/extract_all_rank_comm_trace.py
      --result "$directory/result.jsonl"
      --output "$output"
      --repeat-id "$repeat"
    )
    local rank
    for ((rank = 0; rank < tp; rank++)); do
      args+=(--trace "${rank_zero_trace/_rank0_/_rank${rank}_}")
    done
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
      -name "trace_rank0_*_${phase}.trace.json.gz" -type f | sort
  )
}

validate_directory() {
  local tp="$1"
  local phase="$2"
  local directory="$3"
  local expected="$4"
  python - "$tp" "$phase" "$directory" "$expected" <<'PY'
import json
import sys
from pathlib import Path

tp, phase, directory, expected = (
    int(sys.argv[1]),
    sys.argv[2],
    Path(sys.argv[3]),
    int(sys.argv[4]),
)
results = [
    json.loads(line)
    for line in (directory / "result.jsonl").read_text().splitlines()
    if line.strip()
]
rows = [
    json.loads(line)
    for line in (directory / "all_rank_ground_truth.jsonl").read_text().splitlines()
    if line.strip()
]
assert len(results) == expected, (len(results), expected)
assert len(rows) == expected, (len(rows), expected)
assert all(row["same_shape_workload_warmup"] for row in results)
assert all(row["phase"] == phase for row in rows)
assert all(row["all_rank_ground_truth"]["rank_count"] == tp for row in rows)
assert all(row["alignment"]["exact_count_on_every_rank"] for row in rows)
assert all(row["alignment"]["identical_backend_sequence"] for row in rows)
assert all(
    row["alignment"]["identical_profiled_pattern_demand_on_every_rank"]
    for row in rows
)
assert all(
    row["alignment"]["identical_full_phase_pattern_demand_on_every_rank"]
    for row in rows
)
keys = {
    (
        row["workload"]["batch_size"],
        row["workload"]["input_len"],
        row["workload"]["output_len"],
    )
    for row in rows
}
assert len(keys) == expected, (len(keys), expected)
print(f"validated {directory}: {expected} corrected {phase} workloads")
PY
}

run_phase_grid() {
  local tp="$1"
  local repeat="$2"
  local phase="$3"
  local directory="$OUTPUT_ROOT/tp${tp}/r${repeat}/${phase}"
  local done_marker="$directory/DONE"
  local expected
  local -a grid_args

  if [[ "$phase" == "prefill" ]]; then
    expected=20
    grid_args=(
      --batch-size 1 2 4 8 16
      --input-len 128 512 2048 8192
      --output-len 8
      --profile-stage prefill
    )
  else
    expected=45
    grid_args=(
      --batch-size 1 2 4 8 16
      --input-len 128 2048 8192
      --output-len 32 128 512
      --profile-stage decode
      --profile-start-step "$DECODE_PROFILE_START"
      --profile-steps "$DECODE_PROFILE_STEPS"
    )
  fi

  if [[ -s "$done_marker" ]]; then
    validate_directory "$tp" "$phase" "$directory" "$expected"
    printf 'skip completed TP=%s repeat=%s phase=%s\n' "$tp" "$repeat" "$phase"
    return
  fi

  check_gpus_idle "$tp"
  mkdir -p "$directory/traces"
  rm -f \
    "$directory/result.jsonl" \
    "$directory/all_rank_ground_truth.jsonl" \
    "$directory/run.log" \
    "$directory/extract.log" \
    "$directory/validate.log" \
    "$directory/telemetry.csv"
  find "$directory/traces" -maxdepth 1 -type f \
    -name '*.trace.json.gz' -delete

  printf 'start corrected all-rank TP=%s repeat=%s phase=%s expected=%s\n' \
    "$tp" "$repeat" "$phase" "$expected"
  start_telemetry "$directory/telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$tp" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      "${grid_args[@]}" \
      --warmup-each-workload \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --profile \
      --profile-all-ranks \
      --profile-activities GPU \
      --profile-prefix "$directory/traces/trace" \
      --run-name "qwen3-8b-corrected-all-rank-tp${tp}-${phase}-r${repeat}" \
      --result-filename "$directory/result.jsonl" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM

  extract_directory "$tp" "$phase" "$repeat" "$directory" \
    >"$directory/extract.log" 2>&1
  validate_directory "$tp" "$phase" "$directory" "$expected" \
    | tee "$directory/validate.log"
  printf 'complete\n' >"$done_marker"

  if ((KEEP_TRACES == 0)); then
    find "$directory/traces" -maxdepth 1 -type f \
      -name '*.trace.json.gz' -delete
  fi
}

run_tp() {
  local tp="$1"
  local repeat
  for repeat in 0 1 2; do
    run_phase_grid "$tp" "$repeat" prefill
    run_phase_grid "$tp" "$repeat" decode
  done
}

case "${1:-all}" in
  tp2)
    run_tp 2
    ;;
  tp4)
    run_tp 4
    ;;
  tp8)
    run_tp 8
    ;;
  all)
    run_tp 2
    run_tp 4
    run_tp 8
    ;;
  *)
    printf 'usage: %s [tp2|tp4|tp8|all]\n' "$0" >&2
    exit 64
    ;;
esac
