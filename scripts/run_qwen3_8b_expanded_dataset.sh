#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase4/qwen3_8b_expanded}"
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

extract_directory() {
  local phase="$1"
  local repeat="$2"
  local directory="$3"
  local result="$directory/result.jsonl"
  local output="$directory/comm_ground_truth.jsonl"
  local first=1
  local trace
  mapfile -t traces < <(
    find "$directory/traces" -maxdepth 1 \
      -name "*_${phase}.trace.json.gz" -type f | sort
  )
  if ((${#traces[@]} == 0)); then
    printf 'no %s traces found in %s\n' "$phase" "$directory" >&2
    return 1
  fi
  rm -f "$output"
  for trace in "${traces[@]}"; do
    local args=(
      python scripts/extract_inference_comm_trace.py
      --trace "$trace"
      --result "$result"
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
  done
}

validate_directory() {
  local phase="$1"
  local directory="$2"
  local expected="$3"
  python - "$phase" "$directory" "$expected" <<'PY'
import json
import sys
from pathlib import Path

phase, directory, expected = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
ground_truth = directory / "comm_ground_truth.jsonl"
result = directory / "result.jsonl"
gt_rows = [json.loads(line) for line in ground_truth.read_text().splitlines() if line.strip()]
result_rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
if len(gt_rows) != expected or len(result_rows) != expected:
    raise SystemExit(
        f"{directory}: expected {expected}, got gt={len(gt_rows)} result={len(result_rows)}"
    )
keys = set()
for row in gt_rows:
    if row["phase"] != phase:
        raise SystemExit(f"{directory}: unexpected phase {row['phase']}")
    if not row["alignment"]["exact_one_kernel_per_call"]:
        raise SystemExit(f"{directory}: kernel/call mismatch in {row['run_name']}")
    workload = row["workload"]
    key = (
        workload["batch_size"],
        workload["input_len"],
        workload["output_len"],
    )
    if key in keys:
        raise SystemExit(f"{directory}: duplicate workload {key}")
    keys.add(key)
    full_pattern = row["full_phase_pattern_demand"]
    estimate = row["gpu_ground_truth"]["full_phase_estimate"]
    if full_pattern["all_reduce_calls"] <= 0:
        raise SystemExit(f"{directory}: empty full-phase pattern in {row['run_name']}")
    if estimate["collective_kernel_time_us"] <= 0:
        raise SystemExit(f"{directory}: empty full-phase estimate in {row['run_name']}")
print(f"validated {directory}: {len(gt_rows)} {phase} workloads")
PY
}

run_phase_grid() {
  local tp="$1"
  local repeat="$2"
  local phase="$3"
  local directory="$OUTPUT_ROOT/tp${tp}/r${repeat}/${phase}"
  local done_marker="$directory/DONE"
  local result="$directory/result.jsonl"
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
    validate_directory "$phase" "$directory" "$expected"
    printf 'skip completed TP=%s repeat=%s phase=%s\n' "$tp" "$repeat" "$phase"
    return
  fi

  check_gpus_idle "$tp"
  mkdir -p "$directory/traces"
  rm -f "$result" "$directory/comm_ground_truth.jsonl" "$directory/run.log"
  find "$directory/traces" -maxdepth 1 -type f -name '*.trace.json.gz' -delete

  printf 'start TP=%s repeat=%s phase=%s expected=%s\n' \
    "$tp" "$repeat" "$phase" "$expected"
  CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$tp" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      "${grid_args[@]}" \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --profile \
      --profile-activities GPU \
      --profile-prefix "$directory/traces/trace" \
      --run-name "qwen3-8b-expanded-tp${tp}-${phase}-r${repeat}" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1

  extract_directory "$phase" "$repeat" "$directory" \
    >"$directory/extract.log" 2>&1
  validate_directory "$phase" "$directory" "$expected" \
    | tee "$directory/validate.log"
  printf 'complete\n' >"$done_marker"
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
