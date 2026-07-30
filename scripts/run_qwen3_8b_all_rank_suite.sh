#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase5/qwen3_8b_all_rank}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"
DECODE_PROFILE_START="${DECODE_PROFILE_START:-4}"
DECODE_PROFILE_STEPS="${DECODE_PROFILE_STEPS:-8}"
REEXTRACT="${REEXTRACT:-0}"

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

extract_group() {
  local tp="$1"
  local repeat="$2"
  local phase="$3"
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
      args+=(
        --trace "${rank_zero_trace/_rank0_/_rank${rank}_}"
      )
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

validate_group() {
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
rows = [
    json.loads(line)
    for line in (directory / "all_rank_ground_truth.jsonl").read_text().splitlines()
    if line.strip()
]
assert len(rows) == expected, (len(rows), expected)
assert all(row["schema_version"] == "all-rank-comm-labels-v2" for row in rows)
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
assert all(
    row["all_rank_ground_truth"]["full_phase_estimate"][
        "skew_free_intrinsic_kernel_time_us"
    ]
    <= row["all_rank_ground_truth"]["full_phase_estimate"][
        "synchronization_inclusive_max_duration_sum_us"
    ]
    for row in rows
)
print(f"validated {directory}: {expected} {phase} all-rank workloads")
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
    if ((REEXTRACT)); then
      extract_group "$tp" "$repeat" "$phase" "$directory"
    fi
    validate_group "$tp" "$phase" "$directory" "$expected"
    printf 'skip TP=%s r=%s %s/%s\n' "$tp" "$repeat" "$phase" "$label"
    return
  fi

  check_gpus_idle "$tp"
  mkdir -p "$directory/traces"
  rm -f \
    "$directory/result.jsonl" \
    "$directory/all_rank_ground_truth.jsonl" \
    "$directory/run.log" \
    "$directory/extract.log" \
    "$directory/validate.log"
  find "$directory/traces" -maxdepth 1 -type f \
    -name '*.trace.json.gz' -delete

  local -a profile_args=(--profile-stage "$phase")
  if [[ "$phase" == "decode" ]]; then
    profile_args+=(
      --profile-start-step "$DECODE_PROFILE_START"
      --profile-steps "$DECODE_PROFILE_STEPS"
    )
  fi

  printf 'start all-rank TP=%s r=%s %s/%s expected=%s\n' \
    "$tp" "$repeat" "$phase" "$label" "$expected"
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
      --profile-all-ranks \
      --profile-activities GPU \
      --profile-prefix "$directory/traces/trace" \
      --run-name "qwen3-8b-all-rank-tp${tp}-${phase}-${label}-r${repeat}" \
      --result-filename "$directory/result.jsonl" \
      >"$directory/run.log" 2>&1

  extract_group "$tp" "$repeat" "$phase" "$directory" \
    >"$directory/extract.log" 2>&1
  validate_group "$tp" "$phase" "$directory" "$expected" \
    | tee "$directory/validate.log"
  printf 'complete\n' >"$directory/DONE"
}

run_tp_repeat() {
  local tp="$1"
  local repeat="$2"

  # Common mechanism grid: Prefill scale plus the equal-total-payload Decode
  # contrast B1xM512, B4xM128, and B16xM32.
  run_group "$tp" "$repeat" prefill b1_l128_2048_8192 3 \
    --batch-size 1 --input-len 128 2048 8192 --output-len 8
  run_group "$tp" "$repeat" decode b1_l2048_m512 1 \
    --batch-size 1 --input-len 2048 --output-len 512
  run_group "$tp" "$repeat" decode b4_l2048_m128 1 \
    --batch-size 4 --input-len 2048 --output-len 128
  run_group "$tp" "$repeat" decode b16_l2048_m32 1 \
    --batch-size 16 --input-len 2048 --output-len 32

  # Representative workloads whose original three-repeat rank-0 target had
  # IQR / median above 20%. These determine whether the all-rank critical
  # label removes role-dependent representative-rank variance.
  case "$tp" in
    2)
      run_group "$tp" "$repeat" prefill highvar_b8_l128 1 \
        --batch-size 8 --input-len 128 --output-len 8
      run_group "$tp" "$repeat" decode highvar_b1_l128_m32_128 2 \
        --batch-size 1 --input-len 128 --output-len 32 128
      ;;
    4)
      run_group "$tp" "$repeat" decode highvar_b1_l128_m32_128_512 3 \
        --batch-size 1 --input-len 128 --output-len 32 128 512
      ;;
    8)
      run_group "$tp" "$repeat" decode highvar_b1_l128_m32_512 2 \
        --batch-size 1 --input-len 128 --output-len 32 512
      run_group "$tp" "$repeat" decode highvar_b2_l8192_m32 1 \
        --batch-size 2 --input-len 8192 --output-len 32
      run_group "$tp" "$repeat" decode highvar_b8_l8192_m32 1 \
        --batch-size 8 --input-len 8192 --output-len 32
      run_group "$tp" "$repeat" decode highvar_b16_l128_m128 1 \
        --batch-size 16 --input-len 128 --output-len 128
      ;;
  esac
}

for tp in 2 4 8; do
  for repeat in 0 1 2; do
    run_tp_repeat "$tp" "$repeat"
  done
done
