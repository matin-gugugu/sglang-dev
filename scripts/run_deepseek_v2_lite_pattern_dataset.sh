#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/DeepSeek-V2-Lite}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase8/deepseek_v2_lite_pattern_demand}"
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

check_model() {
  [[ -s "$MODEL_PATH/config.json" ]] || {
    printf 'model is incomplete or missing: %s\n' "$MODEL_PATH" >&2
    return 2
  }
  compgen -G "$MODEL_PATH/model-*.safetensors" >/dev/null || {
    printf 'model shards are missing: %s\n' "$MODEL_PATH" >&2
    return 2
  }
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
    printf 'Refusing to run a PatternDemand experiment on busy GPUs.\n' >&2
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
rows = [
    json.loads(line)
    for line in (directory / "result.jsonl").read_text().splitlines()
    if line.strip()
]
assert len(rows) == expected, (directory, len(rows), expected)
assert all(row["same_shape_workload_warmup"] for row in rows)
assert all(
    row["generated_output_tokens"] == row["output_len"]
    for row in rows
)
assert all(
    len(row["generated_output_tokens_per_request"]) == row["batch_size"]
    and set(row["generated_output_tokens_per_request"]) == {row["output_len"]}
    for row in rows
)

keys = {
    (row["batch_size"], row["input_len"], row["output_len"])
    for row in rows
}
assert len(keys) == expected, (directory, len(keys), expected)

payload_sizes = set()
ops = set()
for row in rows:
    profiles = sorted(row["comm_profile"], key=lambda item: item["tp_rank"])
    assert [profile["tp_rank"] for profile in profiles] == list(range(tp))
    representative = profiles[0]
    assert representative["capture_mode"] == "histogram-only"
    assert representative["raw_events_saved"] is False
    assert representative["events"] == []
    assert representative["events_truncated"] is False
    for profile in profiles[1:]:
        assert profile["stats"] == representative["stats"]
        assert profile["event_histograms"] == representative["event_histograms"]

    phase_histograms = [
        item
        for item in representative["event_histograms"]
        if item["phase"] == phase
    ]
    assert phase_histograms, (row["run_name"], phase)
    assert all(item["count"] > 0 for item in phase_histograms)
    assert all(item["input_payload_bytes"] > 0 for item in phase_histograms)
    assert all(item["group_size"] == tp for item in phase_histograms)
    payload_sizes.update(item["input_payload_bytes"] for item in phase_histograms)
    ops.update(item["op"] for item in phase_histograms)

print(
    f"validated {directory}: rows={len(rows)} phase={phase} "
    f"tp={tp} payload_sizes={len(payload_sizes)} ops={sorted(ops)}"
)
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

  check_model
  check_gpus_idle "$tp"
  mkdir -p "$directory"
  rm -f \
    "$directory/result.jsonl" \
    "$directory/run.log" \
    "$directory/telemetry.csv" \
    "$directory/validate.log"

  printf 'start DeepSeek-V2-Lite PatternDemand TP=%s repeat=%s phase=%s expected=%s\n' \
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
      --run-name "deepseek-v2-lite-pattern-tp${tp}-${phase}-r${repeat}" \
      --result-filename "$directory/result.jsonl" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM

  validate_directory "$tp" "$phase" "$directory" "$expected" \
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
