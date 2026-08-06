#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

RESULT_ROOT=${RESULT_ROOT:-experiment-results/phase15_l1_curve_extension}
WARMUP=${WARMUP:-30}
ITERATIONS=${ITERATIONS:-100}
REPEATS=${REPEATS:-5}
PAYLOADS=(
  167772160
  201326592
  268435456
  335544320
  402653184
  469762048
  536870912
)

visible_devices() {
  case "$1" in
    2) printf '0,1\n' ;;
    4) printf '0,1,2,3\n' ;;
    8) printf '0,1,2,3,4,5,6,7\n' ;;
    *) return 64 ;;
  esac
}

check_gpus_idle() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
  if ((busy > 0)); then
    printf 'refusing to start: found %s active GPU processes\n' "$busy" >&2
    return 1
  fi
}

validate_result() {
  local path="$1"
  local tp="$2"
  local repeat="$3"
  python3 - "$path" "$tp" "$repeat" "$ITERATIONS" "${PAYLOADS[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tp, repeat, iterations = map(int, sys.argv[2:5])
payloads = {int(value) for value in sys.argv[5:]}
if not path.is_file():
    raise SystemExit(1)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(rows) != len(payloads) or {int(row["payload_bytes"]) for row in rows} != payloads:
    raise SystemExit(1)
if any(
    row["schema_version"] != "phase14f-backend-cost-v2"
    or row["op"] != "all_reduce"
    or int(row["group_size"]) != tp
    or int(row["repeat_id"]) != repeat
    or len(row["post_rendezvous_samples_us"]) != iterations
    for row in rows
):
    raise SystemExit(1)
PY
}

run_case() {
  local tp="$1"
  local repeat="$2"
  local directory="$RESULT_ROOT/curve/tp${tp}/all_reduce/r${repeat}"
  local output="$directory/curve.jsonl"
  local temporary="$directory/curve.tmp.jsonl"
  local attempt status
  mkdir -p "$directory"
  if validate_result "$output" "$tp" "$repeat"; then
    printf 'skip TP=%s repeat=%s\n' "$tp" "$repeat"
    return
  fi
  for attempt in 1 2 3; do
    rm -f "$temporary"
    printf 'start TP=%s repeat=%s attempt=%s\n' "$tp" "$repeat" "$attempt"
    set +e
    timeout 1800 env \
      PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES=$(visible_devices "$tp") \
      torchrun --standalone --nproc-per-node="$tp" \
        scripts/benchmark_phase14f_backend_curve.py \
        --output "$temporary" \
        --repeat-id "$repeat" \
        --op all_reduce \
        --payload-bytes "${PAYLOADS[@]}" \
        --warmup "$WARMUP" \
        --iterations "$ITERATIONS" \
        >"$directory/run_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    if ((status == 0)) && validate_result "$temporary" "$tp" "$repeat"; then
      mv "$temporary" "$output"
      printf 'complete TP=%s repeat=%s\n' "$tp" "$repeat"
      return
    fi
    printf 'retry TP=%s repeat=%s status=%s\n' "$tp" "$repeat" "$status" >&2
  done
  return 1
}

main() {
  local tp repeat
  check_gpus_idle
  mkdir -p "$RESULT_ROOT"
  nvidia-smi topo -m >"$RESULT_ROOT/nvidia_topology.txt"
  git rev-parse HEAD >"$RESULT_ROOT/git_commit.txt"
  for tp in 2 4 8; do
    for repeat in $(seq 0 $((REPEATS - 1))); do
      run_case "$tp" "$repeat"
    done
  done
  printf 'complete\n' >"$RESULT_ROOT/DONE"
}

main "$@"
