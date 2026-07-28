#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="$REPO_ROOT/experiment-results/phase2/b200_l1_collective_curve"
MIN_BYTES=1024
MAX_BYTES=134217728
WARMUP=30
ITERATIONS=100
REPEATS=5
EXPECTED_POINTS=19

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/suite.log") 2>&1

validate_result() {
  local path="$1"
  local op="$2"
  local tp="$3"
  local repeat="$4"
  python - "$path" "$op" "$tp" "$repeat" "$EXPECTED_POINTS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
op = sys.argv[2]
tp = int(sys.argv[3])
repeat = int(sys.argv[4])
expected_points = int(sys.argv[5])
if not path.exists():
    raise SystemExit(1)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
expected_sizes = {1 << exponent for exponent in range(10, 28)}
expected_sizes.add(48 * 1024)
if len(rows) != expected_points:
    raise SystemExit(1)
if {row["payload_bytes"] for row in rows} != expected_sizes:
    raise SystemExit(1)
if any(
    row["op"] != op
    or row["group_size"] != tp
    or row["repeat_id"] != repeat
    or len(row["samples_us"]) != 100
    for row in rows
):
    raise SystemExit(1)
PY
}

check_all_gpus_idle() {
  local count=0
  while IFS=, read -r index free util; do
    index="${index//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    count=$((count + 1))
    if (( free < 160000 || util > 20 )); then
      printf 'GPU %s is busy: free=%s MiB util=%s%%\n' "$index" "$free" "$util" >&2
      return 1
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if (( count < 8 )); then
    printf 'Expected 8 visible GPUs, found %s\n' "$count" >&2
    return 1
  fi
}

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

capture_environment() {
  if [[ ! -f "$RESULT_ROOT/nvidia_topology.txt" ]]; then
    nvidia-smi topo -m > "$RESULT_ROOT/nvidia_topology.txt"
  fi
  if [[ ! -f "$RESULT_ROOT/environment.json" ]]; then
    python - "$RESULT_ROOT/environment.json" <<'PY'
import json
import socket
import subprocess
import sys

import torch

record = {
    "hostname": socket.gethostname(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "nccl_version": list(torch.cuda.nccl.version()),
    "device_count": torch.cuda.device_count(),
    "device_names": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ],
}
with open(sys.argv[1], "w") as output:
    json.dump(record, output, indent=2)
    output.write("\n")
PY
  fi
}

run_case() {
  local tp="$1"
  local op="$2"
  local repeat="$3"
  local directory="$RESULT_ROOT/tp${tp}/${op}/r${repeat}"
  local output="$directory/curve.jsonl"
  local temporary="$directory/curve.tmp.jsonl"
  local attempt
  local status

  mkdir -p "$directory"
  if validate_result "$output" "$op" "$tp" "$repeat"; then
    printf 'skip completed TP=%s op=%s repeat=%s\n' "$tp" "$op" "$repeat"
    return
  fi

  for attempt in 1 2 3; do
    rm -f "$temporary"
    printf 'start TP=%s op=%s repeat=%s attempt=%s\n' \
      "$tp" "$op" "$repeat" "$attempt"
    set +e
    timeout 900s env CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" \
      torchrun --standalone --nproc-per-node "$tp" \
      "$REPO_ROOT/scripts/benchmark_collective_curve.py" \
      --output "$temporary" \
      --repeat-id "$repeat" \
      --op "$op" \
      --warmup "$WARMUP" \
      --iterations "$ITERATIONS" \
      --min-bytes "$MIN_BYTES" \
      --max-bytes "$MAX_BYTES" \
      2>&1 | tee "$directory/run_attempt${attempt}.log"
    status="${PIPESTATUS[0]}"
    set -e
    if (( status == 0 )) && validate_result "$temporary" "$op" "$tp" "$repeat"; then
      mv "$temporary" "$output"
      printf 'complete TP=%s op=%s repeat=%s\n' "$tp" "$op" "$repeat"
      return
    fi
    printf 'retry TP=%s op=%s repeat=%s after status=%s\n' \
      "$tp" "$op" "$repeat" "$status" >&2
  done

  printf 'failed TP=%s op=%s repeat=%s after 3 attempts\n' \
    "$tp" "$op" "$repeat" >&2
  return 1
}

run_tp() {
  local tp="$1"
  local repeat
  local op
  for repeat in $(seq 0 $((REPEATS - 1))); do
    for op in all_reduce all_gather; do
      run_case "$tp" "$op" "$repeat"
    done
  done
}

mode="${1:-all}"
check_all_gpus_idle
capture_environment

case "$mode" in
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
    exit 2
    ;;
esac
