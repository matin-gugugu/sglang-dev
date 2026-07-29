#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="$REPO_ROOT/experiment-results/phase2/b200_l1_custom_kernel_curve"
MIN_BYTES=8192
MAX_BYTES=16777216
WARMUP=30
ITERATIONS=100
REPEATS=5
EXPECTED_POINTS=13

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/suite.log") 2>&1

validate_result() {
  local path="$1"
  local tp="$2"
  local repeat="$3"
  python - "$path" "$tp" "$repeat" "$EXPECTED_POINTS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tp = int(sys.argv[2])
repeat = int(sys.argv[3])
expected_points = int(sys.argv[4])
if not path.exists():
    raise SystemExit(1)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
expected_sizes = {1 << exponent for exponent in range(13, 25)}
expected_sizes.add(48 * 1024)
if len(rows) != expected_points:
    raise SystemExit(1)
if {row["payload_bytes"] for row in rows} != expected_sizes:
    raise SystemExit(1)
if any(
    row["schema_version"] != "collective-kernel-cost-v1"
    or row["backend"] != "sglang_custom_all_reduce_v2"
    or row["group_size"] != tp
    or row["repeat_id"] != repeat
    or len(row["samples_us"]) != 100
    or len(row["completion_samples_us"]) != 100
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
    "latency_scope": "skew-free-intrinsic-lower-envelope-across-ranks",
}
with open(sys.argv[1], "w") as output:
    json.dump(record, output, indent=2)
    output.write("\n")
PY
  fi
}

run_case() {
  local tp="$1"
  local repeat="$2"
  local directory="$RESULT_ROOT/tp${tp}/r${repeat}"
  local output="$directory/curve.jsonl"
  local temporary="$directory/curve.tmp.jsonl"
  local attempt
  local status

  mkdir -p "$directory"
  if validate_result "$output" "$tp" "$repeat"; then
    printf 'skip completed TP=%s repeat=%s\n' "$tp" "$repeat"
    return 0
  fi

  for attempt in 1 2 3; do
    rm -f "$temporary"
    printf 'start TP=%s repeat=%s attempt=%s\n' "$tp" "$repeat" "$attempt"
    set +e
    timeout 1200 env \
      PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" \
      torchrun \
        --standalone \
        --nproc-per-node="$tp" \
        "$REPO_ROOT/scripts/benchmark_sglang_custom_allreduce_kernel_curve.py" \
        --output "$temporary" \
        --repeat-id "$repeat" \
        --warmup "$WARMUP" \
        --iterations "$ITERATIONS" \
        --min-bytes "$MIN_BYTES" \
        --max-bytes "$MAX_BYTES" \
        2>&1 | tee "$directory/run_attempt${attempt}.log"
    status=${PIPESTATUS[0]}
    set -e
    if (( status == 0 )) && validate_result "$temporary" "$tp" "$repeat"; then
      mv "$temporary" "$output"
      printf 'complete TP=%s repeat=%s\n' "$tp" "$repeat"
      return 0
    fi
    printf 'failed TP=%s repeat=%s attempt=%s status=%s\n' \
      "$tp" "$repeat" "$attempt" "$status" >&2
  done
  return 1
}

main() {
  local tp
  local repeat
  check_all_gpus_idle
  capture_environment
  for tp in 2 4 8; do
    for ((repeat = 0; repeat < REPEATS; repeat++)); do
      run_case "$tp" "$repeat"
    done
  done
}

main "$@"
