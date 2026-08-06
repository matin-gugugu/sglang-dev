#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/experiment-results/phase14f}"
DATASET="${DATASET:-$REPO_ROOT/experiment-results/phase14c/extended_dataset_analysis/aggregated_configurations.csv}"
WARMUP="${WARMUP:-30}"
ITERATIONS="${ITERATIONS:-100}"
REPEATS="${REPEATS:-5}"

mkdir -p "$RESULT_ROOT/curve"
exec > >(tee -a "$RESULT_ROOT/runner.log") 2>&1

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

build_inventory() {
  python3 - "$DATASET" "$RESULT_ROOT/support_inventory.json" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

dataset = Path(sys.argv[1])
output = Path(sys.argv[2])
supports = defaultdict(set)
rows = list(csv.DictReader(dataset.open()))
for row in rows:
    for key in json.loads(row["calls_by_op_payload_json"]):
        op, payload = key.rsplit(":", 1)
        supports[(int(row["tp"]), op)].add(int(payload))

expected = {
    (tp, "all_reduce"): 25 for tp in (2, 4, 8)
} | {
    (tp, "fused_allreduce_residual_rmsnorm"): 10 for tp in (2, 4, 8)
}
if set(supports) != set(expected):
    raise SystemExit(f"unexpected support keys: {sorted(supports)}")
for key, count in expected.items():
    if len(supports[key]) != count:
        raise SystemExit(f"{key}: expected {count} supports, got {len(supports[key])}")

record = {
    "schema_version": "phase14f-support-inventory-v1",
    "source": str(dataset),
    "source_configurations": len(rows),
    "total_supports": sum(len(values) for values in supports.values()),
    "supports": {
        f"tp{tp}:{op}": sorted(values)
        for (tp, op), values in sorted(supports.items())
    },
}
output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
}

support_args() {
  local tp="$1"
  local op="$2"
  python3 - "$RESULT_ROOT/support_inventory.json" "$tp" "$op" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
values = record["supports"][f"tp{sys.argv[2]}:{sys.argv[3]}"]
print(" ".join(str(value) for value in values))
PY
}

validate_result() {
  local path="$1"
  local tp="$2"
  local op="$3"
  local repeat="$4"
  python3 - "$path" "$RESULT_ROOT/support_inventory.json" "$tp" "$op" "$repeat" "$ITERATIONS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
inventory = json.load(open(sys.argv[2]))
tp = int(sys.argv[3])
op = sys.argv[4]
repeat = int(sys.argv[5])
iterations = int(sys.argv[6])
if not path.exists():
    raise SystemExit(1)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
expected = set(inventory["supports"][f"tp{tp}:{op}"])
if {row["payload_bytes"] for row in rows} != expected:
    raise SystemExit(1)
if len(rows) != len(expected):
    raise SystemExit(1)
if any(
    row["schema_version"] != "phase14f-backend-cost-v2"
    or row["op"] != op
    or row["group_size"] != tp
    or row["repeat_id"] != repeat
    or len(row["completion_samples_us"]) != iterations
    or len(row["intrinsic_samples_us"]) != iterations
    or len(row["post_rendezvous_samples_us"]) != iterations
    for row in rows
):
    raise SystemExit(1)
PY
}

capture_environment() {
  if [[ ! -f "$RESULT_ROOT/environment.json" ]]; then
    python3 - "$RESULT_ROOT/environment.json" <<'PY'
import json
import socket
import subprocess
import sys

import torch

record = {
    "hostname": socket.gethostname(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "nccl_version": list(torch.cuda.nccl.version()),
    "device_count": torch.cuda.device_count(),
    "device_names": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ],
    "topology": "single-node-nvlink",
    "predictive_key": ["raw_op", "payload_bytes", "group_size", "topology"],
    "observed_backend_use": "audit-only",
}
with open(sys.argv[1], "w") as output:
    json.dump(record, output, indent=2)
    output.write("\n")
PY
  fi
  if [[ ! -f "$RESULT_ROOT/nvidia_topology.txt" ]]; then
    nvidia-smi topo -m > "$RESULT_ROOT/nvidia_topology.txt"
  fi
}

run_case() {
  local tp="$1"
  local op="$2"
  local repeat="$3"
  local directory="$RESULT_ROOT/curve/tp${tp}/${op}/r${repeat}"
  local output="$directory/curve.jsonl"
  local temporary="$directory/curve.tmp.jsonl"
  local attempt
  local status
  local payloads

  mkdir -p "$directory"
  if validate_result "$output" "$tp" "$op" "$repeat"; then
    printf 'skip completed TP=%s op=%s repeat=%s\n' "$tp" "$op" "$repeat"
    return 0
  fi
  payloads="$(support_args "$tp" "$op")"

  for attempt in 1 2 3; do
    rm -f "$temporary"
    printf 'start TP=%s op=%s repeat=%s attempt=%s\n' "$tp" "$op" "$repeat" "$attempt"
    set +e
    timeout 1800 env \
      PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" \
      torchrun \
        --standalone \
        --nproc-per-node="$tp" \
        "$REPO_ROOT/scripts/benchmark_phase14f_backend_curve.py" \
        --output "$temporary" \
        --repeat-id "$repeat" \
        --op "$op" \
        --payload-bytes $payloads \
        --warmup "$WARMUP" \
        --iterations "$ITERATIONS" \
        2>&1 | tee "$directory/run_attempt${attempt}.log"
    status=${PIPESTATUS[0]}
    set -e
    if (( status == 0 )) && validate_result "$temporary" "$tp" "$op" "$repeat"; then
      mv "$temporary" "$output"
      printf 'complete TP=%s op=%s repeat=%s\n' "$tp" "$op" "$repeat"
      return 0
    fi
    printf 'retry TP=%s op=%s repeat=%s after status=%s\n' \
      "$tp" "$op" "$repeat" "$status" >&2
  done

  printf 'failed TP=%s op=%s repeat=%s after 3 attempts\n' \
    "$tp" "$op" "$repeat" >&2
  return 1
}

main() {
  local tp
  local op
  local repeat

  check_all_gpus_idle
  build_inventory
  capture_environment
  for tp in 2 4 8; do
    for repeat in $(seq 0 $((REPEATS - 1))); do
      for op in all_reduce fused_allreduce_residual_rmsnorm; do
        run_case "$tp" "$op" "$repeat"
      done
    done
  done

  python3 "$REPO_ROOT/scripts/analyze_phase14f_backend_curve.py" \
    --curve-root "$RESULT_ROOT/curve" \
    --dataset "$DATASET" \
    --phase14d-predictions "$REPO_ROOT/experiment-results/phase14d/tp_phase_interaction_analysis/predictions.csv" \
    --output-dir "$RESULT_ROOT/analysis"
  touch "$RESULT_ROOT/DONE"
}

main "$@"
