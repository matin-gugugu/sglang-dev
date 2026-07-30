#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# != 3 )); then
  printf 'usage: %s <tp2|tp4|tp8> <all_reduce|all_gather> <repeat_id>\n' "$0" >&2
  exit 2
fi

TP="${1#tp}"
OP="$2"
REPEAT_ID="$3"

: "${NODE_RANK:?Set NODE_RANK to 0 on the master and 1 on the peer node}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the node-0 bootstrap IP}"
: "${TOPOLOGY:?Set TOPOLOGY to a verified physical placement label}"

NNODES="${NNODES:-2}"
MASTER_PORT="${MASTER_PORT:-29600}"
TRANSPORT_LABEL="${TRANSPORT_LABEL:-roce}"
TIMING_MODE="${TIMING_MODE:-rendezvous}"
MIN_BYTES="${MIN_BYTES:-1024}"
MAX_BYTES="${MAX_BYTES:-134217728}"
WARMUP="${WARMUP:-30}"
ITERATIONS="${ITERATIONS:-100}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/experiment-results/phase7/b200_${TOPOLOGY}_collective_curve}"

if [[ "$OP" != "all_reduce" && "$OP" != "all_gather" ]]; then
  printf 'unsupported op: %s\n' "$OP" >&2
  exit 2
fi
if [[ ! "$TOPOLOGY" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'TOPOLOGY contains unsupported path characters: %s\n' "$TOPOLOGY" >&2
  exit 2
fi
if [[ "$TP" != "2" && "$TP" != "4" && "$TP" != "8" ]]; then
  printf 'unsupported TP: %s\n' "$TP" >&2
  exit 2
fi
if (( NNODES < 2 || TP % NNODES != 0 )); then
  printf 'TP=%s must be divisible by NNODES=%s\n' "$TP" "$NNODES" >&2
  exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  printf 'NODE_RANK=%s is outside [0, %s)\n' "$NODE_RANK" "$NNODES" >&2
  exit 2
fi

LOCAL_PROCS=$((TP / NNODES))
EXPECTED_POINTS=19
DIRECTORY="$RESULT_ROOT/tp${TP}/${OP}/r${REPEAT_ID}"
OUTPUT="$DIRECTORY/curve.jsonl"
TEMPORARY="$DIRECTORY/curve.tmp.jsonl"
RUN_LOG="$DIRECTORY/node${NODE_RANK}.log"

visible_devices() {
  local devices=""
  local index
  for ((index = 0; index < LOCAL_PROCS; index++)); do
    if [[ -n "$devices" ]]; then
      devices+=","
    fi
    devices+="$index"
  done
  printf '%s' "$devices"
}

check_local_gpus_idle() {
  local count=0
  while IFS=, read -r index free util; do
    index="${index//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    count=$((count + 1))
    if (( index < LOCAL_PROCS && (free < 160000 || util > 20) )); then
      printf 'GPU %s is busy: free=%s MiB util=%s%%\n' "$index" "$free" "$util" >&2
      return 1
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if (( count < LOCAL_PROCS )); then
    printf 'Expected at least %s visible GPUs, found %s\n' "$LOCAL_PROCS" "$count" >&2
    return 1
  fi
}

validate_master_result() {
  local path="$1"
  python - "$path" "$OP" "$TP" "$REPEAT_ID" "$EXPECTED_POINTS" "$ITERATIONS" \
    "$NNODES" "$TOPOLOGY" "$TIMING_MODE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
op = sys.argv[2]
tp = int(sys.argv[3])
repeat_id = int(sys.argv[4])
expected_points = int(sys.argv[5])
iterations = int(sys.argv[6])
node_count = int(sys.argv[7])
topology = sys.argv[8]
timing_mode = sys.argv[9]

if not path.exists():
    raise SystemExit("missing result")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
expected_sizes = {1 << exponent for exponent in range(10, 28)}
expected_sizes.add(48 * 1024)
if len(rows) != expected_points:
    raise SystemExit(f"expected {expected_points} rows, got {len(rows)}")
if {row["payload_bytes"] for row in rows} != expected_sizes:
    raise SystemExit("payload sweep mismatch")
if any(
    row["op"] != op
    or row["group_size"] != tp
    or row["repeat_id"] != repeat_id
    or row["node_count"] != node_count
    or row["topology"] != topology
    or row["timing_mode"] != timing_mode
    or len(row["samples_us"]) != iterations
    or len(row["rank_samples_us"]) != tp
    for row in rows
):
    raise SystemExit("record metadata mismatch")
PY
}

check_local_gpus_idle
mkdir -p "$DIRECTORY"

if (( NODE_RANK == 0 )); then
  rm -f "$TEMPORARY"
fi

printf 'start node_rank=%s nnodes=%s TP=%s op=%s repeat=%s topology=%s timing=%s\n' \
  "$NODE_RANK" "$NNODES" "$TP" "$OP" "$REPEAT_ID" "$TOPOLOGY" "$TIMING_MODE"

set +e
timeout 1800s env CUDA_VISIBLE_DEVICES="$(visible_devices)" \
  torchrun \
  --nnodes "$NNODES" \
  --nproc-per-node "$LOCAL_PROCS" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "$REPO_ROOT/scripts/benchmark_collective_curve.py" \
  --output "$TEMPORARY" \
  --repeat-id "$REPEAT_ID" \
  --op "$OP" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS" \
  --min-bytes "$MIN_BYTES" \
  --max-bytes "$MAX_BYTES" \
  --topology "$TOPOLOGY" \
  --timing-mode "$TIMING_MODE" \
  --transport-label "$TRANSPORT_LABEL" \
  2>&1 | tee "$RUN_LOG"
status="${PIPESTATUS[0]}"
set -e

if (( status != 0 )); then
  printf 'failed node_rank=%s TP=%s op=%s repeat=%s status=%s\n' \
    "$NODE_RANK" "$TP" "$OP" "$REPEAT_ID" "$status" >&2
  exit "$status"
fi

if (( NODE_RANK == 0 )); then
  validate_master_result "$TEMPORARY"
  mv "$TEMPORARY" "$OUTPUT"
fi

printf 'complete node_rank=%s TP=%s op=%s repeat=%s\n' \
  "$NODE_RANK" "$TP" "$OP" "$REPEAT_ID"
