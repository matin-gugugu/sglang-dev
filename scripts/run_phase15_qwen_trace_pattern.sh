#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODEL_PATH=${MODEL_PATH:-/media/ssd1/Qwen3-8B}
PLAN=${PLAN:-experiment-results/phase15_trace_data/smoke_replay_plan.jsonl}
RESULT_ROOT=${RESULT_ROOT:-experiment-results/phase15_qwen_trace_pattern}

visible_devices() {
  case "$1" in
    2) printf '0,1\n' ;;
    4) printf '0,1,2,3\n' ;;
    8) printf '0,1,2,3,4,5,6,7\n' ;;
    *) printf 'unsupported TP: %s\n' "$1" >&2; return 64 ;;
  esac
}

check_gpus_idle() {
  local tp="$1"
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
  if ((busy > 0)); then
    printf 'refusing to start TP=%s: found %s active GPU processes\n' "$tp" "$busy" >&2
    return 1
  fi
}

run_tp() {
  local tp="$1"
  local directory="$RESULT_ROOT/tp${tp}/r0"
  local result="$directory/result.jsonl"
  if [[ -s "$directory/DONE" ]]; then
    python scripts/validate_phase15_trace_pattern.py \
      --result "$result" --plan "$PLAN" --tp "$tp" --output-dir "$directory"
    printf 'skip completed TP=%s\n' "$tp"
    return
  fi
  [[ -d "$MODEL_PATH" ]] || { printf 'missing model: %s\n' "$MODEL_PATH" >&2; return 1; }
  [[ -s "$PLAN" ]] || { printf 'missing plan: %s\n' "$PLAN" >&2; return 1; }
  check_gpus_idle "$tp"
  mkdir -p "$directory"
  rm -f "$result" "$directory/run.log" "$directory/validate.log" "$directory/DONE"
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv -l 1 >"$directory/telemetry.csv" &
  local telemetry_pid=$!
  trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT INT TERM
  CUDA_VISIBLE_DEVICES=$(visible_devices "$tp") PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$tp" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      --batch-size 1 \
      --input-len 16 \
      --output-len 2 \
      --trace-replay-plan "$PLAN" \
      --warmup-each-workload \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --run-name "qwen3-8b-phase15-trace-tp${tp}" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  trap - EXIT INT TERM
  python scripts/validate_phase15_trace_pattern.py \
    --result "$result" --plan "$PLAN" --tp "$tp" --output-dir "$directory" \
    | tee "$directory/validate.log"
  if grep -Eqi 'out of memory|Traceback|CPU fallback|falling back|NCCL error' "$directory/run.log"; then
    printf 'error marker found in %s\n' "$directory/run.log" >&2
    return 1
  fi
  printf 'complete\n' >"$directory/DONE"
}

case "${1:-tp2}" in
  tp2) run_tp 2 ;;
  tp4) run_tp 4 ;;
  tp8) run_tp 8 ;;
  all)
    run_tp 2
    run_tp 4
    run_tp 8
    ;;
  *) printf 'usage: %s [tp2|tp4|tp8|all]\n' "$0" >&2; exit 64 ;;
esac
