#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/sgl-workspace/sglang-src}
MODEL_KEY=${1:-qwen3-8b}
PLAN_KIND=${2:-smoke}
TP=${3:-2}

case "$MODEL_KEY" in
  qwen3-8b) MODEL_PATH=/media/ssd1/Qwen3-8B ;;
  qwen3-30b-a3b) MODEL_PATH=/media/ssd1/Qwen3-30B-A3B ;;
  deepseek-v2-lite) MODEL_PATH=/media/ssd1/DeepSeek-V2-Lite ;;
  *) printf 'unknown model key: %s\n' "$MODEL_KEY" >&2; exit 64 ;;
esac
case "$PLAN_KIND" in
  smoke|full) ;;
  *) printf 'plan kind must be smoke or full\n' >&2; exit 64 ;;
esac
case "$TP" in
  2) DEVICES=0,1 ;;
  4) DEVICES=0,1,2,3 ;;
  8) DEVICES=0,1,2,3,4,5,6,7 ;;
  *) printf 'TP must be 2, 4, or 8\n' >&2; exit 64 ;;
esac

cd "$REPO_ROOT"
PLAN="experiment-results/phase16_profiledemand_plans/${PLAN_KIND}_replay_plan.jsonl"
DIRECTORY="experiment-results/phase16_profiledemand_gpu/${PLAN_KIND}/${MODEL_KEY}/tp${TP}/r0"
RESULT="$DIRECTORY/result.jsonl"
mkdir -p "$DIRECTORY"

validate() {
  python3 scripts/validate_profiledemand_gpu_labels.py \
    --result "$RESULT" \
    --plan "$PLAN" \
    --model "$MODEL_KEY" \
    --tp "$TP" \
    --output-dir "$DIRECTORY" \
    | tee "$DIRECTORY/validate.log"
}

if [[ -s "$DIRECTORY/DONE" ]]; then
  validate
  printf 'already complete: %s\n' "$DIRECTORY"
  exit 0
fi
[[ -d "$MODEL_PATH" ]] || { printf 'missing model: %s\n' "$MODEL_PATH" >&2; exit 2; }
[[ -s "$PLAN" ]] || { printf 'missing plan: %s\n' "$PLAN" >&2; exit 2; }

active_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
if ((active_processes > 0)); then
  printf 'refusing to start: found %s active GPU processes\n' "$active_processes" >&2
  exit 2
fi

rm -f "$RESULT" "$DIRECTORY/run.log" "$DIRECTORY/validate.log" \
  "$DIRECTORY/DONE" "$DIRECTORY/FAIL"
nvidia-smi \
  --query-gpu=timestamp,index,pstate,memory.used,utilization.gpu,power.draw \
  --format=csv -l 1 >"$DIRECTORY/telemetry.csv" 2>&1 &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
CUDA_VISIBLE_DEVICES="$DEVICES" PYTHONPATH=python \
  python3 -m sglang.benchmark.one_batch \
    --model-path "$MODEL_PATH" \
    --tp "$TP" \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    --batch-size 1 \
    --input-len 16 \
    --output-len 2 \
    --trace-replay-plan "$PLAN" \
    --comm-profile \
    --comm-profile-mode histogram-only \
    --run-name "profiledemand-v1-${PLAN_KIND}-${MODEL_KEY}-tp${TP}" \
    --result-filename "$RESULT" \
    >"$DIRECTORY/run.log" 2>&1
run_status=$?
set -e
cleanup
trap - EXIT INT TERM
if ((run_status != 0)); then
  printf 'one_batch exited %s\n' "$run_status" >"$DIRECTORY/FAIL"
  tail -n 80 "$DIRECTORY/run.log" >&2
  exit "$run_status"
fi
if grep -Eqi 'out of memory|Traceback|CPU fallback|falling back|NCCL error' "$DIRECTORY/run.log"; then
  printf 'error marker found in run.log\n' >"$DIRECTORY/FAIL"
  exit 1
fi
validate
printf 'PASS\n' >"$DIRECTORY/DONE"
printf 'complete: %s\n' "$DIRECTORY"
