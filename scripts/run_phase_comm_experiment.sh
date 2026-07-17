#!/usr/bin/env bash
set -euo pipefail

# Phase communication profiling experiment for SGLang one-batch benchmark.
# Run inside the sglang container from /sgl-workspace/sglang-src.
# Override these env vars when needed, e.g.:
#   MODEL_PATH=/media/ssd1/GLM-4.7-FP8 TP_SIZE=8 RESULT_FILE=/sgl-workspace/glm_comm.jsonl ./scripts/run_phase_comm_experiment.sh

MODEL_PATH="${MODEL_PATH:-/media/ssd1/DeepSeek-V3.2}"
TP_SIZE="${TP_SIZE:-8}"
RESULT_FILE="${RESULT_FILE:-/sgl-workspace/phase_comm_results.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"

python -m sglang.benchmark.one_batch \
  --model-path "${MODEL_PATH}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --batch-size 1 2 4 \
  --input-len 256 1024 4096 \
  --output-len 16 64 128 \
  --comm-profile \
  --run-name "phase-comm-tp${TP_SIZE}" \
  --result-filename "${RESULT_FILE}"

printf '\nResult jsonl: %s\n' "${RESULT_FILE}"
