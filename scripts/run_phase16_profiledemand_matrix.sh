#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/sgl-workspace/sglang-src}
cd "$REPO_ROOT"

SMOKE_PARENT=experiment-results/phase16_profiledemand_gpu/smoke/qwen3-8b/tp2
SMOKE_RUN=$SMOKE_PARENT/r0

while [[ ! -s "$SMOKE_RUN/DONE" ]]; do
  if [[ -s "$SMOKE_RUN/FAIL" ]]; then
    printf 'smoke failed: %s\n' "$(cat "$SMOKE_RUN/FAIL")" >&2
    exit 1
  fi
  if [[ -s "$SMOKE_PARENT/runner.pid" ]]; then
    smoke_pid=$(cat "$SMOKE_PARENT/runner.pid")
    if ! kill -0 "$smoke_pid" 2>/dev/null; then
      printf 'smoke process %s exited without DONE\n' "$smoke_pid" >&2
      exit 1
    fi
  fi
  printf '%s waiting for Qwen3-8B TP2 smoke\n' "$(date -Is)"
  sleep 30
done

for model in qwen3-8b deepseek-v2-lite qwen3-30b-a3b; do
  for tp in 2 4 8; do
    printf '%s start full model=%s tp=%s\n' "$(date -Is)" "$model" "$tp"
    bash scripts/run_phase16_profiledemand_gpu.sh "$model" full "$tp"
    printf '%s complete full model=%s tp=%s\n' "$(date -Is)" "$model" "$tp"
  done
done

printf 'PASS\n' >experiment-results/phase16_profiledemand_gpu/MATRIX_DONE
printf '%s matrix complete\n' "$(date -Is)"
