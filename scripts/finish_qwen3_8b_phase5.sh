#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
STABILITY_ROOT="${STABILITY_ROOT:-$REPO_ROOT/experiment-results/phase5/qwen3_8b_stability}"
ALL_RANK_ROOT="${ALL_RANK_ROOT:-$REPO_ROOT/experiment-results/phase5/qwen3_8b_all_rank}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/experiment-results/phase5/qwen3_8b_prediction_eval_stabilized}"
EXPECTED_STABILITY_GROUPS="${EXPECTED_STABILITY_GROUPS:-77}"
STABILITY_PID="${STABILITY_PID:-}"

cd "$REPO_ROOT"

if [[ -n "$STABILITY_PID" ]]; then
  while kill -0 "$STABILITY_PID" 2>/dev/null; do
    sleep 30
  done
fi

done_groups="$(
  find "$STABILITY_ROOT" -name DONE -type f | wc -l
)"
if ((done_groups != EXPECTED_STABILITY_GROUPS)); then
  printf 'stability suite incomplete: expected %s groups, got %s\n' \
    "$EXPECTED_STABILITY_GROUPS" "$done_groups" >&2
  exit 1
fi

mkdir -p \
  "$REPO_ROOT/experiment-results/phase5/qwen3_8b_stability_summary" \
  "$EVAL_ROOT" \
  "$ALL_RANK_ROOT" \
  "$REPO_ROOT/experiment-results/phase5/qwen3_8b_all_rank_summary"

python scripts/summarize_qwen3_8b_stability.py \
  >"$REPO_ROOT/experiment-results/phase5/qwen3_8b_stability_summary/summary.log" \
  2>&1

python scripts/evaluate_qwen3_8b_expanded_models.py \
  --output-dir "$EVAL_ROOT" \
  >"$EVAL_ROOT/evaluation.log" \
  2>&1

bash scripts/run_qwen3_8b_all_rank_suite.sh \
  >"$ALL_RANK_ROOT/suite.log" \
  2>&1

python scripts/summarize_qwen3_8b_all_rank.py \
  >"$REPO_ROOT/experiment-results/phase5/qwen3_8b_all_rank_summary/summary.log" \
  2>&1

printf 'Phase 5 stability, stabilized evaluation, and all-rank suite complete.\n'
