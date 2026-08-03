#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sgl-workspace/sglang-src}"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/Qwen3-30B-A3B}"
MODEL_REVISION="${MODEL_REVISION:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment-results/phase12/qwen3_30b_a3b_admission}"
MIN_FREE_MIB="${MIN_FREE_MIB:-160000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-20}"

cd "$REPO_ROOT"

visible_devices() {
  local tp="$1"
  local devices=""
  local index
  for ((index = 0; index < tp; index++)); do
    [[ -n "$devices" ]] && devices+=","
    devices+="$index"
  done
  printf '%s' "$devices"
}

check_model() {
  python - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
config_path = model / "config.json"
index_path = model / "model.safetensors.index.json"
assert config_path.is_file() and config_path.stat().st_size > 0, config_path
assert index_path.is_file() and index_path.stat().st_size > 0, index_path
config = json.loads(config_path.read_text())
index = json.loads(index_path.read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 16, (len(shards), shards)
missing = [name for name in shards if not (model / name).is_file()]
empty = [name for name in shards if (model / name).is_file() and not (model / name).stat().st_size]
assert not missing and not empty, {"missing": missing, "empty": empty}
assert config["architectures"] == ["Qwen3MoeForCausalLM"]
assert config["hidden_size"] == 2048
assert config["num_attention_heads"] == 32
assert config["num_key_value_heads"] == 4
assert config["num_hidden_layers"] == 48
assert config["num_experts"] == 128
assert config["num_experts_per_tok"] == 8
print(
    f"validated model files: shards={len(shards)} architecture={config['architectures'][0]} "
    f"dtype={config.get('torch_dtype', config.get('dtype'))}"
)
PY
}

check_gpus_idle() {
  local tp="$1"
  local index=0
  local failed=0
  local free_mib utilization
  while IFS=, read -r free_mib utilization; do
    free_mib="${free_mib//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if ((index < tp && (free_mib < MIN_FREE_MIB || utilization > MAX_IDLE_UTIL))); then
      printf 'GPU %d busy: free=%s MiB util=%s%%\n' \
        "$index" "$free_mib" "$utilization" >&2
      failed=1
    fi
    ((index += 1))
  done < <(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if ((failed)); then
    printf 'Refusing to run admission smoke on busy GPUs.\n' >&2
    return 2
  fi
}

start_telemetry() {
  local output="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used \
    --format=csv \
    --loop=1 \
    >"$output" 2>&1 &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID:-}" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill "$TELEMETRY_PID"
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

validate_smoke() {
  local tp="$1"
  local result="$2"
  python - "$tp" "$result" <<'PY'
import json
import sys
from pathlib import Path

tp, result = int(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
assert len(rows) == 1, (result, len(rows))
row = rows[0]
assert row["same_shape_workload_warmup"] is True
assert row["generated_output_tokens"] == row["output_len"] == 8
assert row["generated_output_tokens_per_request"] == [8]
profiles = sorted(row["comm_profile"], key=lambda item: item["tp_rank"])
assert [profile["tp_rank"] for profile in profiles] == list(range(tp))
reference = profiles[0]
assert reference["capture_mode"] == "histogram-only"
assert reference["raw_events_saved"] is False
assert reference["events"] == []
assert reference["events_truncated"] is False
for profile in profiles[1:]:
    assert profile["stats"] == reference["stats"]
    assert profile["event_histograms"] == reference["event_histograms"]
prefill = [item for item in reference["event_histograms"] if item["phase"] == "prefill"]
assert prefill
assert all(item["count"] > 0 for item in prefill)
assert all(item["input_payload_bytes"] > 0 for item in prefill)
assert all(item["group_size"] == tp for item in prefill)
print(
    f"validated admission smoke: tp={tp} rows=1 "
    f"ops={sorted({item['op'] for item in prefill})} "
    f"calls={sum(item['count'] for item in prefill)}"
)
PY
}

write_metadata() {
  mkdir -p "$OUTPUT_ROOT"
  python - "$MODEL_PATH" "$MODEL_REVISION" "$OUTPUT_ROOT/admission_metadata.json" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

model, revision, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
config_path = model / "config.json"
index_path = model / "model.safetensors.index.json"
config = json.loads(config_path.read_text())
index = json.loads(index_path.read_text())
shards = sorted(set(index["weight_map"].values()))
payload = {
    "model_path": str(model),
    "model_revision": revision,
    "architecture": config["architectures"][0],
    "dtype": config.get("torch_dtype", config.get("dtype")),
    "hidden_size": config["hidden_size"],
    "num_hidden_layers": config["num_hidden_layers"],
    "num_attention_heads": config["num_attention_heads"],
    "num_key_value_heads": config["num_key_value_heads"],
    "num_experts": config["num_experts"],
    "num_experts_per_tok": config["num_experts_per_tok"],
    "weight_shards": len(shards),
    "weight_bytes": sum((model / name).stat().st_size for name in shards),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    "python": platform.python_version(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_smoke() {
  local tp="$1"
  local directory="$OUTPUT_ROOT/tp${tp}"
  local result="$directory/result.jsonl"

  if [[ -s "$directory/DONE" ]]; then
    validate_smoke "$tp" "$result"
    printf 'skip completed admission TP=%s\n' "$tp"
    return
  fi

  check_model
  check_gpus_idle "$tp"
  mkdir -p "$directory"
  rm -f "$result" "$directory/run.log" "$directory/telemetry.csv" \
    "$directory/validate.log" "$directory/DONE"

  printf 'start Qwen3-30B-A3B admission TP=%s\n' "$tp"
  start_telemetry "$directory/telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  CUDA_VISIBLE_DEVICES="$(visible_devices "$tp")" PYTHONPATH=python \
    python -m sglang.benchmark.one_batch \
      --model-path "$MODEL_PATH" \
      --tp "$tp" \
      --trust-remote-code \
      --mem-fraction-static 0.85 \
      --disable-cuda-graph \
      --batch-size 1 \
      --input-len 128 \
      --output-len 8 \
      --profile-stage prefill \
      --warmup-each-workload \
      --comm-profile \
      --comm-profile-mode histogram-only \
      --run-name "qwen3-30b-a3b-admission-tp${tp}" \
      --result-filename "$result" \
      >"$directory/run.log" 2>&1
  stop_telemetry
  trap - EXIT INT TERM

  validate_smoke "$tp" "$result" | tee "$directory/validate.log"
  if grep -Eqi 'out of memory|Traceback|CPU fallback|falling back' "$directory/run.log"; then
    printf 'admission log contains an error or fallback marker: %s\n' \
      "$directory/run.log" >&2
    return 1
  fi
  printf 'complete\n' >"$directory/DONE"
}

write_metadata
case "${1:-all}" in
  tp2)
    run_smoke 2
    ;;
  tp8)
    run_smoke 8
    ;;
  all)
    run_smoke 2
    run_smoke 8
    ;;
  *)
    printf 'usage: %s [tp2|tp8|all]\n' "$0" >&2
    exit 64
    ;;
esac
