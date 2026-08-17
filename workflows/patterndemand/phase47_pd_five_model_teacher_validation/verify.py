#!/usr/bin/env python3
"""Independent Phase47 compact-result verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from contracts import read_csv  # noqa: E402


def verify(output: Path) -> dict:
    expected = load_json(HERE / "expected_outputs.json")
    missing = [name for name in expected["required"] if not (output / name).is_file()]
    if missing:
        raise RuntimeError({"missing": missing})
    manifest = verify_result_manifest(output)
    workflow = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    roster = load_json(output / "contracts/models.json")["models"]
    inventory = load_json(output / "contracts/model_inventory.json")["models"]
    summary = load_json(output / "summary.json")
    state = load_json(output / "audit/runtime_state.json")
    smoke = load_json(output / "audit/compatibility_smoke.json")["models"]
    launches = load_json(output / "audit/server_launch.json")["models"]
    raw = load_json(output / "audit/raw_manifests.json")
    alignment = read_csv(output / "analysis/gpu_teacher_alignment.csv")
    histograms = read_csv(output / "analysis/histograms_12bin.csv")
    invariants = read_csv(output / "analysis/formula_invariants.csv")
    model_ids = [row["model_id"] for row in roster]
    expected_policy = {row["model_id"]: (row["attention_backend"], int(row["page_size_tokens"])) for row in roster}
    def command_option(command: list[str], name: str) -> str | None:
        return command[command.index(name) + 1] if name in command and command.index(name) + 1 < len(command) else None
    checks = {
        "manifest": manifest["ok"],
        "status": summary.get("status") == "PASS",
        "done": (output / "DONE").read_text(encoding="utf-8").strip() == "PASS",
        "contract_exact": result_contract == workflow,
        "models_exact": [row.get("model_id") for row in inventory] == model_ids,
        "five_model_state": [row.get("model_id") for row in state.get("models", [])] == model_ids,
        "counts": state.get("counts") == {"models": 5, "requests": 225, "exact_requests": 225, "alignment_rows": 30, "histogram_rows": 360},
        "all_runtime_checks": state.get("all_checks_pass") is True and all(all(row.get("checks", {}).values()) for row in state.get("models", [])),
        "alignment_30": len(alignment) == 30,
        "alignment_exact": all(int(row["requests"]) == int(row["exact_requests"]) and int(row["calls_absolute_error"]) == 0 and int(row["logical_bytes_absolute_error"]) == 0 for row in alignment),
        "histograms_360": len(histograms) == 360,
        "histograms_exact": all(float(row["calls_absolute_error"]) == 0.0 and float(row["logical_bytes_absolute_error"]) == 0.0 for row in histograms),
        "invariants_five": len(invariants) == 5 and all(row["exact"] == "True" for row in invariants),
        "smoke_five_pass": [row.get("model_id") for row in smoke] == model_ids and all(row.get("status") == "PASS" and all(row.get("checks", {}).values()) for row in smoke),
        "launch_policy_exact": [row.get("model_id") for row in launches] == model_ids and all((row.get("attention_backend"), int(row.get("page_size_tokens", -1))) == expected_policy[row["model_id"]] for row in launches),
        "kv_cache_bf16_cli": all(command_option(row.get("commands", {}).get(name, []), "--kv-cache-dtype") == "bf16" for row in launches for name in ("prefill", "decode")),
        "rdma_no_fallback": all(row.get("transport_environment", {}).get("MOONCAKE_PROTOCOL") == "rdma" and row.get("transport_environment", {}).get("WITH_NVIDIA_PEERMEM") == "0" and row.get("transport_environment", {}).get("SGLANG_DISAGG_STAGING_BUFFER") == "0" and all(row.get("transport_environment", {}).get(name) is None for name in ("MC_FORCE_TCP", "MC_FORCE_MNNVL", "MC_INTRANODE_NVLINK", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL")) for row in launches),
        "atomic_barrier": all(row.get("admission_environment", {}).get("SGLANG_PD_BOOTSTRAP_BATCH_BARRIER") == "1" and row.get("admission_environment", {}).get("SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB") is None for row in launches),
        "raw_external": raw.get("raw_committed_to_git") is False and len(raw.get("models", [])) == 5 and all(int(row.get("profiler_event_count", 0)) > 0 for row in raw.get("models", [])),
        "no_raw_jsonl": not list(output.rglob("*.jsonl")),
        "no_model_path": all("model_path" not in row for row in inventory),
        "no_training_or_curve": summary.get("training_performed") is False and summary.get("checkpoint_loaded") is False and summary.get("physical_curve_measured") is False and summary.get("scheduler_or_placement_evaluated") is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "manifest": manifest})
    return {"status": "PASS", "checks": checks, "workflow_commit": summary["workflow_commit"], "models": 5, "requests": 225, "manifest_files": manifest["manifest"]["checked_files"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase47_pd_five_model_teacher_validation")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
