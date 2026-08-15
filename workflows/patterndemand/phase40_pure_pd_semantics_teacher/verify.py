#!/usr/bin/env python3
"""Independent verifier for Phase40 result assets and evidence boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest
from contracts import BIN_EDGES_BYTES, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase40_pure_pd_semantics_teacher",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    expected = load_json(HERE / "expected_outputs.json")
    missing = [relative for relative in expected["required"] if not (output / relative).is_file()]
    if missing:
        raise RuntimeError({"missing_required_outputs": missing})
    manifest = verify_result_manifest(output)
    workflow = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    model = load_json(output / "contracts/model_contract.json")
    workload = load_json(output / "contracts/workload_contract.json")
    freeze = load_json(output / "audit/input_freeze.json")
    environment = load_json(output / "audit/environment.json")
    smoke = load_json(output / "audit/compatibility_smoke.json")
    server_launch = load_json(output / "audit/server_launch.json")
    raw = load_json(output / "audit/raw_manifest.json")
    state = load_json(output / "audit/runtime_state.json")
    summary = load_json(output / "summary.json")
    alignment = read_csv(output / "analysis/gpu_teacher_alignment.csv")
    histograms = read_csv(output / "analysis/histograms_12bin.csv")
    invariants = read_csv(output / "analysis/formula_invariants.csv")
    requests = workload.get("request_order", [])
    prefill_command = server_launch.get("commands", {}).get("prefill", [])
    optimistic_option_index = (
        prefill_command.index("--optimistic-prefill-retries")
        if "--optimistic-prefill-retries" in prefill_command
        else None
    )
    no_optimistic_prefill = (
        optimistic_option_index is not None
        and optimistic_option_index + 1 < len(prefill_command)
        and prefill_command[optimistic_option_index + 1] == "0"
    )
    checks = {
        "manifest": manifest["ok"],
        "status_pass": summary.get("status") == "PASS",
        "done_pass": (output / "DONE").read_text(encoding="utf-8").strip() == "PASS",
        "result_contract_exact": result_contract == workflow,
        "workflow_commit_consistent": summary.get("workflow_commit") == state.get("workflow_commit") == freeze.get("workflow_commit"),
        "parent_result_frozen": freeze.get("workflow_parent_result_commit") == workflow["workflow_parent_result_commit"],
        "all_runtime_checks": all(state.get("checks", {}).values()),
        "requests_45": len(requests) == int(workflow["acceptance_gates"]["expected_requests"]),
        "request_ids_unique": len({row.get("rid") for row in requests}) == len(requests),
        "scenario_repeat_matrix": len({(row.get("scenario"), int(row.get("repeat", -1))) for row in requests}) == 15,
        "alignment_rows_6": len(alignment) == int(workflow["expected_alignment_rows"]),
        "alignment_exact": all(
            int(row["requests"]) == int(row["exact_requests"])
            and int(row["calls_absolute_error"]) == 0
            and int(row["logical_bytes_absolute_error"]) == 0
            for row in alignment
        ),
        "histogram_rows_72": len(histograms) == int(workflow["expected_histogram_rows"]),
        "histogram_edges_exact": all(
            float(row["lower_bytes"]) == BIN_EDGES_BYTES[int(row["bin_index"])]
            and float(row["upper_bytes"]) == BIN_EDGES_BYTES[int(row["bin_index"]) + 1]
            for row in histograms
        ),
        "histograms_exact": all(
            float(row["calls_absolute_error"]) == 0.0
            and float(row["logical_bytes_absolute_error"]) == 0.0
            for row in histograms
        ),
        "invariants_exact": len(invariants) == 5 and all(row["exact"] == "True" for row in invariants),
        "formula_consistent": model["derived"]["kv_bytes_per_page"] == model["derived"]["kv_bytes_per_token"] * model["structure"]["page_size_tokens"],
        "pure_pd_gpu_pair": len(environment.get("gpu_pair", [])) == 2 and environment["gpu_pair"][0] != environment["gpu_pair"][1],
        "compatibility_smoke_pass": smoke.get("status") == "PASS" and all(smoke.get("checks", {}).values()),
        "compatibility_smoke_transport_one_chunk": int(smoke.get("transport_sender_chunks", 0)) == int(workflow["compatibility_smoke_contract"]["expected_transport_sender_chunks"]),
        "compatibility_smoke_total_chunks": int(smoke.get("matching_sender_chunks", 0)) == int(workflow["compatibility_smoke_contract"]["expected_sender_chunks_total"]),
        "compatibility_smoke_backend": smoke.get("attention_backend") == workflow["backend_contract"]["inference_attention_backend"],
        "compatibility_smoke_page_one": all(int(row.get("page_size_tokens", -1)) == int(workflow["measurement_contract"]["page_size_tokens"]) for row in smoke.get("observed", [])),
        "compatibility_smoke_rdma_env": smoke.get("transport_environment", {}).get("MOONCAKE_PROTOCOL") == "rdma" and smoke.get("transport_environment", {}).get("WITH_NVIDIA_PEERMEM") == "0" and all(smoke.get("transport_environment", {}).get(name) is None for name in ("MC_FORCE_TCP", "MC_FORCE_MNNVL", "MC_INTRANODE_NVLINK", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL")),
        "compatibility_smoke_atomic_admission": smoke.get("admission_environment", {}).get("SGLANG_PD_BOOTSTRAP_BATCH_BARRIER") == "1" and smoke.get("admission_environment", {}).get("SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB") is None,
        "formal_server_backend_pinned": server_launch.get("attention_backend") == workflow["backend_contract"]["inference_attention_backend"],
        "formal_server_page_one": int(server_launch.get("page_size_tokens", -1)) == int(workflow["measurement_contract"]["page_size_tokens"]),
        "formal_server_rdma_env": server_launch.get("transport_environment", {}).get("MOONCAKE_PROTOCOL") == "rdma" and server_launch.get("transport_environment", {}).get("WITH_NVIDIA_PEERMEM") == "0" and all(server_launch.get("transport_environment", {}).get(name) is None for name in ("MC_FORCE_TCP", "MC_FORCE_MNNVL", "MC_INTRANODE_NVLINK", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL")),
        "formal_server_atomic_admission": server_launch.get("admission_environment", {}).get("SGLANG_PD_BOOTSTRAP_BATCH_BARRIER") == "1" and server_launch.get("admission_environment", {}).get("SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB") is None,
        "formal_server_no_optimistic_prefill": no_optimistic_prefill,
        "raw_external": raw.get("raw_committed_to_git") is False and int(raw.get("profiler_event_count", 0)) > 0,
        "no_raw_jsonl_in_result": not list(output.rglob("*.jsonl")),
        "no_training": summary.get("training_performed") is False,
        "no_checkpoint": summary.get("checkpoint_loaded") is False,
        "no_curve_or_scheduler": summary.get("physical_curve_measured") is False and summary.get("scheduler_or_placement_evaluated") is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "manifest": manifest})
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "workflow_commit": summary["workflow_commit"],
                "requests": len(requests),
                "gpu_logical_chunks": state["counts"]["gpu_logical_chunks"],
                "teacher_logical_chunks": state["counts"]["teacher_logical_chunks"],
                "histogram_rows": len(histograms),
                "external_raw_files": raw["file_count"],
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
