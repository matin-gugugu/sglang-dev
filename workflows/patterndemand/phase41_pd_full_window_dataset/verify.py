#!/usr/bin/env python3
"""Independent Phase41 result verifier and leakage audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from contracts import BIN_EDGES_BYTES, read_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase41_pd_full_window_dataset",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    expected = load_json(HERE / "expected_outputs.json")
    missing = [relative for relative in expected["required"] if not (output / relative).is_file()]
    if missing:
        raise RuntimeError({"missing_required_outputs": missing})
    manifest = verify_result_manifest(output)
    workflow = load_json(HERE / "experiment.json")
    feature_contract = load_json(HERE / "feature_contract.json")
    result_contract = load_json(output / "contracts/experiment.json")
    result_features = load_json(output / "contracts/feature_contract.json")
    model = load_json(output / "contracts/model_contract.json")
    dataset_contract = load_json(output / "contracts/dataset_contract.json")
    freeze = load_json(output / "audit/input_freeze.json")
    environment = load_json(output / "audit/environment.json")
    source = load_json(output / "audit/source_semantics.json")
    server = load_json(output / "audit/server_launch.json")
    raw = load_json(output / "audit/raw_manifest.json")
    build = load_json(output / "audit/dataset_build.json")
    state = load_json(output / "audit/runtime_state.json")
    summary = load_json(output / "summary.json")
    alignment = read_csv(output / "analysis/gpu_teacher_alignment.csv")
    histograms = read_csv(output / "analysis/gpu_teacher_histograms_12bin.csv")
    cases = read_csv(output / "analysis/gpu_case_inventory.csv")
    inventory = read_csv(output / "analysis/dataset_inventory.csv")
    h0_analysis = read_csv(output / "analysis/h0_vs_hfull.csv")
    profiles = read_csv(output / "profiles/development_profiles.csv.gz")
    targets = read_csv(output / "dataset/pd_development_hfull_targets.csv.gz")
    examples = read_csv(output / "dataset/pd_development_h0_residual_examples.csv.gz")
    blind = read_csv(output / "dataset/pd_blind_target_free_features.csv.gz")

    expected_cases = len(workflow["gpu_sentinel_contract"]["synthetic_cases"]) + len(
        workflow["gpu_sentinel_contract"]["real_full_window_cases"]
    )
    predictor_columns = lambda row: {  # noqa: E731
        name for name in row if name.startswith(("feature_", "h0_"))
    }
    target_ids = {row["profile_id"] for row in targets}
    example_ids = {row["profile_id"] for row in examples}
    blind_ids = {row["profile_id"] for row in blind}
    profile_by_id = {row["profile_id"]: row for row in profiles}
    kv_bytes_per_token = int(model["derived"]["kv_bytes_per_token"])
    bytes_formula_errors = []
    for row in targets:
        profile = profile_by_id[row["profile_id"]]
        expected_bytes = float(profile["input_mean_capped"]) * kv_bytes_per_token * 1000.0
        actual_bytes = float(row["target_total_logical_bytes_per_1000"])
        bytes_formula_errors.append(abs(actual_bytes - expected_bytes) / max(expected_bytes, 1.0))

    prefill_command = server.get("commands", {}).get("prefill", [])
    optimistic_index = (
        prefill_command.index("--optimistic-prefill-retries")
        if "--optimistic-prefill-retries" in prefill_command
        else None
    )
    no_optimistic = (
        optimistic_index is not None
        and optimistic_index + 1 < len(prefill_command)
        and prefill_command[optimistic_index + 1] == "0"
    )
    checks = {
        "manifest": manifest["ok"],
        "status_pass": summary.get("status") == "PASS",
        "done_pass": (output / "DONE").read_text(encoding="utf-8").strip() == "PASS",
        "result_contract_exact": result_contract == workflow,
        "feature_contract_exact": result_features == feature_contract,
        "workflow_commit_consistent": summary.get("workflow_commit")
        == state.get("workflow_commit")
        == freeze.get("workflow_commit"),
        "parent_result_frozen": freeze.get("workflow_parent_result_commit")
        == workflow["workflow_parent_result_commit"],
        "three_gates_pass": all(state.get("gates", {}).values()),
        "all_runtime_checks": all(state.get("checks", {}).values()),
        "gpu_alignment_rows": len(alignment) == expected_cases + 1,
        "gpu_alignment_exact": all(
            int(row["requests"]) == int(row["exact_requests"])
            and int(row["calls_absolute_error"]) == 0
            and int(row["logical_bytes_absolute_error"]) == 0
            for row in alignment
        ),
        "gpu_overall_counts": any(
            row["case"] == "overall"
            and int(row["requests"])
            == int(workflow["gpu_sentinel_contract"]["expected_requests"])
            and int(row["waves"]) == int(workflow["gpu_sentinel_contract"]["expected_waves"])
            for row in alignment
        ),
        "gpu_case_inventory": len(cases) == expected_cases
        and sum(int(row["total_requests"]) for row in cases)
        == int(workflow["gpu_sentinel_contract"]["expected_requests"])
        and sum(int(row["total_waves"]) for row in cases)
        == int(workflow["gpu_sentinel_contract"]["expected_waves"]),
        "histogram_rows": len(histograms) == (expected_cases + 1) * 12,
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
        "profiles_94": len(profiles) == 94,
        "targets_94": len(targets) == 94,
        "examples_94": len(examples) == 94,
        "h0_analysis_94": len(h0_analysis) == 94,
        "blind_12": len(blind) == 12,
        "development_ids_exact": target_ids == example_ids == set(profile_by_id),
        "blind_development_disjoint": not blind_ids.intersection(example_ids),
        "roles_75_19": sum(row["split_role"] == "development_train" for row in examples)
        == 75
        and sum(row["split_role"] == "development_validation" for row in examples)
        == 19,
        "inventory_roles": {row["split_role"] for row in inventory}
        == {"development_train", "development_validation", "blind_confirmation"},
        "blind_target_free": all(
            not any(name.startswith(("target_", "residual_")) for name in row)
            for row in blind
        )
        and int(dataset_contract.get("blind_target_rows", -1)) == 0
        and build.get("blind_target_generated") is False,
        "predictor_schema_same": predictor_columns(examples[0]) == predictor_columns(blind[0]),
        "h0_present": all(
            all(f"h0_calls_bin_{index:02d}" in row for index in range(12))
            and all(f"h0_logical_bytes_bin_{index:02d}" in row for index in range(12))
            for row in examples + blind
        ),
        "targets_and_residuals_present": all(
            all(f"target_calls_bin_{index:02d}" in row for index in range(12))
            and all(f"target_logical_bytes_bin_{index:02d}" in row for index in range(12))
            and all(f"residual_calls_bin_{index:02d}" in row for index in range(12))
            and all(f"residual_logical_bytes_bin_{index:02d}" in row for index in range(12))
            for row in examples
        ),
        "hfull_bytes_formula": max(bytes_formula_errors, default=1.0) < 1e-12,
        "complete_request_columns_absent": not any(
            name in {"requests", "input_lens", "output_lens", "full_request_list"}
            for row in profiles + targets + examples + blind
            for name in row
        ),
        "bundle_external": freeze.get("bundle_external") is True
        and build.get("full_requests_saved_in_git") is False
        and dataset_contract.get("full_request_list_saved_in_result") is False,
        "raw_external": raw.get("raw_committed_to_git") is False
        and raw.get("complete_requests_committed_to_git") is False
        and int(raw.get("profiler_event_count", 0)) > 0,
        "no_raw_jsonl_in_result": not list(output.rglob("*.jsonl")),
        "source_semantics_all_pass": all(source.values()),
        "pure_pd_gpu_pair": len(environment.get("gpu_pair", [])) == 2
        and environment["gpu_pair"][0] != environment["gpu_pair"][1],
        "server_atomic_barrier": server.get("admission_environment", {}).get(
            "SGLANG_PD_BOOTSTRAP_BATCH_BARRIER"
        )
        == "1",
        "server_no_optimistic": no_optimistic,
        "server_rdma": server.get("transport_environment", {}).get("MOONCAKE_PROTOCOL")
        == "rdma"
        and server.get("transport_environment", {}).get("WITH_NVIDIA_PEERMEM") == "0",
        "wave_size_64": int(server.get("wave_size", -1))
        == int(workflow["measurement_contract"]["wave_size"]),
        "no_training": summary.get("training_performed") is False,
        "no_checkpoint_or_blind_eval": summary.get("checkpoint_loaded") is False
        and summary.get("blind_evaluation_performed") is False,
        "no_other_models_or_curve": summary.get("other_models_evaluated") is False
        and summary.get("physical_curve_measured") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(
            {
                "checks": checks,
                "manifest": manifest,
                "max_hfull_bytes_formula_relative_error": max(bytes_formula_errors, default=None),
            }
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "workflow_commit": summary["workflow_commit"],
                "gpu_requests": state["counts"]["gpu_sentinel_requests"],
                "gpu_waves": state["counts"]["gpu_sentinel_waves"],
                "development_profiles": len(examples),
                "blind_features": len(blind),
                "blind_targets": 0,
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
