#!/usr/bin/env python3
"""Independent compact-result verifier for Phase63."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import load_json, repo_root, sha256, verify_result_manifest  # noqa: E402
from contracts import payload_pairs, selected_layouts, validate_pair_contract, validate_plan  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def expected_decision(metrics: list[dict[str, str]], spec: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    overall = next(row for row in metrics if row["slice_type"] == "overall")
    models = [row for row in metrics if row["slice_type"] == "model"]
    fine = [row for row in metrics if row["slice_type"] == "model_configuration_topology"]
    gate = spec["external_validation_gate"]
    prior = load_json(repo_root() / "experiment-results/phase62_pd_contention_fresh_blind/summary.json")
    checks = {
        "four_model_overall_wape": float(overall["corrected_wape"]) <= float(gate["four_model_overall_wape_max"]),
        "each_model_overall_wape": all(float(row["corrected_wape"]) <= float(gate["each_model_overall_wape_max"]) for row in models),
        "each_model_configuration_topology_wape": all(float(row["corrected_wape"]) <= float(gate["each_model_configuration_topology_wape_max"]) for row in fine),
        "four_model_overall_signed_bias": abs(float(overall["corrected_signed_bias"])) <= float(gate["four_model_overall_absolute_signed_bias_max"]),
        "each_model_overall_signed_bias": all(abs(float(row["corrected_signed_bias"])) <= float(gate["each_model_overall_absolute_signed_bias_max"]) for row in models),
        "each_model_configuration_topology_signed_bias": all(abs(float(row["corrected_signed_bias"])) <= float(gate["each_model_configuration_topology_absolute_signed_bias_max"]) for row in fine),
        "all_predictions_positive": True,
        "strictly_improves_uncorrected_phase51_max_overall": float(overall["corrected_wape"]) < float(overall["phase51_wape"]),
        "strictly_improves_uncorrected_phase51_max_each_model": all(float(row["corrected_wape"]) < float(row["phase51_wape"]) for row in models),
        "phase62_two_model_gate_already_passed": prior.get("scientific_outcome") == "FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_PASS" and prior.get("decision", {}).get("fresh_blind_gate_pass") is True and prior.get("training_performed") is False and prior.get("recalibration_performed") is False,
    }
    passed = all(checks.values())
    return ("FROZEN_CONTENTION_CORRECTION_SIX_MODEL_PASS" if passed else "FROZEN_CONTENTION_CORRECTION_FOUR_MODEL_EXTERNAL_FAIL", checks)


def verify(output: Path) -> dict[str, Any]:
    expected = load_json(HERE / "expected_outputs.json")
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    manifest = verify_result_manifest(output)
    spec = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    plan = load_json(output / "contracts/topology_plan.json")
    plan_audit = validate_plan(plan)
    layouts = load_json(output / "contracts/selected_model_transfer_layouts.json")["layouts"]
    pair_grid = load_json(output / "contracts/external_payload_pair_grid.json")
    frozen_model = load_json(output / "contracts/frozen_contention_correction.json")
    source_model = repo_root() / spec["frozen_correction_contract"]["source"]
    evidence = load_json(output / "evidence/four_model_external_points.json")
    summary = load_json(output / "summary.json")
    raw = load_json(output / "audit/external_raw_manifest.json")
    quality = load_json(output / "audit/measurement_quality.json")
    environment = load_json(output / "audit/environment.json")
    freeze = load_json(output / "audit/input_freeze.json")
    points = read_csv(output / "analysis/four_model_external_points.csv")
    metrics = read_csv(output / "analysis/four_model_external_metrics.csv")
    combined = read_csv(output / "analysis/combined_six_model_metrics.csv")
    replicas = read_csv(output / "analysis/replica_points.csv")
    spreads = read_csv(output / "analysis/replica_spread.csv")
    pair_audit = validate_pair_contract(spec)
    outcome, gate_checks = expected_decision(metrics, spec)
    runtime_count = len(quality.get("final_runtime_variance", []))
    placement_count = int(quality.get("placement_points_above_threshold", -1))
    expected_status = (
        "PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE" if runtime_count and placement_count
        else "PASS_WITH_RUNTIME_VARIANCE" if runtime_count
        else "PASS_WITH_PLACEMENT_VARIANCE" if placement_count
        else "PASS"
    )
    repeat_counts = {row["measurement_id"]: int(row["repeat_count"]) for row in quality.get("measurements", [])}
    expected_raw_records = sum(repeat_counts.values()) * 10
    expected_pair_ids = {row["pair_id"] for model in spec["selected_models"] for row in payload_pairs(model)}
    expected_models = set(spec["selected_models"])
    result_pair_ids = {row["pair_id"] for row in points}
    checks = {
        "manifest": manifest["ok"],
        "required_exact": files == set(expected["required"]),
        "status": summary.get("status") == expected_status and summary.get("status") in spec["accepted_result_statuses"] and (output / "DONE").read_text().strip() == expected_status,
        "contract_exact": result_contract == spec,
        "plan": plan_audit["measurements"] == 48 and plan.get("workflow_commit") == summary.get("workflow_commit") and plan_audit["endpoint_slots"] == 24,
        "layouts": layouts == selected_layouts(spec),
        "pair_grid": pair_grid.get("grid_sha256") == pair_audit["grid_sha256"] and pair_grid.get("models") == {model: payload_pairs(model) for model in spec["selected_models"]} and pair_grid.get("selection_uses_phase63_concurrent_targets") is False,
        "frozen_model": frozen_model == load_json(source_model) and sha256(source_model) == spec["frozen_correction_contract"]["sha256"] and freeze.get("frozen_model_file_sha256") == spec["frozen_correction_contract"]["sha256"],
        "point_counts": len(points) == 240 and len(replicas) == 480 and len(spreads) == 240 and len(evidence.get("points", [])) == 240 and len(evidence.get("replica_points", [])) == 480,
        "models_and_pairs": {row["model_id"] for row in points} == expected_models and result_pair_ids == expected_pair_ids and not (expected_models & set(spec["prior_validated_models"])),
        "metric_counts": len(metrics) == 34 and sum(row["slice_type"] == "overall" for row in metrics) == 1 and sum(row["slice_type"] == "model" for row in metrics) == 4 and sum(row["slice_type"] == "model_configuration_topology" for row in metrics) == 24,
        "combined_metric_counts": len(combined) == 7 and sum(row["slice_type"] == "six_model_overall" and int(row["points"]) == 360 for row in combined) == 1 and {row["slice_value"] for row in combined if row["slice_type"] == "six_model_model"} == set(spec["prior_validated_models"] + spec["selected_models"]),
        "positive_values": all(float(row[key]) > 0 for row in points for key in ("phase51_max_us", "frozen_corrected_us", "actual_concurrent_wave_us")),
        "official_replica_policy": all(math.isclose(float(row["official_us"]), max(float(row["replica0_us"]), float(row["replica1_us"])), rel_tol=1e-12, abs_tol=1e-7) for row in spreads),
        "decision": summary.get("scientific_outcome") == outcome and summary.get("decision", {}).get("scientific_outcome") == outcome and summary.get("decision", {}).get("checks") == gate_checks and summary.get("decision", {}).get("four_model_external_gate_pass") == all(gate_checks.values()) and summary.get("decision", {}).get("six_model_evidence_gate_pass") == all(gate_checks.values()),
        "raw_external": raw.get("raw_committed_to_git") is False and raw.get("counts", {}).get("measurements_with_data") == 48 and raw.get("counts", {}).get("expected_measurements") == 48 and raw.get("counts", {}).get("files") == sum(repeat_counts.values()) and raw.get("counts", {}).get("records") == expected_raw_records and len(raw.get("files", [])) == raw.get("counts", {}).get("files") and all(row.get("path", "").endswith(".jsonl") and len(row.get("sha256", "")) == 64 and row.get("records") == 10 for row in raw.get("files", [])),
        "quality": len(quality.get("measurements", [])) == 48 and all(int(row.get("repeat_count", 0)) in (5, 7, 9) for row in quality.get("measurements", [])),
        "runtime_endpoints": len(environment.get("gpu_measurement_runtime_endpoints", [])) == 48 and all(len(row.get("endpoints", [])) == 3 and all(endpoint.get("mooncake_protocol") == "rdma" and endpoint.get("with_nvidia_peermem") == "0" for endpoint in row.get("endpoints", [])) for row in environment.get("gpu_measurement_runtime_endpoints", [])),
        "resource_contract": summary.get("counts", {}).get("world_size_per_shard") == 3 and summary.get("counts", {}).get("maximum_simultaneous_nodes_per_shard") == 2 and summary.get("counts", {}).get("global_peak_simultaneous_nodes") == 2 and summary.get("counts", {}).get("global_peak_simultaneous_gpu_processes") == 3 and summary.get("counts", {}).get("maximum_concurrent_measurement_shards") == 1 and summary.get("counts", {}).get("single_scheduler_allocation_for_all_topologies_required") is False and summary.get("counts", {}).get("endpoint_slots") == 24 and summary.get("counts", {}).get("measurement_shards") == 48,
        "no_raw": not list(output.rglob("*.jsonl")),
        "scope": summary.get("training_performed") is False and summary.get("recalibration_performed") is False and summary.get("model_weights_loaded") is False and summary.get("inference_performed") is False and summary.get("histograms_recomputed") is False and summary.get("phase63_labels_used_for_fitting") is False and freeze.get("threshold_tuning_performed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"phase63_checks": checks, "manifest": manifest})
    overall = next(row for row in metrics if row["slice_type"] == "overall")
    combined_overall = next(row for row in combined if row["slice_type"] == "six_model_overall")
    return {
        "status": "PASS",
        "checks": checks,
        "workflow_commit": summary["workflow_commit"],
        "result_status": summary["status"],
        "scientific_outcome": outcome,
        "four_model_phase51_max_overall_wape": float(overall["phase51_wape"]),
        "four_model_frozen_corrected_overall_wape": float(overall["corrected_wape"]),
        "combined_six_model_frozen_corrected_overall_wape": float(combined_overall["corrected_wape"]),
        "points": 240,
        "manifest_files": manifest["manifest"]["checked_files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase63_pd_contention_four_model_external")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
