#!/usr/bin/env python3
"""Independent compact-result verifier for Phase62."""
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
from contracts import canonical_sha, development_pair_ids, payload_pairs, selected_layouts, validate_pair_contract, validate_plan  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.lower() == "true"


def expected_decision(metrics: list[dict[str, str]], contract: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    overall = next(row for row in metrics if row["slice_type"] == "overall")
    slices = [row for row in metrics if row["slice_type"] == "configuration_topology"]
    gate = contract["fresh_blind_acceptance_gate"]
    checks = {
        "corrected_overall_wape": float(overall["corrected_wape"]) <= float(gate["corrected_overall_wape_max"]),
        "corrected_configuration_topology_wape": all(float(row["corrected_wape"]) <= float(gate["corrected_each_configuration_topology_wape_max"]) for row in slices),
        "corrected_overall_signed_bias": abs(float(overall["corrected_signed_bias"])) <= float(gate["corrected_overall_absolute_signed_bias_max"]),
        "corrected_configuration_topology_signed_bias": all(abs(float(row["corrected_signed_bias"])) <= float(gate["corrected_each_configuration_topology_absolute_signed_bias_max"]) for row in slices),
        "all_predictions_positive": True,
        "strictly_improves_uncorrected_phase51_max": float(overall["corrected_wape"]) < float(overall["phase51_wape"]),
    }
    passed = all(checks.values())
    return ("FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_PASS" if passed else "FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_FAIL", checks)


def verify(output: Path) -> dict[str, Any]:
    expected = load_json(HERE / "expected_outputs.json")
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    manifest = verify_result_manifest(output)
    contract = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    plan = load_json(output / "contracts/topology_plan.json")
    plan_audit = validate_plan(plan)
    layouts = load_json(output / "contracts/selected_model_transfer_layouts.json")["layouts"]
    pair_grid = load_json(output / "contracts/reserved_payload_pair_grid.json")
    frozen_model = load_json(output / "contracts/frozen_contention_correction.json")
    source_model = repo_root() / contract["frozen_correction_contract"]["source"]
    evidence = load_json(output / "evidence/fresh_blind_points.json")
    summary = load_json(output / "summary.json")
    raw = load_json(output / "audit/external_raw_manifest.json")
    quality = load_json(output / "audit/measurement_quality.json")
    environment = load_json(output / "audit/environment.json")
    freeze = load_json(output / "audit/input_freeze.json")
    points = read_csv(output / "analysis/fresh_blind_points.csv")
    metrics = read_csv(output / "analysis/fresh_blind_metrics.csv")
    replicas = read_csv(output / "analysis/replica_points.csv")
    spreads = read_csv(output / "analysis/replica_spread.csv")
    pair_audit = validate_pair_contract(contract)
    outcome, gate_checks = expected_decision(metrics, contract)
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
    reserved_ids = {row["pair_id"] for model in contract["selected_models"] for row in payload_pairs(model)}
    development_ids = development_pair_ids()
    checks = {
        "manifest": manifest["ok"],
        "required_exact": files == set(expected["required"]),
        "status": summary.get("status") == expected_status and summary.get("status") in contract["accepted_result_statuses"] and (output / "DONE").read_text().strip() == expected_status,
        "contract_exact": result_contract == contract,
        "plan": plan_audit["measurements"] == 24 and plan.get("workflow_commit") == summary.get("workflow_commit") and plan_audit["fresh_endpoint_slots"] == 24,
        "layouts": layouts == selected_layouts(contract),
        "pair_grid": pair_grid.get("reserved_sha256") == pair_audit["reserved_sha256"] and pair_grid.get("development_pair_ids_sha256") == pair_audit["development_pair_ids_sha256"] and pair_grid.get("development_pairs_measured") == 0,
        "frozen_model": frozen_model == load_json(source_model) and sha256(source_model) == contract["frozen_correction_contract"]["sha256"] and freeze.get("frozen_model_file_sha256") == contract["frozen_correction_contract"]["sha256"],
        "point_counts": len(points) == 120 and len(replicas) == 240 and len(spreads) == 120 and len(evidence.get("points", [])) == 120 and len(evidence.get("replica_points", [])) == 240,
        "pair_roles": {row["pair_id"] for row in points} == reserved_ids and not ({row["pair_id"] for row in points} & development_ids),
        "metric_counts": len(metrics) == 14 and sum(row["slice_type"] == "overall" for row in metrics) == 1 and sum(row["slice_type"] == "configuration_topology" for row in metrics) == 6,
        "positive_values": all(float(row[key]) > 0 for row in points for key in ("phase51_max_us", "frozen_corrected_us", "actual_concurrent_wave_us")),
        "official_replica_policy": all(math.isclose(float(row["official_us"]), max(float(row["replica0_us"]), float(row["replica1_us"])), rel_tol=1e-12, abs_tol=1e-7) for row in spreads),
        "decision": summary.get("scientific_outcome") == outcome and summary.get("decision", {}).get("scientific_outcome") == outcome and summary.get("decision", {}).get("checks") == gate_checks and summary.get("decision", {}).get("fresh_blind_gate_pass") == all(gate_checks.values()),
        "raw_external": raw.get("raw_committed_to_git") is False and raw.get("counts", {}).get("measurements_with_data") == 24 and raw.get("counts", {}).get("files") == sum(repeat_counts.values()) and raw.get("counts", {}).get("records") == expected_raw_records and len(raw.get("files", [])) == raw.get("counts", {}).get("files") and all(row.get("path", "").endswith(".jsonl") and len(row.get("sha256", "")) == 64 and row.get("records") == 10 for row in raw.get("files", [])),
        "quality": len(quality.get("measurements", [])) == 24 and all(int(row.get("repeat_count", 0)) in (5, 7, 9) for row in quality.get("measurements", [])),
        "runtime_endpoints": len(environment.get("gpu_measurement_runtime_endpoints", [])) == 24 and all(row.get("freshness", {}).get("all_four_endpoint_tuples_fresh") is True and len(row.get("endpoints", [])) == 3 and all(endpoint.get("mooncake_protocol") == "rdma" and endpoint.get("with_nvidia_peermem") == "0" for endpoint in row.get("endpoints", [])) for row in environment.get("gpu_measurement_runtime_endpoints", [])),
        "resource_contract": summary.get("counts", {}).get("world_size_per_shard") == 3 and summary.get("counts", {}).get("maximum_simultaneous_nodes_per_shard") == 2 and summary.get("counts", {}).get("fresh_endpoint_slots") == 24 and all(int(summary.get("counts", {}).get("new_host_signatures", {}).get(level, 0)) >= 1 for level in ("L1", "L2", "L3")),
        "no_raw": not list(output.rglob("*.jsonl")),
        "scope": summary.get("training_performed") is False and summary.get("recalibration_performed") is False and summary.get("model_weights_loaded") is False and summary.get("inference_performed") is False and summary.get("histograms_recomputed") is False and summary.get("blind_labels_used_for_fitting") is False and summary.get("counts", {}).get("development_pairs_measured") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError({"phase62_checks": checks, "manifest": manifest})
    overall = next(row for row in metrics if row["slice_type"] == "overall")
    return {
        "status": "PASS",
        "checks": checks,
        "workflow_commit": summary["workflow_commit"],
        "result_status": summary["status"],
        "scientific_outcome": outcome,
        "phase51_max_overall_wape": float(overall["phase51_wape"]),
        "frozen_corrected_overall_wape": float(overall["corrected_wape"]),
        "points": 120,
        "manifest_files": manifest["manifest"]["checked_files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase62_pd_contention_fresh_blind")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
