#!/usr/bin/env python3
"""Independent verifier for Phase39 compact result tree."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest
from analysis import read_csv
from contracts import validate_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    expected = load_json(HERE / "expected_outputs.json")
    missing = [relative for relative in expected["required"] if not (output / relative).is_file()]
    if missing:
        raise RuntimeError({"missing_required_outputs": missing})
    manifest = verify_result_manifest(output)
    summary = load_json(output / "summary.json")
    state = load_json(output / "audit/runtime_state.json")
    freeze = load_json(output / "audit/input_freeze.json")
    raw_manifest = load_json(output / "audit/raw_manifest.json")
    quality = load_json(output / "audit/measurement_quality.json")
    workflow_contract = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    plan = load_json(output / "contracts/topology_plan.json")
    plan_audit = validate_plan(plan, workflow_contract)
    curves_payload = load_json(output / "curves/physical_curves.json")
    registry = load_json(output / "contracts/physical_curve_registry.json")
    phase_rows = read_csv(output / "analysis/phase_costs.csv.gz")
    total_rows = read_csv(output / "analysis/combined_costs.csv.gz")
    metrics = read_csv(output / "analysis/cost_metrics.csv")
    histogram_metrics = read_csv(output / "analysis/frozen_histogram_metrics.csv")
    proxy = read_csv(output / "analysis/physical_vs_phase35.csv")
    rankings = read_csv(output / "analysis/placement_rankings.csv.gz")
    decisions = read_csv(output / "analysis/placement_decisions.csv.gz")
    decision_metrics = read_csv(output / "analysis/placement_decision_metrics.csv")
    done = (output / "DONE").read_text(encoding="utf-8").strip()
    curves = curves_payload.get("curves", [])
    payload_grid = [int(value) for value in workflow_contract["measurement_contract"]["payload_bytes"]]
    curve_matrix = {(row.get("parallelism"), row.get("topology_level"), int(row.get("group_size", 0))) for row in curves}
    expected_matrix = {(row["parallelism"], row["topology_level"], int(row["world_size"])) for row in workflow_contract["required_measurement_matrix"]}
    curve_contract = all(
        row.get("curve_evidence") == "physical_measurement"
        and [int(knot["payload_bytes"]) for knot in row.get("knots", [])] == payload_grid
        and all(
            math.isfinite(float(knot[field])) and float(knot[field]) > 0
            for knot in row["knots"]
            for field in ("official_latency_us", "lower_latency_us", "upper_latency_us")
        )
        and all(float(knot["lower_latency_us"]) <= float(knot["official_latency_us"]) <= float(knot["upper_latency_us"]) for knot in row["knots"])
        and all(len(knot.get("replicas", [])) == int(workflow_contract["minimum_placement_replicas_per_case"]) for knot in row["knots"])
        for row in curves
    )
    checks = {
        "manifest": manifest["ok"],
        "summary_status_accepted": summary.get("status") in workflow_contract["accepted_result_statuses"],
        "done_matches": done == summary.get("status"),
        "result_contract_matches_workflow": result_contract == workflow_contract,
        "runtime_checks_pass": all(state.get("checks", {}).values()),
        "summary_counts_match_state": summary.get("counts") == state.get("counts"),
        "plan_valid": plan_audit["ok"],
        "plan_sha_consistent": plan.get("plan_sha256") == curves_payload.get("topology_plan_sha256") == registry.get("topology_plan_sha256"),
        "plan_workflow_commit": plan.get("workflow_commit") == summary.get("workflow_commit"),
        "measurement_shards_24": len(plan.get("measurements", [])) == int(workflow_contract["expected_measurement_shards"]),
        "raw_manifest_external_only": raw_manifest.get("raw_committed_to_git") is False,
        "raw_manifest_nonempty": int(raw_manifest.get("file_count", 0)) >= int(workflow_contract["expected_measurement_shards"]) * int(workflow_contract["measurement_contract"]["minimum_independent_repeats"]),
        "raw_quality_complete": quality.get("complete") is True and not quality.get("missing") and not quality.get("needs_extra_repeats"),
        "curves_12": len(curves) == int(workflow_contract["expected_physical_curves"]),
        "curve_matrix_exact": curve_matrix == expected_matrix,
        "curve_contract": curve_contract,
        "registry_all_physical": len(registry.get("curves", [])) == len(curves) and all(row.get("evidence") == "physical_measurement" for row in registry.get("curves", [])),
        "phase_rows": len(phase_rows) == int(workflow_contract["expected_phase_cost_rows"]),
        "total_rows": len(total_rows) == int(workflow_contract["expected_total_cost_rows"]),
        "metric_rows": len(metrics) == int(workflow_contract["expected_cost_metric_rows"]),
        "histogram_rows": len(histogram_metrics) == int(workflow_contract["expected_histogram_metric_rows"]),
        "proxy_rows": len(proxy) == int(workflow_contract["expected_proxy_comparison_rows"]),
        "ranking_rows": len(rankings) == int(workflow_contract["expected_placement_ranking_rows"]),
        "decision_rows": len(decisions) == int(workflow_contract["expected_placement_decision_rows"]),
        "decision_metric_rows": len(decision_metrics) == int(workflow_contract["expected_decision_metric_rows"]),
        "cost_rows_physical": all(row.get("curve_evidence") == "physical_measurement" and row.get("topology_level") in {"L1", "L2", "L3"} for row in phase_rows),
        "costs_finite_nonnegative": all(
            math.isfinite(float(row[field])) and float(row[field]) >= 0
            for row in [*phase_rows, *total_rows]
            for field in ("predicted_cost_us_per_1000", "teacher_cost_us_per_1000", "absolute_error_us_per_1000")
        ),
        "decisions_fixed_parallel_configuration": all(row.get("ranking_scope") == "communication_only_fixed_parallel_configuration" for row in decisions),
        "histogram_invariance": state.get("histogram_invariance", {}).get("ok") is True,
        "phase38_parent_frozen": freeze.get("workflow_parent_result_commit") == workflow_contract["workflow_parent_result_commit"],
        "all_static_pins_valid": all(row.get("ok") is True for row in freeze.get("static_pinned_inputs", {}).values()),
        "no_training": summary.get("training_performed") is False,
        "no_checkpoint": summary.get("checkpoint_loaded") is False,
        "no_prediction_recompute": summary.get("prediction_recomputation_performed") is False,
        "diagnostic_not_gate": summary.get("diagnostic_reference_is_pass_fail_gate") is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "manifest": manifest})
    print(json.dumps({
        "status": summary["status"],
        "output": str(output),
        "workflow_commit": summary["workflow_commit"],
        "measurements": len(plan["measurements"]),
        "raw_files": raw_manifest["file_count"],
        "raw_records": raw_manifest["record_count"],
        "curves": len(curves),
        "phase_cost_rows": len(phase_rows),
        "total_cost_rows": len(total_rows),
        "decision_rows": len(decisions),
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
