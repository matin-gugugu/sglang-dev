#!/usr/bin/env python3
"""Validate Phase39 raw shards, build physical curves, and run CPU placement analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from common import environment_record, load_json, repo_root, sha256, utc_now, write_json
from analysis import (
    aggregate_cost_metrics,
    combined_cost_rows,
    compare_histogram_metrics,
    compare_phase35,
    cost_phase_rows,
    frozen_histogram_metrics,
    input_rows,
    make_figure,
    placement_decisions,
    read_csv,
    vector,
    write_csv,
)
from contracts import validate_plan
from finalize import finalize
from measurement import build_curves, validate_raw
from preflight import static_checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--topology-plan", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation")
    args = parser.parse_args()
    spec = load_json(HERE / "experiment.json")
    output = args.output_dir.resolve()
    expected_output = (repo_root() / spec["result_dir"]).resolve()
    if output != expected_output:
        raise RuntimeError(f"Phase39正式结果目录不可修改：{output}")
    if output.exists():
        raise RuntimeError(f"正式结果目录已存在，拒绝覆盖：{output}")
    static = static_checks(args.expected_workflow_commit)
    plan_path = args.topology_plan.expanduser().resolve()
    plan = load_json(plan_path)
    plan_audit = validate_plan(plan, spec)
    if plan["workflow_commit"] != static["workflow_commit"]:
        raise RuntimeError("topology plan不是从当前W39冻结")
    preflight_path = args.preflight_audit.expanduser().resolve()
    preflight = load_json(preflight_path)
    raw_dir = args.raw_dir.expanduser().resolve()
    if (
        preflight.get("workflow_commit") != static["workflow_commit"]
        or preflight.get("topology_plan_sha256") != plan["plan_sha256"]
        or Path(preflight.get("external_raw_dir", "")).resolve() != raw_dir
    ):
        raise RuntimeError("preflight audit与W39/plan/raw目录不一致")
    raw_audit = validate_raw(plan, raw_dir, spec)
    if not raw_audit["complete"]:
        raise RuntimeError({
            "missing": raw_audit["missing"],
            "needs_extra_repeats": raw_audit["needs_extra_repeats"],
        })
    curves_payload, registry = build_curves(plan, raw_audit, spec)
    curves = curves_payload["curves"]

    predictions, targets = input_rows(ROOT, spec)
    prediction_ids = {row["example_id"] for row in predictions}
    target_ids = {row["example_id"] for row in targets}
    phase_rows, interpolation_audit = cost_phase_rows(predictions, targets, curves)
    total_rows = combined_cost_rows(phase_rows)
    cost_metrics = aggregate_cost_metrics([*phase_rows, *total_rows])
    histogram_metrics = frozen_histogram_metrics(predictions, targets, spec["phase34_bin_edges_bytes"])
    pinned_paths = {item["name"]: ROOT / item["path"] for item in spec["pinned_inputs"]}
    histogram_audit = compare_histogram_metrics(histogram_metrics, read_csv(pinned_paths["phase34d_histogram_metrics"]))
    proxy_comparison = compare_phase35(cost_metrics, read_csv(pinned_paths["phase35_cost_metrics"]))
    ranking_rows, decision_rows, decision_metrics = placement_decisions(total_rows)

    scalar_max_relative = 0.0
    for row in predictions:
        for vector_name, total_name in (
            ("predicted_calls_by_12bin_json", "predicted_total_calls_per_1000"),
            ("predicted_logical_bytes_by_12bin_json", "predicted_total_logical_bytes_per_1000"),
        ):
            scalar_max_relative = max(scalar_max_relative, abs(sum(vector(row, vector_name)) - float(row[total_name])) / max(abs(float(row[total_name])), 1.0))
    for row in targets:
        for vector_name, total_name in (
            ("target_calls_by_12bin_json", "target_total_calls_per_1000"),
            ("target_logical_bytes_by_12bin_json", "target_total_logical_bytes_per_1000"),
        ):
            scalar_max_relative = max(scalar_max_relative, abs(sum(vector(row, vector_name)) - float(row[total_name])) / max(abs(float(row[total_name])), 1.0))

    phase35_registry = load_json(pinned_paths["phase35_curve_registry"])
    old_l2_l3 = [
        row for row in phase35_registry["curves"]
        if row.get("parallelism") in {"tp", "pp"}
        and ("l2_" in row.get("placement_id", "") or "l3_" in row.get("placement_id", ""))
    ]
    proxy_boundary_ok = (
        "Only TP L1 is a physical measurement" in phase35_registry.get("boundary", "")
        and len(old_l2_l3) == 4
        and all(row.get("evidence") == "not_physical_measurement" for row in old_l2_l3)
    )
    matrix_pairs = {(curve["parallelism"], curve["topology_level"], int(curve["group_size"])) for curve in curves}
    expected_pairs = {(row["parallelism"], row["topology_level"], int(row["world_size"])) for row in spec["required_measurement_matrix"]}
    all_costs_finite = all(
        math.isfinite(float(row[field])) and float(row[field]) >= 0
        for row in [*phase_rows, *total_rows]
        for field in (
            "predicted_cost_us_per_1000", "teacher_cost_us_per_1000", "absolute_error_us_per_1000",
            "predicted_cost_lower_us_per_1000", "predicted_cost_upper_us_per_1000",
            "teacher_cost_lower_us_per_1000", "teacher_cost_upper_us_per_1000",
        )
    )
    checks = {
        "static_preflight_pass": static["status"] == "PASS",
        "topology_plan_valid": plan_audit["ok"],
        "topology_plan_frozen_before_raw": plan.get("classification_frozen_before_measurement") is True,
        "topology_labels_not_inferred_from_benchmark": plan.get("classification_not_inferred_from_benchmark") is True,
        "raw_complete": raw_audit["complete"],
        "raw_not_committed_to_git": raw_audit["raw_manifest"]["raw_committed_to_git"] is False,
        "measurement_shards_24": len(plan["measurements"]) == int(spec["expected_measurement_shards"]),
        "physical_curves_12": len(curves) == int(spec["expected_physical_curves"]),
        "physical_curve_matrix_exact": matrix_pairs == expected_pairs,
        "all_curve_evidence_physical": all(curve["curve_evidence"] == "physical_measurement" for curve in curves),
        "prediction_phase_rows_2592": len(predictions) == int(spec["expected_prediction_phase_rows"]),
        "target_phase_rows_2592": len(targets) == int(spec["expected_target_phase_rows"]),
        "prediction_ids_unique": len(prediction_ids) == len(predictions),
        "target_ids_unique": len(target_ids) == len(targets),
        "prediction_target_ids_exact": prediction_ids == target_ids,
        "frozen_histogram_metrics_reproduced": histogram_audit["ok"],
        "frozen_histogram_metric_rows_84": len(histogram_metrics) == int(spec["expected_histogram_metric_rows"]),
        "saved_totals_match_vectors": scalar_max_relative <= 1e-12,
        "phase_cost_rows_7776": len(phase_rows) == int(spec["expected_phase_cost_rows"]),
        "total_cost_rows_3888": len(total_rows) == int(spec["expected_total_cost_rows"]),
        "cost_metric_rows_234": len(cost_metrics) == int(spec["expected_cost_metric_rows"]),
        "placement_ranking_rows_3888": len(ranking_rows) == int(spec["expected_placement_ranking_rows"]),
        "placement_decision_rows_1296": len(decision_rows) == int(spec["expected_placement_decision_rows"]),
        "decision_metric_rows_24": len(decision_metrics) == int(spec["expected_decision_metric_rows"]),
        "proxy_comparison_rows_180": len(proxy_comparison) == int(spec["expected_proxy_comparison_rows"]),
        "phase35_proxy_boundary_preserved": proxy_boundary_ok,
        "all_costs_finite_nonnegative": all_costs_finite,
        "parallel_configuration_never_selected": all(row["ranking_scope"] == "communication_only_fixed_parallel_configuration" for row in decision_rows),
        "no_training_checkpoint_inference_or_prediction_recompute": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"phase39_runtime_checks": checks})

    for name in ("contracts", "curves", "audit", "analysis", "figures", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    write_json(output / "contracts/experiment.json", spec)
    write_json(output / "contracts/topology_plan.json", plan)
    write_json(output / "contracts/physical_curve_registry.json", registry)
    write_json(output / "curves/physical_curves.json", curves_payload)
    write_json(output / "audit/input_freeze.json", {
        "schema_version": "phase39-input-freeze-v1",
        "workflow_commit": static["workflow_commit"],
        "workflow_parent_result_commit": static["workflow_parent_result_commit"],
        "static_pinned_inputs": static["pinned_inputs"],
        "phase38": static["phase38"],
        "source_semantics": static["source_semantics"],
        "topology_plan_sha256": plan["plan_sha256"],
        "preflight_audit_sha256": sha256(preflight_path),
    })
    write_json(output / "audit/environment.json", {
        **environment_record(),
        "analysis_gpu_used": False,
        "training_performed": False,
        "checkpoint_loaded": False,
        "prediction_recomputation_performed": False,
        "measurement_hosts": sorted({rank["host"] for measurement in plan["measurements"] for rank in measurement["ranks"]}),
        "raw_torch_versions": sorted({str(row.get("environment", {}).get("torch")) for row in raw_audit["records"]}),
        "raw_cuda_versions": sorted({str(row.get("environment", {}).get("cuda")) for row in raw_audit["records"]}),
        "raw_nccl_versions_json": sorted({json.dumps(row.get("environment", {}).get("nccl"), sort_keys=True) for row in raw_audit["records"]}),
    })
    write_json(output / "audit/raw_manifest.json", raw_audit["raw_manifest"])
    quality_output = {key: value for key, value in raw_audit.items() if key not in {"records", "repeat_values", "raw_manifest"}}
    write_json(output / "audit/measurement_quality.json", quality_output)
    write_json(output / "audit/interpolation_audit.json", interpolation_audit)
    write_json(output / "audit/histogram_invariance.json", histogram_audit)
    write_csv(output / "analysis/phase_costs.csv.gz", phase_rows)
    write_csv(output / "analysis/combined_costs.csv.gz", total_rows)
    write_csv(output / "analysis/cost_metrics.csv", cost_metrics)
    write_csv(output / "analysis/frozen_histogram_metrics.csv", histogram_metrics)
    write_csv(output / "analysis/physical_vs_phase35.csv", proxy_comparison)
    write_csv(output / "analysis/placement_rankings.csv.gz", ranking_rows)
    write_csv(output / "analysis/placement_decisions.csv.gz", decision_rows)
    write_csv(output / "analysis/placement_decision_metrics.csv", decision_metrics)
    make_figure(output / "figures/placement_regret.svg", decision_metrics)

    cost_headline = [row for row in cost_metrics if row["phase"] == "total" and row["slice_type"] == "overall"]
    decision_headline = [row for row in decision_metrics if row["slice_type"] in {"overall", "parallelism"}]
    runtime_state = {
        "schema_version": "phase39-runtime-state-v1",
        "workflow_commit": static["workflow_commit"],
        "counts": {
            "measurement_cases": len(spec["required_measurement_matrix"]),
            "measurement_shards": len(plan["measurements"]),
            "raw_files": raw_audit["raw_manifest"]["file_count"],
            "raw_records": raw_audit["raw_manifest"]["record_count"],
            "physical_curves": len(curves),
            "prediction_phase_rows": len(predictions), "target_phase_rows": len(targets),
            "phase_cost_rows": len(phase_rows), "total_cost_rows": len(total_rows),
            "cost_metric_rows": len(cost_metrics), "frozen_histogram_metric_rows": len(histogram_metrics),
            "proxy_comparison_rows": len(proxy_comparison), "placement_ranking_rows": len(ranking_rows),
            "placement_decision_rows": len(decision_rows), "decision_metric_rows": len(decision_metrics),
        },
        "cost_headline": cost_headline,
        "decision_headline": decision_headline,
        "histogram_invariance": histogram_audit,
        "final_high_runtime_variance": raw_audit["final_high_variance"],
        "high_cross_replica_spread": registry["high_cross_replica_spread"],
        "scalar_max_relative_difference_vs_saved_totals": scalar_max_relative,
        "checks": checks,
    }
    write_json(output / "audit/runtime_state.json", runtime_state)
    write_json(output / "logs/runtime.log", {
        "event": "phase39_tp_pp_l1_l3_physical_placement_validation_complete",
        "completed_at_utc": utc_now(),
        "workflow_commit": static["workflow_commit"],
        "topology_plan_sha256": plan["plan_sha256"],
        "raw_files": raw_audit["raw_manifest"]["file_count"],
        "raw_records": raw_audit["raw_manifest"]["record_count"],
        "training_performed": False,
        "prediction_recomputation_performed": False,
    })
    summary = finalize(output)
    print(json.dumps({
        "status": summary["status"], "output": str(output),
        "workflow_commit": static["workflow_commit"],
        "curves": len(curves), "cost_headline": cost_headline,
        "decision_headline": decision_headline,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
