#!/usr/bin/env python3
"""Aggregate Phase63 raw and evaluate the frozen correction on four held-out models."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import environment_record, load_json, refresh_manifest, repo_root, require_clean_before_run, require_expected_head, utc_now, validate_result_tree, verify_pinned_inputs, write_json  # noqa: E402
from contracts import file_sha, payload_pairs, selected_layouts, validate_pair_contract, validate_plan  # noqa: E402
from measurement import build_external_analysis, validate_raw  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty Phase63 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def result_status(runtime_variance: bool, placement_variance: bool) -> str:
    if runtime_variance and placement_variance:
        return "PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE"
    if runtime_variance:
        return "PASS_WITH_RUNTIME_VARIANCE"
    if placement_variance:
        return "PASS_WITH_PLACEMENT_VARIANCE"
    return "PASS"


def run(expected: str, plan_path: Path, raw_dir: Path, preflight_path: Path, output: Path) -> dict[str, Any]:
    head = require_expected_head(expected)
    require_clean_before_run()
    spec = load_json(HERE / "experiment.json")
    pins = verify_pinned_inputs(spec)
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    plan_path = plan_path.expanduser().resolve()
    raw_dir = raw_dir.expanduser().resolve()
    preflight_path = preflight_path.expanduser().resolve()
    plan = load_json(plan_path)
    plan_audit = validate_plan(plan)
    if plan["workflow_commit"] != head:
        raise RuntimeError({"plan_workflow": plan["workflow_commit"], "HEAD": head})
    preflight = load_json(preflight_path)
    check_groups = [preflight.get(name, {}) for name in ("environment_checks", "module_checks", "runtime_checks")]
    if (
        preflight.get("status") != "PASS"
        or preflight.get("workflow_commit") != head
        or preflight.get("plan_audit", {}).get("plan_sha256") != plan["plan_sha256"]
        or preflight.get("plan_file_sha256") != file_sha(plan_path)
        or Path(preflight.get("raw_dir", "")).resolve() != raw_dir
        or preflight.get("environment", {}).get("declared_container_image") != spec["container_contract"]["image"]
        or preflight.get("frozen_model", {}).get("file_sha256") != spec["frozen_correction_contract"]["sha256"]
        or any(not values or not all(values.values()) for values in check_groups)
    ):
        raise RuntimeError("Phase63 preflight/plan/raw/model binding failed")
    raw = validate_raw(plan, raw_dir, require_complete=True)
    analysis = build_external_analysis(plan, raw)
    runtime_variance = bool(raw["final_runtime_variance"])
    placement_variance = any(row["above_threshold"] for row in analysis["spreads"])
    status = result_status(runtime_variance, placement_variance)
    pair_audit = validate_pair_contract(spec)
    output.mkdir(parents=True)
    write_json(output / "contracts/experiment.json", spec)
    write_json(output / "contracts/topology_plan.json", plan)
    write_json(output / "contracts/selected_model_transfer_layouts.json", {
        "schema_version": "phase63-selected-model-layouts-v1",
        "layouts": selected_layouts(spec),
    })
    write_json(output / "contracts/external_payload_pair_grid.json", {
        "schema_version": "phase63-external-payload-pair-grid-result-v1",
        "models": {model: payload_pairs(model) for model in spec["selected_models"]},
        "grid_sha256": pair_audit["grid_sha256"],
        "selection_frozen_before_phase63_raw": True,
        "selection_uses_phase63_concurrent_targets": False,
    })
    frozen_model = load_json(repo_root() / spec["frozen_correction_contract"]["source"])
    write_json(output / "contracts/frozen_contention_correction.json", frozen_model)
    write_json(output / "evidence/four_model_external_points.json", {
        "schema_version": "phase63-four-model-external-points-v1",
        "workflow_commit": head,
        "source_correction_result_commit": spec["source_correction_result_commit"],
        "source_two_model_validation_result_commit": spec["source_two_model_validation_result_commit"],
        "plan_sha256": plan["plan_sha256"],
        "points": analysis["points"],
        "replica_points": analysis["replica_points"],
    })
    write_csv(output / "analysis/four_model_external_points.csv", analysis["points"])
    write_csv(output / "analysis/four_model_external_metrics.csv", analysis["metrics"])
    write_csv(output / "analysis/combined_six_model_metrics.csv", analysis["combined_six_model_metrics"])
    write_csv(output / "analysis/replica_points.csv", analysis["replica_points"])
    write_csv(output / "analysis/replica_spread.csv", analysis["spreads"])
    write_json(output / "audit/input_freeze.json", {
        "schema_version": "phase63-input-freeze-v1",
        "workflow_commit": head,
        "workflow_parent_result_commit": spec["workflow_parent_result_commit"],
        "source_correction_result_commit": spec["source_correction_result_commit"],
        "source_two_model_validation_result_commit": spec["source_two_model_validation_result_commit"],
        "pinned_inputs": pins,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": file_sha(plan_path),
        "preflight_file_sha256": file_sha(preflight_path),
        "frozen_model_file_sha256": spec["frozen_correction_contract"]["sha256"],
        "external_pair_grid_sha256": pair_audit["grid_sha256"],
        "training_performed": False,
        "recalibration_performed": False,
        "threshold_tuning_performed": False,
        "phase63_labels_used_for_fitting": False,
    })
    write_json(output / "audit/external_raw_manifest.json", {
        "schema_version": "phase63-external-raw-manifest-v1",
        "raw_committed_to_git": False,
        "raw_root_recorded": str(raw_dir),
        "counts": raw["counts"],
        "files": raw["files"],
    })
    write_json(output / "audit/measurement_quality.json", {
        "schema_version": "phase63-measurement-quality-v1",
        "repeat_policy": spec["measurement_contract"],
        "measurements": raw["measurements"],
        "final_runtime_variance": raw["final_runtime_variance"],
        "placement_spread_threshold": spec["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"],
        "placement_points_above_threshold": sum(row["above_threshold"] for row in analysis["spreads"]),
    })
    safe_keys = ("rank", "role", "hostname", "expected_host", "physical_gpu", "visible_gpu", "gpu_name", "gpu_uuid", "ib_device", "mooncake_protocol", "with_nvidia_peermem", "torch", "cuda", "python")
    runtime_endpoints = []
    for measurement in plan["measurements"]:
        first = next(iter(next(iter(raw["records"][measurement["measurement_id"]].values()))))
        runtime_endpoints.append({
            "measurement_id": measurement["measurement_id"],
            "phase62_comparability": measurement["phase62_comparability"],
            "endpoints": [{key: endpoint.get(key) for key in safe_keys} for endpoint in first["runtime_endpoints"]],
        })
    write_json(output / "audit/environment.json", {
        "aggregation": environment_record(),
        "gpu_measurement_preflight": preflight,
        "gpu_measurement_runtime_endpoints": runtime_endpoints,
    })
    overall = next(row for row in analysis["metrics"] if row["slice_type"] == "overall")
    combined = next(row for row in analysis["combined_six_model_metrics"] if row["slice_type"] == "six_model_overall")
    summary = {
        "schema_version": "phase63-pd-contention-four-model-external-result-v1",
        "status": status,
        "scientific_outcome": analysis["decision"]["scientific_outcome"],
        "workflow_commit": head,
        "workflow_parent_result_commit": spec["workflow_parent_result_commit"],
        "source_correction_result_commit": spec["source_correction_result_commit"],
        "completed_at_utc": utc_now(),
        "counts": {
            "held_out_models": 4,
            "combined_validated_models": 6,
            "configurations": 2,
            "topology_levels": 3,
            "placement_replicas": 2,
            "measurement_shards": 48,
            "world_size_per_shard": 3,
            "maximum_simultaneous_nodes_per_shard": 2,
            "endpoint_slots": plan["placement_summary"]["endpoint_slots"],
            "exact_phase62_placements": plan["placement_summary"]["exact_phase62_placements"],
            "phase62_endpoint_slots_reused": plan["placement_summary"]["phase62_endpoint_slots_reused"],
            "official_external_points": len(analysis["points"]),
            "replica_points": len(analysis["replica_points"]),
            "combined_six_model_official_points": len(analysis["points"]) + 120,
            "raw_files": raw["counts"]["files"],
            "raw_records": raw["counts"]["records"],
        },
        "metrics": {
            "four_model_phase51_max_overall_wape": overall["phase51_wape"],
            "four_model_frozen_corrected_overall_wape": overall["corrected_wape"],
            "four_model_frozen_corrected_overall_signed_bias": overall["corrected_signed_bias"],
            "combined_six_model_phase51_max_overall_wape": combined["phase51_wape"],
            "combined_six_model_frozen_corrected_overall_wape": combined["corrected_wape"],
            "combined_six_model_frozen_corrected_overall_signed_bias": combined["corrected_signed_bias"],
        },
        "decision": analysis["decision"],
        "runtime_variance_measurements": len(raw["final_runtime_variance"]),
        "placement_variance_points": sum(row["above_threshold"] for row in analysis["spreads"]),
        "transport": "SGLang production MooncakeTransferEngine.batch_transfer_sync over RDMA/dma-buf",
        "training_performed": False,
        "recalibration_performed": False,
        "model_weights_loaded": False,
        "inference_performed": False,
        "histograms_recomputed": False,
        "phase63_concurrent_targets_opened_for_measurement": True,
        "phase63_labels_used_for_fitting": False,
        "proved": "external-model physical evaluation of the frozen R61 two-flow correction on four held-out KV layouts; combined with pinned R62 for six-model evidence",
        "not_proved": "P2D2, more than two flows, end-to-end latency, compute, memory, queueing, unrelated-job congestion, online arrivals or scheduling",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Phase63：四模型两流并发外部验证\n\n"
        f"执行状态：{status}；科学结论：{summary['scientific_outcome']}。"
        f"四个held-out模型未修正overall WAPE={overall['phase51_wape']:.6f}，冻结修正overall WAPE={overall['corrected_wape']:.6f}。"
        f"合并R62后的六模型overall WAPE={combined['corrected_wape']:.6f}。"
        "完成48个三rank shard；不训练、不调参、不加载模型权重，raw仅保存在Git外。\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir()
    (output / "logs/runtime.log").write_text(
        f"completed={utc_now()} workflow_commit={head}\n"
        f"status={status} scientific_outcome={summary['scientific_outcome']}\n"
        f"measurements=48 external_points=240 replica_points=480 raw_files={raw['counts']['files']} raw_records={raw['counts']['records']}\n"
        f"four_model_phase51_wape={overall['phase51_wape']} four_model_corrected_wape={overall['corrected_wape']} combined_six_model_corrected_wape={combined['corrected_wape']}\n"
        "training=false recalibration=false phase63_labels_used_for_fitting=false raw_committed=false\n",
        encoding="utf-8",
    )
    (output / "DONE").write_text(status + "\n", encoding="utf-8")
    refresh_manifest(output)
    tree = validate_result_tree(output)
    if not tree["ok"]:
        raise RuntimeError(tree)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--topology-plan", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase63_pd_contention_four_model_external")
    args = parser.parse_args()
    print(json.dumps(run(args.expected_workflow_commit, args.topology_plan, args.raw_dir, args.preflight_audit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
