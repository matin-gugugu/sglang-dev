#!/usr/bin/env python3
"""Run the CPU-only Phase71 multiflow cost and placement integration."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json  # noqa:E402
from analysis import build_analysis, read_csv, write_csv  # noqa:E402
from preflight import run_checks  # noqa:E402


def run(expected: str, output: Path) -> dict:
    started = time.monotonic()
    preflight = run_checks(expected)
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    root = repo_root()
    spec = load_json(HERE / "experiment.json")
    pins = {row["name"]: root / row["path"] for row in spec["pinned_inputs"]}
    predictions = read_csv(pins["phase49_frozen_predictions"])
    targets = read_csv(pins["phase50_hfull_targets"])
    curves = load_json(pins["phase51_curves"])["curves"]
    layouts = load_json(pins["phase51_layouts"])["layouts"]
    result = build_analysis(predictions, targets, curves, layouts, load_json(pins["r61_model"]), load_json(pins["r67_model"]), load_json(pins["r69_model"]), spec)
    output.mkdir(parents=True)
    write_json(output / "contracts/experiment.json", spec)
    write_json(output / "contracts/wave_policies.json", load_json(HERE / "wave_policies.json"))
    write_json(output / "contracts/coverage_matrix.json", load_json(HERE / "coverage_matrix.json"))
    write_csv(output / "analysis/unit_configuration_topology_costs.csv.gz", result["costs"])
    write_csv(output / "analysis/cost_metrics.csv", result["cost_metrics"])
    write_csv(output / "analysis/cost_method_comparison.csv", result["cost_comparison"])
    write_csv(output / "analysis/placement_decisions.csv.gz", result["decisions"])
    write_csv(output / "analysis/placement_metrics.csv", result["placement_metrics"])
    write_csv(output / "analysis/placement_method_comparison.csv", result["placement_comparison"])
    write_csv(output / "analysis/wave_sensitivity.csv", result["wave_sensitivity"])
    write_json(output / "audit/input_freeze.json", {"schema_version": "phase71-input-freeze-v1", "workflow_commit": preflight["workflow_commit"], "source_result_commit": spec["workflow_parent_result_commit"], "pinned_inputs": preflight["pinned_inputs"], "input_audit": result["input_audit"], "official_wave_policy": "bin_aligned", "diagnostic_wave_policies_selected": False, "original_request_order_recovered": False, "r69_models": ["qwen3-8b", "deepseek-v2-lite"], "training_performed": False, "prediction_recomputed": False, "teacher_recomputed": False, "gpu_used": False})
    write_json(output / "audit/interpolation.json", result["interpolation"])
    environment = environment_record()
    environment.update({"gpu_used": False, "network_used": False, "model_weights_loaded": False, "physical_measurement_performed": False})
    write_json(output / "audit/environment.json", environment)
    elapsed = time.monotonic() - started
    official_cost = [row for row in result["cost_comparison"] if row["slice_type"] == "overall"]
    official_placement = [row for row in result["placement_comparison"] if row["slice_type"] == "overall"]
    summary = {
        "schema_version": "phase71-pd-multiflow-cost-placement-integration-result-v1", "status": "PASS",
        "scientific_outcome": result["decision"]["scientific_outcome"], "workflow_commit": preflight["workflow_commit"],
        "source_result_commit": spec["workflow_parent_result_commit"], "completed_at_utc": utc_now(), "runtime_seconds": elapsed,
        "counts": {**spec["expected_counts"], "models": 6, "r69_models": 2, "configurations": 7, "topologies": 3, "wave_policies": 3},
        "decision": result["decision"],
        "headline": {
            "cost_config_topology_checks_passed": sum(bool(row["strict_mape_and_wape_improvement"]) for row in official_cost),
            "cost_config_topology_checks_total": len(official_cost),
            "placement_configuration_checks_passed": sum(bool(row["weak_agreement_and_regret_improvement"]) for row in official_placement),
            "placement_configuration_checks_total": len(official_placement),
            "maximum_dnn_cost_wape": max(float(row["dnn_cost_wape"]) for row in official_cost),
            "maximum_h0_cost_wape": max(float(row["h0_cost_wape"]) for row in official_cost),
            "minimum_dnn_placement_agreement": min(float(row["dnn_agreement_rate"]) for row in official_placement),
            "maximum_dnn_mean_teacher_regret": max(float(row["dnn_mean_teacher_regret"]) for row in official_placement),
            "maximum_wave_policy_relative_cost_range": max(float(row["max_relative_cost_range"]) for row in result["wave_sensitivity"]),
            "minimum_wave_policy_placement_stability": min(float(row["placement_stability_rate"]) for row in result["wave_sensitivity"]),
        },
        "official_wave_policy": "bin_aligned", "diagnostic_wave_policies_selected": False,
        "training_performed": False, "prediction_recomputed": False, "teacher_recomputed": False,
        "gpu_used": False, "network_used": False, "physical_measurement_performed": False,
        "proved": "deterministic communication-only integration of frozen histograms with Phase51/R61/R69 under the preregistered marginal-wave contract",
        "not_proved": "original message concurrency recovery, fresh blind prediction, R69 on four unmeasured models, larger graphs, full scheduler or end-to-end latency",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        f"# Phase71：PD多流通信代价与placement集成\n\n状态：`PASS`；科学结论：`{summary['scientific_outcome']}`。"
        f"共{len(result['costs'])}行配置×拓扑代价、{len(result['decisions'])}个placement决策。"
        f"H0+DNN严格改善cost的配置×拓扑为{summary['headline']['cost_config_topology_checks_passed']}/{summary['headline']['cost_config_topology_checks_total']}；"
        f"弱改善placement的配置为{summary['headline']['placement_configuration_checks_passed']}/{summary['headline']['placement_configuration_checks_total']}。"
        "正式wave策略固定为bin_aligned，另外两种只做敏感性；没有训练、GPU、网络、teacher重算或原始请求顺序恢复。\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir()
    (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={preflight['workflow_commit']} status=PASS outcome={summary['scientific_outcome']} runtime_seconds={elapsed:.6f}\ngpu=false network=false training=false teacher_recomputed=false\n", encoding="utf-8")
    (output / "DONE").write_text("PASS\n", encoding="utf-8")
    refresh_manifest(output)
    tree = validate_result_tree(output)
    if not tree["ok"]:
        raise RuntimeError(tree)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase71_pd_multiflow_cost_placement_integration")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_workflow_commit, arguments.output_dir.resolve()), ensure_ascii=False, indent=2))
