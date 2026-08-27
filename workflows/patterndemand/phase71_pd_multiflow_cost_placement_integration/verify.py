#!/usr/bin/env python3
"""Verify Phase71 result tree and deterministically recompute all analysis."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import load_json, repo_root, verify_result_manifest  # noqa:E402
from analysis import build_analysis, read_csv  # noqa:E402


def rows_equal(actual: list[dict[str, str]], expected: list[dict[str, Any]]) -> bool:
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if set(left) != set(right):
            return False
        for key, expected_value in right.items():
            actual_value = left[key]
            if isinstance(expected_value, bool):
                if actual_value != str(expected_value):
                    return False
            elif isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                try:
                    if not math.isclose(float(actual_value), float(expected_value), rel_tol=1e-11, abs_tol=1e-10):
                        return False
                except ValueError:
                    return False
            elif actual_value != str(expected_value):
                return False
    return True


def verify(output: Path) -> dict:
    root = repo_root()
    spec = load_json(HERE / "experiment.json")
    expected = load_json(HERE / "expected_outputs.json")
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    manifest = verify_result_manifest(output)
    result_spec = load_json(output / "contracts/experiment.json")
    wave = load_json(output / "contracts/wave_policies.json")
    coverage = load_json(output / "contracts/coverage_matrix.json")
    summary = load_json(output / "summary.json")
    freeze = load_json(output / "audit/input_freeze.json")
    interpolation = load_json(output / "audit/interpolation.json")
    environment = load_json(output / "audit/environment.json")
    paths = {row["name"]: root / row["path"] for row in spec["pinned_inputs"]}
    recomputed = build_analysis(
        read_csv(paths["phase49_frozen_predictions"]), read_csv(paths["phase50_hfull_targets"]),
        load_json(paths["phase51_curves"])["curves"], load_json(paths["phase51_layouts"])["layouts"],
        load_json(paths["r61_model"]), load_json(paths["r67_model"]), load_json(paths["r69_model"]), spec,
    )
    file_checks = {
        "costs": rows_equal(read_csv(output / "analysis/unit_configuration_topology_costs.csv.gz"), recomputed["costs"]),
        "cost_metrics": rows_equal(read_csv(output / "analysis/cost_metrics.csv"), recomputed["cost_metrics"]),
        "cost_comparison": rows_equal(read_csv(output / "analysis/cost_method_comparison.csv"), recomputed["cost_comparison"]),
        "decisions": rows_equal(read_csv(output / "analysis/placement_decisions.csv.gz"), recomputed["decisions"]),
        "placement_metrics": rows_equal(read_csv(output / "analysis/placement_metrics.csv"), recomputed["placement_metrics"]),
        "placement_comparison": rows_equal(read_csv(output / "analysis/placement_method_comparison.csv"), recomputed["placement_comparison"]),
        "wave_sensitivity": rows_equal(read_csv(output / "analysis/wave_sensitivity.csv"), recomputed["wave_sensitivity"]),
    }
    counts = spec["expected_counts"]
    summary_counts = summary.get("counts", {})
    checks = {
        "manifest": manifest["ok"],
        "required_exact": files == set(expected["required"]),
        "status": summary.get("status") == "PASS" and (output / "DONE").read_text().strip() == "PASS",
        "contract_exact": result_spec == spec,
        "wave_contract_exact": wave == load_json(HERE / "wave_policies.json") and wave.get("official_policy") == "bin_aligned" and wave.get("selection_after_results_forbidden") is True,
        "coverage_exact": coverage == load_json(HERE / "coverage_matrix.json") and len(coverage.get("rows", [])) == 7,
        "analysis_recomputed": all(file_checks.values()),
        "interpolation_recomputed": interpolation == recomputed["interpolation"] and interpolation.get("exact_piecewise_cdf_integration") is True,
        "counts": all(int(summary_counts.get(key, -1)) == value for key, value in counts.items()),
        "decision": summary.get("scientific_outcome") == recomputed["decision"]["scientific_outcome"] and summary.get("decision") == recomputed["decision"],
        "freeze": freeze.get("official_wave_policy") == "bin_aligned" and freeze.get("diagnostic_wave_policies_selected") is False and freeze.get("original_request_order_recovered") is False and freeze.get("r69_models") == ["qwen3-8b", "deepseek-v2-lite"],
        "scope": summary.get("training_performed") is False and summary.get("prediction_recomputed") is False and summary.get("teacher_recomputed") is False and summary.get("gpu_used") is False and summary.get("network_used") is False and summary.get("physical_measurement_performed") is False and environment.get("gpu_used") is False and environment.get("network_used") is False,
        "no_forbidden_assets": not list(output.rglob("*.jsonl")) and not list(output.rglob("*.safetensors")) and not list(output.rglob("*.pt")),
    }
    if not all(checks.values()):
        raise RuntimeError({"phase71_checks": checks, "file_checks": file_checks, "manifest": manifest})
    return {"status": "PASS", "checks": checks, "workflow_commit": summary["workflow_commit"], "scientific_outcome": summary["scientific_outcome"], "cost_rows": len(recomputed["costs"]), "placement_decisions": len(recomputed["decisions"]), "manifest_files": manifest["manifest"]["checked_files"], "headline": summary["headline"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase71_pd_multiflow_cost_placement_integration")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.output_dir.resolve()), ensure_ascii=False, indent=2))
