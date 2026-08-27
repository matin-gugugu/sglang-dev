#!/usr/bin/env python3
"""Phase71 read-only pins, scope and CPU-only preflight."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, utc_now, verify_pinned_inputs  # noqa:E402
from analysis import read_csv, validate_inputs  # noqa:E402


def run_checks(expected: str) -> dict:
    head = require_expected_head(expected)
    require_clean_before_run()
    root = repo_root()
    spec = load_json(HERE / "experiment.json")
    pins = verify_pinned_inputs(spec)
    paths = {row["name"]: root / row["path"] for row in spec["pinned_inputs"]}
    predictions = read_csv(paths["phase49_frozen_predictions"])
    targets = read_csv(paths["phase50_hfull_targets"])
    curves = load_json(paths["phase51_curves"])["curves"]
    layouts = load_json(paths["phase51_layouts"])["layouts"]
    input_audit = validate_inputs(predictions, targets, curves, layouts, spec)
    wave = load_json(HERE / "wave_policies.json")
    coverage = load_json(HERE / "coverage_matrix.json")
    source = {
        "phase50": load_json(paths["phase50_summary"]),
        "phase51": load_json(paths["phase51_summary"]),
        "phase52": load_json(paths["phase52_summary"]),
        "phase63": load_json(paths["phase63_summary"]),
        "phase70": load_json(paths["phase70_summary"]),
    }
    source_checks = {
        "phase50": source["phase50"].get("scientific_outcome") == "CONFIRMS_SIX_MODEL_H0_PROTECTED_IMPROVEMENT",
        "phase51": source["phase51"].get("status") in ("PASS", "PASS_WITH_RUNTIME_VARIANCE", "PASS_WITH_PLACEMENT_VARIANCE", "PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE"),
        "phase52": source["phase52"].get("scientific_outcome") == {"cost": "CONFIRMED", "placement": "CONFIRMED"},
        "phase63": source["phase63"].get("scientific_outcome") == "FROZEN_CONTENTION_CORRECTION_SIX_MODEL_PASS",
        "phase70": source["phase70"].get("scientific_outcome") == "MULTIFLOW_HIGH_PAGE_RESIDUAL_THIRD_FRESH_BLIND_PASS",
    }
    wave_checks = {
        "schema": wave.get("schema_version") == "phase71-wave-policies-v1",
        "official": wave.get("official_policy") == "bin_aligned",
        "policy_order": [row.get("name") for row in wave.get("policies", [])] == ["bin_aligned", "cyclic_staggered", "opposed_extremes"],
        "no_post_selection": wave.get("selection_after_results_forbidden") is True,
        "no_order_recovery": wave.get("original_request_order_recovery_claimed") is False,
    }
    coverage_checks = {
        "schema": coverage.get("schema_version") == "phase71-coverage-matrix-v1",
        "rows": len(coverage.get("rows", [])) == 7,
        "r69_two_models": all(row["models"] == "qwen3-8b,deepseek-v2-lite" for row in coverage["rows"] if row["physical_model"] == "R69"),
        "unsupported_explicit": "R69 on the other four models" in coverage.get("unsupported", []),
    }
    execution_checks = {
        "cuda_hidden_or_unset": os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"),
        "network_not_required": spec["network_required"] is False,
        "gpu_not_required": spec["gpu_required"] is False,
        "training_forbidden": spec["training_permitted"] is False,
    }
    if not all(source_checks.values()) or not all(wave_checks.values()) or not all(coverage_checks.values()) or not all(execution_checks.values()):
        raise RuntimeError({"source": source_checks, "wave": wave_checks, "coverage": coverage_checks, "execution": execution_checks})
    return {"schema_version": "phase71-preflight-v1", "status": "PASS", "workflow_commit": head, "captured_at_utc": utc_now(), "pinned_inputs": pins, "input_audit": input_audit, "source_checks": source_checks, "wave_checks": wave_checks, "coverage_checks": coverage_checks, "execution_checks": execution_checks, "execution": {"gpu_used": False, "network_used": False, "training_performed": False, "prediction_recomputed": False, "teacher_recomputed": False, "physical_measurement_performed": False}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_checks(arguments.expected_workflow_commit), ensure_ascii=False, indent=2))
