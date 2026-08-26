#!/usr/bin/env python3
"""Independent deterministic verifier for the Phase69 compact result."""
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
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import CANDIDATES, evaluate, fit_model, predict, read_development  # noqa: E402
from preflight import validate_phase70  # noqa: E402
from run import metric_rows  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def scalar_equal(left: str, right: Any) -> bool:
    if isinstance(right, bool):
        return left.lower() == str(right).lower()
    if isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    if right is None:
        return left == ""
    return left == str(right)


def rows_equal(left: list[dict[str, str]], right: list[dict[str, Any]]) -> bool:
    return len(left) == len(right) and all(
        all(key in actual and scalar_equal(actual[key], value) for key, value in expected.items())
        for actual, expected in zip(left, right)
    )


def verify(output: Path) -> dict[str, Any]:
    expected = load_json(HERE / "expected_outputs.json")
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    manifest = verify_result_manifest(output)
    contract = load_json(HERE / "experiment.json")
    summary = load_json(output / "summary.json")
    preflight = load_json(output / "audit/preflight.json")
    freeze = load_json(output / "audit/input_freeze.json")
    model = load_json(output / "model/multiflow_high_page_residual.json")
    root = repo_root()
    r65 = load_json(root / "experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json")
    r67 = load_json(root / "experiment-results/phase67_pd_graph_page_shape_refinement/model/multiflow_graph_page_correction.json")
    rows = read_development(
        root / contract["dataset_contract"]["phase64_source"],
        root / contract["dataset_contract"]["phase66_source"],
        root / contract["dataset_contract"]["phase68_source"],
        r65,
        r67,
    )
    evaluation = evaluate(rows, contract)
    selected = evaluation["selected"]
    status = "PASS" if selected else "PASS_TARGET_NOT_MET"
    selected_id = None if selected is None else selected["candidate_id"]
    expected_model = None if selected is None else fit_model(rows, next(candidate for candidate in CANDIDATES if candidate["candidate_id"] == selected_id))
    blind = validate_phase70(contract)
    checks = {
        "manifest": manifest["ok"],
        "required_exact": files == set(expected["required"]),
        "contract_exact": load_json(output / "contracts/experiment.json") == contract,
        "grid_exact": load_json(output / "contracts/phase70_reserved_blind_grid.json") == load_json(HERE / "phase70_reserved_blind_grid.json") and blind["status"] == "PASS",
        "status": summary.get("status") == status and (output / "DONE").read_text().strip() == status and status in contract["accepted_result_statuses"],
        "preflight": preflight.get("status") == "PASS" and all(preflight.get("checks", {}).values()) and preflight.get("execution", {}).get("phase70_targets_read") is False,
        "candidate_metrics": rows_equal(read_csv(output / "analysis/candidate_metrics.csv"), metric_rows(evaluation)),
        "oof_predictions": rows_equal(read_csv(output / "analysis/oof_predictions.csv"), evaluation["predictions"]),
        "oof_slices": rows_equal(read_csv(output / "analysis/oof_slice_metrics.csv"), evaluation["slices"]),
        "selection": selected_id == "r67_high_page_linear" and summary.get("selection", {}).get("first_simplest_passing") is True,
        "model": status != "PASS" or (
            model.get("candidate_id") == selected_id and model.get("groups") == expected_model["groups"]
            and model.get("anchor_model") == "frozen_phase67_graph_page_shape"
            and model.get("activation_page_threshold") == 32
        ),
        "anchor_preserved": status != "PASS" or all(
            (max(row["pages_list"]) > 32 and row["configuration"] != "P2D2_MATCHING")
            or math.isclose(predict(model, row), row["r67_prediction_us"], rel_tol=0.0, abs_tol=0.0)
            for row in rows
        ),
        "blind_boundary": freeze.get("phase70_measurements_or_targets_read") is False and summary.get("phase70_targets_opened") is False and summary.get("fresh_blind_validated") is False,
        "scope": summary.get("gpu_used") is False and summary.get("network_used") is False and summary.get("new_physical_measurement") is False and summary.get("counts", {}).get("gpu_measurements") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError({"phase69_checks": checks, "manifest": manifest})
    return {
        "status": "PASS", "checks": checks, "workflow_commit": summary["workflow_commit"],
        "result_status": status, "selected_candidate_id": selected_id,
        "validation": summary["selection"]["schemes"], "next_phase_permitted": summary["next_phase_permitted"],
        "manifest_files": manifest["manifest"]["checked_files"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase69_pd_high_page_residual_refinement")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.output_dir.resolve()), ensure_ascii=False, indent=2))
