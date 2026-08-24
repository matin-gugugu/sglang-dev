#!/usr/bin/env python3
"""Independent verifier for compact Phase61 results."""
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
from model import CANDIDATES, predict, read_points  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.lower() == "true"


def verify(output: Path) -> dict[str, Any]:
    expected = load_json(HERE / "expected_outputs.json")
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    manifest = verify_result_manifest(output)
    contract = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    summary = load_json(output / "summary.json")
    model = load_json(output / "model/contention_correction.json")
    preflight = load_json(output / "audit/preflight.json")
    freeze = load_json(output / "audit/input_freeze.json")
    candidates = read_csv(output / "analysis/oof_candidate_metrics.csv")
    oof_predictions = read_csv(output / "analysis/oof_predictions.csv")
    oof_slices = read_csv(output / "analysis/oof_slice_metrics.csv")
    refit_predictions = read_csv(output / "analysis/refit_predictions.csv")
    refit_slices = read_csv(output / "analysis/refit_slice_metrics.csv")
    points = read_points(repo_root() / contract["dataset_contract"]["source"])
    pair_grid = load_json(repo_root() / "experiment-results/phase60_pd_multi_endpoint_composability/contracts/payload_pair_grid.json")
    reserved_ids = {
        row["pair_id"]
        for values in pair_grid["reserved_future_blind"].values()
        for row in values
    }
    ordered = [
        next(row for row in candidates if row["candidate_id"] == specification["candidate_id"])
        for specification in CANDIDATES
    ]
    first_passing = next((row for row in ordered if truth(row["target_guard"])), None)
    selected_rows = [row for row in candidates if truth(row["selected"])]
    selected = selected_rows[0] if len(selected_rows) == 1 else None
    baseline = next(row for row in candidates if row["candidate_id"] == "phase51_max")
    status = summary.get("status")
    status_consistent = (
        status == "PASS" and selected is not None and first_passing is not None
        or status == "PASS_TARGET_NOT_MET" and selected is None and first_passing is None
    )
    selected_id = None if selected is None else selected["candidate_id"]
    acceptance = contract["acceptance_gate"]
    selected_gate = selected is not None and (
        float(selected["overall_wape"]) <= float(acceptance["oof_overall_wape_max"])
        and float(selected["max_configuration_topology_wape"]) <= float(acceptance["oof_each_configuration_topology_wape_max"])
        and abs(float(selected["overall_signed_bias"])) <= float(acceptance["oof_overall_absolute_signed_bias_max"])
        and float(selected["max_configuration_topology_absolute_signed_bias"]) <= float(acceptance["oof_each_configuration_topology_absolute_signed_bias_max"])
        and truth(selected["all_predictions_positive"])
        and float(selected["overall_wape"]) < float(baseline["overall_wape"])
    )
    recorded_refit = {
        row["pair_id"] + "|" + row["configuration"] + "|" + row["topology_level"]: float(row["predicted_concurrent_wave_us"])
        for row in refit_predictions
        if row["candidate_id"] == selected_id
    }
    recomputed_refit = {}
    if status == "PASS":
        for row in points:
            key = row["pair_id"] + "|" + row["configuration"] + "|" + row["topology_level"]
            recomputed_refit[key] = predict(model, row)
    checks = {
        "manifest": manifest["ok"],
        "required_exact": files == set(expected["required"]),
        "contract_exact": result_contract == contract,
        "status": status in contract["accepted_result_statuses"] and status_consistent and (output / "DONE").read_text().strip() == status,
        "source_provenance": summary.get("source_result_commit") == contract["workflow_parent_result_commit"] and freeze.get("source_result_commit") == contract["workflow_parent_result_commit"],
        "preflight": preflight.get("status") == "PASS" and preflight.get("workflow_commit") == summary.get("workflow_commit") and all(preflight.get("checks", {}).values()),
        "cardinality": len(candidates) == 7 and len(oof_predictions) == 840 and len(oof_slices) == 98 and len(refit_predictions) == (240 if status == "PASS" else 120) and len(refit_slices) == (28 if status == "PASS" else 14),
        "fold_isolation": all(row["candidate_id"] == "phase51_max" or row["fold_pair_id"] == row["pair_id"] for row in oof_predictions),
        "zero_reserved_overlap": not ({row["pair_id"] for row in oof_predictions} & reserved_ids) and not ({row["pair_id"] for row in refit_predictions} & reserved_ids),
        "baseline_reproduced": math.isclose(float(baseline["overall_wape"]), 0.27908318299410906, rel_tol=1e-12, abs_tol=1e-12),
        "selection_order": selected is None or selected["candidate_id"] == first_passing["candidate_id"],
        "selected_gate": status != "PASS" or selected_gate,
        "model_identity": status != "PASS" or (
            model.get("candidate_id") == selected_id
            and model.get("required_runtime_inputs") == ["phase51_flow0_us", "phase51_flow1_us"]
            and model.get("blind_status", {}).get("fresh_blind_validated") is False
            and model.get("selection_evidence", {}).get("simplest_passing_candidate") is True
        ),
        "refit_reproducible": status != "PASS" or (
            recorded_refit.keys() == recomputed_refit.keys()
            and all(math.isclose(recorded_refit[key], recomputed_refit[key], rel_tol=1e-12, abs_tol=1e-9) for key in recorded_refit)
        ),
        "scope": summary.get("gpu_used") is False and summary.get("network_used") is False and summary.get("new_physical_measurement") is False and summary.get("reserved_future_blind_opened") is False and summary.get("fresh_blind_validated") is False and summary.get("counts", {}).get("reserved_future_blind_points_used") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError({"phase61_checks": checks, "selected": selected, "first_passing": first_passing, "manifest": manifest})
    return {
        "status": "PASS",
        "checks": checks,
        "workflow_commit": summary["workflow_commit"],
        "result_status": status,
        "selected_candidate_id": selected_id,
        "baseline_wape": float(baseline["overall_wape"]),
        "selected_oof_wape": None if selected is None else float(selected["overall_wape"]),
        "manifest_files": manifest["manifest"]["checked_files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase61_pd_contention_correction",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
