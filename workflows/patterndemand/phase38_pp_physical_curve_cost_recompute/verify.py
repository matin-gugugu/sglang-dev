#!/usr/bin/env python3
"""验证Phase38正式紧凑结果。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, sha256, verify_result_manifest
from preflight import validate_phase37_curves


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase38_pp_physical_curve_cost_recompute",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    expected = load_json(HERE / "expected_outputs.json")
    missing = [relative for relative in expected["required"] if not (output / relative).is_file()]
    manifest = verify_result_manifest(output)
    summary = load_json(output / "summary.json")
    state = load_json(output / "audit/runtime_state.json")
    freeze = load_json(output / "audit/input_freeze.json")
    workflow_contract = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    curve_snapshot = load_json(output / "contracts/phase37_curve_snapshot.json")
    done = (output / "DONE").read_text(encoding="utf-8").strip()
    phase_rows = read_csv(output / "analysis/phase_costs.csv.gz")
    total_rows = read_csv(output / "analysis/combined_costs.csv.gz")
    metrics = read_csv(output / "analysis/cost_metrics.csv")
    histogram_metrics = read_csv(output / "analysis/frozen_histogram_metrics.csv")
    proxy_rows = read_csv(output / "analysis/physical_vs_phase35_proxy.csv")
    curve_count = len(curve_snapshot.get("curves", []))
    snapshot_sha = sha256(output / "contracts/phase37_curve_snapshot.json")
    frozen_phase37 = freeze.get("phase37", {})
    curve_contract_audit = validate_phase37_curves(workflow_contract, curve_snapshot)
    checks = {
        "required_outputs_present": not missing,
        "result_manifest": manifest["ok"],
        "summary_pass": summary.get("status") == "PASS",
        "done_matches": done == summary.get("status"),
        "runtime_checks_pass": all(state.get("checks", {}).values()),
        "result_contract_matches_workflow": result_contract == workflow_contract,
        "phase37_curve_snapshot_sha": snapshot_sha == freeze.get("phase37_curve_snapshot_sha256"),
        "phase37_curve_source_sha": snapshot_sha == frozen_phase37.get("curve_sha256"),
        "phase37_curve_source_snapshot_match": freeze.get("phase37_source_and_snapshot_sha_match") is True,
        "phase37_manifest_was_valid": frozen_phase37.get("manifest", {}).get("ok") is True,
        "phase37_result_commit_was_valid": frozen_phase37.get("result_commit_audit", {}).get("ok") is True,
        "phase37_result_is_ancestor_of_w38": frozen_phase37.get("result_commit_is_ancestor_of_w38") is True,
        "phase37_commit_matches_summary": summary.get("phase37_result_commit") == frozen_phase37.get("result_commit"),
        "all_static_pins_were_valid": all(
            value.get("ok") is True for value in freeze.get("static_pinned_inputs", {}).values()
        ),
        "curve_evidence_physical": curve_snapshot.get("curve_evidence") == "physical_measurement",
        "phase37_curve_contract": curve_contract_audit["ok"],
        "curve_count_nonzero": curve_count > 0,
        "phase_row_count": len(phase_rows) == 1296 * curve_count,
        "total_row_count": len(total_rows) == 648 * curve_count,
        "metric_row_count": len(metrics) == 30 * curve_count,
        "histogram_metric_row_count": len(histogram_metrics) == 42,
        "proxy_comparison_row_count": len(proxy_rows) == 30 * curve_count,
        "phase_rows_only_physical": all(
            row.get("curve_evidence") == "physical_measurement"
            and row.get("topology_scope") == "single_node"
            and row.get("parallelism") == "pp"
            for row in phase_rows
        ),
        "costs_finite_nonnegative": all(
            math.isfinite(float(row[name])) and float(row[name]) >= 0
            for row in [*phase_rows, *total_rows]
            for name in ("predicted_cost_us_per_1000", "teacher_cost_us_per_1000", "absolute_error_us_per_1000")
        ),
        "no_training": summary.get("training_performed") is False,
        "no_checkpoint": summary.get("checkpoint_loaded") is False,
        "no_prediction_recompute": summary.get("prediction_recomputation_performed") is False,
        "diagnostic_not_pass_fail_gate": summary.get("diagnostic_reference_is_pass_fail_gate") is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "missing": missing, "manifest": manifest})
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "phase37_result_commit": summary["phase37_result_commit"],
        "curves": curve_count,
        "phase_cost_rows": len(phase_rows),
        "total_cost_rows": len(total_rows),
        "metric_rows": len(metrics),
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
