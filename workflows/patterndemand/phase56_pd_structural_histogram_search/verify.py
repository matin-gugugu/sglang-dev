#!/usr/bin/env python3
"""Verify Phase56 structural-search integrity and development gates."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import load_json, repo_root, run_git, verify_result_manifest  # noqa: E402
from model_loader import read_csv, read_csv_gz  # noqa: E402


def verify(output: Path) -> dict:
    expected = load_json(HERE / "expected_outputs.json"); manifest = verify_result_manifest(output); summary = load_json(output / "summary.json")
    required = set(expected["required"]); actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    trace = read_csv(output / "analysis/search_trace.csv"); candidates = read_csv(output / "analysis/oof_candidate_metrics.csv"); groups = read_csv(output / "analysis/oof_group_metrics.csv"); predictions = read_csv_gz(output / "predictions/development_validation_predictions.csv.gz")
    with gzip.open(output / "checkpoints/pd_structural_histogram_search.json.gz", "rt", encoding="utf-8") as source:
        checkpoint = json.load(source)
    gates = summary["gates"]; search = load_json(output / "audit/search.json"); environment = load_json(output / "audit/environment.json")
    expected_head = run_git(["rev-parse", "HEAD"])
    checks = {
        "manifest": manifest["ok"],
        "required_outputs": required.issubset(actual),
        "status": summary.get("status") == "PASS",
        "done": (output / "DONE").read_text(encoding="utf-8").strip() == "PASS",
        "workflow_head_exact": summary.get("workflow_commit") == expected_head and checkpoint.get("workflow_commit") == expected_head,
        "candidate_budget_32": len(candidates) == 32 and len(trace) == 32 and summary["counts"]["candidates"] == 32,
        "stage_counts": search["stage_a_count"] == 20 and search["stage_b_count"] == 12,
        "prediction_rows_2880": len(predictions) == 2880,
        "prediction_methods": {row["method"] for row in predictions} == {"h0", "h0_plus_dnn_structural"},
        "profiles_240": len({row["profile_id"] for row in predictions}) == 240,
        "models_6": len({row["model"] for row in predictions}) == 6,
        "counts_exact": summary["counts"] == {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "segments": 3, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "candidates": 32, "complete_request_rows_in_git": 0},
        "selected_candidate_recorded": sum(1 for row in candidates if row["selected"] == "True") == 1 and checkpoint["selected_candidate"]["candidate_id"] == summary["selected"]["candidate_id"],
        "group_alpha_audit_present": len(groups) >= 1 and all(row["alpha"] for row in groups),
        "validation_after_freeze": search["validation_opened_once_after_freeze"] is True,
        "no_phase50_blind": checkpoint["phase50_blind_accessed"] is False and search["phase50_blind_accessed"] is False and gates["next_phase_permitted"] == gates["target_met"],
        "no_complete_requests": checkpoint["complete_requests_accessed"] is False and search["complete_requests_accessed"] is False,
        "gate_consistency": gates["target_met"] == all([gates["development_overall"], gates["development_all_models"], gates["development_all_segments"]]),
        "environment_local": environment["gpu_used"] is False and environment["network_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {"status": "PASS", "checks": checks, "manifest_files": manifest["manifest"]["checked_files"], "scientific_outcome": summary["scientific_outcome"], "target_met": gates["target_met"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase56_pd_structural_histogram_search")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
