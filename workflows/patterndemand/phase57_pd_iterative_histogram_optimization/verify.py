#!/usr/bin/env python3
"""Verify Phase57 integrity, OOF search contract and one-shot validation gates."""

from __future__ import annotations

import argparse
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
    trace = read_csv(output / "analysis/round_trace.csv"); candidates = read_csv(output / "analysis/oof_candidate_metrics.csv"); predictions = read_csv_gz(output / "predictions/development_validation_predictions.csv.gz")
    checkpoint = json.load(gzip.open(output / "checkpoints/pd_iterative_histogram_optimization.json.gz", "rt", encoding="utf-8"))
    search = load_json(output / "audit/search.json"); environment = load_json(output / "audit/environment.json"); gates = summary["gates"]
    expected_head = run_git(["rev-parse", "HEAD"])
    candidate_count = len(candidates)
    checks = {
        "manifest": manifest["ok"],
        "required_outputs": required.issubset(actual),
        "status": summary.get("status") == "PASS",
        "done": (output / "DONE").read_text(encoding="utf-8").strip() == "PASS",
        "workflow_head_exact": summary.get("workflow_commit") == expected_head and checkpoint.get("workflow_commit") == expected_head,
        "candidate_budget": 0 < candidate_count <= 36 and candidate_count == len(trace) and candidate_count % 12 == 0,
        "round_count": 1 <= int(search["rounds_completed"]) <= 3 and int(search["candidate_budget"]) == candidate_count,
        "prediction_rows_2880": len(predictions) == 2880,
        "prediction_methods": {row["method"] for row in predictions} == {"h0", "h0_plus_dnn_iterative"},
        "profiles_240": len({row["profile_id"] for row in predictions}) == 240,
        "models_6": len({row["model"] for row in predictions}) == 6,
        "counts_exact": summary["counts"]["profiles"] == 1200 and summary["counts"]["train_profiles"] == 960 and summary["counts"]["validation_profiles"] == 240 and summary["counts"]["models"] == 6 and summary["counts"]["segments"] == 3 and summary["counts"]["example_rows"] == 7200 and summary["counts"]["train_rows"] == 5760 and summary["counts"]["validation_rows"] == 1440 and summary["counts"]["candidates"] == candidate_count and summary["counts"]["complete_request_rows_in_git"] == 0,
        "one_selected": sum(1 for row in candidates if row.get("selected") == "True") == 1 and checkpoint["selected_candidate"]["candidate_id"] == summary["selected"]["candidate_id"],
        "validation_after_freeze": search["validation_opened_once_after_freeze"] is True,
        "no_phase50_blind": checkpoint["phase50_blind_accessed"] is False and search["phase50_blind_accessed"] is False,
        "no_complete_requests": checkpoint["complete_requests_accessed"] is False and search["complete_requests_accessed"] is False,
        "gate_consistency": gates["target_met"] == all([gates["development_overall"], gates["development_all_models"], gates["development_all_segments"]]),
        "next_permission_consistent": gates["next_phase_permitted"] == gates["target_met"],
        "environment_cpu_only": environment["gpu_used"] is False and environment["network_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {"status": "PASS", "checks": checks, "manifest_files": manifest["manifest"]["checked_files"], "scientific_outcome": summary["scientific_outcome"], "target_met": gates["target_met"], "candidate_count": candidate_count}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase57_pd_iterative_histogram_optimization")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
