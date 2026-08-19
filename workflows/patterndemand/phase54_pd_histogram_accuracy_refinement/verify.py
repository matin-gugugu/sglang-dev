#!/usr/bin/env python3
"""Verify Phase54 development result integrity and target gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "phase42_pd_residual_training"))

from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import read_csv_gz, read_json_gz  # noqa: E402


def verify(output: Path) -> dict:
    manifest = verify_result_manifest(output); summary = load_json(output / "summary.json"); expected = load_json(HERE / "expected_outputs.json")
    predictions = read_csv_gz(output / "predictions/development_validation_predictions.csv.gz"); checkpoint = read_json_gz(output / "checkpoints/pd_histogram_refinement.json.gz")
    required = set(expected["required"]); actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    gates = summary["gates"]
    checks = {
        "manifest": manifest["ok"],
        "required_outputs": required.issubset(actual),
        "status": summary.get("status") == "PASS",
        "done": (output / "DONE").read_text(encoding="utf-8").strip() == "PASS",
        "predictions_2880": len(predictions) == 2880,
        "prediction_methods": {row["method"] for row in predictions} == {"h0", "h0_plus_dnn_refined"},
        "profiles_240": len({row["profile_id"] for row in predictions}) == 240,
        "models_6": len({row["model"] for row in predictions}) == 6,
        "counts_exact": summary["counts"] == {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "complete_request_rows_in_git": 0},
        "no_phase50_blind": checkpoint["phase50_blind_accessed"] is False and summary["gates"]["phase55_permitted"] == gates["target_met"],
        "no_complete_requests": checkpoint["complete_requests_accessed"] is False,
        "gate_consistency": gates["target_met"] == all([gates["development_overall"], gates["development_all_models"], gates["development_all_segments"]]),
        "environment_local": load_json(output / "audit/environment.json")["gpu_used"] is False and load_json(output / "audit/environment.json")["network_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {"status": "PASS", "checks": checks, "manifest_files": manifest["manifest"]["checked_files"], "scientific_outcome": summary["scientific_outcome"], "target_met": gates["target_met"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase54_pd_histogram_accuracy_refinement")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
