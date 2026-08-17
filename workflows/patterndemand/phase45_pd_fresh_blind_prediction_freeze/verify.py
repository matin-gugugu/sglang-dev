#!/usr/bin/env python3
"""Verify Phase45 target-free blind prediction freeze."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent; P42 = HERE.parent / "phase42_pd_residual_training"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import read_csv_gz  # noqa: E402


def verify(output: Path) -> dict:
    manifest = verify_result_manifest(output); summary = load_json(output / "summary.json")
    features = read_csv_gz(output / "dataset/pd_fresh_blind_target_free_features.csv.gz"); profiles = read_csv_gz(output / "profiles/fresh_blind_lowdim_profiles.csv.gz"); predictions = read_csv_gz(output / "predictions/pd_fresh_blind_frozen_predictions.csv.gz")
    forbidden_prefixes = ("target_", "residual_", "future_"); forbidden_exact = {"requests", "full_request_list", "input_lens", "output_lens", "timestamps", "arrival_times"}
    all_fields = set(features[0]) | set(profiles[0]) | set(predictions[0]); methods = Counter(row["method"] for row in predictions)
    checks = {
        "manifest": manifest["ok"], "status": summary.get("status") == "PASS", "done": (output / "DONE").read_text().strip() == "PASS",
        "features_300": len(features) == 300, "profiles_300": len(profiles) == 300, "predictions_600": len(predictions) == 600,
        "methods_exact": methods == Counter({"h0": 300, "h0_plus_dnn_residual": 300}),
        "profile_ids_exact": len({row["profile_id"] for row in features}) == 300 and {row["profile_id"] for row in features} == {row["profile_id"] for row in profiles} == {row["profile_id"] for row in predictions},
        "no_target_or_residual": not any(name.startswith(forbidden_prefixes) for name in all_fields),
        "no_complete_requests": not forbidden_exact.intersection(all_fields) and int(summary["counts"]["complete_request_rows_in_git"]) == 0,
        "target_rows_zero": int(summary["counts"]["target_rows"]) == 0,
    }
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status": "PASS", "checks": checks, "manifest_files": manifest["manifest"]["checked_files"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
