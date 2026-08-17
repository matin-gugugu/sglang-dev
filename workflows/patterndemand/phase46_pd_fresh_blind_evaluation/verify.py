#!/usr/bin/env python3
"""Verify Phase46 blind labels, metrics and frozen-gate outcome."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; P42 = HERE.parent / "phase42_pd_residual_training"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import read_csv_gz  # noqa: E402


def verify(output: Path) -> dict:
    manifest = verify_result_manifest(output); summary = load_json(output / "summary.json"); labels = read_csv_gz(output / "labels/pd_fresh_blind_hfull_targets.csv.gz"); per_profile = read_csv_gz(output / "analysis/per_profile_metrics.csv.gz")
    with (output / "analysis/aggregate_metrics.csv").open(newline="") as source: aggregate = list(csv.DictReader(source))
    forbidden = {"requests", "full_request_list", "input_lens", "output_lens", "timestamps", "arrival_times"}; gates = summary["gates"]
    checks = {
        "manifest": manifest["ok"], "status": summary.get("status") == "PASS", "done": (output / "DONE").read_text().strip() == "PASS",
        "labels_300": len(labels) == 300, "aggregate_2": len(aggregate) == 2, "per_profile_600": len(per_profile) == 600,
        "profile_ids_300": len({row["profile_id"] for row in labels}) == 300,
        "no_complete_requests": not forbidden.intersection(labels[0]) and int(summary["counts"]["complete_request_rows_in_git"]) == 0,
        "outcome_matches_gates": (summary["scientific_outcome"] == "CONFIRMS_H0_PROTECTED_IMPROVEMENT") == bool(gates["overall_strict_four_metrics"] and gates["all_segments"]),
        "segment_count": len(summary["segments"]) == 3,
    }
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status": "PASS", "checks": checks, "manifest_files": manifest["manifest"]["checked_files"], "scientific_outcome": summary["scientific_outcome"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase46_pd_fresh_blind_evaluation")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
