#!/usr/bin/env python3
"""Verify Phase43 labels, frozen join and no full-request leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P42 = HERE.parent / "phase42_pd_residual_training"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import read_csv_gz  # noqa: E402


def verify(output: Path) -> dict:
    manifest = verify_result_manifest(output); summary = load_json(output / "summary.json")
    targets = read_csv_gz(output / "labels/pd_blind_hfull_targets.csv.gz")
    per_profile = read_csv_gz(output / "analysis/per_profile_metrics.csv.gz")
    aggregate = []
    import csv
    with (output / "analysis/aggregate_metrics.csv").open(newline="") as source: aggregate = list(csv.DictReader(source))
    forbidden_names = {"requests", "full_request_list", "input_lens", "output_lens", "timestamp", "arrival_time"}
    checks = {
        "manifest": manifest["ok"], "status": summary.get("status") == "PASS", "done": (output / "DONE").read_text().strip() == "PASS",
        "targets_12": len(targets) == 12, "target_ids_unique": len({row["profile_id"] for row in targets}) == 12,
        "targets_schema_safe": not forbidden_names.intersection(targets[0]) and all(sum(name.startswith(f"target_{kind}_bin_") for name in targets[0]) == 12 for kind in ("calls", "logical_bytes")),
        "per_profile_24": len(per_profile) == 24, "aggregate_2": len(aggregate) == 2,
        "methods_exact": {row["method"] for row in aggregate} == {"h0", "h0_plus_dnn_residual"},
        "no_full_requests": int(summary["counts"]["full_request_rows_in_git"]) == 0,
        "no_training_or_recompute": load_json(output / "audit/environment.json")["checkpoint_loaded"] is False and load_json(output / "audit/environment.json")["prediction_recomputed"] is False,
    }
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status": "PASS", "checks": checks, "manifest_files": manifest["manifest"]["checked_files"], "scientific_outcome": summary["blind_metrics"]["outcome"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase43_pd_blind_evaluation")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
