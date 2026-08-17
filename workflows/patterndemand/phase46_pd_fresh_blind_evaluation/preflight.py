#!/usr/bin/env python3
"""Verify the immutable R45 prediction freeze before Phase46 target access."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent; P41 = HERE.parent / "phase41_pd_full_window_dataset"; P42 = HERE.parent / "phase42_pd_residual_training"; P45 = HERE.parent / "phase45_pd_fresh_blind_prediction_freeze"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42)); sys.path.insert(0, str(P41)); sys.path.insert(0, str(P45)); sys.path.insert(0, str(HERE))
from build_selection import select  # noqa: E402
from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_pinned_inputs  # noqa: E402
from model import read_csv_gz  # noqa: E402
from prepare_bundle import raw_source_audit  # noqa: E402


def run_checks(expected: str, raw_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json"); phase41 = load_json(P41 / "experiment.json")
    head = require_expected_head(expected); require_clean_before_run(allowed_untracked_prefixes=()); pins = verify_pinned_inputs(contract)
    frozen_features = read_csv_gz(repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze/dataset/pd_fresh_blind_target_free_features.csv.gz")
    frozen_predictions = read_csv_gz(repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze/predictions/pd_fresh_blind_frozen_predictions.csv.gz")
    selection = select(repo_root()); forbidden = [name for row in frozen_features + frozen_predictions for name in row if name.startswith("target_") or name.startswith("residual_") or name.startswith("future_")]
    methods = Counter(row["method"] for row in frozen_predictions); selection_ids = {row["profile_id"] for row in selection}; feature_ids = {row["profile_id"] for row in frozen_features}; prediction_ids = {row["profile_id"] for row in frozen_predictions}
    checks = {"selection_300": len(selection) == 300, "features_300": len(frozen_features) == 300, "predictions_600": len(frozen_predictions) == 600, "methods_exact": methods == Counter({"h0": 300, "h0_plus_dnn_residual": 300}), "ids_exact": selection_ids == feature_ids == prediction_ids, "no_target_or_residual": not forbidden}
    if not all(checks.values()): raise RuntimeError({"freeze_checks": checks, "forbidden": sorted(set(forbidden))})
    raw = raw_source_audit(phase41, raw_dir.expanduser().resolve())
    return {"status": "PASS", "workflow_commit": head, "pinned_inputs": pins, "freeze_checks": checks, "raw_source_audit": raw, "targets_accessed": False, "checkpoint_loaded": False, "prediction_recomputed": False, "gpu_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run_checks(args.expected_workflow_commit, args.raw_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
