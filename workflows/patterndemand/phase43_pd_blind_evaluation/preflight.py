#!/usr/bin/env python3
"""Phase43 sealed-target preflight and protected-raw audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P41 = HERE.parent / "phase41_pd_full_window_dataset"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P41))
from common import load_json, require_clean_before_run, require_expected_head, verify_pinned_inputs  # noqa: E402
from prepare_bundle import raw_source_audit  # noqa: E402


def run_checks(expected: str, raw_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json")
    phase41 = load_json(P41 / "experiment.json")
    head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=())
    pins = verify_pinned_inputs(contract)
    raw = raw_source_audit(phase41, raw_dir.expanduser().resolve())
    if len(raw) != 6 or not all(row["exact"] for row in raw): raise RuntimeError("raw source audit failed")
    return {"status": "PASS", "workflow_commit": head, "pinned_inputs": pins, "raw_source_audit": raw, "raw_read_only": True, "gpu_used": False, "checkpoint_loaded": False, "predictions_recomputed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run_checks(args.expected_workflow_commit, args.raw_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
