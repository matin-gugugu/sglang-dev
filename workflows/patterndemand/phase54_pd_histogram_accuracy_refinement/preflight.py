#!/usr/bin/env python3
"""Read-only Phase54 preflight; no raw, GPU, network or Phase50 access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_manifest, verify_pinned_inputs  # noqa: E402
from contracts import validate_rows, loss_contract_self_check  # noqa: E402
from model import read_csv_gz  # noqa: E402


def run_checks(expected: str) -> dict:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=("data/",))
    source = repo_root() / contract["pinned_inputs"][2]["path"]
    rows = read_csv_gz(source)
    phase48_audit = verify_manifest(repo_root() / contract["dataset_contract"]["source_result_dir"])
    return {
        "status": "PASS",
        "workflow_commit": head,
        "pinned_inputs": verify_pinned_inputs(contract),
        "phase48_result_manifest": phase48_audit,
        "source_schema": validate_rows(rows),
        "loss_contract": loss_contract_self_check(),
        "gpu_used": False,
        "network_used": False,
        "raw_accessed": False,
        "phase50_blind_accessed": False,
        "complete_requests_accessed": False,
        "training_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
