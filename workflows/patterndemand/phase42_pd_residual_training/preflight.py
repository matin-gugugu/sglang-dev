#!/usr/bin/env python3
"""Phase42 CPU-only, target-isolated preflight."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_pinned_inputs  # noqa: E402


def run_checks(expected: str) -> dict:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=())
    if (repo_root() / "data").exists():
        raise RuntimeError("Phase42 must run in an isolated worktree without data/")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError("Phase42 is CPU-only; unset CUDA_VISIBLE_DEVICES or set -1")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Phase42 requires NumPy but not torch/sklearn") from error
    pins = verify_pinned_inputs(contract)
    return {"status": "PASS", "workflow_commit": head, "numpy": np.__version__, "pinned_inputs": pins, "raw_visible": False, "gpu_used": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
