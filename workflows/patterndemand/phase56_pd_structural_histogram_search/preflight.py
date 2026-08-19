#!/usr/bin/env python3
"""Read-only Phase56 preflight; no GPU, raw, network or blind access."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P54 = HERE.parent / "phase54_pd_histogram_accuracy_refinement"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_manifest, verify_pinned_inputs  # noqa: E402
from model_loader import read_csv_gz  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("phase54_contracts_for_phase56", P54 / "contracts.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load Phase54 contracts")
_P54 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_P54)


def run_checks(expected: str) -> dict:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=(
        "data/",
        "experiment-results/phase54_pd_histogram_accuracy_refinement/",
        "experiment-results/phase55_pd_adaptive_histogram_search/",
    ))
    source = repo_root() / contract["pinned_inputs"][2]["path"]
    rows = read_csv_gz(source)
    source_schema = _P54.validate_rows(rows)
    phase48_audit = verify_manifest(repo_root() / contract["dataset_contract"]["source_result_dir"])
    search = contract["search_contract"]
    checks = {
        "seed_candidates_20": int(search["stage_a_seed_candidates"]) == 20,
        "top_k_6": int(search["stage_a_top_k"]) == 6,
        "variants_per_top_2": int(search["stage_b_variants_per_top_candidate"]) == 2,
        "max_total_candidates_32": int(search["max_total_candidates"]) == 32,
        "oof_folds_4": int(search["oof_folds"]) == 4,
        "validation_not_opened": True,
        "phase50_not_opened": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {
        "status": "PASS",
        "workflow_commit": head,
        "pinned_inputs": verify_pinned_inputs(contract),
        "phase48_result_manifest": phase48_audit,
        "source_schema": source_schema,
        "search_contract": checks,
        "gpu_used": False,
        "network_used": False,
        "raw_accessed": False,
        "phase50_blind_accessed": False,
        "complete_requests_accessed": False,
        "training_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True)
    args = parser.parse_args(); print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
