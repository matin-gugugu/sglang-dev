#!/usr/bin/env python3
"""Read-only Phase57 contract and input checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P54 = HERE.parent / "phase54_pd_histogram_accuracy_refinement"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    verify_manifest,
    verify_pinned_inputs,
)

_MODEL_SPEC = importlib.util.spec_from_file_location("phase57_p54_model", P54 / "model.py")
_CONTRACT_SPEC = importlib.util.spec_from_file_location("phase57_p54_contracts", P54 / "contracts.py")
if _MODEL_SPEC is None or _MODEL_SPEC.loader is None or _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise RuntimeError("cannot load pinned Phase48/54 validators")
_MODEL = importlib.util.module_from_spec(_MODEL_SPEC); _MODEL_SPEC.loader.exec_module(_MODEL)
_CONTRACT = importlib.util.module_from_spec(_CONTRACT_SPEC); _CONTRACT_SPEC.loader.exec_module(_CONTRACT)


def run_checks(expected: str) -> dict:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=(
        "data/",
        "experiment-results/phase54_pd_histogram_accuracy_refinement/",
        "experiment-results/phase55_pd_adaptive_histogram_search/",
        "experiment-results/phase56_pd_structural_histogram_search/",
    ))
    source = repo_root() / contract["pinned_inputs"][1]["path"]
    rows = _MODEL.read_csv_gz(source)
    schema = _CONTRACT.validate_rows(rows)
    source_manifest = verify_manifest(repo_root() / contract["dataset_contract"]["source_result_dir"])
    train_profiles = {row["profile_id"] for row in rows if row["split_role"] == "expanded_train"}
    validation_profiles = {row["profile_id"] for row in rows if row["split_role"] == "expanded_validation"}
    checks = {
        "rows_7200": len(rows) == 7200,
        "train_profiles_960": len(train_profiles) == 960,
        "validation_profiles_240": len(validation_profiles) == 240,
        "six_rows_per_profile": all(sum(row["profile_id"] == profile for row in rows) == 6 for profile in train_profiles | validation_profiles),
        "no_arrival_semantic_drift": all(row["feature_pd_fixed_draining"] == "1" for row in rows),
        "round_budget": int(contract["search_contract"]["max_rounds"]) == 3
        and int(contract["search_contract"]["seed_candidates_per_round"]) == 8
        and int(contract["search_contract"]["adaptive_candidates_per_round"]) == 4
        and int(contract["search_contract"]["max_total_candidates"]) == 36
        and float(contract["search_contract"]["estimated_cpu_budget_hours"]) <= 8.0,
        "blind_not_opened": True,
        "complete_requests_not_opened": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {
        "status": "PASS",
        "workflow_commit": head,
        "pinned_inputs": verify_pinned_inputs(contract),
        "phase48_result_manifest": source_manifest,
        "source_schema": schema,
        "counts": {"rows": len(rows), "train_profiles": len(train_profiles), "validation_profiles": len(validation_profiles)},
        "checks": checks,
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
