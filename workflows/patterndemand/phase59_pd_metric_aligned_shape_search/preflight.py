#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P54 = HERE.parent / "phase54_pd_histogram_accuracy_refinement"
sys.path.insert(0, str(P54)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import load_json, repo_root, require_expected_head, run_git, verify_manifest, verify_pinned_inputs  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


CONTRACTS = load_module("phase59_p54_contracts", P54 / "contracts.py")
MODEL = load_module("phase59_p54_model", P54 / "model.py")


def require_safe_worktree(contract: dict) -> dict:
    allowed_tracked = set(contract.get("allowed_protected_dirty_paths", []))
    unstaged = run_git(["-c", "core.quotePath=false", "diff", "--name-only"]).splitlines()
    staged = run_git(["-c", "core.quotePath=false", "diff", "--cached", "--name-only"]).splitlines()
    tracked_paths = sorted({path for path in unstaged + staged if path})
    unexpected_tracked = [path for path in tracked_paths if path not in allowed_tracked]
    allowed_untracked = tuple(contract.get("allowed_protected_untracked_prefixes", []))
    untracked = [path for path in run_git(["ls-files", "--others", "--exclude-standard"]).splitlines() if path]
    unexpected_untracked = [path for path in untracked if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in allowed_untracked)]
    if unexpected_tracked or unexpected_untracked:
        raise RuntimeError({"unexpected_tracked": unexpected_tracked, "unexpected_untracked": unexpected_untracked})
    return {"allowed_protected_tracked": tracked_paths, "allowed_untracked_prefixes": list(allowed_untracked)}


def run_checks(expected: str) -> dict:
    contract = load_json(HERE / "experiment.json"); head = require_expected_head(expected); worktree = require_safe_worktree(contract)
    rows = MODEL.read_csv_gz(repo_root() / contract["pinned_inputs"][1]["path"]); schema = CONTRACTS.validate_rows(rows)
    train_profiles = {row["profile_id"] for row in rows if row["split_role"] == "expanded_train"}; validation_profiles = {row["profile_id"] for row in rows if row["split_role"] == "expanded_validation"}
    search = contract["search_contract"]
    checks = {
        "rows_7200": len(rows) == 7200,
        "train_profiles_960": len(train_profiles) == 960,
        "validation_profiles_240": len(validation_profiles) == 240,
        "six_rows_per_profile": all(sum(row["profile_id"] == profile for row in rows) == 6 for profile in train_profiles | validation_profiles),
        "fixed_draining": all(row["feature_pd_fixed_draining"] == "1" for row in rows),
        "runtime_budget": int(search["search_time_budget_seconds"]) == 32400 and int(search["hard_total_runtime_seconds"]) <= 37800,
        "candidate_budget": int(search["max_rounds"]) * (int(search["train_candidates_per_round"]) + int(search["blend_candidates_per_round"])) == int(search["max_total_candidates"]),
        "blind_not_opened": True,
        "complete_requests_not_opened": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {
        "status": "PASS", "workflow_commit": head, "pinned_inputs": verify_pinned_inputs(contract),
        "phase48_result_manifest": verify_manifest(repo_root() / contract["dataset_contract"]["source_result_dir"]),
        "phase58_result_manifest": verify_manifest(repo_root() / "experiment-results/phase58_pd_shape_aware_iterative_refinement"),
        "source_schema": schema, "counts": {"rows": len(rows), "train_profiles": len(train_profiles), "validation_profiles": len(validation_profiles)},
        "worktree": worktree, "checks": checks, "gpu_used": False, "network_used": False, "raw_accessed": False,
        "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); args = parser.parse_args()
    print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
