#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P54 = HERE.parent / "phase54_pd_histogram_accuracy_refinement"
sys.path.insert(0, str(P54)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_manifest, verify_pinned_inputs  # noqa: E402

spec = importlib.util.spec_from_file_location("phase58_p54_contracts", P54 / "contracts.py")
if spec is None or spec.loader is None: raise RuntimeError("cannot load Phase54 contracts")
contracts = importlib.util.module_from_spec(spec); spec.loader.exec_module(contracts)
spec = importlib.util.spec_from_file_location("phase58_p54_model", P54 / "model.py")
if spec is None or spec.loader is None: raise RuntimeError("cannot load Phase54 model")
model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)

def run_checks(expected: str) -> dict:
    contract = load_json(HERE / "experiment.json"); head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=("data/", "experiment-results/phase54_pd_histogram_accuracy_refinement/", "experiment-results/phase55_pd_adaptive_histogram_search/", "experiment-results/phase56_pd_structural_histogram_search/", "experiment-results/phase57_pd_iterative_histogram_optimization/"))
    rows = model.read_csv_gz(repo_root() / contract["pinned_inputs"][1]["path"]); schema = contracts.validate_rows(rows); result_manifest = verify_manifest(repo_root() / contract["dataset_contract"]["source_result_dir"])
    train_profiles = {r["profile_id"] for r in rows if r["split_role"] == "expanded_train"}; validation_profiles = {r["profile_id"] for r in rows if r["split_role"] == "expanded_validation"}; search = contract["search_contract"]
    checks = {"rows_7200": len(rows) == 7200, "train_profiles_960": len(train_profiles) == 960, "validation_profiles_240": len(validation_profiles) == 240, "six_rows_per_profile": all(sum(r["profile_id"] == p for r in rows) == 6 for p in train_profiles | validation_profiles), "fixed_draining": all(r["feature_pd_fixed_draining"] == "1" for r in rows), "budget": int(search["max_rounds"]) == 3 and int(search["seed_candidates_per_round"]) == 8 and int(search["adaptive_candidates_per_round"]) == 4 and int(search["max_total_candidates"]) == 36 and float(search["estimated_cpu_budget_hours"]) <= 8.0, "blind_not_opened": True, "complete_requests_not_opened": True}
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status": "PASS", "workflow_commit": head, "pinned_inputs": verify_pinned_inputs(contract), "phase48_result_manifest": result_manifest, "source_schema": schema, "counts": {"rows": len(rows), "train_profiles": len(train_profiles), "validation_profiles": len(validation_profiles)}, "checks": checks, "gpu_used": False, "network_used": False, "raw_accessed": False, "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": False}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); args = parser.parse_args(); print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
