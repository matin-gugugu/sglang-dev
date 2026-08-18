#!/usr/bin/env python3
"""CPU-only Phase53 ancestry, source-result, manifest and conclusion audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    run_git,
    verify_pinned_inputs,
)
from report import load_source_summaries  # noqa: E402


def run_checks(expected: str) -> dict:
    spec = load_json(HERE / "experiment.json")
    head = require_expected_head(expected)
    require_clean_before_run(allowed_untracked_prefixes=())
    parents = subprocess.check_output(
        ["git", "show", "-s", "--format=%P", head],
        cwd=repo_root(),
        text=True,
    ).strip().split()
    required_ancestors = spec["required_workflow_ancestors"]
    if parents != [required_ancestors[-1]]:
        raise RuntimeError({"W53_fix_parent": parents, "expected_W53": required_ancestors[-1]})
    lineage_commits = [spec["workflow_base_result_commit"], *required_ancestors]
    lineage_audit = {
        commit: subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=repo_root(),
            check=False,
        ).returncode
        == 0
        for commit in lineage_commits
    }
    if not all(lineage_audit.values()):
        raise RuntimeError({"workflow_lineage": lineage_audit})
    output = repo_root() / spec["result_dir"]
    if output.exists():
        raise RuntimeError(f"formal result already exists: {output}")

    pins = verify_pinned_inputs(spec)
    summaries = load_source_summaries(repo_root(), spec)
    result_audits = []
    for item in spec["source_results"]:
        phase = item["phase"]
        summary_path = f"experiment-results/{item['directory']}/summary.json"
        latest_commit = run_git(["log", "-1", "--format=%H", "--", summary_path])
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", item["result_commit"], head],
            cwd=repo_root(),
            check=False,
        ).returncode == 0
        result_audits.append(
            {
                "phase": phase,
                "directory": item["directory"],
                "expected_result_commit": item["result_commit"],
                "latest_summary_commit": latest_commit,
                "commit_matches": latest_commit == item["result_commit"],
                "commit_is_ancestor": ancestor,
                "expected_status": item["expected_status"],
                "actual_status": summaries[phase].get("status"),
                "status_matches": summaries[phase].get("status") == item["expected_status"],
            }
        )
    if not all(
        row["commit_matches"] and row["commit_is_ancestor"] and row["status_matches"]
        for row in result_audits
    ):
        raise RuntimeError({"source_result_audits": result_audits})

    p34 = summaries["Phase34D"]
    p35 = summaries["Phase35"]
    p39 = summaries["Phase39"]
    p40 = summaries["Phase40"]
    p41 = summaries["Phase41"]
    p43 = summaries["Phase43"]
    p46 = summaries["Phase46"]
    p47 = summaries["Phase47"]
    p48 = summaries["Phase48"]
    p50 = summaries["Phase50"]
    p51 = summaries["Phase51"]
    p52 = summaries["Phase52"]
    scientific_checks = {
        "source_results_19": len(spec["source_results"]) == 19 and len(summaries) == 19,
        "source_manifests_19": sum("verify_manifest_directory" in item for item in spec["pinned_inputs"]) == 19,
        "phase34_target_isolated": p34["checks"].get("phase34c_archived_before_target_generation") is True,
        "phase34_six_model_rows": p34["counts"].get("blind_target_phase_rows") == 2592,
        "phase35_replay_exact": p35["replay_audit"].get("max_scalar_relative_difference") == 0.0,
        "phase39_physical_curves_12": p39["counts"].get("physical_curves") == 12,
        "phase39_communication_top1": p39.get("overall_top1_agreement") == 1.0,
        "phase40_teacher_exact": p40["checks"].get("all_requests_exact") is True
        and p40["checks"].get("histogram_calls_exact") is True
        and p40["checks"].get("histogram_bytes_exact") is True,
        "phase41_sentinel_exact": p41["gates"].get("GateB_GPU_SENTINEL") is True
        and p41["gpu_sentinel"].get("histogram_l1") == 0.0,
        "phase43_negative_preserved": p43["blind_metrics"].get("composite_ratio", 0.0) > 1.0,
        "phase46_qwen_confirmed": p46.get("scientific_outcome") == "CONFIRMS_H0_PROTECTED_IMPROVEMENT",
        "phase47_five_models": p47["counts"].get("models") == 5,
        "phase48_six_models": p48["counts"].get("models") == 6,
        "phase50_six_model_confirmed": p50.get("scientific_outcome") == "CONFIRMS_SIX_MODEL_H0_PROTECTED_IMPROVEMENT",
        "phase50_all_model_gate": p50["gates"].get("all_models_strict_four_metrics") is True,
        "phase51_physical_curves_18": p51["counts"].get("physical_curves") == 18,
        "phase52_cost_confirmed": p52["scientific_outcome"].get("cost") == "CONFIRMED",
        "phase52_placement_confirmed": p52["scientific_outcome"].get("placement") == "CONFIRMED",
    }
    if not all(scientific_checks.values()):
        raise RuntimeError({"scientific_checks": scientific_checks})
    return {
        "schema_version": "phase53-preflight-v1",
        "status": "PASS",
        "workflow_commit": head,
        "workflow_base_result_commit": spec["workflow_base_result_commit"],
        "workflow_parent_commits": parents,
        "workflow_lineage": lineage_audit,
        "pinned_inputs": pins,
        "source_result_audits": result_audits,
        "scientific_checks": scientific_checks,
        "gpu_used": False,
        "network_used": False,
        "training_used": False,
        "checkpoint_loaded": False,
        "prediction_recomputed": False,
        "teacher_recomputed": False,
        "physical_measurement_performed": False,
        "scheduler_simulation_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
