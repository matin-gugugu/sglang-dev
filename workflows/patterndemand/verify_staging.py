#!/usr/bin/env python3
"""执行Agent提交前检查暂存区仅含对应Phase结果目录。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, repo_root, validate_staged_allowlist, verify_result_manifest


PHASES = {
    "phase36": "experiment-results/phase36_cross_environment_replay/",
    "phase37": "experiment-results/phase37_pp_single_node_p2p_curve/",
    "phase38": "experiment-results/phase38_pp_physical_curve_cost_recompute/",
    "phase39": "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/",
    "phase40": "experiment-results/phase40_pure_pd_semantics_teacher/",
    "phase41": "experiment-results/phase41_pd_full_window_dataset/",
    "phase42": "experiment-results/phase42_pd_residual_training/",
    "phase43": "experiment-results/phase43_pd_blind_evaluation/",
    "phase44": "experiment-results/phase44_pd_expanded_protected_training/",
    "phase45": "experiment-results/phase45_pd_fresh_blind_prediction_freeze/",
    "phase46": "experiment-results/phase46_pd_fresh_blind_evaluation/",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    args = parser.parse_args()
    prefix = PHASES[args.phase]
    output = repo_root() / prefix
    staging = validate_staged_allowlist(prefix)
    blocked = (output / "BLOCKED.json").is_file()
    actual_result_paths = {
        str(path.relative_to(repo_root()))
        for path in output.rglob("*")
        if path.is_file()
    }
    staged_result_paths = set(staging["paths"])
    staging_completeness = {
        "ok": staged_result_paths == actual_result_paths,
        "missing_from_staging": sorted(actual_result_paths - staged_result_paths),
        "unexpected_in_staging": sorted(staged_result_paths - actual_result_paths),
    }
    result = {
        "staging": staging,
        "staging_completeness": staging_completeness,
        "blocked": blocked,
    }
    if not blocked:
        result["result_manifest"] = verify_result_manifest(output)
        summary = load_json(output / "summary.json")
        result["summary_status"] = summary["status"]
        if not result["result_manifest"]["ok"]:
            raise RuntimeError(result)
    elif not (output / "manifest.sha256").is_file():
        raise RuntimeError("BLOCKED结果也必须生成manifest.sha256")
    if not staging["ok"] or not staging_completeness["ok"]:
        raise RuntimeError(result)
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
