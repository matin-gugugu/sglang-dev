#!/usr/bin/env python3
"""在控制环境验证执行Agent回传的单个result commit。"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import repo_root, run_git


PHASE_PREFIX = {
    "phase36": "experiment-results/phase36_cross_environment_replay/",
    "phase37": "experiment-results/phase37_pp_single_node_p2p_curve/",
    "phase38": "experiment-results/phase38_pp_physical_curve_cost_recompute/",
    "phase39": "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/",
    "phase40": "experiment-results/phase40_pure_pd_semantics_teacher/",
    "phase41": "experiment-results/phase41_pd_full_window_dataset/",
    "phase42": "experiment-results/phase42_pd_residual_training/",
    "phase43": "experiment-results/phase43_pd_blind_evaluation/",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PHASE_PREFIX), required=True)
    parser.add_argument("--workflow-commit", required=True)
    parser.add_argument("--result-commit", required=True)
    args = parser.parse_args()

    workflow = run_git(["rev-parse", args.workflow_commit])
    result = run_git(["rev-parse", args.result_commit])
    parents = run_git(["show", "-s", "--format=%P", result]).split()
    if parents != [workflow]:
        raise RuntimeError(f"result commit必须只有一个父提交W：parents={parents}, W={workflow}")
    paths = run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", result]).splitlines()
    prefix = PHASE_PREFIX[args.phase]
    invalid = [path for path in paths if not path.startswith(prefix)]
    if not paths or invalid:
        raise RuntimeError(f"result commit路径越界：invalid={invalid}, paths={paths}")
    forbidden = [
        path
        for path in paths
        if Path(path).suffix.lower() in {".pid", ".pt", ".pth", ".ckpt", ".safetensors", ".jsonl"}
        or any(part.lower() in {"data", "raw_samples", "raw_trace", "cache"} for part in Path(path).parts)
    ]
    if forbidden:
        raise RuntimeError(f"result commit含禁止资产：{forbidden}")
    manifest = prefix + "manifest.sha256"
    done = prefix + "DONE"
    tree = set(run_git(["ls-tree", "-r", "--name-only", result]).splitlines())
    missing = [path for path in (manifest, done) if path not in tree]
    blocked = prefix + "BLOCKED.json" in tree
    if missing and not blocked:
        raise RuntimeError(f"正式结果缺少交付文件：{missing}")
    manifest_bytes = subprocess.check_output(["git", "show", f"{result}:{manifest}"], cwd=repo_root())
    manifest_entries = {}
    for line in manifest_bytes.decode("utf-8").splitlines():
        expected_sha, relative = line.split("  ", 1)
        manifest_entries[prefix + relative] = expected_sha
    result_files = {path for path in tree if path.startswith(prefix) and path != manifest}
    if set(manifest_entries) != result_files:
        raise RuntimeError({"manifest_missing_files": sorted(result_files - set(manifest_entries)), "manifest_extra_files": sorted(set(manifest_entries) - result_files)})
    mismatches = []
    for path, expected_sha in sorted(manifest_entries.items()):
        content = subprocess.check_output(["git", "show", f"{result}:{path}"], cwd=repo_root())
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            mismatches.append({"path": path, "expected": expected_sha, "actual": actual_sha})
    if mismatches:
        raise RuntimeError(f"result commit manifest校验失败：{mismatches}")
    print({"status": "PASS", "workflow_commit": workflow, "result_commit": result, "files": len(paths), "manifest_files": len(manifest_entries), "blocked_evidence": blocked})


if __name__ == "__main__":
    main()
