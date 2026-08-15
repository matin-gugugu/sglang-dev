#!/usr/bin/env python3
"""Phase38运行前审计：只接受已验收合入的Phase37物理曲线。"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    run_git,
    sha256,
    verify_pinned_inputs,
    verify_result_manifest,
)


def validate_phase37_curves(contract: dict, payload: dict) -> dict:
    required = contract["required_phase37_curve_contract"]
    checks = {
        "schema_version": payload.get("schema_version") == required["schema_version"],
        "curve_evidence": payload.get("curve_evidence") == required["curve_evidence"],
        "metadata_overhead_excluded": payload.get("metadata_overhead_included") is required["metadata_overhead_included"],
        "direction_policy": payload.get("direction_policy") == required["direction_policy"],
        "nonempty_curves": isinstance(payload.get("curves"), list) and bool(payload.get("curves")),
    }
    curve_audits = []
    curve_ids = set()
    expected_payloads = [int(value) for value in required["required_payload_bytes"]]
    for curve in payload.get("curves", []):
        curve_id = curve.get("curve_id")
        knots = curve.get("knots", [])
        payloads = [int(knot.get("payload_bytes", -1)) for knot in knots]
        latencies = [float(knot.get("median_latency_us", float("nan"))) for knot in knots]
        audit = {
            "curve_id": curve_id,
            "unique_curve_id": bool(curve_id) and curve_id not in curve_ids,
            "topology_scope": curve.get("topology_scope") == required["topology_scope"],
            "measurement_scope": curve.get("measurement_scope") == required["measurement_scope"],
            "backend": curve.get("backend") == required["backend"],
            "direction_policy": curve.get("direction_policy") == required["direction_policy"],
            "interpolation": curve.get("interpolation") == required["interpolation"],
            "payload_grid": payloads == expected_payloads,
            "positive_finite_latencies": len(latencies) == len(expected_payloads)
            and all(math.isfinite(value) and value > 0 for value in latencies),
            "repeat_counts_present": len(knots) == len(expected_payloads)
            and all(int(knot.get("repeat_count", 0)) >= 5 for knot in knots),
            "topology_category_present": bool(curve.get("topology_category")),
            "physical_gpu_pair_present": len(curve.get("physical_gpu_pair", [])) == 2,
        }
        curve_ids.add(curve_id)
        audit["ok"] = all(value for key, value in audit.items() if key not in {"curve_id", "ok"})
        curve_audits.append(audit)
    checks["all_curve_contracts"] = bool(curve_audits) and all(audit["ok"] for audit in curve_audits)
    return {"ok": all(checks.values()), "checks": checks, "curves": curve_audits}


def verify_phase37_result_commit(summary: dict, result_commit: str) -> dict:
    command = [
        sys.executable,
        str(HERE.parent / "verify_result_commit.py"),
        "--phase",
        "phase37",
        "--workflow-commit",
        summary["workflow_commit"],
        "--result-commit",
        result_commit,
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_checks(expected_commit: str, output_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json")
    expected_output = (repo_root() / contract["result_dir"]).resolve()
    if output_dir != expected_output:
        raise RuntimeError(f"Phase38正式结果目录不可修改：expected={expected_output}, actual={output_dir}")
    head = require_expected_head(expected_commit)
    require_clean_before_run()
    if output_dir.exists():
        raise RuntimeError(f"正式结果目录已存在，拒绝覆盖：{output_dir}")
    pinned = verify_pinned_inputs(contract)

    phase37_dir = (repo_root() / contract["phase37_result_dir"]).resolve()
    if not phase37_dir.is_dir():
        raise RuntimeError("Phase37正式结果目录不存在；Phase38必须等待R37验收并ff-only合入")
    phase37_manifest = verify_result_manifest(phase37_dir)
    summary = load_json(phase37_dir / "summary.json")
    done = (phase37_dir / "DONE").read_text(encoding="utf-8").strip()
    status_ok = summary.get("status") in set(contract["accepted_phase37_statuses"])
    if not phase37_manifest["ok"] or not status_ok or done != summary.get("status"):
        raise RuntimeError({
            "phase37_manifest": phase37_manifest,
            "phase37_status": summary.get("status"),
            "phase37_done": done,
        })

    curve_path = (repo_root() / contract["phase37_curve_path"]).resolve()
    curve_payload = load_json(curve_path)
    curve_contract = validate_phase37_curves(contract, curve_payload)
    if not curve_contract["ok"]:
        raise RuntimeError({"phase37_curve_contract": curve_contract})

    result_commit = run_git(["log", "-1", "--format=%H", "--", contract["phase37_result_dir"]])
    if not result_commit:
        raise RuntimeError("找不到Phase37 result commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", result_commit, head],
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).returncode == 0
    result_commit_audit = verify_phase37_result_commit(summary, result_commit)
    if not ancestor or not result_commit_audit["ok"]:
        raise RuntimeError({
            "phase37_result_commit": result_commit,
            "ancestor_of_w38": ancestor,
            "result_commit_audit": result_commit_audit,
        })

    return {
        "status": "PASS",
        "workflow_commit": head,
        "output_dir": str(output_dir.relative_to(repo_root())),
        "pinned_inputs": pinned,
        "phase37": {
            "result_commit": result_commit,
            "result_commit_is_ancestor_of_w38": ancestor,
            "workflow_commit": summary["workflow_commit"],
            "status": summary["status"],
            "topology_categories": summary["topology_categories_measured"],
            "manifest": phase37_manifest,
            "result_commit_audit": result_commit_audit,
            "curve_path": contract["phase37_curve_path"],
            "curve_sha256": sha256(curve_path),
            "curve_contract": curve_contract,
            "curve_count": len(curve_payload["curves"]),
        },
        "curve_payload": curve_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase38_pp_physical_curve_cost_recompute",
    )
    args = parser.parse_args()
    print(run_checks(args.expected_workflow_commit, args.output_dir.resolve()))


if __name__ == "__main__":
    main()
