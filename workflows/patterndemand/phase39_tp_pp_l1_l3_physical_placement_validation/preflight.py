#!/usr/bin/env python3
"""Phase39 static lineage audit and pre-measurement topology freeze."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (
    environment_record,
    ensure_external_raw_dir,
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    run_git,
    verify_pinned_inputs,
    verify_result_manifest,
    write_json,
)
from contracts import canonical_sha, validate_plan


def source_semantics_audit() -> dict:
    root = repo_root()
    parallel_state = (root / "python/sglang/srt/distributed/parallel_state.py").read_text(encoding="utf-8")
    communication_op = (root / "python/sglang/srt/distributed/communication_op.py").read_text(encoding="utf-8")
    pp_profile = (root / "python/sglang/srt/distributed/pp_comm_profile.py").read_text(encoding="utf-8")
    checks = {
        "tp_frontend_uses_group_all_reduce": "return get_tp_group().all_reduce(input_)" in communication_op,
        "group_all_reduce_records_all_reduce": 'self._record_comm("all_reduce", input_, output_value=input_)' in parallel_state,
        "group_all_reduce_custom_path": "self.ca_comm.should_custom_ar(input_)" in parallel_state,
        "group_all_reduce_nccl_fallback": "torch.distributed.all_reduce(input_, group=self.device_group)" in parallel_state,
        "pp_histogram_sender_only": "Only tensor sends are counted" in pp_profile,
        "pp_histogram_raw_op": '"raw_op": "p2p_send_tensor"' in pp_profile,
    }
    if not all(checks.values()):
        raise RuntimeError({"production_source_semantics": checks})
    return checks


def static_checks(expected_commit: str) -> dict:
    spec = load_json(HERE / "experiment.json")
    head = require_expected_head(expected_commit)
    require_clean_before_run()
    parents = run_git(["show", "-s", "--format=%P", head]).split()
    if parents != [spec["workflow_parent_result_commit"]]:
        raise RuntimeError({"w39_parent_mismatch": {"parents": parents, "required": spec["workflow_parent_result_commit"]}})
    pinned = verify_pinned_inputs(spec)
    phase38_dir = repo_root() / "experiment-results/phase38_pp_physical_curve_cost_recompute"
    phase38_manifest = verify_result_manifest(phase38_dir)
    phase38_summary = load_json(phase38_dir / "summary.json")
    phase38_commit = run_git(["log", "-1", "--format=%H", "--", str(phase38_dir.relative_to(repo_root()))])
    phase38_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", phase38_commit, head],
        cwd=repo_root(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).returncode == 0
    phase38_ok = (
        phase38_manifest["ok"]
        and phase38_summary.get("status") == "PASS"
        and phase38_commit == spec["workflow_parent_result_commit"]
        and phase38_ancestor
    )
    if not phase38_ok:
        raise RuntimeError({
            "phase38_lineage": {
                "manifest": phase38_manifest,
                "summary_status": phase38_summary.get("status"),
                "result_commit": phase38_commit,
                "required_commit": spec["workflow_parent_result_commit"],
            }
        })
    matrix = spec["required_measurement_matrix"]
    matrix_ok = (
        len(matrix) == int(spec["expected_measurement_cases"])
        and len({row["case_key"] for row in matrix}) == len(matrix)
        and {row["topology_level"] for row in matrix} == {"L1", "L2", "L3"}
        and {row["world_size"] for row in matrix if row["parallelism"] == "tp"} == {2, 4, 8}
        and len([row for row in matrix if row["parallelism"] == "pp"]) == 3
    )
    if not matrix_ok:
        raise RuntimeError("Phase39 required measurement matrix contract is inconsistent")
    return {
        "status": "PASS",
        "workflow_commit": head,
        "workflow_parent_result_commit": parents[0],
        "pinned_inputs": pinned,
        "phase38": {
            "result_commit": phase38_commit,
            "summary_status": phase38_summary["status"],
            "manifest": phase38_manifest,
        },
        "source_semantics": source_semantics_audit(),
        "matrix_cases": len(matrix),
        "expected_measurement_shards": int(spec["expected_measurement_shards"]),
    }


def full_checks(expected_commit: str, topology_plan: Path, raw_dir: Path, output_dir: Path) -> dict:
    result = static_checks(expected_commit)
    spec = load_json(HERE / "experiment.json")
    expected_output = (repo_root() / spec["result_dir"]).resolve()
    if output_dir != expected_output:
        raise RuntimeError(f"Phase39正式结果目录不可修改：expected={expected_output}, actual={output_dir}")
    if output_dir.exists():
        raise RuntimeError(f"正式结果目录已存在，拒绝覆盖：{output_dir}")
    plan_path = topology_plan.expanduser().resolve()
    root = repo_root().resolve()
    if plan_path == root or root in plan_path.parents:
        raise RuntimeError("topology plan必须在Git仓库外冻结")
    plan = load_json(plan_path)
    plan_audit = validate_plan(plan, spec)
    if plan.get("workflow_commit") != result["workflow_commit"]:
        raise RuntimeError({"topology_plan_workflow_commit": plan.get("workflow_commit"), "expected": result["workflow_commit"]})
    raw = ensure_external_raw_dir(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-smi") is None or shutil.which("torchrun") is None:
        raise RuntimeError("Phase39 GPU环境缺少nvidia-smi或torchrun")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Phase39 GPU环境缺少torch") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("Phase39完整矩阵包含TP8 L1；单节点preflight至少需要8张可见CUDA GPU")
    result.update({
        "topology_plan_path": str(plan_path),
        "topology_plan_sha256": plan["plan_sha256"],
        "topology_plan_canonical_sha256": canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"}),
        "topology_plan_audit": plan_audit,
        "topology_plan_generated_at_utc": plan["generated_at_utc"],
        "external_raw_dir": str(raw),
        "output_dir": str(output_dir.relative_to(repo_root())),
        "environment": environment_record(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "nvidia_smi_list": subprocess.check_output(["nvidia-smi", "-L"], text=True),
        "nccl_environment_keys": sorted(key for key in os.environ if key.startswith("NCCL_")),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--topology-plan", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation")
    args = parser.parse_args()
    if args.static_only:
        result = static_checks(args.expected_workflow_commit)
    else:
        if not args.topology_plan or not args.raw_dir or not args.audit_output:
            parser.error("full preflight requires --topology-plan, --raw-dir and --audit-output")
        audit_output = args.audit_output.expanduser().resolve()
        root = repo_root().resolve()
        if audit_output == root or root in audit_output.parents:
            raise RuntimeError("preflight audit必须保存在Git仓库外")
        if audit_output.exists():
            raise RuntimeError(f"拒绝覆盖已有preflight audit：{audit_output}")
        result = full_checks(args.expected_workflow_commit, args.topology_plan, args.raw_dir, args.output_dir.resolve())
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(audit_output, result)
        result = {**result, "audit_output": str(audit_output)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
