#!/usr/bin/env python3
"""Phase36结果收口、中文说明与manifest生成。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json


def finalize(output_dir: Path) -> dict:
    state = load_json(output_dir / "audit/runtime_state.json")
    checks = state["checks"]
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase36-cross-environment-replay-result-v1",
        "status": status,
        "completed_at_utc": utc_now(),
        "objective": "另一GPU环境对Phase34冻结六模型TP/PP直方图做target-free复播和Git回传演练",
        "workflow_commit": state["workflow_commit"],
        "counts": state["counts"],
        "replay_audit": state["replay_audit"],
        "runtime_audit": state["runtime_audit"],
        "input_audit": state["input_audit"],
        "checks": checks,
        "evidence_boundary": {
            "training": "none",
            "teacher_target": "not_read",
            "claim": "跨环境复播和结果回传工程证据，不是新模型精度盲测"
        }
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# Phase36：跨环境冻结推理与结果commit回传演练\n\n"
        f"最终状态：`{status}`。本阶段没有训练、没有读取teacher或target，也没有修改Phase34 checkpoint。\n\n"
        f"共复播{state['counts']['prediction_rows']:,}条六模型TP/PP phase预测；与Phase34冻结预测比较的最大相对差为"
        f"`{state['replay_audit']['max_scalar_relative_difference']:.3e}`，合同容差为`1e-6`。\n\n"
        "该结果只能证明另一环境能够按冻结输入复播并按统一目录回传结果，不能作为新的预测精度盲测。"
        "GPU Agent必须只提交本目录；模型权重、data、raw、缓存和PID不得进入Git。\n",
        encoding="utf-8",
    )
    tree = validate_result_tree(output_dir)
    if not tree["ok"]:
        raise RuntimeError(f"结果树含禁止资产：{tree['violations']}")
    (output_dir / "DONE").write_text(status + "\n", encoding="utf-8")
    refresh_manifest(output_dir)
    if status != "PASS":
        raise RuntimeError(f"Phase36未通过：{checks}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase36_cross_environment_replay")
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
